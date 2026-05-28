"""
Flow Trajectory Analysis

Three-phase analysis of a trained DiT model:
  Phase 1: CFG Scale Sweep (8-step Euler, 256 val samples + 8-sample visual grids)
  Phase 2: Step Reduction Comparison
  Phase 3: Trajectory Straightness (4-step and 8-step)

Usage:
    python scripts/analysis/flow_trajectory_analysis.py --config flowdit.yaml --output results/trajectory_analysis/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import csv
import textwrap
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.flowdit import DiTAir, DiT, UViT, MMDiT
from models.theia_decoder import Decoder as TheiaDecoder
from utils.datasets import MemoryMappedDataset, mmap_collate_fn

MODEL_CLASSES = {
    "cross_attn": DiT,
    "uvit": UViT,
    "full_mmdit": MMDiT,
    "dit_air": DiTAir,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='flowdit.yaml')
    parser.add_argument('--output', type=str, default='results/trajectory_analysis/')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using {torch.cuda.get_device_name(args.gpu)}")

    # Load config
    config_path = os.path.join('config', args.config)
    params = yaml.safe_load(open(config_path, 'r'))

    # ========================================================================
    # Setup: model, data, decoder (matching train_flowdit.py)
    # ========================================================================

    dataset_path = params["dataset_path"]
    vision_model = params["vision_model"]
    text_model = params["text_model"]
    scale_factor = params.get("scale_factor", 1.0)
    B = params["batch_size"]

    # Dataset
    test_dataset = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, split='test')

    # 8 vis samples (same indices as train_flowdit.py)
    vis_indices = [2, 7, 10, 16, 17, 18, 19, 20]
    vis_batch = mmap_collate_fn([test_dataset[i] for i in vis_indices])
    x0_224, z0_vis, xt_224, zt_vis, c_txt, c_hidden_vis, c_mask_vis, c_pooled_vis = vis_batch
    n_vis = len(vis_indices)

    print(f"Vis samples: {n_vis} (indices={vis_indices})")

    # Get text dims from data
    sample_c_hidden = np.load(f"{dataset_path}/labels_hidden_{text_model}.npy", mmap_mode='r')
    text_dim = sample_c_hidden.shape[2]
    max_text_len = sample_c_hidden.shape[1]
    del sample_c_hidden
    sample_c_pooled = np.load(f"{dataset_path}/labels_pooled_{text_model}.npy", mmap_mode='r')
    pooled_text_dim = sample_c_pooled.shape[1]
    del sample_c_pooled

    # Model
    mp = params["model_params"]
    model_type = mp.get("model_type", "cross_attn")
    ModelClass = MODEL_CLASSES[model_type]

    model = ModelClass(
        latent_dim=mp["latent_dim"],
        num_patches=mp["num_patches"],
        hidden_dim=mp["hidden_dim"],
        depth=mp["depth"],
        num_heads=mp["num_heads"],
        text_dim=text_dim,
        pooled_text_dim=pooled_text_dim,
        max_text_len=max_text_len,
        mlp_ratio=mp.get("mlp_ratio", 4.0),
        dropout=mp.get("dropout", 0.0),
        cfg_drop_prompt=mp.get("cfg_drop_prompt", 0.05),
        cfg_drop_context=mp.get("cfg_drop_context", 0.05),
        cfg_drop_both=mp.get("cfg_drop_both", 0.05),
        use_pooled_text=mp.get("use_pooled_text", True),
    ).to(device)

    checkpoint = torch.load(params["model_path"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    ckpt_epoch = checkpoint.get('epoch', '?')
    ckpt_loss = checkpoint.get('test_loss', '?')
    del checkpoint
    print(f"Loaded checkpoint: epoch {ckpt_epoch}, val loss {ckpt_loss}")

    # Theia decoder
    theia_decoder = None
    if "theia_decoder" in params and os.path.exists(params["theia_decoder"]["model_path"]):
        theia_decoder = TheiaDecoder(**params["theia_decoder"]["model_params"])
        dec_ckpt = torch.load(params["theia_decoder"]["model_path"], map_location=device, weights_only=True)
        theia_decoder.load_state_dict(dec_ckpt['model'])
        theia_decoder = theia_decoder.to(device)
        theia_decoder.eval()
        for p in theia_decoder.parameters():
            p.requires_grad = False
        print("Theia decoder loaded")

    def decode(z):
        with torch.no_grad():
            return theia_decoder(z)

    # ========================================================================
    # Prepare vis data + pre-compute text projections
    # ========================================================================
    z_init_vis = z0_vis.to(device) * scale_factor
    z_target_vis = zt_vis.to(device) * scale_factor

    print("Pre-computing text projections (vis set)...")
    with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
        c_h = c_hidden_vis.to(device)
        c_m = c_mask_vis.to(device)
        c_p = c_pooled_vis.to(device)
        text_ctx_vis = model.text_proj(c_h) * c_m.unsqueeze(-1)
        null_text_ctx_vis = model.null_text_emb.expand(n_vis, text_ctx_vis.shape[1], -1)
        pooled_cond_vis = model._get_pooled_conditioning(c_p, use_null=False)
        null_pooled_cond_vis = model._get_pooled_conditioning(c_p, use_null=True)

    del c_h, c_m, c_p
    torch.cuda.empty_cache()

    # Shared noise for vis samples
    shared_noise_vis = torch.randn(n_vis, mp["num_patches"], mp["latent_dim"])

    def euler_generate_vis(context_cfg, prompt_cfg, num_steps=8, noise=None):
        """Euler on vis set (8 samples). Returns [n_vis, N, D]."""
        z = (noise if noise is not None else shared_noise_vis).to(device)
        dt = 1.0 / num_steps
        with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
            for step_i in range(num_steps):
                t = torch.ones(n_vis, device=device) * (1.0 - step_i * dt)
                v_uncond = model._forward_with_ctx(z, t, z_init_vis, null_text_ctx_vis, pooled_cond=null_pooled_cond_vis, use_null_context=True)
                v_context = model._forward_with_ctx(z, t, z_init_vis, null_text_ctx_vis, pooled_cond=null_pooled_cond_vis, use_null_context=False)
                v_full = model._forward_with_ctx(z, t, z_init_vis, text_ctx_vis, pooled_cond=pooled_cond_vis, use_null_context=False)
                v = v_uncond + context_cfg * (v_context - v_uncond) + prompt_cfg * (v_full - v_context)
                z = z - v * dt
        return z.float()

    # ========================================================================
    # Phase 1: CFG Scale Sweep (8-step Euler)
    # ========================================================================

    print("\n" + "=" * 60)
    print("Phase 1: CFG Scale Sweep (8-step Euler)")
    print("=" * 60)

    context_cfgs = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    prompt_cfgs = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]

    n_ctx = len(context_cfgs)
    n_pmt = len(prompt_cfgs)

    cfg_results = []
    z_vis_all = []  # for visual grids
    cos_vis_all = []
    mse_vis_all = []
    best_cos_sim = -1.0
    best_cfg = (1.0, 1.0)

    t0 = time.time()
    pbar = tqdm(total=n_ctx * n_pmt, desc="CFG sweep")
    for ctx_cfg in context_cfgs:
        for pmt_cfg in prompt_cfgs:
            z_vis = euler_generate_vis(ctx_cfg, pmt_cfg, num_steps=8)
            cos_sim = F.cosine_similarity(z_vis.flatten(1), z_target_vis.flatten(1), dim=1).mean().item()
            mse = F.mse_loss(z_vis, z_target_vis, reduction='none').mean(dim=[1, 2]).mean().item()

            cfg_results.append({
                'context_cfg': ctx_cfg,
                'prompt_cfg': pmt_cfg,
                'cos_sim': cos_sim,
                'mse': mse,
            })

            if cos_sim > best_cos_sim:
                best_cos_sim = cos_sim
                best_cfg = (ctx_cfg, pmt_cfg)

            cos_vis = F.cosine_similarity(z_vis.flatten(1), z_target_vis.flatten(1), dim=1)
            mse_vis = F.mse_loss(z_vis, z_target_vis, reduction='none').mean(dim=[1, 2])
            z_vis_all.append(z_vis)
            cos_vis_all.append(cos_vis)
            mse_vis_all.append(mse_vis)

            pbar.update(1)
            pbar.set_postfix(best=f"ctx={best_cfg[0]:.2f},pmt={best_cfg[1]:.1f},cos={best_cos_sim:.4f}")
    pbar.close()
    print(f"Phase 1: {time.time() - t0:.1f}s")

    # Save CSV
    csv_path = os.path.join(args.output, 'cfg_sweep.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['context_cfg', 'prompt_cfg', 'cos_sim', 'mse'])
        writer.writeheader()
        writer.writerows(cfg_results)
    print(f"Saved {csv_path}")

    # Top 5
    sorted_results = sorted(cfg_results, key=lambda x: x['cos_sim'], reverse=True)
    print("\nTop 5 CFG combinations by cos_sim:")
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i+1}. context={r['context_cfg']:.1f}, prompt={r['prompt_cfg']:.1f} -> cos_sim={r['cos_sim']:.4f}, mse={r['mse']:.4f}")

    # Heatmaps
    cos_sim_grid = np.array([r['cos_sim'] for r in cfg_results]).reshape(n_ctx, n_pmt)
    mse_grid = np.array([r['mse'] for r in cfg_results]).reshape(n_ctx, n_pmt)
    prompt_labels = [f"{v:.1f}" for v in prompt_cfgs]
    context_labels = [f"{v:.1f}" for v in context_cfgs]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cos_sim_grid, annot=True, fmt='.4f', xticklabels=prompt_labels,
                yticklabels=context_labels, cmap='viridis', ax=ax, annot_kws={'size': 8})
    ax.set_xlabel('Prompt CFG')
    ax.set_ylabel('Context CFG')
    ax.set_title(f'Cosine Similarity vs GT (8-step Euler, {n_vis} vis samples)')
    best_idx = np.unravel_index(cos_sim_grid.argmax(), cos_sim_grid.shape)
    ax.add_patch(plt.Rectangle((best_idx[1], best_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
    plt.tight_layout()
    fig.savefig(os.path.join(args.output, 'cfg_sweep_cos_sim.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mse_grid, annot=True, fmt='.4f', xticklabels=prompt_labels,
                yticklabels=context_labels, cmap='viridis_r', ax=ax, annot_kws={'size': 8})
    ax.set_xlabel('Prompt CFG')
    ax.set_ylabel('Context CFG')
    ax.set_title(f'MSE vs GT (8-step Euler, {n_vis} vis samples)')
    best_mse_idx = np.unravel_index(mse_grid.argmin(), mse_grid.shape)
    ax.add_patch(plt.Rectangle((best_mse_idx[1], best_mse_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
    plt.tight_layout()
    fig.savefig(os.path.join(args.output, 'cfg_sweep_mse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved cfg_sweep_cos_sim.png and cfg_sweep_mse.png")

    # Per-sample visual grids (matching train_flowdit layout)
    if theia_decoder is not None:
        print("Generating per-sample CFG grids...")
        cfg_grid_dir = os.path.join(args.output, 'cfg_grids')
        os.makedirs(cfg_grid_dir, exist_ok=True)

        for i in tqdm(range(n_vis), desc="Decoding grids"):
            n_rows = n_ctx + 1
            n_cols = n_pmt + 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

            wrapped_text = '\n'.join(textwrap.wrap(c_txt[i], width=60))
            fig.suptitle(wrapped_text, fontsize=10, fontweight='bold', y=1.02)

            x0_img = np.clip(x0_224[i].permute(1, 2, 0).numpy(), 0, 1)
            xt_img = np.clip(xt_224[i].permute(1, 2, 0).numpy(), 0, 1)

            axes[0, 0].imshow(x0_img)
            axes[0, 0].set_title('Initial', fontsize=8, fontweight='bold')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(xt_img)
            axes[0, 1].set_title('Target', fontsize=8, fontweight='bold')
            axes[0, 1].axis('off')

            for pc_idx in range(2, n_cols):
                axes[0, pc_idx].set_title(f'P={prompt_cfgs[pc_idx - 1]:.1f}', fontsize=8, fontweight='bold')
                axes[0, pc_idx].axis('off')

            for cc_idx, ctx_cfg in enumerate(context_cfgs):
                row = cc_idx + 1
                axes[row, 0].text(0.5, 0.5, f'C={ctx_cfg:.1f}', ha='center', va='center',
                                  fontsize=9, fontweight='bold', transform=axes[row, 0].transAxes)
                axes[row, 0].axis('off')

                for pc_idx, pmt_cfg in enumerate(prompt_cfgs):
                    col = pc_idx + 1
                    flat_idx = cc_idx * n_pmt + pc_idx
                    z_gen = z_vis_all[flat_idx]
                    cos_sim = cos_vis_all[flat_idx][i].item()
                    mse_err = mse_vis_all[flat_idx][i].item()

                    with torch.autocast(str(device), dtype=torch.bfloat16):
                        xg_recon = decode(z_gen[i:i+1] / scale_factor)
                    img = xg_recon[0].float().cpu().permute(1, 2, 0).numpy()
                    axes[row, col].imshow(np.clip(img, 0, 1))
                    axes[row, col].set_title(f'cos={cos_sim:.3f}\nMSE={mse_err:.3f}', fontsize=7)
                    axes[row, col].axis('off')

            plt.tight_layout()
            fig.savefig(os.path.join(cfg_grid_dir, f'sample_{i}.png'), dpi=200, bbox_inches='tight')
            plt.close()

        print(f"Saved {n_vis} grids to {cfg_grid_dir}/")

    del z_vis_all, cos_vis_all, mse_vis_all
    torch.cuda.empty_cache()

    best_context_cfg_sweep, best_prompt_cfg_sweep = best_cfg
    print(f"\nSweep best CFG: context={best_context_cfg_sweep:.2f}, prompt={best_prompt_cfg_sweep:.1f} (cos_sim={best_cos_sim:.4f})")

    # Hardcode trajectory CFGs regardless of sweep results
    best_context_cfg = 2.5
    best_prompt_cfg = 6.0
    print(f"Using hardcoded CFG for Phase 2+3: context={best_context_cfg}, prompt={best_prompt_cfg}")

    # ========================================================================
    # Phase 2: Step Reduction Comparison
    # ========================================================================

    print("\n" + "=" * 60)
    print("Phase 2: Step Reduction Comparison")
    print("=" * 60)

    step_counts = [50, 25, 10, 8, 4, 2, 1]

    step_results = {}
    for steps in tqdm(step_counts, desc="Step sweep"):
        z_gen = euler_generate_vis(best_context_cfg, best_prompt_cfg, num_steps=steps)
        step_results[steps] = z_gen

    z_50step = step_results[50]
    step_csv_rows = []
    for steps in step_counts:
        z_gen = step_results[steps]
        cos_vs_gt = F.cosine_similarity(z_gen.flatten(1), z_target_vis.flatten(1), dim=1).mean().item()
        mse_vs_gt = F.mse_loss(z_gen, z_target_vis, reduction='none').mean(dim=[1, 2]).mean().item()
        cos_vs_50 = F.cosine_similarity(z_gen.flatten(1), z_50step.flatten(1), dim=1).mean().item()
        mse_vs_50 = F.mse_loss(z_gen, z_50step, reduction='none').mean(dim=[1, 2]).mean().item()
        step_csv_rows.append({
            'steps': steps,
            'cos_sim_vs_gt': cos_vs_gt,
            'mse_vs_gt': mse_vs_gt,
            'cos_sim_vs_50step': cos_vs_50,
            'mse_vs_50step': mse_vs_50,
        })
        print(f"  {steps:2d} steps: cos_gt={cos_vs_gt:.4f}  mse_gt={mse_vs_gt:.4f}  cos_50={cos_vs_50:.4f}  mse_50={mse_vs_50:.4f}")

    csv_path = os.path.join(args.output, 'step_reduction.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['steps', 'cos_sim_vs_gt', 'mse_vs_gt', 'cos_sim_vs_50step', 'mse_vs_50step'])
        writer.writeheader()
        writer.writerows(step_csv_rows)
    print(f"Saved {csv_path}")

    # Dual y-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    steps_arr = [r['steps'] for r in step_csv_rows]
    l1, = ax1.plot(steps_arr, [r['cos_sim_vs_gt'] for r in step_csv_rows], 'b-o', label='cos_sim vs GT')
    l2, = ax1.plot(steps_arr, [r['cos_sim_vs_50step'] for r in step_csv_rows], 'b--s', label='cos_sim vs 50-step')
    l3, = ax2.plot(steps_arr, [r['mse_vs_gt'] for r in step_csv_rows], 'r-o', label='MSE vs GT')
    l4, = ax2.plot(steps_arr, [r['mse_vs_50step'] for r in step_csv_rows], 'r--s', label='MSE vs 50-step')
    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('Cosine Similarity', color='b')
    ax2.set_ylabel('MSE', color='r')
    ax1.set_xticks(steps_arr)
    ax1.legend(handles=[l1, l2, l3, l4], loc='center right')
    ax1.set_title(f'Step Reduction (CFG: ctx={best_context_cfg:.1f}, pmt={best_prompt_cfg:.1f})')
    plt.tight_layout()
    fig.savefig(os.path.join(args.output, 'step_reduction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved step_reduction.png")

    # Decoded image grid
    if theia_decoder is not None:
        print("Generating decoded step grid...")
        n_cols = len(step_counts) + 2
        fig, axes = plt.subplots(n_vis, n_cols, figsize=(n_cols * 2.2, n_vis * 2.2))
        col_labels = ['Source'] + [f'{s} steps' for s in step_counts] + ['Target']
        for col, label in enumerate(col_labels):
            axes[0, col].set_title(label, fontsize=9, fontweight='bold')

        for row in range(n_vis):
            axes[row, 0].imshow(np.clip(x0_224[row].permute(1, 2, 0).numpy(), 0, 1))
            axes[row, 0].axis('off')
            for col_offset, steps in enumerate(step_counts):
                with torch.autocast(str(device), dtype=torch.bfloat16):
                    img = decode(step_results[steps][row:row+1] / scale_factor)
                axes[row, col_offset + 1].imshow(np.clip(img[0].float().cpu().permute(1, 2, 0).numpy(), 0, 1))
                axes[row, col_offset + 1].axis('off')
            axes[row, n_cols - 1].imshow(np.clip(xt_224[row].permute(1, 2, 0).numpy(), 0, 1))
            axes[row, n_cols - 1].axis('off')

        plt.tight_layout()
        fig.savefig(os.path.join(args.output, 'step_reduction_decoded.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved step_reduction_decoded.png")

    del step_results
    torch.cuda.empty_cache()

    # ========================================================================
    # Phase 3: Trajectory Straightness (4-step and 8-step)
    # ========================================================================

    print("\n" + "=" * 60)
    print("Phase 3: Trajectory Straightness")
    print("=" * 60)

    for num_steps_traj in [4, 8]:
        print(f"\n--- {num_steps_traj}-step trajectory ---")
        num_intermediates = num_steps_traj + 1

        z = shared_noise_vis.to(device)
        intermediates = [z.clone().float()]
        dt = 1.0 / num_steps_traj

        with torch.no_grad(), torch.autocast(str(device), dtype=torch.bfloat16):
            for step_i in range(num_steps_traj):
                t = torch.ones(n_vis, device=device) * (1.0 - step_i * dt)
                v_uncond = model._forward_with_ctx(z, t, z_init_vis, null_text_ctx_vis, pooled_cond=null_pooled_cond_vis, use_null_context=True)
                v_context = model._forward_with_ctx(z, t, z_init_vis, null_text_ctx_vis, pooled_cond=null_pooled_cond_vis, use_null_context=False)
                v_full = model._forward_with_ctx(z, t, z_init_vis, text_ctx_vis, pooled_cond=pooled_cond_vis, use_null_context=False)
                v = v_uncond + best_context_cfg * (v_context - v_uncond) + best_prompt_cfg * (v_full - v_context)
                z = z - v * dt
                intermediates.append(z.clone().float())

        intermediates = torch.stack(intermediates, dim=0)  # [steps+1, n_vis, N, D]
        z_start = intermediates[0]
        z_end = intermediates[-1]
        total_disp = (z_end - z_start).flatten(1).norm(dim=1)

        deviations = torch.zeros(n_vis, num_intermediates, device=device)
        for step_idx in range(num_intermediates):
            frac = step_idx / float(num_steps_traj)
            z_interp = z_start + frac * (z_end - z_start)
            deviation = (intermediates[step_idx] - z_interp).flatten(1).norm(dim=1)
            deviations[:, step_idx] = deviation

        max_deviation = deviations.max(dim=1).values
        straightness = max_deviation / (total_disp + 1e-8)

        straightness_np = straightness.cpu().numpy()
        max_dev_np = max_deviation.cpu().numpy()
        total_disp_np = total_disp.cpu().numpy()

        csv_path = os.path.join(args.output, f'straightness_{num_steps_traj}step.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_idx', 'straightness', 'max_deviation', 'total_displacement'])
            writer.writeheader()
            for i in range(n_vis):
                writer.writerow({
                    'sample_idx': i,
                    'straightness': float(straightness_np[i]),
                    'max_deviation': float(max_dev_np[i]),
                    'total_displacement': float(total_disp_np[i]),
                })
        print(f"Saved {csv_path}")
        print(f"Straightness: mean={straightness_np.mean():.4f}, median={np.median(straightness_np):.4f}, std={straightness_np.std():.4f}")

        # Trajectory deviation plot
        t_values = np.linspace(1.0, 0.0, num_intermediates)
        deviations_np = deviations.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in range(n_vis):
            ax.plot(t_values, deviations_np[idx], alpha=0.4, color='steelblue', linewidth=1)
        ax.plot(t_values, deviations_np.mean(axis=0), color='darkred', linewidth=2.5, label='Mean')
        ax.set_xlabel('t (1 → 0)')
        ax.set_ylabel('L2 Deviation from Straight Line')
        ax.set_title(f'Trajectory Deviation ({num_steps_traj}-step Euler)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(args.output, f'trajectory_deviation_{num_steps_traj}step.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory_deviation_{num_steps_traj}step.png")

    # ========================================================================
    # Final Summary
    # ========================================================================

    step_50_cos = next(r['cos_sim_vs_gt'] for r in step_csv_rows if r['steps'] == 50)
    step_8_cos_gt = next(r['cos_sim_vs_gt'] for r in step_csv_rows if r['steps'] == 8)
    step_4_cos_gt = next(r['cos_sim_vs_gt'] for r in step_csv_rows if r['steps'] == 4)

    print("\n" + "=" * 60)
    print("=== Trajectory Analysis Summary ===")
    print("=" * 60)
    print(f"Sweep best CFG: context={best_context_cfg_sweep:.2f}, prompt={best_prompt_cfg_sweep:.1f} (cos_sim={best_cos_sim:.4f}, {n_vis} vis samples)")
    print(f"Hardcoded CFG for Phase 2+3: context={best_context_cfg}, prompt={best_prompt_cfg}")
    print()
    print("Step Reduction (vis samples):")
    print(f"  50-step cos_sim vs GT: {step_50_cos:.4f}")
    print(f"   8-step cos_sim vs GT: {step_8_cos_gt:.4f}")
    print(f"   4-step cos_sim vs GT: {step_4_cos_gt:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

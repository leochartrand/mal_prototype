"""
Per-timestep flow matching loss evaluation.

Evaluates trained DiT checkpoints on the full validation set across a
uniform grid of fixed timesteps.  Produces a CSV + line plot showing where
each model is strongest/weakest along the ODE trajectory.

Usage:
    python scripts/analysis/eval_timestep_loss.py \
        --checkpoints models/round4/uniform/model.pt models/round4/rae_shift/model.pt models/round4/logit_normal/model.pt \
        --labels uniform rae_shift logit_normal \
        --params round4/rae_shift.yaml \
        --num_bins 20 \
        --output results/timestep_loss_analysis \
        --gpu 0

To redraw plots from existing CSV (no GPU needed):
    python scripts/analysis/eval_timestep_loss.py --redraw results/timestep_loss_analysis
"""

import argparse
import csv
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add src/ to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.flowdit import DiT, UViT, MMDiT, DiTAir
from utils.datasets import MemoryMappedDataset, mmap_collate_fn
from utils.checkpoint import load_checkpoint

MODEL_CLASSES = {
    "cross_attn": DiT,
    "uvit": UViT,
    "full_mmdit": MMDiT,
    "dit_air": DiTAir,
}


# ---------------------------------------------------------------------------
# Fixed-timestep loss (matches flow_matching_loss in flowdit.py exactly)
# ---------------------------------------------------------------------------
def fixed_timestep_loss(
    model,
    z_init: torch.Tensor,
    z_target: torch.Tensor,
    text_emb: torch.Tensor,
    t_fixed: float,
    text_mask: Optional[torch.Tensor] = None,
    pooled_text_emb: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """MSE loss at a single fixed timestep (same maths as flow_matching_loss)."""
    B = z_target.shape[0]
    device = z_target.device

    t = torch.full((B,), t_fixed, device=device, dtype=z_target.dtype)
    z_noise = torch.randn_like(z_target)

    t_exp = t.view(B, 1, 1)
    z_t = (1 - t_exp) * z_target + (eps + (1 - eps) * t_exp) * z_noise

    target_v = (1 - eps) * z_noise - z_target

    v_pred = model(z_t, t, z_init, text_emb,
                   text_mask=text_mask, pooled_text_emb=pooled_text_emb)

    return F.mse_loss(v_pred, target_v)


# ---------------------------------------------------------------------------
# Build model from YAML (same logic as train_flowdit.py)
# ---------------------------------------------------------------------------
def build_model(params, text_dim, pooled_text_dim, max_text_len, device):
    mp = params["model_params"]
    model_type = mp.get("model_type", "cross_attn")
    ModelClass = MODEL_CLASSES[model_type]

    cfg_drop_prompt = mp.get("cfg_drop_prompt", 0.05)
    cfg_drop_context = mp.get("cfg_drop_context", 0.05)
    cfg_drop_both = mp.get("cfg_drop_both", 0.05)
    cond_drop_prob = mp.get("cond_drop_prob", None)
    if cond_drop_prob is not None and "cfg_drop_prompt" not in mp:
        context_drop_prob = mp.get("context_drop_prob", 0.0)
        cfg_drop_prompt = cond_drop_prob
        cfg_drop_context = context_drop_prob
        cfg_drop_both = cfg_drop_prompt * cfg_drop_context
        cfg_drop_prompt -= cfg_drop_both
        cfg_drop_context -= cfg_drop_both

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
        cfg_drop_prompt=cfg_drop_prompt,
        cfg_drop_context=cfg_drop_context,
        cfg_drop_both=cfg_drop_both,
        use_pooled_text=mp.get("use_pooled_text", True),
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# Plotting (from CSV data only — no GPU needed)
# ---------------------------------------------------------------------------
def draw_plots(csv_path, png_path):
    """Read CSV and produce the two-panel timestep loss plot."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    t = np.array([float(r['timestep']) for r in rows])
    col_names = [k for k in rows[0].keys() if k != 'timestep']
    data = {}
    for key in col_names:
        data[key] = np.array([float(r[key]) for r in rows])

    palette = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f']

    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw={'height_ratios': [2, 1]})

    # --- Top: absolute loss ---
    ax = axes[0]
    for idx, lb in enumerate(col_names):
        ax.plot(t, data[lb], marker='o', markersize=4, linewidth=1.8,
                color=palette[idx % len(palette)], label=lb)
    ax.set_ylabel('Mean MSE Loss')
    ax.set_title('Per-Timestep Flow Matching Loss (uniform eval grid)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(t)
    ax.set_xticklabels(['%.2f' % v for v in t], rotation=45, fontsize=7)
    ax.set_xlim(t[0] - 0.02, t[-1] + 0.02)

    # --- Bottom: difference vs first column (baseline) ---
    ax2 = axes[1]
    baseline_name = col_names[0]
    baseline = data[baseline_name]
    for idx, lb in enumerate(col_names[1:], start=1):
        diff = data[lb] - baseline
        ax2.plot(t, diff, marker='o', markersize=4, linewidth=1.8,
                 color=palette[idx % len(palette)],
                 label='%s - %s' % (lb, baseline_name))
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Timestep  (0 = clean target, 1 = pure noise)')
    ax2.set_ylabel('Delta MSE vs %s' % baseline_name)
    ax2.set_title('Difference from %s' % baseline_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(t)
    ax2.set_xticklabels(['%.2f' % v for v in t], rotation=45, fontsize=7)
    ax2.set_xlim(t[0] - 0.02, t[-1] + 0.02)

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Plot saved: %s' % png_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Per-timestep loss evaluation")
    parser.add_argument("--checkpoints", nargs="+",
                        help="Paths to checkpoint .pt files")
    parser.add_argument("--labels", nargs="+",
                        help="Label for each checkpoint (same order)")
    parser.add_argument("--params", type=str,
                        help="YAML config (for model arch + dataset)")
    parser.add_argument("--num_bins", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/timestep_loss_analysis")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--redraw", type=str, default=None,
                        help="Redraw plots from existing CSV (path without extension)")
    args = parser.parse_args()

    # ---- Redraw mode: just re-plot from CSV ----
    if args.redraw is not None:
        csv_path = args.redraw + '/loss_by_timestep.csv'
        png_path = args.redraw + '/loss_comparison.png'
        draw_plots(csv_path, png_path)
        return

    # ---- Full evaluation mode ----
    assert args.checkpoints and args.labels, \
        "Provide --checkpoints and --labels (or use --redraw)"
    assert len(args.checkpoints) == len(args.labels), \
        "--checkpoints and --labels must have the same length"

    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    params = yaml.safe_load(open("./config/%s" % args.params, "r"))
    scale_factor = params.get("scale_factor", 1.0)
    eps = 1e-5

    # ---- Data ----
    dataset_path = params["dataset_path"]
    vision_model = params["vision_model"]
    text_model = params["text_model"]

    test_dataset = MemoryMappedDataset(
        dataset_path, vision_model=vision_model,
        text_model=text_model, split="test",
    )
    B = params["batch_size"]
    test_loader = DataLoader(
        test_dataset, batch_size=B, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=mmap_collate_fn,
    )

    sample_c_hidden = np.load(
        "%s/labels_hidden_%s.npy" % (dataset_path, text_model), mmap_mode="r")
    text_dim = sample_c_hidden.shape[2]
    max_text_len = sample_c_hidden.shape[1]
    del sample_c_hidden
    sample_c_pooled = np.load(
        "%s/labels_pooled_%s.npy" % (dataset_path, text_model), mmap_mode="r")
    pooled_text_dim = sample_c_pooled.shape[1]
    del sample_c_pooled

    print("Val samples: %d, Batch size: %d" % (len(test_dataset), B))

    # ---- Timestep grid ----
    t_values = torch.linspace(0.025, 0.975, args.num_bins).tolist()
    print("Timestep bins (%d): %.3f ... %.3f" % (args.num_bins, t_values[0], t_values[-1]))

    # ---- Evaluate each checkpoint ----
    all_results = {}

    for ckpt_path, label in zip(args.checkpoints, args.labels):
        print("\n" + "=" * 60)
        print("Evaluating: %s  (%s)" % (label, ckpt_path))
        print("=" * 60)

        model = build_model(params, text_dim, pooled_text_dim, max_text_len, device)
        load_checkpoint(ckpt_path, {"model": model})
        model.eval()

        losses_per_t = []

        for t_fixed in tqdm(t_values, desc="  [%s] timesteps" % label):
            batch_losses = []

            for batch in test_loader:
                x0, z0, xt, zt, c_txt, c_hidden, c_mask, c_pooled = batch
                z0 = z0.to(device) * scale_factor
                zt = zt.to(device) * scale_factor
                c_hidden = c_hidden.to(device)
                c_mask = c_mask.to(device)
                c_pooled = c_pooled.to(device)

                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = fixed_timestep_loss(
                        model, z0, zt, c_hidden, t_fixed,
                        text_mask=c_mask, pooled_text_emb=c_pooled, eps=eps,
                    )
                batch_losses.append(loss.item())

            mean_loss = sum(batch_losses) / len(batch_losses)
            losses_per_t.append(mean_loss)

        all_results[label] = losses_per_t

        del model
        torch.cuda.empty_cache()

    # ---- Save CSV ----
    os.makedirs(args.output, exist_ok=True)
    csv_path = args.output + "/loss_by_timestep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep"] + args.labels)
        for i, t_val in enumerate(t_values):
            row = ["%.4f" % t_val] + ["%.6f" % all_results[lb][i] for lb in args.labels]
            writer.writerow(row)
    print("\nCSV saved: %s" % csv_path)

    # ---- Plot ----
    draw_plots(csv_path, args.output + "/loss_comparison.png")


if __name__ == "__main__":
    main()

"""
Training script for FlowDiT.

Trains a flow matching DiT model to predict target affordance states from
initial observations and text commands.

Usage:
    Single GPU:  python train_flowdit.py --params flowdit.yaml [--gpu 0] [--frontend] [--dummy]
    Multi-GPU:   torchrun --nproc_per_node=N src/train_flowdit.py --params flowdit.yaml [--frontend]
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import sys
import os
import math
import yaml
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import textwrap
import warnings

from models.flowdit import FlowDiT, flow_matching_loss
from models.theia_decoder import Decoder as TheiaDecoder
from utils.args import parse_args
from utils.ema import EMA
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import visualize_flowdit_samples, plot_loss_curves

# Suppress transformers warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', message='Some weights of.*were not initialized')
warnings.filterwarnings('ignore', message='find_unused_parameters=True was specified')

# ============================================================================
# Setup
# ============================================================================

args = parse_args(sys.argv[1:])
params = yaml.safe_load(open("./config/" + args.config, 'r'))

model_path = params["model_path"]
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
results_path = params["results_path"]
if not os.path.exists(os.path.dirname(results_path)):
    os.makedirs(os.path.dirname(results_path))

# Detect DDP (set automatically by torchrun)
ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1
if ddp:
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    is_main = (rank == 0)
    if is_main:
        print(f"DDP: {world_size} GPUs")
    else:
        # Silence non-rank-0 print output, keep stderr for error tracebacks
        sys.stdout = open(os.devnull, 'w')
else:
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    is_main = True
    rank = 0
    world_size = 1
    print(f"Using GPU {args.gpu}")

# Frontend monitor (optional, rank 0 only)
if args.frontend and is_main:
    from frontend.training_monitor import TrainingMonitor
    monitor = TrainingMonitor(params)

# ============================================================================
# Load Theia decoder (for visualization only - encoder not needed with pre-embedded data)
# ============================================================================

# Load Theia decoder for visualization (if available)
theia_decoder = None
if "theia_decoder" in params and os.path.exists(params["theia_decoder"]["model_path"]):
    # print("Loading Theia decoder for visualization...")
    theia_decoder = TheiaDecoder(**params["theia_decoder"]["model_params"])
    checkpoint = torch.load(params["theia_decoder"]["model_path"], map_location=device, weights_only=True)
    theia_decoder.load_state_dict(checkpoint['model'])
    theia_decoder = theia_decoder.to(device)
    theia_decoder.eval()
    for p in theia_decoder.parameters():
        p.requires_grad = False

# ============================================================================
# Load and process data
# ============================================================================

dataset_path = params["dataset_path"]
vision_model = params["vision_model"]
text_model = params["text_model"]

# Create memory-mapped datasets using split files
from utils.datasets import MemoryMappedDataset, mmap_collate_fn
train_dataset = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, split='train')
test_dataset = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, split='test')

n_train = len(train_dataset)
n_test = len(test_dataset)
n_total = n_train + n_test

# Dummy mode: load only one batch for testing
if args.dummy:
    B = params["batch_size"]
    train_dataset = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, indices=train_dataset.indices[:B])
    test_dataset = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, indices=test_dataset.indices[:B])
    n_train = len(train_dataset)
    n_test = len(test_dataset)
    print(f"[DUMMY MODE] Using only {B} samples per split")

print(f"Total samples: {n_total}, Train: {n_train}, Test: {n_test}")

# Get text_dim from first sample
sample_c_hidden = np.load(f"{dataset_path}/labels_hidden_{text_model}.npy", mmap_mode='r')
text_dim = sample_c_hidden.shape[2]  # CLIP hidden dim (768)
max_text_len = sample_c_hidden.shape[1]  # 25
del sample_c_hidden

B = params["batch_size"]
if ddp:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
else:
    train_sampler = None
    test_sampler = None
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=(train_sampler is None),
                          sampler=train_sampler, num_workers=0, pin_memory=False, collate_fn=mmap_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False,
                         sampler=test_sampler, num_workers=0, pin_memory=False, collate_fn=mmap_collate_fn)

# ============================================================================
# Initialize model
# ============================================================================

# Get dimensions from config
latent_dim = params["model_params"]["latent_dim"]
num_patches = params["model_params"]["num_patches"]
hidden_dim = params["model_params"]["hidden_dim"]
depth = params["model_params"]["depth"]
num_heads = params["model_params"]["num_heads"]
# text_dim already loaded from data above
mlp_ratio = params["model_params"].get("mlp_ratio", 4.0)
dropout = params["model_params"].get("dropout", 0.0)
cond_drop_prob = params["model_params"].get("cond_drop_prob", None)  # deprecated
context_drop_prob = params["model_params"].get("context_drop_prob", None)  # deprecated
cfg_drop_prompt = params["model_params"].get("cfg_drop_prompt", 0.05)
cfg_drop_context = params["model_params"].get("cfg_drop_context", 0.05)
cfg_drop_both = params["model_params"].get("cfg_drop_both", 0.05)

# Backward compat: if old params present and new ones absent, convert
if cond_drop_prob is not None and "cfg_drop_prompt" not in params["model_params"]:
    cfg_drop_prompt = cond_drop_prob
    cfg_drop_context = context_drop_prob or 0.0
    cfg_drop_both = cfg_drop_prompt * cfg_drop_context  # approximate old independent behavior
    cfg_drop_prompt -= cfg_drop_both
    cfg_drop_context -= cfg_drop_both

model = FlowDiT(
    latent_dim=latent_dim,
    num_patches=num_patches,
    hidden_dim=hidden_dim,
    depth=depth,
    num_heads=num_heads,
    text_dim=text_dim,
    max_text_len=max_text_len,
    mlp_ratio=mlp_ratio,
    dropout=dropout,
    cfg_drop_prompt=cfg_drop_prompt,
    cfg_drop_context=cfg_drop_context,
    cfg_drop_both=cfg_drop_both,
).to(device)

# Enable gradient checkpointing if specified
if params.get("gradient_checkpointing", False):
    print("Enabling gradient checkpointing...")
    model.enable_gradient_checkpointing()

# Scale factor for Theia latents
scale_factor = params.get("scale_factor", 1.0)

# Two-scale CFG parameters for inference
context_cfg_scale = params.get("context_cfg_scale", None)
prompt_cfg_scale = params.get("prompt_cfg_scale", None)

# Register frontend chart
if args.frontend and is_main:
    monitor.register_chart('Loss', [
        {'label': 'Train', 'color': '#b86934'},
        {'label': 'Val',   'color': '#b8bb26'},
    ], csv_column='loss')

# ============================================================================
# Optimizer and scheduler (linear warmup, then constant LR)
# ============================================================================

lr = float(params["lr"])
warmup_steps = int(params.get("warmup_steps", 2000))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=float(params.get("weight_decay", 0.0))
)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if is_main:
    print(f"LR schedule: linear warmup {warmup_steps} steps, then constant lr={lr}")

# Optional EMA (not a wrapper, just for parameter swapping)
use_ema = params.get("use_ema", False)
if use_ema:
    ema = EMA(optimizer)
else:
    ema = None

# ============================================================================
# Checkpoint loading
# ============================================================================

resume_from_checkpoint = params.get("resume_from_checkpoint", True)
checkpoint_path = params["model_path"]

if resume_from_checkpoint:
    meta = load_checkpoint(checkpoint_path, {'model': model, 'optimizer': optimizer, 'scheduler': scheduler})
else:
    print("No checkpoint found - Starting training from scratch")
    meta = {}

start_epoch = meta.get('epoch', -1) + 1
best_loss = meta.get('test_loss', float('inf'))
best_loss_epoch = -1
train_losses = meta.get('train_losses', [])
test_losses = meta.get('test_losses', [])

# Early stopping
patience = params.get('patience', 5)
patience_counter = 0

# ============================================================================
# DDP wrapping (after checkpoint loading, before training)
# ============================================================================

raw_model = model
if ddp:
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# ============================================================================
# CSV logging setup
# ============================================================================

log_file = results_path + 'training_log.csv'
if is_main and (not os.path.exists(log_file) or start_epoch == 0):
    with open(log_file, 'w') as f:
        f.write('epoch,split,loss\n')

# ============================================================================
# Helper functions
# ============================================================================


def decode(z: torch.Tensor) -> torch.Tensor:
    """Decode Theia latents to images."""
    with torch.no_grad():
        return theia_decoder(z)


if args.dummy and not ddp:
    torch.cuda.reset_peak_memory_stats()
    dummy_batch = next(iter(train_loader))

    # I/O: transfer to device
    t0 = time.perf_counter()
    x0, z0, xt, zt, text_txt, text_hidden, text_mask = dummy_batch
    z0 = z0.to(device) * scale_factor
    zt = zt.to(device) * scale_factor
    text_hidden = text_hidden.to(device)
    text_mask = text_mask.to(device)
    torch.cuda.synchronize()
    t_io = time.perf_counter() - t0

    # Flow matching forward
    t0 = time.perf_counter()
    loss = flow_matching_loss(model, z0, zt, text_hidden, text_mask=text_mask)
    torch.cuda.synchronize()
    t_flow_fwd = time.perf_counter() - t0

    # Flow matching backward
    t0 = time.perf_counter()
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()
    t_flow_bwd = time.perf_counter() - t0

    print(f"--- Flow Matching ---")
    print(f"  I/O:      {t_io*1000:7.1f} ms")
    print(f"  Forward:  {t_flow_fwd*1000:7.1f} ms")
    print(f"  Backward: {t_flow_bwd*1000:7.1f} ms")
    print(f"  Peak mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\nTotal peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")



# ============================================================================
# Training loop
# ============================================================================

print("Starting training...")
eps = 1e-5  # For flow matching interpolation

pbar = tqdm(range(start_epoch, params["num_epochs"]), desc="Training", initial=start_epoch, total=params["num_epochs"], disable=not is_main)

for epoch in pbar:
    if ddp:
        train_sampler.set_epoch(epoch)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    model.train()
    total_train_loss = 0
    
    for batch_idx, (x0, z0, xt, zt, c_txt, c_hidden, c_mask) in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training", disable=not is_main)):
        if args.frontend and is_main:
            monitor.update_batch(batch_idx, total_batches=len(train_loader), mode='train')

        # Use pre-computed embeddings directly
        z0 = z0.to(device) * scale_factor
        zt = zt.to(device) * scale_factor
        c_hidden = c_hidden.to(device)
        c_mask = c_mask.to(device)

        loss = flow_matching_loss(model, z0, zt, c_hidden, text_mask=c_mask, eps=eps)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        if params.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])

        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_loader)

    # Average losses across DDP workers
    if ddp:
        loss_tensor = torch.tensor([avg_train_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size
        avg_train_loss = loss_tensor.item()

    train_losses.append(avg_train_loss)

    # Log to CSV (rank 0 only)
    if is_main:
        with open(log_file, 'a') as f:
            f.write(f'{epoch},train,{avg_train_loss:.6f}\n')
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    model.eval()
    total_test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (x0, z0, xt, zt, c_txt, c_hidden, c_mask) in enumerate(tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation", disable=not is_main)):
            if args.frontend and is_main:
                monitor.update_batch(batch_idx, total_batches=len(test_loader), mode='val')
            
            # Use pre-computed embeddings directly
            z0 = z0.to(device) * scale_factor
            zt = zt.to(device) * scale_factor
            c_hidden = c_hidden.to(device)
            c_mask = c_mask.to(device)
            
            # Compute flow matching loss
            test_loss = flow_matching_loss(model, z0, zt, c_hidden, text_mask=c_mask, eps=eps)
            total_test_loss += test_loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)

    # Average val losses across DDP workers
    if ddp:
        val_tensor = torch.tensor([avg_test_loss], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        avg_test_loss = val_tensor.item() / world_size

    test_losses.append(avg_test_loss)
    
    # Log to CSV (rank 0 only)
    if is_main:
        with open(log_file, 'a') as f:
            f.write(f'{epoch},val,{avg_test_loss:.6f}\n')
    
    # -------------------------------------------------------------------------
    # Visualization (rank 0 only)
    # -------------------------------------------------------------------------
    if is_main:
        # Swap to EMA weights if using EMA
        if use_ema:
            ema.swap_parameters_with_ema(store_params_in_ema=True)
        
        test_x0, test_z0, test_xt, test_zt, test_c_txt, test_c_hidden, test_c_mask = next(iter(test_loader))
        visualize_flowdit_samples(
            raw_model, test_x0, test_z0, test_xt, test_zt, test_c_txt, test_c_hidden, test_c_mask,
            epoch=epoch,
            save_dir=results_path + 'decoded',
            device=device,
            scale_factor=scale_factor,
            decode_fn=decode if theia_decoder is not None else None,
            num_vis=min(4, B),
            num_steps=params.get("sample_steps", 50),
            cfg_scale=1.0,
        )
        
        if use_ema:
            ema.swap_parameters_with_ema(store_params_in_ema=True)
    
    # -------------------------------------------------------------------------
    # Update frontend (rank 0 only)
    # -------------------------------------------------------------------------
    if args.frontend and is_main:
        monitor.update_epoch(
            epoch,
            charts={'Loss': {'Train': avg_train_loss, 'Val': avg_test_loss}},
            tables={'Loss': {
                'Train': {'Loss': avg_train_loss},
                'Val':   {'Loss': avg_test_loss},
            }},
        )
    
    # Update progress bar
    pbar.set_postfix({
        'Train': f'{avg_train_loss:.4f}',
        'Test': f'{avg_test_loss:.4f}',
        'LR': f'{scheduler.get_last_lr()[0]:.2e}',
    })
    
    # -------------------------------------------------------------------------
    # Save checkpoint (rank 0 only) + early stopping (all ranks)
    # -------------------------------------------------------------------------
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        patience_counter = 0
        if ddp:
            dist.barrier()
        if is_main:
            if use_ema:
                ema.swap_parameters_with_ema(store_params_in_ema=True)
            save_checkpoint(params["model_path"], {'model': raw_model, 'optimizer': optimizer, 'scheduler': scheduler}, {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'train_losses': train_losses,
                'test_losses': test_losses,
            })
            if use_ema:
                ema.swap_parameters_with_ema(store_params_in_ema=True)
    else:
        patience_counter += 1
        if ddp:
            dist.barrier()

    if patience_counter >= patience:
        if is_main:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break

# ============================================================================
# Cleanup & post-training (rank 0 only for visualization)
# ============================================================================

if is_main:
    print("\nTraining complete!")
    print(f"Best test loss: {best_loss:.6f} at epoch {best_loss_epoch}")

    plot_loss_curves(train_losses, test_losses, results_path + 'training_losses.png', best_loss_epoch)

    # ========================================================================
    # Visualize some samples
    # ========================================================================

    print("Generating final samples...")

    # Load best model (use raw_model for consistent state_dict keys)
    checkpoint = torch.load(params["model_path"], map_location=device, weights_only=True)
    raw_model.load_state_dict(checkpoint['model'])

    if use_ema:
        ema.swap_parameters_with_ema(store_params_in_ema=True)

    raw_model.eval()

    # Generate samples from test set
    test_x0, test_z0, test_xt, test_zt, test_c_txt, test_c_hidden, test_c_mask = next(iter(test_loader))
    n_samples = min(8, test_x0.shape[0])

    # Images for display (CHW float32 [0,1] tensors)
    x0_vis = test_x0[:n_samples]
    xt_vis = test_xt[:n_samples]
    # Pre-computed embeddings for generation
    z_init = test_z0[:n_samples].to(device) * scale_factor
    z_target_gt = test_zt[:n_samples].to(device) * scale_factor
    c_hidden_vis = test_c_hidden[:n_samples].to(device)
    c_mask_vis = test_c_mask[:n_samples].to(device)
    c_txt_vis = test_c_txt[:n_samples]

    # Generate samples at different CFG scales
    context_cfg_scales = [0.0, 1.0, 3.0, 5.0, 7.0]
    prompt_cfg_scales = [0.0, 1.0, 3.0, 5.0, 7.0]
    z_generated_all = []
    cos_sims_all = []
    mse_errors_all = []

    print(f"\nFinal generation metrics:")
    print(f"Best test loss: {best_loss:.4f} at epoch {best_loss_epoch}\n")

    # Matrix of CFG scale combinations (context vs prompt)
    for context_cfg_scale in context_cfg_scales:
        for prompt_cfg_scale in prompt_cfg_scales:
            with torch.no_grad():
                z_generated = raw_model.sample_adaptive(
                    z_init, c_hidden_vis, text_mask=c_mask_vis, cfg_scale=1.0,
                    context_cfg_scale=context_cfg_scale, prompt_cfg_scale=prompt_cfg_scale,
                )
    
            # Compute metrics
            cos_sims = F.cosine_similarity(z_generated.flatten(1), z_target_gt.flatten(1), dim=1)
            mse_errors = F.mse_loss(z_generated, z_target_gt, reduction='none').mean(dim=[1, 2])
            
            z_generated_all.append(z_generated)
            cos_sims_all.append(cos_sims)
            mse_errors_all.append(mse_errors)
            
            print(f"Context CFG={context_cfg_scale:.1f}, Prompt CFG={prompt_cfg_scale:.1f}: cos_sim={cos_sims.mean():.4f}±{cos_sims.std():.4f}, MSE={mse_errors.mean():.4f}±{mse_errors.std():.4f}")

    # Visualize: one figure per sample, each a grid of context_cfg x prompt_cfg
    if theia_decoder is not None:
        n_ctx = len(context_cfg_scales)
        n_pmt = len(prompt_cfg_scales)

        for i in range(n_samples):
            # Grid: rows = context_cfg, cols = prompt_cfg, plus label row/col + initial/target
            n_rows = n_ctx + 1  # +1 for header row (Initial / Target / prompt labels)
            n_cols = n_pmt + 1  # +1 for label column (context labels)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

            # Sample title
            wrapped_text = '\n'.join(textwrap.wrap(c_txt_vis[i], width=60))
            fig.suptitle(wrapped_text, fontsize=10, fontweight='bold', y=1.02)

            # Header row: col 0 = Initial, col 1 = Target, cols 2.. = prompt_cfg labels
            # Images are CHW float [0,1] tensors — transpose to HWC for imshow
            x0_img = np.clip(x0_vis[i].cpu().numpy().transpose(1, 2, 0), 0, 1)
            xt_img = np.clip(xt_vis[i].cpu().numpy().transpose(1, 2, 0), 0, 1)
            axes[0, 0].imshow(x0_img)
            axes[0, 0].set_title('Initial', fontsize=8, fontweight='bold')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(xt_img)
            axes[0, 1].set_title('Target', fontsize=8, fontweight='bold')
            axes[0, 1].axis('off')

            for pc_idx in range(2, n_cols):
                axes[0, pc_idx].set_title(f'P={prompt_cfg_scales[pc_idx - 1]:.0f}', fontsize=8, fontweight='bold')
                axes[0, pc_idx].axis('off')

            # Fill generation grid (rows 1..n_ctx, cols 1..n_pmt)
            for cc_idx, ctx_cfg in enumerate(context_cfg_scales):
                row = cc_idx + 1

                # Row label in column 0
                axes[row, 0].text(0.5, 0.5, f'C={ctx_cfg:.0f}', ha='center', va='center',
                                  fontsize=9, fontweight='bold', transform=axes[row, 0].transAxes)
                axes[row, 0].axis('off')

                for pc_idx, pmt_cfg in enumerate(prompt_cfg_scales):
                    col = pc_idx + 1
                    flat_idx = cc_idx * n_pmt + pc_idx
                    z_gen = z_generated_all[flat_idx]
                    cos_sim = cos_sims_all[flat_idx][i].item()
                    mse_err = mse_errors_all[flat_idx][i].item()

                    xg_recon = decode(z_gen[i:i+1] / scale_factor)
                    img = xg_recon[0].cpu().permute(1, 2, 0).numpy()
                    axes[row, col].imshow(np.clip(img, 0, 1))
                    axes[row, col].set_title(f'cos={cos_sim:.3f}\nMSE={mse_err:.3f}', fontsize=7)
                    axes[row, col].axis('off')

            plt.tight_layout()
            save_path = results_path + f'generated_sample_{i}.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved {save_path}")

if ddp:
    dist.destroy_process_group()

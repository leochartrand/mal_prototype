"""
Training script for DiT.

Trains a flow-matching DiT to predict target affordance states from initial
observations and text commands. Works for DROID-style endgoal data and
CALVIN endgoal/subgoal fine-tuning — they all share the MemoryMappedDataset
format; only `dataset_path` (and optionally `pretrained_flowdit_path`) differ
in the config.

Usage:
    Single GPU:  python src/train_flowdit.py --config <name>.yaml [--gpu 0]
    Multi-GPU:   torchrun --nproc_per_node=N src/train_flowdit.py --config <name>.yaml
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import os
import yaml
from tqdm import tqdm
import warnings

from models.flowdit import DiTAir, flow_matching_loss
from utils.args import parse_args
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.datasets import mmap_collate_fn
from utils.training import (
    init_distributed, init_frontend_monitor, load_theia_decoder,
    build_dataloaders, load_pretrained, reduce_mean, init_csv_log,
)
from utils.visualization import visualize_flowdit_samples

warnings.filterwarnings('ignore', message='Some weights of.*were not initialized')
warnings.filterwarnings('ignore', message='find_unused_parameters=True was specified')

MODEL_CLASSES = {
    "dit_air": DiTAir,
}

# ============================================================================
# Setup
# ============================================================================

args = parse_args(sys.argv[1:])
params = yaml.safe_load(open("./config/" + args.config, 'r'))

model_path = params["model_path"]
os.makedirs(os.path.dirname(model_path), exist_ok=True)
results_path = params["results_path"]
os.makedirs(results_path, exist_ok=True)

dd = init_distributed(args.gpu)
ddp, device, is_main = dd['ddp'], dd['device'], dd['is_main']
rank, world_size, local_rank = dd['rank'], dd['world_size'], dd['local_rank']

monitor = init_frontend_monitor(params, args.frontend, is_main)

# ============================================================================
# Theia decoder (visualization only)
# ============================================================================

theia_decoder = load_theia_decoder(params, device, is_main=is_main)

# ============================================================================
# Data
# ============================================================================

(train_loader, val_loader, train_sampler, train_dataset, val_dataset,
 text_dim, max_text_len, pooled_text_dim) = build_dataloaders(params, args.dummy, ddp)
n_train, n_val = len(train_dataset), len(val_dataset)

# ============================================================================
# Model
# ============================================================================

mp = params["model_params"]
model_type = mp.get("model_type", "dit_air")
ModelClass = MODEL_CLASSES[model_type]
print(f"Using model type: {model_type} -> {ModelClass.__name__}")

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

if params.get("gradient_checkpointing", False):
    model.enable_gradient_checkpointing()

scale_factor      = params.get("scale_factor", 1.0)
context_cfg_scale = params.get("cfg_ctx", params.get("context_cfg_scale", None))
prompt_cfg_scale  = params.get("cfg_prompt", params.get("prompt_cfg_scale", None))

# Warm-start from pretrained weights (optional)
pretrained_path = params.get("pretrained_flowdit_path")
if pretrained_path:
    load_pretrained(model, pretrained_path, is_main)
else:
    print("No pretrained_flowdit_path — training from scratch.")

if monitor and is_main:
    monitor.register_chart('Loss', [
        {'label': 'Train', 'color': '#b86934'},
        {'label': 'Val',   'color': '#b8bb26'},
    ], csv_column='loss')

# ============================================================================
# Optimizer and scheduler
# ============================================================================

lr           = float(params["lr"])
warmup_steps = int(params.get("warmup_steps", 500))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=float(params.get("weight_decay", 0.01)),
)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
if is_main:
    print(f"LR schedule: linear warmup {warmup_steps} steps, then constant lr={lr}")

# ============================================================================
# Checkpoint resume (fine-tune checkpoint, not pretrained)
# ============================================================================

resume_from_checkpoint = params.get("resume_from_checkpoint", True)
if resume_from_checkpoint:
    meta = load_checkpoint(model_path, {'model': model, 'optimizer': optimizer, 'scheduler': scheduler})
else:
    print("Starting fine-tuning from pretrained weights (no resume).")
    meta = {}

start_epoch = meta.get('epoch', -1) + 1
# Accept legacy `test_loss`/`test_losses` keys for back-compat with older checkpoints
best_loss    = meta.get('val_loss', meta.get('test_loss', float('inf')))
train_losses = meta.get('train_losses', [])
val_losses   = meta.get('val_losses', meta.get('test_losses', []))

patience = params.get('patience', 5)
patience_counter = 0

# ============================================================================
# DDP wrapping
# ============================================================================

raw_model = model
if ddp:
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)

# ============================================================================
# CSV logging
# ============================================================================

log_file = os.path.join(results_path, 'training_log.csv')
if is_main:
    init_csv_log(log_file, start_epoch)

# ============================================================================
# Training loop
# ============================================================================

print("Starting training...")
eps = 1e-5
timestep_distribution = params.get("timestep_distribution", params.get("noise_schedule", "rae_shift"))
if is_main:
    print(f"Timestep distribution: {timestep_distribution}")

# Fixed vis indices into the val split (configurable; sensible default spreads across val set)
vis_indices = params.get("vis_indices", [0, 100, 200, 300, 400, 600, 800, 1000])
vis_indices = [i for i in vis_indices if i < n_val]

pbar = tqdm(range(start_epoch, params["num_epochs"]), desc="Training",
            initial=start_epoch, total=params["num_epochs"], disable=not is_main)

for epoch in pbar:
    if ddp:
        train_sampler.set_epoch(epoch)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    model.train()
    total_train_loss = 0.0

    train_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Train", disable=not is_main)
    for batch_idx, (x0, z0, xt, zt, c_txt, c_hidden, c_mask, c_pooled) in enumerate(train_pbar):
        if args.frontend and is_main:
            monitor.update_batch(batch_idx, total_batches=len(train_loader), mode='train')

        z0       = z0.to(device) * scale_factor
        zt       = zt.to(device) * scale_factor
        c_hidden = c_hidden.to(device)
        c_mask   = c_mask.to(device)
        c_pooled = c_pooled.to(device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = flow_matching_loss(
                model, z0, zt, c_hidden,
                text_mask=c_mask, pooled_text_emb=c_pooled,
                eps=eps, timestep_distribution=timestep_distribution,
            )

        total_train_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        if params.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
        optimizer.step()
        scheduler.step()

    avg_train = reduce_mean(total_train_loss / len(train_loader), ddp, world_size, device)
    train_losses.append(avg_train)

    if is_main:
        with open(log_file, 'a') as f:
            f.write(f"{epoch},train,{avg_train:.6f}\n")

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------
    model.eval()
    total_val_loss = 0.0

    val_pbar = tqdm(val_loader, leave=False, desc=f"Epoch {epoch}: Val", disable=not is_main)
    with torch.no_grad():
        for batch_idx, (x0, z0, xt, zt, c_txt, c_hidden, c_mask, c_pooled) in enumerate(val_pbar):
            if args.frontend and is_main:
                monitor.update_batch(batch_idx, total_batches=len(val_loader), mode='val')

            z0       = z0.to(device) * scale_factor
            zt       = zt.to(device) * scale_factor
            c_hidden = c_hidden.to(device)
            c_mask   = c_mask.to(device)
            c_pooled = c_pooled.to(device)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                val_loss = flow_matching_loss(
                    model, z0, zt, c_hidden,
                    text_mask=c_mask, pooled_text_emb=c_pooled,
                    eps=eps, timestep_distribution=timestep_distribution,
                )
            total_val_loss += val_loss.item()
            val_pbar.set_postfix(loss=f"{val_loss.item():.4f}")

    avg_val = reduce_mean(total_val_loss / len(val_loader), ddp, world_size, device)
    val_losses.append(avg_val)

    if is_main:
        with open(log_file, 'a') as f:
            f.write(f"{epoch},val,{avg_val:.6f}\n")

    pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{avg_val:.4f}",
                     lr=f"{scheduler.get_last_lr()[0]:.2e}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    if is_main and theia_decoder is not None and vis_indices:
        vx0, vz0, vxt, vzt, vc_txt, vc_hidden, vc_mask, vc_pooled = mmap_collate_fn(
            [val_dataset[i] for i in vis_indices]
        )
        visualize_flowdit_samples(
            raw_model, vx0, vz0, vxt, vzt, vc_txt, vc_hidden, vc_mask,
            pooled_text_batch=vc_pooled,
            epoch=epoch,
            save_dir=os.path.join(results_path, "decoded"),
            device=device,
            scale_factor=scale_factor,
            decode_fn=lambda z: theia_decoder(z).detach(),
            num_vis=min(4, len(vis_indices)),
            num_steps=params.get("sample_steps", 8),
            cfg_scale=1.0,
            context_cfg_scale=context_cfg_scale,
            prompt_cfg_scale=prompt_cfg_scale,
        )

    if args.frontend and is_main:
        monitor.update_epoch(
            epoch,
            charts={'Loss': {'Train': avg_train, 'Val': avg_val}},
            tables={'Loss': {
                'Train': {'Loss': avg_train},
                'Val':   {'Loss': avg_val},
            }},
        )

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------
    if avg_val < best_loss:
        best_loss = avg_val
        patience_counter = 0
        if ddp:
            dist.barrier()
        if is_main:
            save_checkpoint(model_path, {'model': raw_model, 'optimizer': optimizer, 'scheduler': scheduler}, {
                'epoch': epoch,
                'train_loss': avg_train,
                'val_loss': avg_val,
                'train_losses': train_losses,
                'val_losses': val_losses,
            })
            print(f"  Saved checkpoint (val={avg_val:.4f})")
    else:
        patience_counter += 1
        if ddp:
            dist.barrier()

    if patience_counter >= patience:
        if is_main:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break

# ============================================================================
# Summary
# ============================================================================

if is_main:
    print(f"\nTraining complete! Best val loss: {best_loss:.6f}")
    from utils.visualization import plot_loss_curves
    plot_loss_curves(train_losses, val_losses, os.path.join(results_path, 'training_losses.png'))

if ddp:
    dist.destroy_process_group()
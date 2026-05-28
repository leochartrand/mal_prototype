"""
Guidance distillation for DiT (Meng et al. 2023, fixed-w form).

Trains a student DiTAir to mimic the two-scale CFG-guided velocity of a
frozen teacher in a single forward pass. At inference the student is sampled
with `cfg_scale=1.0` (no CFG forwards) — 3× fewer forwards per Euler step.

Architecture: unchanged. Student warm-starts from teacher weights. Loss is
MSE between student velocity and the teacher's three-fwd CFG combination at
the same (z_t, t) point. Validation runs under EMA-swapped weights.

Usage:
    Single GPU:  python src/flowdit_cfg_distill.py --config <name>.yaml [--gpu 0]
    Multi-GPU:   torchrun --nproc_per_node=N src/flowdit_cfg_distill.py --config <name>.yaml
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import os
import yaml
from tqdm import tqdm
import warnings

from models.flowdit import DiTAir, _sample_timesteps
from utils.args import parse_args
from utils.checkpoint import load_checkpoint, save_checkpoint, _strip_module_prefix
from utils.datasets import mmap_collate_fn
from utils.ema import EMA
from utils.training import (
    init_distributed, init_frontend_monitor, load_theia_decoder,
    build_dataloaders, reduce_mean, init_csv_log,
)
from utils.visualization import visualize_distill_samples

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

save_dir = params["save_dir"]
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "distill_latest.pt")
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
# Model (teacher + student)
# ============================================================================

mp = params["model_params"]
model_type = mp.get("model_type", "dit_air")
ModelClass = MODEL_CLASSES[model_type]
print(f"Using model type: {model_type} -> {ModelClass.__name__}")

def _build():
    return ModelClass(
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
        cfg_drop_prompt=0.0,
        cfg_drop_context=0.0,
        cfg_drop_both=0.0,
        use_pooled_text=mp.get("use_pooled_text", True),
    ).to(device)

teacher = _build()
student = _build()

teacher_path = params["teacher_path"]
print(f"Loading teacher checkpoint from {teacher_path}...")
ckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
teacher_sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
teacher_sd = _strip_module_prefix(teacher_sd)
missing, unexpected = teacher.load_state_dict(teacher_sd, strict=False)
if is_main and (missing or unexpected):
    print(f"  Missing: {missing}\n  Unexpected: {unexpected}")
student.load_state_dict(teacher_sd, strict=False)  # warm-start student from same weights
del ckpt, teacher_sd

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

n_params = sum(p.numel() for p in student.parameters())
print(f"Student params: {n_params:,} ({n_params/1e6:.1f}M)")

scale_factor      = float(params.get("scale_factor", 1.0))
context_cfg_scale = float(params.get("cfg_ctx", params.get("context_cfg_scale", 2.5)))
prompt_cfg_scale  = float(params.get("cfg_prompt", params.get("prompt_cfg_scale", 6.0)))

if monitor and is_main:
    monitor.register_chart('Loss', [
        {'label': 'Train', 'color': '#b86934'},
        {'label': 'Val',   'color': '#b8bb26'},
    ], csv_column='loss')

# ============================================================================
# Optimizer and scheduler (with EMA)
# ============================================================================

lr           = float(params["lr"])
warmup_steps = int(params.get("warmup_steps", 500))

base_opt = torch.optim.AdamW(
    student.parameters(),
    lr=lr,
    weight_decay=float(params.get("weight_decay", 0.0)),
)
optimizer = EMA(base_opt, ema_decay=float(params.get("ema_decay", 0.9999)))

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(base_opt, lr_lambda)
if is_main:
    print(f"LR schedule: linear warmup {warmup_steps} steps, then constant lr={lr}")

# ============================================================================
# Checkpoint resume (fine-tune checkpoint, not pretrained)
# ============================================================================

resume_from_checkpoint = params.get("resume_from_checkpoint", True)
if resume_from_checkpoint:
    meta = load_checkpoint(model_path, {'model': student, 'optimizer': optimizer, 'scheduler': scheduler})
else:
    print("Starting distillation from teacher (no resume).")
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

raw_student = student
if ddp:
    student = DDP(student, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)

# ============================================================================
# CSV logging
# ============================================================================

log_file = os.path.join(results_path, 'training_log.csv')
if is_main:
    init_csv_log(log_file, start_epoch)

# ============================================================================
# Training loop
# ============================================================================

print("Starting distillation...")
eps = float(params.get("eps", 1e-5))
timestep_distribution = params.get("timestep_distribution", params.get("noise_schedule", "rae_shift"))
if is_main:
    print(f"Timestep distribution: {timestep_distribution}, CFG=(ctx={context_cfg_scale}, prompt={prompt_cfg_scale})")

vis_indices = params.get("vis_indices", [0, 100, 200, 300, 400, 600, 800, 1000])
vis_indices = [i for i in vis_indices if i < n_val]

@torch.no_grad()
def _teacher_cfg_velocity(z_t, t, z_init, c_hidden, c_mask, c_pooled):
    """Three-fwd CFG combination of teacher velocity at (z_t, t)."""
    B_now = z_t.shape[0]
    text_ctx = teacher.text_proj(c_hidden)
    if c_mask is not None:
        text_ctx = text_ctx * c_mask.unsqueeze(-1)
    pooled_cond      = teacher._get_pooled_conditioning(c_pooled, use_null=False)
    null_pooled_cond = teacher._get_pooled_conditioning(c_pooled, use_null=True)
    null_text_ctx    = teacher.null_text_emb.expand(B_now, text_ctx.shape[1], -1)

    v_uncond = teacher._forward_with_ctx(z_t, t, z_init, null_text_ctx,
                                         pooled_cond=null_pooled_cond, use_null_context=True)
    v_ctx    = teacher._forward_with_ctx(z_t, t, z_init, null_text_ctx,
                                         pooled_cond=null_pooled_cond, use_null_context=False)
    v_full   = teacher._forward_with_ctx(z_t, t, z_init, text_ctx,
                                         pooled_cond=pooled_cond, use_null_context=False)
    return v_uncond + context_cfg_scale * (v_ctx - v_uncond) + prompt_cfg_scale * (v_full - v_ctx)

pbar = tqdm(range(start_epoch, params["num_epochs"]), desc="Distilling",
            initial=start_epoch, total=params["num_epochs"], disable=not is_main)

for epoch in pbar:
    if ddp:
        train_sampler.set_epoch(epoch)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    student.train()
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

        B_now = zt.shape[0]
        latent_numel = zt.shape[1] * zt.shape[2]
        t = _sample_timesteps(B_now, device, schedule=timestep_distribution, latent_numel=latent_numel)
        z_noise = torch.randn_like(zt)
        t_exp = t.view(B_now, 1, 1)
        z_t = (1 - t_exp) * zt + (eps + (1 - eps) * t_exp) * z_noise

        with torch.autocast('cuda', dtype=torch.bfloat16):
            v_target = _teacher_cfg_velocity(z_t, t, z0, c_hidden, c_mask, c_pooled)
        v_target = v_target.detach().float()

        with torch.autocast('cuda', dtype=torch.bfloat16):
            v_student = student(z_t, t, z0, c_hidden, text_mask=c_mask, pooled_text_emb=c_pooled)
        loss = F.mse_loss(v_student.float(), v_target)

        total_train_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        if params.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), params["grad_clip"])
        optimizer.step()
        scheduler.step()

    avg_train = reduce_mean(total_train_loss / len(train_loader), ddp, world_size, device)
    train_losses.append(avg_train)

    if is_main:
        with open(log_file, 'a') as f:
            f.write(f"{epoch},train,{avg_train:.6f}\n")

    # -------------------------------------------------------------------------
    # Validate (under EMA-swapped weights)
    # -------------------------------------------------------------------------
    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    student.eval()
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

            B_now = zt.shape[0]
            latent_numel = zt.shape[1] * zt.shape[2]
            t = _sample_timesteps(B_now, device, schedule=timestep_distribution, latent_numel=latent_numel)
            z_noise = torch.randn_like(zt)
            t_exp = t.view(B_now, 1, 1)
            z_t = (1 - t_exp) * zt + (eps + (1 - eps) * t_exp) * z_noise

            with torch.autocast('cuda', dtype=torch.bfloat16):
                v_target = _teacher_cfg_velocity(z_t, t, z0, c_hidden, c_mask, c_pooled)
                v_student = student(z_t, t, z0, c_hidden, text_mask=c_mask, pooled_text_emb=c_pooled)
            val_loss = F.mse_loss(v_student.float(), v_target.float())
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
    # Visualization (with EMA weights)
    # -------------------------------------------------------------------------
    if is_main and theia_decoder is not None and vis_indices:
        vx0, vz0, vxt, vzt, vc_txt, vc_hidden, vc_mask, vc_pooled = mmap_collate_fn(
            [val_dataset[i] for i in vis_indices]
        )
        visualize_distill_samples(
            teacher, raw_student, vx0, vz0, vxt, vzt, vc_txt, vc_hidden, vc_mask,
            pooled_text_batch=vc_pooled,
            epoch=epoch,
            save_dir=os.path.join(results_path, "decoded"),
            device=device,
            scale_factor=scale_factor,
            decode_fn=lambda z: theia_decoder(z).detach(),
            num_vis=min(4, len(vis_indices)),
            num_steps=params.get("sample_steps", 4),
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
    # `distill_latest.pt` carries full state (model + EMA + scheduler) for resume.
    # `distill_best.pt` carries the same full payload at the best-val epoch.
    if is_main:
        ckpt_meta = {
            'epoch': epoch,
            'train_loss': avg_train,
            'val_loss': avg_val,
            'best_val_loss': min(best_loss, avg_val),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        save_checkpoint(model_path,
                        {'model': raw_student, 'optimizer': optimizer, 'scheduler': scheduler},
                        ckpt_meta)
    if ddp:
        dist.barrier()

    if avg_val < best_loss:
        best_loss = avg_val
        patience_counter = 0
        if is_main:
            best_path = os.path.join(save_dir, "distill_best.pt")
            save_checkpoint(best_path,
                            {'model': raw_student, 'optimizer': optimizer, 'scheduler': scheduler},
                            ckpt_meta)
            print(f"  Saved best checkpoint (val={avg_val:.4f})")
    else:
        patience_counter += 1

    # Restore non-EMA weights for next training epoch
    optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    if patience_counter >= patience:
        if is_main:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break

# ============================================================================
# Summary
# ============================================================================

if is_main:
    print(f"\nDistillation complete! Best val loss: {best_loss:.6f}")
    optimizer.swap_parameters_with_ema(store_params_in_ema=True)  # student → EMA weights
    final_path = os.path.join(save_dir, "distill_final.pt")
    save_checkpoint(final_path, {'model': raw_student}, {
        'epoch': epoch,
        'best_val_loss': best_loss,
        'config': params,
    })
    print(f"Final EMA checkpoint → {final_path}")
    from utils.visualization import plot_loss_curves
    plot_loss_curves(train_losses, val_losses, os.path.join(results_path, 'training_losses.png'))

if ddp:
    dist.destroy_process_group()

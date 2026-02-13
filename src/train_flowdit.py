"""
Training script for AffordanceFlowDiT.

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
from contextlib import nullcontext
import numpy as np
import sys
import os
import yaml
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import textwrap
import warnings

from models.flowdit import AffordanceFlowDiT, flow_matching_loss
from models.theia_decoder import Decoder as TheiaDecoder
from utils.args import parse_args
from utils.ema import EMA
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import visualize_flowdit_samples, plot_loss_curves
from models.discriminator import (
    EmbeddingDiscriminator,
    discriminator_loss_bce, generator_loss_bce,
    discriminator_loss_hinge, generator_loss_hinge,
)

# Suppress transformers warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', message='Some weights of.*were not initialized')

# ============================================================================
# Setup
# ============================================================================

args = parse_args(sys.argv[1:])
params = yaml.safe_load(open("./params/" + args.params, 'r'))

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
        # Silence non-rank-0 processes (catches all print + library output)
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
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
cond_drop_prob = params["model_params"].get("cond_drop_prob", 0.1)
context_drop_prob = params["model_params"].get("context_drop_prob", 0.1)

model = AffordanceFlowDiT(
    latent_dim=latent_dim,
    num_patches=num_patches,
    hidden_dim=hidden_dim,
    depth=depth,
    num_heads=num_heads,
    text_dim=text_dim,
    max_text_len=max_text_len,
    mlp_ratio=mlp_ratio,
    dropout=dropout,
    cond_drop_prob=cond_drop_prob,
    context_drop_prob=context_drop_prob,
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

# ============================================================================
# Adversarial components (Phase 2)
# ============================================================================

adv_config = params.get("adversarial", {})
adv_start_epoch = adv_config.get("start_epoch", -1)
use_adversarial = adv_start_epoch >= 0

discriminator = None
disc_optimizer = None

if use_adversarial:
    discriminator = EmbeddingDiscriminator(
        latent_dim=latent_dim,
        channels=adv_config.get("disc_channels", [256, 512]),
    ).to(device)

    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(adv_config.get("disc_lr", 2e-4)),
        betas=(0.0, 0.99),  # Standard GAN discriminator betas
        weight_decay=0.0,
    )

    lambda_adv = adv_config.get("lambda_adv", 0.05)
    adv_euler_steps = adv_config.get("euler_steps", 4)
    adv_loss_type = adv_config.get("loss_type", 'bce')
    adv_cfg_scale = 1.0
    ramp_start = adv_config.get("ramp_start_epoch", 2)   # epochs after start_epoch
    ramp_end = adv_config.get("ramp_end_epoch", 10)       # epochs after start_epoch

    # Select loss functions
    if adv_loss_type == 'hinge':
        disc_loss_fn = discriminator_loss_hinge
        gen_loss_fn = generator_loss_hinge
    else:
        disc_loss_fn = discriminator_loss_bce
        gen_loss_fn = generator_loss_bce

# Register frontend charts (after adversarial config is known)
if args.frontend and is_main:
    monitor.register_chart('Generator', [
        {'label': 'Train', 'color': '#c88650'},
        {'label': 'Val',   'color': '#b8bb26'},
    ])
    if use_adversarial:
        monitor.register_chart('Discriminator', [
            {'label': 'Train', 'color': '#d3869b'},
            {'label': 'Val',   'color': '#83a598'},
        ])

# ============================================================================
# Optimizer and scheduler
# ============================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(params["lr"]),
    weight_decay=float(params.get("weight_decay", 0.0))
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=params["num_epochs"],
    eta_min=float(params.get("min_lr", 1e-6))
)

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
    ckpt_modules = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
    if use_adversarial:
        ckpt_modules['discriminator'] = discriminator
        ckpt_modules['disc_optimizer'] = disc_optimizer
    meta = load_checkpoint(checkpoint_path, ckpt_modules)
else:
    print("No checkpoint found - Starting training from scratch")
    meta = {}

start_epoch = meta.get('epoch', -1) + 1
best_loss = meta.get('test_loss', float('inf'))
best_loss_epoch = -1
train_losses = meta.get('train_losses', [])
test_losses = meta.get('test_losses', [])

# ============================================================================
# DDP wrapping (after checkpoint loading, before training)
# ============================================================================

raw_model = model
raw_discriminator = discriminator
if ddp:
    model = DDP(model, device_ids=[local_rank])
    if discriminator is not None:
        discriminator = DDP(discriminator, device_ids=[local_rank])

# ============================================================================
# CSV logging setup
# ============================================================================

log_file = results_path + 'training_log.csv'
if is_main and (not os.path.exists(log_file) or start_epoch == 0):
    with open(log_file, 'w') as f:
        f.write('epoch,split,total_loss,flow_loss,disc_loss,adv_loss\n')

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

    if use_adversarial:
        # Free flow graph & grads, enable checkpointing like real training loop
        optimizer.zero_grad(set_to_none=True)
        del loss
        torch.cuda.empty_cache()
        if not model.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        torch.cuda.reset_peak_memory_stats()

        # Generation (inference)
        t0 = time.perf_counter()
        generated = model.generate_fixed_steps(
            z0, text_hidden, text_mask=text_mask,
            num_steps=adv_euler_steps,
            cfg_scale=1.0
        )
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t0

        # Discriminator forward + backward
        t0 = time.perf_counter()
        disc_optimizer.zero_grad()
        D_real = discriminator(zt)
        D_fake = discriminator(generated.detach())
        loss_disc = disc_loss_fn(D_real, D_fake)
        loss_disc.backward()
        disc_optimizer.step()
        torch.cuda.synchronize()
        t_disc = time.perf_counter() - t0

        disc_loss_item = loss_disc.item()
        d_real_mean = D_real.mean().item()
        d_fake_mean = D_fake.mean().item()
        del D_real, D_fake, loss_disc

        # Generator adversarial forward + backward
        t0 = time.perf_counter()
        optimizer.zero_grad()
        D_fake_gen = discriminator(generated)
        loss_adv = gen_loss_fn(D_fake_gen)
        (lambda_adv * loss_adv).backward()
        torch.cuda.synchronize()
        t_adv = time.perf_counter() - t0

        print(f"\n--- Adversarial ---")
        print(f"  Generate ({adv_euler_steps} steps): {t_gen*1000:7.1f} ms")
        print(f"  Disc update:          {t_disc*1000:7.1f} ms")
        print(f"  Gen adv backward:     {t_adv*1000:7.1f} ms")
        print(f"  Peak mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        val_batch = next(iter(test_loader))
        train_batch = next(iter(train_loader))

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
    total_flow_loss = 0
    total_disc_loss = 0
    total_adv_loss = 0

    if use_adversarial and epoch >= adv_start_epoch and not raw_model.gradient_checkpointing:
        raw_model.enable_gradient_checkpointing()

    # Compute effective lambda with ramp schedule
    adv_active = use_adversarial and epoch >= adv_start_epoch
    effective_lambda = 0.0
    if adv_active:
        adv_epoch = epoch - adv_start_epoch  # epochs since adversarial activated
        if adv_epoch < ramp_start:
            effective_lambda = 0.0
        elif adv_epoch >= ramp_end:
            effective_lambda = lambda_adv
        else:
            effective_lambda = lambda_adv * (adv_epoch - ramp_start) / (ramp_end - ramp_start)
    
    for batch_idx, (x0, z0, xt, zt, c_txt, c_hidden, c_mask) in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training", disable=not is_main)):
        if args.frontend and is_main:
            monitor.update_batch(batch_idx, total_batches=len(train_loader), mode='train')

        # Use pre-computed embeddings directly
        z0 = z0.to(device) * scale_factor
        zt = zt.to(device) * scale_factor
        c_hidden = c_hidden.to(device)
        c_mask = c_mask.to(device)

        # ---- Flow matching loss (always) ----
        loss_flow = flow_matching_loss(model, z0, zt, c_hidden, text_mask=c_mask, eps=eps)
        total_flow_loss += loss_flow.item()

        # ---- Backward flow loss (frees graph before generation in adv branch) ----
        optimizer.zero_grad()
        # In DDP + adversarial mode, defer gradient sync until after both backwards
        no_sync_ctx = model.no_sync() if (ddp and adv_active) else nullcontext()
        with no_sync_ctx:
            loss_flow.backward()

        if adv_active:

            # ---- Generate goal states (use raw_model to bypass DDP forward) ----
            if effective_lambda > 0:
                # Need gradients for generator update
                generated = raw_model.generate_fixed_steps(
                    z0, c_hidden, text_mask=c_mask,
                    num_steps=adv_euler_steps,
                    cfg_scale=1.0,
                )
            else:
                # Disc warmup: no generator grads needed, save memory
                with torch.no_grad():
                    generated = raw_model.generate_fixed_steps(
                        z0, c_hidden, text_mask=c_mask,
                        num_steps=adv_euler_steps,
                        cfg_scale=1.0,
                    )

            # ---- Discriminator update (always runs) ----
            disc_optimizer.zero_grad()
            D_real = discriminator(zt)
            D_fake = discriminator(generated.detach())
            loss_disc = disc_loss_fn(D_real, D_fake)
            loss_disc.backward()
            disc_optimizer.step()

            del D_real, D_fake
            total_disc_loss += loss_disc.item()
            del loss_disc

            # ---- Generator adversarial loss (only when lambda > 0) ----
            if effective_lambda > 0:
                D_fake_for_gen = discriminator(generated)
                loss_adv = gen_loss_fn(D_fake_for_gen)
                total_adv_loss += effective_lambda * loss_adv.item()

                (effective_lambda * loss_adv).backward()

        # Manually sync generator gradients across DDP workers (deferred from no_sync)
        if ddp and adv_active:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)

        if params.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])

        optimizer.step()
    
    total_train_loss = total_flow_loss + total_adv_loss

    avg_train_loss = total_train_loss / len(train_loader)
    avg_flow_loss = total_flow_loss / len(train_loader)
    avg_disc_loss = total_disc_loss / len(train_loader)
    avg_adv_loss = total_adv_loss / len(train_loader)

    # Average losses across DDP workers
    if ddp:
        loss_tensor = torch.tensor([avg_train_loss, avg_flow_loss, avg_disc_loss, avg_adv_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size
        avg_train_loss, avg_flow_loss, avg_disc_loss, avg_adv_loss = loss_tensor.tolist()

    train_losses.append(avg_train_loss)

    # Log to CSV (rank 0 only)
    if is_main:
        with open(log_file, 'a') as f:
            if adv_active:
                f.write(f'{epoch},train,{avg_train_loss:.6f},{avg_flow_loss:.6f},{avg_disc_loss:.6f},{avg_adv_loss:.6f}\n')
            else:
                f.write(f'{epoch},train,{avg_train_loss:.6f},{avg_flow_loss:.6f},,\n')
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    model.eval()
    total_test_loss = 0
    total_val_flow_loss = 0
    total_val_disc_loss = 0
    total_val_adv_loss = 0
    val_disc_correct = 0
    val_disc_total = 0
    
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
            total_val_flow_loss += test_loss.item()

            # Discriminator evaluation on val set
            if adv_active:
                z_gen = raw_model.sample_euler(
                    z0, c_hidden, text_mask=c_mask,
                    num_steps=adv_euler_steps,
                    cfg_scale=1.0,
                )
                D_real_val = discriminator(zt)
                D_fake_val = discriminator(z_gen)
                val_d_loss = disc_loss_fn(D_real_val, D_fake_val)
                val_g_loss = gen_loss_fn(D_fake_val)
                total_val_disc_loss += val_d_loss.item()
                total_val_adv_loss += effective_lambda * val_g_loss.item()
                # Discriminator accuracy (logit > 0 = real, logit < 0 = fake)
                val_disc_correct += (D_real_val > 0).sum().item() + (D_fake_val < 0).sum().item()
                val_disc_total += D_real_val.numel() + D_fake_val.numel()
    
    total_test_loss = total_val_flow_loss + total_val_adv_loss

    avg_test_loss = total_test_loss / len(test_loader)
    avg_val_flow_loss = total_val_flow_loss / len(test_loader)
    avg_val_disc_loss = total_val_disc_loss / len(test_loader)
    avg_val_adv_loss = total_val_adv_loss / len(test_loader)
    disc_accuracy = val_disc_correct / max(val_disc_total, 1)

    # Average val losses across DDP workers
    if ddp:
        val_tensor = torch.tensor([avg_test_loss, avg_val_flow_loss, avg_val_disc_loss, avg_val_adv_loss, val_disc_correct, val_disc_total], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        avg_test_loss = val_tensor[0].item() / world_size
        avg_val_flow_loss = val_tensor[1].item() / world_size
        avg_val_disc_loss = val_tensor[2].item() / world_size
        avg_val_adv_loss = val_tensor[3].item() / world_size
        disc_accuracy = val_tensor[4].item() / max(val_tensor[5].item(), 1)

    test_losses.append(avg_test_loss)
    
    # Log to CSV (rank 0 only)
    if is_main:
        with open(log_file, 'a') as f:
            if adv_active:
                f.write(f'{epoch},val,{avg_test_loss:.6f},{avg_val_flow_loss:.6f},{avg_val_disc_loss:.6f},{avg_val_adv_loss:.6f}\n')
            else:
                f.write(f'{epoch},val,{avg_test_loss:.6f},{avg_val_flow_loss:.6f},,\n')
    
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
        chart_data = {
            'Generator': {'Train': avg_train_loss, 'Val': avg_test_loss},
        }
        table_data = {
            'Generator': {
                'Train': {'Flow Matching': avg_flow_loss},
                'Val':   {'Flow Matching': avg_test_loss},
            },
        }
        if adv_active:
            chart_data['Discriminator'] = {'Train': avg_disc_loss, 'Val': avg_val_disc_loss}
            table_data['Generator']['Train'] = {
                'Total': avg_train_loss,
                'Flow Matching': avg_flow_loss,
                'Adversarial': avg_adv_loss,
            }
            table_data['Generator']['Val'] = {
                'Total': avg_test_loss,
                'Flow Matching': avg_val_flow_loss,
                'Adversarial': avg_val_adv_loss,
            }
            table_data['Discriminator'] = {
                'Train': {'Loss': avg_disc_loss},
                'Val':   {'Loss': avg_val_disc_loss},
            }
        monitor.update_epoch(epoch, charts=chart_data, tables=table_data)
    
    # Update scheduler
    scheduler.step()
    
    # Update progress bar
    postfix = {
        'Train': f'{avg_train_loss:.4f}',
        'Test': f'{avg_test_loss:.4f}',
        'LR': f'{scheduler.get_last_lr()[0]:.2e}',
    }
    if adv_active:
        postfix['D_acc'] = f'{disc_accuracy:.1%}'
    pbar.set_postfix(postfix)
    
    # -------------------------------------------------------------------------
    # Save checkpoint (rank 0 only, use raw model to avoid module. prefix)
    # -------------------------------------------------------------------------
    if ddp:
        dist.barrier()  # Ensure all ranks finish the epoch before saving
    if is_main and avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        if use_ema:
            ema.swap_parameters_with_ema(store_params_in_ema=True)
        save_modules = {'model': raw_model, 'optimizer': optimizer, 'scheduler': scheduler}
        if use_adversarial and raw_discriminator is not None:
            save_modules['discriminator'] = raw_discriminator
            save_modules['disc_optimizer'] = disc_optimizer
        save_checkpoint(params["model_path"], save_modules, {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
        })
        if use_ema:
            ema.swap_parameters_with_ema(store_params_in_ema=True)

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

    # Raw images for display (numpy uint8 [H, W, C])
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
                    z_init, c_hidden_vis, text_mask=c_mask_vis, cfg_scale=cfg_scale,
                    context_cfg_scale=context_cfg_scale, prompt_cfg_scale=prompt_cfg_scale,
                )
    
            # Compute metrics
            cos_sims = F.cosine_similarity(z_generated.flatten(1), z_target_gt.flatten(1), dim=1)
            mse_errors = F.mse_loss(z_generated, z_target_gt, reduction='none').mean(dim=[1, 2])
            
            z_generated_all.append(z_generated)
            cos_sims_all.append(cos_sims)
            mse_errors_all.append(mse_errors)
            
            print(f"CFG={cfg_scale:.1f}: cos_sim={cos_sims.mean():.4f}±{cos_sims.std():.4f}, MSE={mse_errors.mean():.4f}±{mse_errors.std():.4f}")

    # Visualize
    if theia_decoder is not None:
        n_rows = 2 + len(cfg_scales)  # Initial + Target + Generated for each CFG
        fig, axes = plt.subplots(n_rows, n_samples + 1, figsize=((n_samples + 1) * 2, n_rows * 2))
        
        # Row labels in first column
        row_labels = ['Initial', 'Target'] + [f'Gen (CFG={cfg:.1f})' for cfg in cfg_scales]
        
        for row_idx in range(n_rows):
            # Label column
            axes[row_idx, 0].text(0.5, 0.5, row_labels[row_idx], 
                                 ha='center', va='center', fontsize=10, fontweight='bold',
                                 transform=axes[row_idx, 0].transAxes, rotation=0)
            axes[row_idx, 0].axis('off')
        
        for i in range(n_samples):
            col_idx = i + 1  # Offset by 1 for label column
            
            # Task description above first row
            wrapped_text = '\n'.join(textwrap.wrap(c_txt_vis[i], width=25))
            axes[0, col_idx].text(0.5, 1.15, wrapped_text, transform=axes[0, col_idx].transAxes,
                                 fontsize=8, fontweight='bold', ha='center', va='bottom')
            
            # Initial (numpy uint8 [H, W, C])
            axes[0, col_idx].imshow(x0_vis[i])
            axes[0, col_idx].axis('off')
            
            # Target (numpy uint8 [H, W, C])
            axes[1, col_idx].imshow(xt_vis[i])
            axes[1, col_idx].axis('off')
            
            # Generated for each CFG
            for cfg_idx, (z_gen, cos_sims, mse_errors) in enumerate(zip(z_generated_all, cos_sims_all, mse_errors_all)):
                row_idx = 2 + cfg_idx
                xg_recon = decode(z_gen / scale_factor)
                img = xg_recon[i].cpu().permute(1, 2, 0).numpy()
                axes[row_idx, col_idx].imshow(np.clip(img, 0, 1))
                axes[row_idx, col_idx].set_title(f'cos={cos_sims[i]:.3f}\nMSE={mse_errors[i]:.3f}', fontsize=8)
                axes[row_idx, col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(results_path + 'generated_samples.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved visualization to {results_path}generated_samples.png")

if ddp:
    dist.destroy_process_group()

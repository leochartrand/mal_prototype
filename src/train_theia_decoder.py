"""
Training script for Theia Decoder.

Trains a small decoder for visualization of Theia latent representations.

Usage:
    Single GPU:  python train_theia_decoder.py --params <config>.yaml [--gpu 0] [--frontend] [--dummy]
    Multi-GPU:   torchrun --nproc_per_node=N src/train_theia_decoder.py --params <config>.yaml [--frontend]
"""
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import sys
import os
import time

from models.theia_decoder import Decoder
from utils.args import parse_args
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import visualize_decoder_samples, plot_loss_curves

import warnings

args = parse_args(sys.argv[1:])
params = yaml.safe_load(open("./params/"+args.params, 'r'))

dataset_path = params["dataset_path"]
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
        # Silence non-rank-0 processes
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
    monitor.register_chart('Loss', [
        {'label': 'Train', 'color': '#c88650'},
        {'label': 'Val',   'color': '#b8bb26'},
    ])

# ============================================================================
# Load data
# ============================================================================

print("Loading memory-mapped dataset...")
decoder_dataset_path = params["decoder_dataset_path"]
vision_model = params["vision_model"]

from utils.datasets import DecoderMemoryMappedDataset
train_dataset = DecoderMemoryMappedDataset(decoder_dataset_path, vision_model=vision_model, split='train')
test_dataset = DecoderMemoryMappedDataset(decoder_dataset_path, vision_model=vision_model, split='test')

n_train = len(train_dataset)
n_test = len(test_dataset)
print(f"Train samples: {n_train}, Test samples: {n_test}")

# Dummy mode: limit samples
if args.dummy:
    B = params["batch_size"]
    train_dataset = DecoderMemoryMappedDataset(decoder_dataset_path, vision_model=vision_model, indices=train_dataset.indices[:B * 2])
    test_dataset = DecoderMemoryMappedDataset(decoder_dataset_path, vision_model=vision_model, indices=test_dataset.indices[:B * 2])
    print(f"[DUMMY MODE] Using only {len(train_dataset)} train, {len(test_dataset)} test samples")

batch_size = params["batch_size"]
if ddp:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
else:
    train_sampler = None
    test_sampler = None
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                          sampler=train_sampler, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         sampler=test_sampler, num_workers=0, pin_memory=False)

print("Building Decoder model...")

model = Decoder(**params["model_params"])

model = model.to(device)

# Optimizer and scheduler
lr = float(params.get("lr", 1e-4))
weight_decay = float(params.get("weight_decay", 0.0))
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["num_epochs"])

print("Checking for existing checkpoint to resume from...")
train_losses = []
test_losses = []
best_loss = float('inf')
best_loss_epoch = -1
start_epoch = 0

resume_from_checkpoint = True
checkpoint_path = params["model_path"]

if resume_from_checkpoint:
    meta = load_checkpoint(checkpoint_path, {'model': model, 'optimizer': optimizer})
else:
    meta = {}

start_epoch = meta.get('epoch', -1) + 1
best_loss = meta.get('test_loss', float('inf'))
train_losses = meta.get('train_losses', [])
test_losses = meta.get('test_losses', [])

# Early stopping
patience = params.get('patience', 5)
patience_counter = 0

print("Theia Decoder model built.")

# DDP wrapping (after checkpoint loading, before training)
raw_model = model
if ddp:
    model = DDP(model, device_ids=[local_rank])

print("Setting up training...")
model.train()
for p in model.parameters():
    p.requires_grad_(True)

# Set up CSV logging
log_file = results_path + 'training_log.csv'
if is_main and (not os.path.exists(log_file) or start_epoch == 0):
    with open(log_file, 'w') as f:
        f.write('epoch,split,total_loss\n')

print("Starting training...")

# ============================================================================
# Dummy mode: profile one step and exit
# ============================================================================

if args.dummy and not ddp:
    torch.cuda.reset_peak_memory_stats()
    dummy_batch = next(iter(train_loader))

    # I/O: transfer to device
    t0 = time.perf_counter()
    z, target = dummy_batch
    z = z.to(device)
    target = target.to(device)
    torch.cuda.synchronize()
    t_io = time.perf_counter() - t0

    # Forward pass
    t0 = time.perf_counter()
    recon_frames = model.forward(z)
    loss = F.mse_loss(recon_frames, target)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0

    # Backward pass
    t0 = time.perf_counter()
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0

    print(f"\n--- Theia Decoder Dummy Step ---")
    print(f"  Batch:    {z.shape[0]} samples, z={list(z.shape)}, target={list(target.shape)}")
    print(f"  I/O:      {t_io*1000:7.1f} ms")
    print(f"  Forward:  {t_fwd*1000:7.1f} ms")
    print(f"  Backward: {t_bwd*1000:7.1f} ms")
    print(f"  Total:    {(t_io+t_fwd+t_bwd)*1000:7.1f} ms")
    print(f"  Loss:     {loss.item():.6f}")
    print(f"  Peak mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    sys.exit(0)

pbar = tqdm(range(start_epoch, params["num_epochs"]), desc="Training", initial=start_epoch, total=params["num_epochs"], disable=not is_main)
for epoch in pbar:
    if ddp:
        train_sampler.set_epoch(epoch)

    # Training
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()
    
    pbar2 = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training", disable=not is_main)
    for batch_idx, (z, target) in enumerate(pbar2):
        if args.frontend and is_main:
            monitor.update_batch(batch_idx, total_batches=len(train_loader), mode='train')

        z = z.to(device)
        target = target.to(device)
        
        recon_frames = model.forward(z)
        loss = F.mse_loss(recon_frames, target)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Average losses across DDP workers
    if ddp:
        loss_tensor = torch.tensor([avg_train_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size
        avg_train_loss = loss_tensor.item()

    train_losses.append(avg_train_loss)

    # Log train losses to CSV immediately
    if is_main:
        with open(log_file, 'a') as f:
            f.write(f'{epoch},train,{avg_train_loss:.6f}\n')
    
    # Validation
    model.eval()
    total_test_loss = 0
    
    with torch.no_grad():
        pbar3 = tqdm(test_loader, leave=False, desc=f"Epoch {epoch} Validation", disable=not is_main)
        for batch_idx, (z, target) in enumerate(pbar3):
            if args.frontend and is_main:
                monitor.update_batch(batch_idx, total_batches=len(test_loader), mode='val')
            
            z = z.to(device)
            target = target.to(device)
            
            recon_frames = model.forward(z)
            loss = F.mse_loss(recon_frames, target)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)

    # Average val losses across DDP workers
    if ddp:
        val_tensor = torch.tensor([avg_test_loss], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        avg_test_loss = val_tensor.item() / world_size

    test_losses.append(avg_test_loss)
    
    # Log val losses to CSV immediately
    if is_main:
        with open(log_file, 'a') as f:
            f.write(f'{epoch},val,{avg_test_loss:.6f}\n')
    
    # Save example validation reconstructions (rank 0 only)
    if is_main:
        with torch.no_grad():
            sample_z, sample_target = next(iter(test_loader))
            sample_z = sample_z.to(device)
            sample_target = sample_target.to(device)
            visualize_decoder_samples(
                raw_model, sample_z, sample_target, epoch,
                save_dir=results_path + '/decoded', num_vis=4,
            )
    
    # Update frontend with epoch-level averages (after visual is saved)
    if args.frontend and is_main:
        monitor.update_epoch(
            epoch,
            charts={'Loss': {'Train': avg_train_loss, 'Val': avg_test_loss}},
            tables={'Loss': {
                'Train': {'loss': avg_train_loss},
                'Val':   {'loss': avg_test_loss},
            }},
        )
    
    # Update scheduler
    scheduler.step()
    
    # Print progress
    pbar.set_postfix({'Train': avg_train_loss, 'Test': avg_test_loss})
      
    # Save best model (rank 0 only)
    if ddp:
        dist.barrier()
    if is_main and avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        patience_counter = 0
        save_checkpoint(params["model_path"], {'model': raw_model, 'optimizer': optimizer}, {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
        })
        print(f"\u2713 Saved best model with test loss: {best_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break


if is_main:
    print("Finetuning complete!")
    print(f"Best test loss: {best_loss:.4f}")

    plot_loss_curves(train_losses, test_losses, results_path + 'training_losses.png',
                     best_loss_epoch, title='Theia Decoder Training')

    raw_model.eval()

    # Visualize some reconstructions and associated latent codes
    viz_z, viz_data = next(iter(test_loader))
    viz_z = viz_z.to(device)
    viz_data = viz_data.to(device)
    visualize_decoder_samples(
        raw_model, viz_z, viz_data, epoch=-1,
        save_dir=results_path, num_vis=8,
    )

if ddp:
    dist.destroy_process_group()

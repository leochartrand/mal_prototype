"""
Training script for VQ-VAE.

Trains a Vector Quantized Variational Autoencoder for image reconstruction
and discrete latent representation learning.

Usage:
    Single GPU:  python train_vqvae.py --config val_vqvae.yaml [--gpu 0] [--frontend] [--dummy]
    Multi-GPU:   torchrun --nproc_per_node=N src/train_vqvae.py --config val_vqvae.yaml [--frontend]
"""

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import glob
import yaml
import sys
import os
import warnings

from models.vqvae import VQ_VAE
from utils.args import parse_args
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import plot_loss_curves
from utils.datasets import FrameDataset, augment_batch, resize_and_normalize_batch

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
    monitor.register_chart('Loss', [
        {'label': 'Train', 'color': '#c88650'},
        {'label': 'Val',   'color': '#b8bb26'},
    ], csv_column='loss')

# ============================================================================
# Load data
# ============================================================================

print("Loading and processing data...")
data_files = sorted(glob.glob(f"{params['dataset_path']}*.pkl"))
data = []
for file in data_files:
    with open(file, 'rb') as f:
        part = pickle.load(f)
        part_trajs = [torch.FloatTensor(np.array(traj)).permute(0, 3, 1, 2) / 255.0 for traj in part]
        data.extend(part_trajs)
        del part, part_trajs
print(f"Total trajectories: {len(data)}, Total frames: {sum(len(traj) for traj in data)}")

# Data split
split_indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(42))
n_train = int(0.8 * len(data))
train_data = [data[i] for i in split_indices[:n_train]]
test_data = [data[i] for i in split_indices[n_train:]]
del data, split_indices

train_dataset = FrameDataset(train_data)
test_dataset = FrameDataset(test_data)
del train_data, test_data

# Dummy mode
if args.dummy:
    B = params["batch_size"]
    train_dataset = FrameDataset(train_dataset.frames[:B * 2])
    test_dataset = FrameDataset(test_dataset.frames[:B * 2])
    print(f"[DUMMY MODE] Using {len(train_dataset)} train, {len(test_dataset)} test samples")

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

# ============================================================================
# Initialize model
# ============================================================================

print("Initializing VQ-VAE model...")
model = VQ_VAE(**params["model_params"])
model = model.to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=float(params["lr"]),
    weight_decay=float(params["weight_decay"]),
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["num_epochs"])

# ============================================================================
# Checkpoint loading
# ============================================================================

resume_from_checkpoint = params.get("resume_from_checkpoint", True)
if resume_from_checkpoint:
    meta = load_checkpoint(params["model_path"], {'model': model, 'optimizer': optimizer, 'scheduler': scheduler})
else:
    meta = {}

start_epoch = meta.get('epoch', -1) + 1
best_loss = meta.get('test_loss', float('inf'))
best_loss_epoch = -1
train_losses = meta.get('train_losses', [])
test_losses = meta.get('test_losses', [])

# Early stopping
patience = params.get('patience', 5)
patience_counter = 0

# Note: VQ-VAE uses compute_loss() instead of forward(), so we skip DDP wrapping
# and manually sync gradients. DistributedSampler handles data sharding.

# ============================================================================
# CSV logging setup
# ============================================================================

log_file = results_path + 'training_log.csv'
if is_main and (not os.path.exists(log_file) or start_epoch == 0):
    with open(log_file, 'w') as f:
        f.write('epoch,split,loss\n')

# ============================================================================
# Training loop
# ============================================================================

print("Starting training...")

pbar = tqdm(range(start_epoch, params["num_epochs"]), desc="Training",
            initial=start_epoch, total=params["num_epochs"], disable=not is_main)

for epoch in pbar:
    if ddp:
        train_sampler.set_epoch(epoch)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    model.train()
    total_train_loss = 0
    epoch_encoding_indices = []

    for batch_idx, data in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training", disable=not is_main)):
        if args.frontend and is_main:
            monitor.update_batch(batch_idx, total_batches=len(train_loader), mode='train')

        data = augment_batch(data.to(device), size=(model.imsize, model.imsize))
        outputs, losses = model.compute_loss(data)
        loss = losses['vq_loss'] + losses['recon_loss']

        epoch_encoding_indices.append(outputs['encoding_indices'].cpu().numpy())

        optimizer.zero_grad()
        loss.backward()

        # Manual gradient sync for DDP (compute_loss bypasses forward/DDP wrapper)
        if ddp:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)

        optimizer.step()
        total_train_loss += loss.item()

    # Embedding usage stats
    all_indices = np.concatenate(epoch_encoding_indices)
    embedding_counts = np.bincount(all_indices.flatten(), minlength=model.codebook_size)
    unique_embeddings = np.count_nonzero(embedding_counts)

    avg_train_loss = total_train_loss / len(train_loader)

    if ddp:
        loss_tensor = torch.tensor([avg_train_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = loss_tensor.item() / world_size

    train_losses.append(avg_train_loss)

    if is_main:
        with open(log_file, 'a') as f:
            f.write(f'{epoch},train,{avg_train_loss:.6f}\n')

    scheduler.step()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation", disable=not is_main)):
            if args.frontend and is_main:
                monitor.update_batch(batch_idx, total_batches=len(test_loader), mode='val')

            data = resize_and_normalize_batch(data.to(device), size=(model.imsize, model.imsize))
            outputs, losses = model.compute_loss(data)
            loss = losses['vq_loss'] + losses['recon_loss']
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)

    if ddp:
        val_tensor = torch.tensor([avg_test_loss], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        avg_test_loss = val_tensor.item() / world_size

    test_losses.append(avg_test_loss)

    if is_main:
        with open(log_file, 'a') as f:
            f.write(f'{epoch},val,{avg_test_loss:.6f}\n')

    # -------------------------------------------------------------------------
    # Frontend update (rank 0 only)
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

    # Progress bar
    pbar.set_postfix({
        'Train': f'{avg_train_loss:.4f}',
        'Test': f'{avg_test_loss:.4f}',
        'Emb': f'{unique_embeddings}/{model.codebook_size}',
    })

    # -------------------------------------------------------------------------
    # Save checkpoint + early stopping
    # -------------------------------------------------------------------------
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        patience_counter = 0
        if ddp:
            dist.barrier()
        if is_main:
            save_checkpoint(params["model_path"], {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}, {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'train_losses': train_losses,
                'test_losses': test_losses,
            })
    else:
        patience_counter += 1
        if ddp:
            dist.barrier()

    if patience_counter >= patience:
        if is_main:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break

# ============================================================================
# Post-training (rank 0 only)
# ============================================================================

if is_main:
    print("\nTraining complete!")
    print(f"Best test loss: {best_loss:.4f} at epoch {best_loss_epoch}")

    plot_loss_curves(train_losses, test_losses, results_path + 'training_losses.png',
                     best_loss_epoch, title='VQ-VAE Training')

    # Load best model
    checkpoint = torch.load(params["model_path"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Visualize reconstructions
    viz_data = next(iter(test_loader))
    viz_data = resize_and_normalize_batch(viz_data.to(device), size=(model.imsize, model.imsize))

    with torch.no_grad():
        outputs, _ = model.compute_loss(viz_data)
        n = min(viz_data.size(0), 8)

        fig, axes = plt.subplots(2, n, figsize=(2 * n, 6))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for i in range(n):
            img = viz_data[i].squeeze()
            axes[0, i].imshow(img.permute(1, 2, 0).cpu().numpy())
            axes[0, i].axis('off')
        axes[0, 0].set_title('Original')

        for i in range(n):
            img = outputs['reconstructions'][i].squeeze()
            axes[1, i].imshow(img.permute(1, 2, 0).detach().cpu().numpy())
            axes[1, i].axis('off')
        axes[1, 0].set_title('Reconstructed')

        plt.tight_layout()
        plt.savefig(results_path + 'final_reconstructions.png', dpi=300, bbox_inches='tight')
        plt.close()

if ddp:
    dist.destroy_process_group()

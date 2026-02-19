"""
Training script for PixelCNN (text-conditioned).

Trains a Gated PixelCNN to generate target VQ-VAE latent codes conditioned on
initial observation codes and text commands.

Usage:
    Single GPU:  python train_pixelcnn.py --config val_pixelcnn.yaml [--gpu 0] [--frontend] [--dummy]
    Multi-GPU:   torchrun --nproc_per_node=N src/train_pixelcnn.py --config val_pixelcnn.yaml [--frontend]
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pickle
import numpy as np
import sys
import os
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

from models.vqvae import VQ_VAE
from models.pixelcnn import GatedPixelCNN
from utils.nlp_utils import prepare_fewhot_commands, encode_commands, decode_commands
from utils.args import parse_args
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import plot_loss_curves
from utils.datasets import augment_batch, resize_and_normalize_batch

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
# Load VQ-VAE encoder (frozen)
# ============================================================================

vqvae = VQ_VAE(**params["vqvae"]["model_params"])
vqvae.load_state_dict(torch.load(params["vqvae"]["model_path"], weights_only=True))
vqvae = vqvae.to(device)
vqvae.eval()
for p in vqvae.parameters():
    p.requires_grad = False

# ============================================================================
# Load and process data
# ============================================================================

print("Loading and processing data...")

with open(params["dataset_path"], 'rb') as f:
    data = pickle.load(f)

x0_data = torch.FloatTensor(np.array([x[0] for x in data])).permute(0, 3, 1, 2) / 255.0
xt_data = torch.FloatTensor(np.array([x[1] for x in data])).permute(0, 3, 1, 2) / 255.0

commands_txt = [x[2] for x in data]
commands, vocab, word2idx, idx2word = prepare_fewhot_commands(commands_txt)
del data


class TransitionDataset(Dataset):
    """Simple dataset for (x0, xt, command) transitions."""
    def __init__(self, x0, xt, commands):
        self.x0 = x0
        self.xt = xt
        self.commands = commands

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return self.x0[idx], self.xt[idx], self.commands[idx]


# Random split
indices = torch.randperm(len(x0_data), generator=torch.Generator().manual_seed(42))
n_train = int(0.8 * len(x0_data))
train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_dataset = TransitionDataset(x0_data[train_indices], xt_data[train_indices], commands[train_indices])
test_dataset = TransitionDataset(x0_data[test_indices], xt_data[test_indices], commands[test_indices])
del x0_data, xt_data, commands

# Dummy mode
if args.dummy:
    B = params["batch_size"]
    train_dataset = TransitionDataset(train_dataset.x0[:B * 2], train_dataset.xt[:B * 2], train_dataset.commands[:B * 2])
    test_dataset = TransitionDataset(test_dataset.x0[:B * 2], test_dataset.xt[:B * 2], test_dataset.commands[:B * 2])
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
# Initialize PixelCNN model
# ============================================================================

initial_cont_size = vqvae.codebook_embed_dim * vqvae.discrete_size
command_size = len(vocab)
total_cond_size = initial_cont_size + command_size

pixelcnn = GatedPixelCNN(
    vqvae=vqvae,
    input_dim=vqvae.codebook_size,
    dim=params["model_params"]["dim"],
    n_layers=params["model_params"]["n_layers"],
    n_classes=total_cond_size,
    criterion=nn.CrossEntropyLoss().to(device),
).to(device)

optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=float(params["lr"]))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["num_epochs"])

# ============================================================================
# Checkpoint loading
# ============================================================================

resume_from_checkpoint = params.get("resume_from_checkpoint", True)
if resume_from_checkpoint:
    meta = load_checkpoint(params["model_path"], {'model': pixelcnn, 'optimizer': optimizer, 'scheduler': scheduler})
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

# Note: PixelCNN uses compute_loss() instead of forward(), so we skip DDP wrapping
# and manually sync gradients. DistributedSampler handles data sharding.

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


def augment_and_encode_batch(x0, xt):
    """Augment and encode image pairs through VQ-VAE."""
    seed = torch.randint(0, 100000, (1,)).item()
    torch.manual_seed(seed)
    x0 = augment_batch(x0, size=(vqvae.imsize, vqvae.imsize))
    torch.manual_seed(seed)
    xt = augment_batch(xt, size=(vqvae.imsize, vqvae.imsize))

    with torch.no_grad():
        outputs_0, _ = vqvae.compute_loss(x0)
        outputs_t, _ = vqvae.compute_loss(xt)
        z0 = outputs_0["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
        zt = outputs_t["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)

    return z0, zt


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
    pixelcnn.train()
    total_train_loss = 0

    for batch_idx, (x0, xt, c) in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training", disable=not is_main)):
        if args.frontend and is_main:
            monitor.update_batch(batch_idx, total_batches=len(train_loader), mode='train')

        z0, zt = augment_and_encode_batch(x0.to(device), xt.to(device))
        loss = pixelcnn.compute_loss(z0.to(device), zt.to(device), c.to(device), False)

        optimizer.zero_grad()
        loss.backward()

        if ddp:
            for p in pixelcnn.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)

        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    if ddp:
        loss_tensor = torch.tensor([avg_train_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = loss_tensor.item() / world_size

    train_losses.append(avg_train_loss)

    if is_main:
        with open(log_file, 'a') as f:
            f.write(f'{epoch},train,{avg_train_loss:.6f}\n')

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    pixelcnn.eval()
    total_test_loss = 0

    with torch.no_grad():
        for batch_idx, (x0, xt, c) in enumerate(tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation", disable=not is_main)):
            if args.frontend and is_main:
                monitor.update_batch(batch_idx, total_batches=len(test_loader), mode='val')

            z0, zt = augment_and_encode_batch(x0.to(device), xt.to(device))
            test_loss = pixelcnn.compute_loss(z0.to(device), zt.to(device), c.to(device), True)
            total_test_loss += test_loss.item()

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

    scheduler.step()

    pbar.set_postfix({
        'Train': f'{avg_train_loss:.4f}',
        'Test': f'{avg_test_loss:.4f}',
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
            save_checkpoint(params["model_path"], {'model': pixelcnn, 'optimizer': optimizer, 'scheduler': scheduler}, {
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
                     best_loss_epoch, title='PixelCNN Training')

    # Load best model
    checkpoint = torch.load(params["model_path"], map_location=device, weights_only=True)
    pixelcnn.load_state_dict(checkpoint['model'])
    pixelcnn.eval()

    # ========================================================================
    # Generate text-conditioned samples
    # ========================================================================

    print("Generating text-conditioned samples...")
    n_samples = 8

    with torch.no_grad():
        test_x0, test_xt, test_commands = next(iter(test_loader))
        test_z0, _ = augment_and_encode_batch(test_x0.to(device), test_xt.to(device))

        z0 = test_z0[:n_samples].long().to(device)
        c = test_commands[:n_samples].to(device)

        with torch.no_grad():
            z0_cont = F.normalize(vqvae.discrete_to_cont(z0).reshape(n_samples, -1), dim=1)
        cond = torch.cat([z0_cont, c], dim=1)

        zg, conf = pixelcnn.safe_generate(
            n_trials=5,
            min_confidence=-3.0,
            shape=(vqvae.root_len, vqvae.root_len),
            batch_size=n_samples,
            cond=cond,
            temperature=0.001,
        )

        x0_img = vqvae.decode(z0, cont=False).squeeze().permute(0, 2, 3, 1).detach().cpu().numpy()
        xg_img = vqvae.decode(zg.reshape(n_samples, -1), cont=False).squeeze().permute(0, 2, 3, 1).detach().cpu().numpy()
        command_strs = decode_commands(c, idx2word)

        samples = []
        for i in range(n_samples):
            samples.append({
                'initial': x0_img[i].squeeze(),
                'generated': xg_img[i].squeeze(),
                'command': command_strs[i],
                'confidence': conf[i].item(),
            })

        # Random (unconditional) samples
        samples2 = []
        for i in range(n_samples):
            z0_i = test_z0[n_samples + i].long().unsqueeze(0).to(device)
            with torch.no_grad():
                z0_cont_i = F.normalize(vqvae.discrete_to_cont(z0_i).reshape(1, -1), dim=1)
            dummy_cmd = torch.zeros(1, len(vocab), device=device)
            cond_i = torch.cat([z0_cont_i, dummy_cmd], dim=1)

            zg_i, conf_i = pixelcnn.safe_generate(
                n_trials=10,
                min_confidence=-3.0,
                shape=(vqvae.root_len, vqvae.root_len),
                batch_size=1,
                cond=cond_i,
                temperature=0.01,
            )

            x0_dec = vqvae.decode(z0_i, cont=False).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            xg_dec = vqvae.decode(zg_i.reshape(1, -1), cont=False).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            samples2.append({
                'initial': x0_dec.squeeze(),
                'generated': xg_dec.squeeze(),
                'confidence': conf_i.item(),
            })

        fig, axes = plt.subplots(4, n_samples, figsize=(16, 4))
        for i, sample in enumerate(samples):
            axes[0, i].imshow(sample['initial'])
            axes[0, i].set_title("Initial")
            axes[0, i].axis('off')
            axes[1, i].imshow(sample['generated'])
            axes[1, i].set_title(f"→ {sample['command']}")
            axes[1, i].axis('off')
            axes[1, i].text(0.5, -0.15, f"conf: {sample['confidence']:.2f}",
                            ha='center', va='top', transform=axes[1, i].transAxes, fontsize=8)

        for i, sample in enumerate(samples2):
            axes[2, i].imshow(sample['initial'])
            axes[2, i].set_title("Initial")
            axes[2, i].axis('off')
            axes[3, i].imshow(sample['generated'])
            axes[3, i].set_title("random")
            axes[3, i].axis('off')
            axes[3, i].text(0.5, -0.15, f"conf: {sample['confidence']:.2f}",
                            ha='center', va='top', transform=axes[3, i].transAxes, fontsize=8)

        plt.tight_layout()
        plt.savefig(results_path + 'generated_samples.png', dpi=300, bbox_inches='tight')
        plt.close()

if ddp:
    dist.destroy_process_group()

# Small Theia decoder for visualization
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import glob
import yaml
import sys
import os
import time

from models.theia_decoder import Decoder
from utils.args import parse_args
from frontend.training_monitor import TrainingMonitor

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

use_parallel = args.gpu == -1
if use_parallel:
    print("Using DataParallel for multiple GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

size=(224, 224)  # Image size for resizing

if args.frontend:
    monitor = TrainingMonitor(params)

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
# Data is pre-shuffled, use sequential access for fast mmap reads
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Building Decoder model...")

model = Decoder(**params["model_params"])

model = model.to(device)

# Optimizer and scheduler
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

print("Checking for existing checkpoint to resume from...")
# Before training loop, add lists to store losses
train_losses = []
test_losses = []
embedding_usage_history = []
best_loss = float('inf')
best_loss_epoch = -1
start_epoch = 0

# Check if we should resume from checkpoint (controlled by config)
resume_from_checkpoint = True
checkpoint_path = params["model_path"]

if resume_from_checkpoint and os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model'], strict=True)
    except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
        print(f"⚠ Warning: Checkpoint appears corrupted ({e})")
        print(f"⚠ Removing corrupted checkpoint and starting fresh")
        os.remove(checkpoint_path)
        checkpoint = None
    
    # Load optimizer state if available
    if checkpoint and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("✓ Loaded optimizer state")
    
    # Load training progress
    if checkpoint and 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        print(f"✓ Resuming from epoch {start_epoch}")
    
    if checkpoint and 'train_loss' in checkpoint and 'test_loss' in checkpoint:
        best_loss = checkpoint['test_loss']
        print(f"✓ Best loss so far: {best_loss:.4f}")
    
    # Load loss history if available
    if checkpoint and 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        print(f"✓ Loaded loss history ({len(train_losses)} epochs)")
else:
    if resume_from_checkpoint and not os.path.exists(checkpoint_path):
        print(f"Warning: resume_from_checkpoint=True but no checkpoint found at {checkpoint_path}")
    print("Starting training from scratch")

print("VQ-VAE model built.")

# Finetune VQ-VAE on the dataset
print("Setting up training...")
model.train()
for p in model.parameters():
    p.requires_grad_(True)

# Set up CSV logging
log_file = results_path + 'training_log.csv'
if not os.path.exists(log_file) or start_epoch == 0:
    with open(log_file, 'w') as f:
        f.write('epoch,split,total_loss\n')

print("Starting training...")

pbar = tqdm(range(start_epoch, params["num_epochs"]), desc="Training", initial=start_epoch, total=params["num_epochs"])
for epoch in pbar:    
    # Training
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()
    
    pbar2 = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training")
    for batch_idx, (z, target) in enumerate(pbar2):
        if args.frontend:
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
    train_losses.append(avg_train_loss)

    # Log train losses to CSV immediately
    with open(log_file, 'a') as f:
        f.write(f'{epoch},train,{avg_train_loss:.6f}\n')
    
    # Validation
    model.eval()
    total_test_loss = 0
    
    with torch.no_grad():
        pbar3 = tqdm(test_loader, leave=False, desc=f"Epoch {epoch} Validation")
        for batch_idx, (z, target) in enumerate(pbar3):
            if args.frontend:
                monitor.update_batch(batch_idx, total_batches=len(test_loader), mode='val')
            
            z = z.to(device)
            target = target.to(device)
            
            recon_frames = model.forward(z)
            loss = F.mse_loss(recon_frames, target)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Log val losses to CSV immediately
    with open(log_file, 'a') as f:
        f.write(f'{epoch},val,{avg_test_loss:.6f}\n')
    
    # Save example validation reconstructions
    model.eval()
    with torch.no_grad():
        # Get a batch from test loader
        sample_z, sample_target = next(iter(test_loader))
        sample_z = sample_z.to(device)
        sample_target = sample_target.to(device)
        frames = sample_target
        
        # Reconstruct from pre-embedded features
        sample_recons = model.forward(sample_z)
        
        # Save first 4 examples as a grid
        n_samples = min(4, frames.shape[0])
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*3, 6))
        
        for i in range(n_samples):
            # Original
            orig_img = frames[i].cpu().permute(1, 2, 0).numpy()
            orig_img = np.clip(orig_img, 0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstruction
            recon_img = sample_recons[i].cpu().permute(1, 2, 0).numpy()
            recon_img = np.clip(recon_img, 0, 1)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        if not os.path.exists(f'{results_path}/decoded'):
            os.makedirs(f'{results_path}/decoded')
        plt.savefig(f'{results_path}/decoded/epoch_{epoch+1:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    # Update frontend with epoch-level averages (after visual is saved)
    if args.frontend:
        monitor.update_epoch(
            epoch,
            avg_train_loss,
            {
                'loss': avg_train_loss
            },
            avg_test_loss,
            {
                'loss': avg_test_loss
            }
        )
    
    # Update scheduler
    scheduler.step()
    
    # Print progress
    pbar.set_postfix({'Train': avg_train_loss, 'Test': avg_test_loss})
      
    # Save best model
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
        }, params["model_path"])
        print(f"✓ Saved best model with test loss: {best_loss:.4f}")


print("Finetuning complete!")
print(f"Best test loss: {best_loss:.4f}")

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='orange')
plt.plot(test_losses, label='Test Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VQ-VAE Finetuning Loss')
plt.legend()
plt.grid(True)
plt.savefig(results_path + 'flexvae_finetuning_loss.png')
plt.show()

model.eval()

# Visualize some reconstructions and associated latent codes
data_iter = iter(test_loader)
viz_z, viz_data = next(data_iter)
viz_z = viz_z.to(device)
viz_data = viz_data.to(device)

with torch.no_grad():
    recons = model.forward(viz_z)
    
    n_imgs = min(8, viz_data.shape[0])  # Number of images to show
    
    fig, axes = plt.subplots(2, n_imgs, figsize=(2*n_imgs, 4))
    
    # Plot originals
    for i in range(n_imgs):
        img = viz_data[i].permute(1,2,0).cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, 0].set_ylabel('Original\n224×224', fontsize=10)
    
    # Plot reconstructions
    res = recons.shape[2]
    for i in range(n_imgs):
        img = recons[i].permute(1,2,0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, 0].set_ylabel(f'Recon\n{res}×{res}', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_path + 'final_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.show()


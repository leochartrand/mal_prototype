# Small Theia decoder for visualization
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
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
    with torch.no_grad():
        sample_z, sample_target = next(iter(test_loader))
        sample_z = sample_z.to(device)
        sample_target = sample_target.to(device)
        visualize_decoder_samples(
            model, sample_z, sample_target, epoch,
            save_dir=results_path + '/decoded', num_vis=4,
        )
    
    # Update frontend with epoch-level averages (after visual is saved)
    if args.frontend:
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
      
    # Save best model
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        save_checkpoint(params["model_path"], {'model': model, 'optimizer': optimizer}, {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
        })
        print(f"✓ Saved best model with test loss: {best_loss:.4f}")


print("Finetuning complete!")
print(f"Best test loss: {best_loss:.4f}")

plot_loss_curves(train_losses, test_losses, results_path + 'training_losses.png',
                 best_loss_epoch, title='Theia Decoder Training')

model.eval()

# Visualize some reconstructions and associated latent codes
model.eval()
viz_z, viz_data = next(iter(test_loader))
viz_z = viz_z.to(device)
viz_data = viz_data.to(device)
visualize_decoder_samples(
    model, viz_z, viz_data, epoch=-1,
    save_dir=results_path, num_vis=8,
)

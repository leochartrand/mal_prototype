"""
Training script for AffordanceFlowDiT.

Trains a flow matching DiT model to predict target affordance states from
initial observations and text commands.

Usage:
    python train_flowdit.py --params flowdit.yaml [--gpu 0] [--frontend] [--dummy]
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np
import sys
import os
import yaml
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import textwrap
import warnings

from models.flowdit import AffordanceFlowDiT, flow_matching_loss
from models.theia_decoder import Decoder as TheiaDecoder
from utils.datasets import MultiModalDataset
from utils.args import parse_args
from utils.ema import EMA

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

use_parallel = args.gpu == -1
if use_parallel:
    print("Using DataParallel for multiple GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
    # TODO: implement DataParallel
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using GPU {args.gpu}")

# Frontend monitor (optional)
if args.frontend:
    from frontend.training_monitor import TrainingMonitor
    monitor = TrainingMonitor(params)

# ============================================================================
# Load Theia decoder (for visualization only - encoder not needed with pre-embedded data)
# ============================================================================

# print("Loading Theia encoder...")
# from transformers import AutoModel
# theia = AutoModel.from_pretrained(params["theia"]["model_path"], trust_remote_code=True)
# theia = theia.to(device)
# theia.eval()
# for p in theia.parameters():
#     p.requires_grad = False

# Load Theia decoder for visualization (if available)
theia_decoder = None
if "theia_decoder" in params and os.path.exists(params["theia_decoder"]["model_path"]):
    print("Loading Theia decoder for visualization...")
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

print("Loading memory-mapped dataset...")
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
max_text_len = sample_c_hidden.shape[1]  # Sequence length (77)
del sample_c_hidden

B = params["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=0, pin_memory=False, collate_fn=mmap_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False, num_workers=0, pin_memory=False, collate_fn=mmap_collate_fn)

# ============================================================================
# Initialize model
# ============================================================================

print("Initializing AffordanceFlowDiT model...")

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
).to(device)

# Enable gradient checkpointing if specified
if params.get("gradient_checkpointing", False):
    print("Enabling gradient checkpointing...")
    model.enable_gradient_checkpointing()

# Scale factor for Theia latents
scale_factor = params.get("scale_factor", 1.0)
print(f"Using scale factor: {scale_factor}")

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

print("Checking for existing checkpoint...")
train_losses = []
test_losses = []
best_loss = float('inf')
best_loss_epoch = -1
start_epoch = 0

resume_from_checkpoint = params.get("resume_from_checkpoint", True)
checkpoint_path = params["model_path"]

if resume_from_checkpoint and os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        print("✓ Loaded model state")
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("✓ Loaded optimizer state")
        
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("✓ Loaded scheduler state")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"✓ Resuming from epoch {start_epoch}")
        
        if 'test_loss' in checkpoint:
            best_loss = checkpoint['test_loss']
            print(f"✓ Best loss so far: {best_loss:.6f}")
        
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            print(f"✓ Loaded loss history ({len(train_losses)} epochs)")
            
    except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
        print(f"⚠ Warning: Checkpoint appears corrupted ({e})")
        print("Starting training from scratch")
        start_epoch = 0
else:
    if resume_from_checkpoint:
        print(f"No checkpoint found at {checkpoint_path}")
    print("Starting training from scratch")

# ============================================================================
# CSV logging setup
# ============================================================================

log_file = results_path + 'training_log.csv'
if not os.path.exists(log_file) or start_epoch == 0:
    with open(log_file, 'w') as f:
        f.write('epoch,split,total_loss\n')

# ============================================================================
# Helper functions
# ============================================================================

# def encode_with_theia(images: torch.Tensor) -> torch.Tensor:
#     """Encode images with Theia encoder."""
#     # Theia expects: do_resize=False (already 224), do_rescale=False (already 0-1), do_normalize=True
#     with torch.no_grad():
#         z = theia.forward_feature(images, do_resize=False, do_rescale=False, do_normalize=True)
#     return z  # [B, 196, 384]


def decode_with_theia(z: torch.Tensor) -> torch.Tensor:
    """Decode Theia latents to images (if decoder available)."""
    if theia_decoder is None:
        return None
    with torch.no_grad():
        images = theia_decoder(z)
    return images


@torch.no_grad()
def visualize_samples(
    model: AffordanceFlowDiT,
    x0_raw_batch: torch.Tensor,
    x0_embed_batch: torch.Tensor,
    xt_raw_batch: torch.Tensor,
    xt_embed_batch: torch.Tensor,
    text_txt_batch: list,
    text_hidden_batch: torch.Tensor,
    text_mask_batch: torch.Tensor,
    epoch: int,
    num_vis: int = 4,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
):
    """Generate and visualize samples using pre-embedded data."""
    model.eval()
    
    n_vis = min(num_vis, x0_raw_batch.shape[0])
    
    # Raw images for display (numpy uint8 [H, W, C])
    x0_vis = x0_raw_batch[:n_vis]
    xt_vis = xt_raw_batch[:n_vis]
    # Pre-computed embeddings for generation
    z_init = x0_embed_batch[:n_vis].to(device) * scale_factor
    z_target_gt = xt_embed_batch[:n_vis].to(device) * scale_factor
    text_hidden_vis = text_hidden_batch[:n_vis].to(device)
    text_mask_vis = text_mask_batch[:n_vis].to(device)
    text_txt_vis = text_txt_batch[:n_vis]
    
    # Generate
    z_generated = model.sample_euler(z_init, text_hidden_vis, text_mask=text_mask_vis, num_steps=num_steps, cfg_scale=cfg_scale)
    
    # Create figure
    if theia_decoder is not None:
        # Decode latents to images for visualization
        xg_recon = decode_with_theia(z_generated / scale_factor)
        
        fig, axes = plt.subplots(3, n_vis, figsize=(n_vis * 2, 6))
        row_labels = ['Initial', 'Target', 'Generated']
        
        for i in range(n_vis):
            # Task description
            wrapped_text = '\n'.join(textwrap.wrap(text_txt_vis[i], width=15))
            axes[0, i].text(0.5, 1.15, wrapped_text, transform=axes[0, i].transAxes,
                           fontsize=8, fontweight='bold', ha='center', va='bottom')
            
            # Initial (numpy uint8 [H, W, C])
            axes[0, i].imshow(x0_vis[i])
            axes[0, i].axis('off')
            if i == 0:
                axes[0, 0].set_ylabel(row_labels[0], fontsize=9)
            
            # Target (numpy uint8 [H, W, C])
            axes[1, i].imshow(xt_vis[i])
            axes[1, i].axis('off')
            if i == 0:
                axes[1, 0].set_ylabel(row_labels[1], fontsize=9)
            
            # Generated (decoded tensor)
            img = xg_recon[i].cpu().permute(1, 2, 0).numpy()
            axes[2, i].imshow(np.clip(img, 0, 1))
            axes[2, i].axis('off')
            if i == 0:
                axes[2, 0].set_ylabel(row_labels[2], fontsize=9)
    else:
        # No decoder available - show latent space visualization
        # Compute cosine similarity between generated and GT
        cos_sims = F.cosine_similarity(
            z_generated.flatten(1), z_target_gt.flatten(1), dim=1
        )
        mse_errors = F.mse_loss(z_generated, z_target_gt, reduction='none').mean(dim=[1, 2])
        
        fig, axes = plt.subplots(2, n_vis, figsize=(n_vis * 2, 4))
        
        for i in range(n_vis):
            # Task description
            wrapped_text = '\n'.join(textwrap.wrap(text_txt_vis[i], width=15))
            axes[0, i].text(0.5, 1.15, wrapped_text, transform=axes[0, i].transAxes,
                           fontsize=8, fontweight='bold', ha='center', va='bottom')
            
            # Initial (numpy uint8 [H, W, C])
            axes[0, i].imshow(x0_vis[i])
            axes[0, i].set_title('Initial', fontsize=9)
            axes[0, i].axis('off')
            
            # Target (numpy uint8 [H, W, C])
            axes[1, i].imshow(xt_vis[i])
            axes[1, i].set_title(f'Target\ncos={cos_sims[i]:.3f}', fontsize=9)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    if not os.path.exists(f'{results_path}decoded'):
        os.makedirs(f'{results_path}decoded')
    plt.savefig(f'{results_path}decoded/epoch_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    model.train()

if args.dummy:
    dummy_batch = next(iter(train_loader))
    x0, z0, xt, zt, text_txt, text_hidden, text_mask = dummy_batch
    z0 = z0.to(device) * scale_factor
    zt = zt.to(device) * scale_factor
    text_hidden = text_hidden.to(device)
    text_mask = text_mask.to(device)
    with torch.cuda.amp.autocast():
        loss = flow_matching_loss(model, z0, zt, text_hidden, text_mask=text_mask)
        loss.backward()
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

# ============================================================================
# Training loop
# ============================================================================

print("Starting training...")
eps = 1e-5  # For flow matching interpolation

pbar = tqdm(range(start_epoch, params["num_epochs"]), desc="Training", initial=start_epoch, total=params["num_epochs"])

for epoch in pbar:
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    model.train()
    total_train_loss = 0
    
    for batch_idx, (x0, z0, xt, zt, c_txt, c_hidden, c_mask) in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training")):
        if args.frontend:
            monitor.update_batch(batch_idx, total_batches=len(train_loader), mode='train')
        
        # Use pre-computed embeddings directly
        z0 = z0.to(device) * scale_factor
        zt = zt.to(device) * scale_factor
        c_hidden = c_hidden.to(device)
        c_mask = c_mask.to(device)
        
        # Compute flow matching loss
        loss = flow_matching_loss(model, z0, zt, c_hidden, text_mask=c_mask, eps=eps)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Optional gradient clipping
        if params.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
        
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Log to CSV
    with open(log_file, 'a') as f:
        f.write(f'{epoch},train,{avg_train_loss:.6f}\n')
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    model.eval()
    total_test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (x0, z0, xt, zt, c_txt, c_hidden, c_mask) in enumerate(tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation")):
            if args.frontend:
                monitor.update_batch(batch_idx, total_batches=len(test_loader), mode='val')
            
            # Use pre-computed embeddings directly
            z0 = z0.to(device) * scale_factor
            zt = zt.to(device) * scale_factor
            c_hidden = c_hidden.to(device)
            c_mask = c_mask.to(device)
            
            # Compute loss
            test_loss = flow_matching_loss(model, z0, zt, c_hidden, text_mask=c_mask, eps=eps)
            total_test_loss += test_loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    # print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    
    # Log to CSV
    with open(log_file, 'a') as f:
        f.write(f'{epoch},val,{avg_test_loss:.6f}\n')
    
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    # Swap to EMA weights if using EMA
    if use_ema:
        ema.swap_parameters_with_ema(store_params_in_ema=True)
    
    test_x0, test_z0, test_xt, test_zt, test_c_txt, test_c_hidden, test_c_mask = next(iter(test_loader))
    visualize_samples(
        model, test_x0, test_z0, test_xt, test_zt, test_c_txt, test_c_hidden, test_c_mask,
        epoch=epoch,
        num_vis=min(4, B),
        num_steps=params.get("sample_steps", 50),
        cfg_scale=params.get("cfg_scale", 1.0),
    )
    
    if use_ema:
        ema.swap_parameters_with_ema(store_params_in_ema=True)
    
    # -------------------------------------------------------------------------
    # Update frontend
    # -------------------------------------------------------------------------
    if args.frontend:
        monitor.update_epoch(
            epoch,
            avg_train_loss,
            {'total_loss': avg_train_loss},
            avg_test_loss,
            {'total_loss': avg_test_loss}
        )
    
    # Update scheduler
    scheduler.step()
    
    # Update progress bar
    pbar.set_postfix({
        'Train': f'{avg_train_loss:.4f}',
        'Test': f'{avg_test_loss:.4f}',
        'LR': f'{scheduler.get_last_lr()[0]:.2e}'
    })
    
    # -------------------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------------------
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        
        # Swap to EMA weights if using EMA
        if use_ema:
            ema.swap_parameters_with_ema(store_params_in_ema=True)
        
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
        }, params["model_path"])
        print(f"✓ Saved best model with test loss: {best_loss:.4f}")
        
        if use_ema:
            ema.swap_parameters_with_ema(store_params_in_ema=True)

# ============================================================================
# Visualize loss history
# ============================================================================

print("\nTraining complete!")
print(f"Best test loss: {best_loss:.6f} at epoch {best_loss_epoch}")

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='orange')
plt.plot(test_losses, label='Test Loss', color='green')
# Add marker for best loss and epoch
if best_loss_epoch >= 0:
    plt.scatter([best_loss_epoch], [test_losses[best_loss_epoch]], color='green', zorder=5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
if best_loss_epoch >= 0:
    plt.title('Training and Test Losses (Best loss: {:.4f})'.format(test_losses[best_loss_epoch]))
else:
    plt.title('Training and Test Losses')
plt.legend()
plt.grid(True)
plt.savefig(results_path + 'training_losses.png')
plt.close()

# ============================================================================
# Visualize some samples
# ============================================================================

print("Generating final samples...")

# Load best model
checkpoint = torch.load(params["model_path"], map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model'])

if use_ema:
    ema.swap_parameters_with_ema(store_params_in_ema=True)

model.eval()

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
cfg_scales = [0.0, 1.0, 3.0, 5.0, 7.0]
z_generated_all = []
cos_sims_all = []
mse_errors_all = []

print(f"\nFinal generation metrics:")
print(f"Best test loss: {best_loss:.4f} at epoch {best_loss_epoch}\n")

for cfg_scale in cfg_scales:
    with torch.no_grad():
        z_generated = model.sample_euler(z_init, c_hidden_vis, text_mask=c_mask_vis, num_steps=50, cfg_scale=cfg_scale)
    
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
        wrapped_text = '\n'.join(textwrap.wrap(c_txt_vis[i], width=15))
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
            xg_recon = decode_with_theia(z_gen / scale_factor)
            img = xg_recon[i].cpu().permute(1, 2, 0).numpy()
            axes[row_idx, col_idx].imshow(np.clip(img, 0, 1))
            axes[row_idx, col_idx].set_title(f'cos={cos_sims[i]:.3f}\nMSE={mse_errors[i]:.3f}', fontsize=8)
            axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(results_path + 'generated_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization to {results_path}generated_samples.png")

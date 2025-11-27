import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import glob
import yaml
import sys
import os

from models.vq_llama import VQModel
from utils.datasets import FrameDataset

torch.cuda.empty_cache()

args = sys.argv
if len(args) > 1:
    params = yaml.safe_load(open("./params/"+args[1], 'r'))

dataset_path = params["dataset_path"]
model_path = params["model_path"]
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
results_path = params["results_path"]
if not os.path.exists(os.path.dirname(results_path)):
    os.makedirs(os.path.dirname(results_path))

size=(params["imsize"], params["imsize"])  # Image size for resizing

print("Loading and processing data...")
data_files = sorted(glob.glob(f"{dataset_path}/*.pkl"))  # Adjust path/pattern as needed
data = []
for file in tqdm(data_files):
    with open(file, 'rb') as f:
        part = pickle.load(f)
        # Each frame to float tensor, concat at dim 0, list of frames to float tensor traj
        # part_trajs = [torch.FloatTensor(np.array(traj)).permute(0,3,1,2) / 255.0 for traj in part]
        data.extend(part) 
        del part  # Free memory
        # del part_trajs
print(f"Total trajectories: {len(data)}, Total frames: {sum(len(traj) for traj in data)}")

# Data split
split_indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(1)) # For reproducibility
n_train = int(0.8 * len(data))
train_dataset = [data[i] for i in split_indices[:n_train]]
test_dataset = [data[i] for i in split_indices[n_train:]]
del data  # Free memory
del split_indices

train_dataset = FrameDataset(train_dataset)
test_dataset = FrameDataset(test_dataset)

batch_size = params["batch_size"] 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Building VQ-VAE model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VQModel(**params["model_params"])

# Old code (commented out for reference):
# model.eval()
# [p.requires_grad_(False) for p in model.parameters()]
# model.load_state_dict(torch.load("src/FlexVAR/pretrained/FlexVAE.pth", map_location='cpu', weights_only=True)['model'], strict=True)

model = model.to(device)

# Optimizer and scheduler
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

patch_nums = tuple(params["patch_nums"])

print("Starting training...")
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model'], strict=True)
    
    # Load optimizer state if available
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("✓ Loaded optimizer state")
    
    # Load training progress
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        print(f"✓ Resuming from epoch {start_epoch}")
    
    if 'train_loss' in checkpoint and 'test_loss' in checkpoint:
        best_loss = checkpoint['test_loss']
        print(f"✓ Best loss so far: {best_loss:.4f}")
    
    # Load loss history if available
    if 'train_losses' in checkpoint:
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

print("Starting training...")

print("Starting finetuning...")
pbar = tqdm(range(start_epoch, params["num_epochs"]), desc="Training", initial=start_epoch, total=params["num_epochs"])
for epoch in pbar:
    # Training
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()
    max_codebook_usage = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_commit_loss = 0
    total_diversity_loss = 0
    total_confidence_loss = 0
    
    pbar2 = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training")
    for data in pbar2:
        # data = augment_batch(data.to(device), size=size)
        data = data.to(device)

        # Forward pass returns (dec, diff, quant)
        # When v_patch_nums is used, diff is a list of (codebook_loss, commit_loss, diversity_loss, confidence_loss, codebook_usage) tuples
        recon_frames, diff_list, _ = model.forward(data, v_patch_nums=patch_nums)
        
        # Average losses across all scales
        codebook_loss = sum(d[0] for d in diff_list) / len(diff_list)
        commit_loss = sum(d[1] for d in diff_list) / len(diff_list)
        diversity_loss = sum(d[2] for d in diff_list) / len(diff_list)
        confidence_loss = sum(d[3] for d in diff_list) / len(diff_list)
        codebook_usage = max(d[4] for d in diff_list)  # Use max usage across scales
        
        # Reconstruction loss (MSE)
        target_size = data.shape[2]
        recon_loss = sum(
            F.mse_loss(
                recon, 
                F.interpolate(data, size=(recon.shape[2], recon.shape[3]), mode='bilinear', align_corners=False)
            )
            for recon in recon_frames
        ) / len(recon_frames)

        # Total loss: reconstruction + VQ losses
        total_loss = recon_loss * params["recon_loss_weight"] + codebook_loss + commit_loss + diversity_loss + confidence_loss
        
        # Normalize by accumulation steps
        loss = total_loss
        
        # Backward pass
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_train_loss += loss.item()

        if codebook_usage is not None:
            max_codebook_usage = max(max_codebook_usage, codebook_usage)
        
        total_recon_loss += recon_loss.item()
        total_vq_loss += codebook_loss.item()
        total_commit_loss += commit_loss.item()
        total_diversity_loss += diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
        total_confidence_loss += confidence_loss.item() if isinstance(confidence_loss, torch.Tensor) else confidence_loss

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_vq_loss = total_vq_loss / len(train_loader)
    avg_commit_loss = total_commit_loss / len(train_loader)
    avg_diversity_loss = total_diversity_loss / len(train_loader)
    avg_confidence_loss = total_confidence_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader, leave=False, desc=f"Epoch {epoch} Validation"):
            # data = resize_and_normalize_batch(data.to(device), size=size)
            data = data.to(device)
            
            # Forward pass returns (dec, diff_list, quant)
            recon_frames, diff_list, _ = model.forward(data, v_patch_nums=patch_nums)
            
            # Average losses across all scales
            codebook_loss = sum(d[0] for d in diff_list) / len(diff_list)
            commit_loss = sum(d[1] for d in diff_list) / len(diff_list)
            diversity_loss = sum(d[2] for d in diff_list) / len(diff_list)
            confidence_loss = sum(d[3] for d in diff_list) / len(diff_list)
            
            # Reconstruction loss (MSE) across scales
            recon_loss = sum(
                F.mse_loss(
                    recon, 
                    F.interpolate(data, size=(recon.shape[2], recon.shape[3]), mode='bilinear', align_corners=False)
                )
                for recon in recon_frames
            ) / len(recon_frames)

            # Total loss
            loss = recon_loss + codebook_loss + commit_loss + diversity_loss + confidence_loss
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Save example validation reconstructions
    model.eval()
    with torch.no_grad():
        # Get a batch from test loader
        sample_frames = next(iter(test_loader)).to(device)
        # frames = resize_and_normalize_batch(sample_frames, size=size)
        frames = sample_frames
        
        # Reconstruct - returns list of reconstructions at different scales
        sample_recons, _, _ = model.forward(frames, v_patch_nums=patch_nums)
        sample_recon = sample_recons[-1]  # Use highest resolution reconstruction
        
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
            recon_img = sample_recon[i].cpu().permute(1, 2, 0).numpy()
            recon_img = np.clip(recon_img, 0, 1)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{results_path}epoch_{epoch+1:03d}_reconstruction.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    # Update scheduler
    scheduler.step()
    
    # Print progress
    pbar.set_postfix({'Train': avg_train_loss, 'Test': avg_test_loss, 
                      'Recon': avg_recon_loss,
                      'Cb': avg_vq_loss, 'Commit': avg_commit_loss, 
                      'Div': avg_diversity_loss, 'Conf': avg_confidence_loss,
                      'Usage': f"{int(max_codebook_usage*model.codebook_size)}/{model.codebook_size}"})
      
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
viz_data = next(data_iter)  # Note the comma after data
# viz_data = resize_and_normalize_batch(viz_data.to(device), size=size)
viz_data = viz_data.to(device)

with torch.no_grad():
    recons, _, _ = model.forward(viz_data, v_patch_nums=patch_nums)

    # Select a few scales to show progression
    # e.g., indices [0, 4, 9] = 16x16, 48x48, 128x128
    selected_indices = [0, len(recons)//2, -1]
    
    n_imgs = 8  # Number of images to show
    n_scales = len(selected_indices) + 1  # +1 for original
    
    fig, axes = plt.subplots(n_scales, n_imgs, figsize=(2*n_imgs, 2*n_scales))
    
    # Plot originals
    for i in range(n_imgs):
        img = viz_data[i].permute(1,2,0).cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, 0].set_ylabel('Original\n256×256', fontsize=10)
    
    # Plot selected scales
    for row, idx in enumerate(selected_indices, start=1):
        recon = recons[idx]
        res = recon.shape[2]
        for i in range(n_imgs):
            img = recon[i].permute(1,2,0).detach().cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[row, i].imshow(img)
            axes[row, i].axis('off')
            if i == 0:
                axes[row, 0].set_ylabel(f'Recon\n{res}×{res}', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_path + 'final_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.show()

torch.save(model.state_dict(), model_path)


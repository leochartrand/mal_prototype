import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np
import sys
import os
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vq_llama import VQModel as VQ_LLAMA
from models.flexvar import FlexVAR
from utils.nlp_utils import prepare_fewhot_commands, encode_commands, decode_commands
from utils.datasets import MultiModalDataset, augment_batch, resize_and_normalize_batch

args = sys.argv
if len(args) > 1:
    params = yaml.safe_load(open("./params/"+args[1], 'r'))

model_path = params["model_path"]
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
results_path = params["results_path"]
if not os.path.exists(os.path.dirname(results_path)):
    os.makedirs(os.path.dirname(results_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

vq_llama = VQ_LLAMA(**params["vq_llama"]["model_params"])
vq_llama.load_state_dict(torch.load(params["vq_llama"]["model_path"], weights_only=True)["model"])
vq_llama = vq_llama.to(device)
vq_llama.eval()
for p in vq_llama.parameters():
    p.requires_grad = False

# Load and process data
print("Loading and processing data...")

with open(params["dataset_path"], 'rb') as f:
    data = pickle.load(f)

# Convert data to torch tensors and create datasets
x0_data = torch.FloatTensor(np.array([x[0] for x in data])).permute(0,3,1,2) / 255.0 # [N, C, H, W]
xt_data = torch.FloatTensor(np.array([x[1] for x in data])).permute(0,3,1,2) / 255.0


commands_txt = [x[2] for x in data]  # list of strings
commands, vocab, word2idx, idx2word = prepare_fewhot_commands(commands_txt)

# Random split
indices = torch.randperm(len(x0_data), generator=torch.Generator().manual_seed(1)) # For reproducibility
n_train = int(0.8 * len(x0_data))
train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_dataset = MultiModalDataset(
    x0_data[train_indices], 
    xt_data[train_indices], 
    commands[train_indices])
test_dataset = MultiModalDataset(
    x0_data[test_indices], 
    xt_data[test_indices], 
    commands[test_indices])

B = params["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False)

size = (params["imsize"], params["imsize"])
patch_nums = tuple(params["model_params"]["patch_nums"])
depth = params["model_params"]["depth"]
heads = depth
width = depth * 64
dpr = 0.1 * depth/24
token_dropout_p=params["model_params"]["token_dropout_p"]
flash_if_available = params["model_params"]["flash_if_available"]
fused_if_available = params["model_params"]["fused_if_available"]

# Calculate actual conditioning size
last_scale = patch_nums[-1]
last_scale_tokens = last_scale * last_scale  # 16*16 = 256
initial_cont_size = vq_llama.codebook_embed_dim * last_scale_tokens
command_size = len(vocab) 
total_cond_size = initial_cont_size + command_size

# Initialize PixelCNN model
model = FlexVAR(
    vae_local=vq_llama,
    num_classes=total_cond_size, 
    depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
    norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, token_dropout_p=token_dropout_p,
    attn_l2_norm=True,
    patch_nums=patch_nums,
    flash_if_available=flash_if_available, fused_if_available=fused_if_available,
).to(device)

criterion=nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


print("Starting training...")
train_losses = []
test_losses = []
best_loss = float('inf')
best_loss_epoch = -1

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

def prepare_multiscale_for_flexvar(vq_llama, x, patch_nums):
    """
    Encode at multiple scales with FlexVAR's bicubic interpolation strategy.
    Returns continuous embeddings and discrete indices.
    """
    all_quant, _, all_indices = vq_llama.multi_scale_encode(x, patch_nums)
    
    # Apply bicubic interpolation between scales (FlexVAR strategy)
    interpolated_quant = []
    for num in range(len(all_quant) - 1):  # EXCLUDE LAST SCALE
        quant = all_quant[num]
        
        # Interpolate to next scale
        next_hw = patch_nums[num + 1]
        next_quant = F.interpolate(quant, size=(next_hw, next_hw), mode='bicubic')
        next_quant = next_quant.reshape(quant.shape[0], quant.shape[1], -1)
        interpolated_quant.append(next_quant)
    
    # Concatenate all scales
    all_quant_concat = torch.cat(interpolated_quant, dim=2).permute(0, 2, 1)  # [B, L, C]
    
    return all_quant_concat, all_indices

def gen_curr_patch_nums(patch_nums, full_prob=0.05):
    """Generate random subset of scales for training robustness"""
    if torch.rand(1).item() < full_prob:
        return patch_nums  # 5%: use full sequence
    
    # 95%: random subset, but always keep first and last
    if len(patch_nums) <= 3:
        return patch_nums
    
    # Sample roughly half of middle scales
    mid_indices = list(range(1, len(patch_nums) - 1))
    num_to_keep = max(1, len(mid_indices) // 2)
    kept_indices = sorted(torch.randperm(len(mid_indices))[:num_to_keep].tolist())
    kept_indices = [mid_indices[i] for i in kept_indices]
    
    result = [patch_nums[0]] + [patch_nums[i] for i in kept_indices] + [patch_nums[-1]]
    
    # Randomly drop 1-2 more scales (except first and last)
    x = torch.rand(1).item()
    if x > 0.9 and len(result) > 3:
        result.pop(torch.randint(1, len(result) - 1, (1,)).item())
    if x > 0.95 and len(result) > 3:
        result.pop(torch.randint(1, len(result) - 1, (1,)).item())
    
    return tuple(result)

pbar = tqdm(range(params["num_epochs"]), desc="Training")
for epoch in pbar:
    # Training loop
    model.train() # Set model to training mode
    total_train_loss = 0
    for x0, xt, c, in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training"):
        x0, xt, c = x0.to(device), xt.to(device), c.to(device)

        # Random scale sampling
        curr_patch_nums = gen_curr_patch_nums(patch_nums)

        # Augment with same seed
        seed = torch.randint(0, 100000, (1,)).item()
        torch.manual_seed(seed)
        x0_aug = augment_batch(x0, size=(128, 128))
        torch.manual_seed(seed)
        xt_aug = augment_batch(xt, size=(128, 128))
        
        # Encode at multiple scales
        with torch.no_grad():
            z0_quant, _, _ = vq_llama.encode(x0_aug)
            z0_quant_flat = z0_quant.reshape(x0.shape[0], z0_quant.shape[1], -1).permute(0, 2, 1)
            zt_quant, zt_indices = prepare_multiscale_for_flexvar(vq_llama, xt_aug, curr_patch_nums)
        
        # Prepare conditioning: flatten z0 + command (FIXED SIZE)
        z0_flat = z0_quant.reshape(x0.shape[0], -1)
        z0_flat = F.normalize(z0_flat, dim=1)
        conditioning = torch.cat([z0_flat, c], dim=1)  # [B, total_cond_size] - always same size
        
        # Forward through FlexVAR
        logits = model(
            label_B=conditioning,
            x_BLCv_wo_first_l=zt_quant,  
            infer_patch_nums=curr_patch_nums
        )

        # Loss
        loss = criterion(
            logits.reshape(-1, vq_llama.codebook_size),
            zt_indices.flatten()
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for x0, xt, c in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            x0, xt, c = x0.to(device), xt.to(device), c.to(device)
            
            # Resize (no augmentation for validation)
            x0_resized = resize_and_normalize_batch(x0, size=(128, 128))
            xt_resized = resize_and_normalize_batch(xt, size=(128, 128))
            
            # Use full scales for validation
            z0_quant, _, _ = vq_llama.encode(x0_resized)
            z0_quant_flat = z0_quant.reshape(x0.shape[0], z0_quant.shape[1], -1).permute(0, 2, 1)
            zt_quant, zt_indices = prepare_multiscale_for_flexvar(vq_llama, xt_resized, patch_nums)
            
            # Prepare conditioning
            z0_flat = z0_quant.reshape(x0.shape[0], -1)
            z0_flat = F.normalize(z0_flat, dim=1)
            conditioning = torch.cat([z0_flat, c], dim=1)
            
            # Forward
            logits = model(
                label_B=conditioning,
                x_BLCv_wo_first_l=zt_quant,
                infer_patch_nums=patch_nums
            )
            
            # Loss
            test_loss = criterion(
                logits.reshape(-1, vq_llama.codebook_size),
                zt_indices.flatten()
            )
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Save example validation samples after each epoch
    model.eval()
    vq_llama.eval()
    with torch.no_grad():
        # Get some test data for visualization
        test_x0, test_xt, test_commands = next(iter(test_loader))
        
        # Limit to available samples
        n_vis = min(4, test_x0.shape[0])
        
        # Resize
        test_x0_vis = resize_and_normalize_batch(test_x0[:n_vis].to(device), size=(128, 128))
        test_xt_vis = resize_and_normalize_batch(test_xt[:n_vis].to(device), size=(128, 128))
        test_commands_vis = test_commands[:n_vis].to(device)
        
        # Generate
        generated_images = model.generate_from_conditioning(
            vq_llama=vq_llama,
            z0_images=test_x0_vis,
            commands=test_commands_vis,
            patch_nums=patch_nums,
            device=device,
            cfg=1.5,
            temperature=0.001
        )
        
        # Get command names
        command_names = decode_commands(test_commands_vis, idx2word)
        
        # Visualize: initial, ground truth, generated
        fig, axes = plt.subplots(3, n_vis, figsize=(n_vis*2, 6))
        
        for i in range(n_vis):
            # Initial state
            x0_img = test_x0_vis[i].cpu().permute(1, 2, 0).numpy()
            x0_img = np.clip(x0_img, 0, 1)
            axes[0, i].imshow(x0_img)
            axes[0, i].set_title(f"Initial")
            axes[0, i].axis('off')
            
            # Ground truth
            xt_img = test_xt_vis[i].cpu().permute(1, 2, 0).numpy()
            xt_img = np.clip(xt_img, 0, 1)
            axes[1, i].imshow(xt_img)
            axes[1, i].set_title(f"Ground Truth")
            axes[1, i].axis('off')
            
            # Generated
            xg_img = generated_images[i].cpu().permute(1, 2, 0).numpy()
            xg_img = np.clip(xg_img, 0, 1)
            axes[2, i].imshow(xg_img)
            axes[2, i].set_title(f"→ {command_names[i]}")
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/flexvar/epoch_{epoch:03d}_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    # Print progress
    pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})

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
    
    if avg_test_loss < 1e-5:  # Early stopping condition
        print(f"Early stopping at epoch {epoch} with test loss {avg_test_loss}")
        break

# Visualize loss history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='orange')
plt.plot(test_losses, label='Test Loss', color='green')
# Add marker for best loss and epoch
plt.scatter([best_loss_epoch], [test_losses[best_loss_epoch]], color='green', zorder=5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses (Best loss: {:.4f})'.format(test_losses[best_loss_epoch]))
plt.legend()
plt.grid(True)
plt.savefig(results_path + 'training_losses.png')
plt.show()

model.load_state_dict(torch.load(model_path, weights_only=True)['model'])

# Visualize some samples
print("Generating text-conditioned samples...")
n_samples = 8
model.eval()
vq_llama.eval()
with torch.no_grad():
    # Get some test data for conditioning
    test_x0, test_xt, test_commands = next(iter(test_loader))
    
    # Resize
    test_x0_resized = resize_and_normalize_batch(test_x0[:n_samples].to(device), size=(128, 128))
    test_xt_resized = resize_and_normalize_batch(test_xt[:n_samples].to(device), size=(128, 128))
    test_commands_sample = test_commands[:n_samples].to(device)
    
    generated_images = model.generate_from_conditioning(
        vq_llama=vq_llama,
        z0_images=test_x0_resized,
        commands=test_commands_sample,
        patch_nums=patch_nums,
        device=device,
        cfg=1.5,
        temperature=0.001
    )

    # Decode initial states for visualization
    
    # Get command names
    command_names = decode_commands(test_commands_sample, idx2word)
    
    # Visualize: initial state + generated state
    samples = []
    for i in range(n_samples):
        x0_img = test_x0_resized[i].cpu().permute(1, 2, 0).numpy()
        x0_img = np.clip(x0_img, 0, 1)
        
        # Ground truth
        xt_img = test_xt_resized[i].cpu().permute(1, 2, 0).numpy()
        xt_img = np.clip(xt_img, 0, 1)
        
        # Generated
        xg_img = generated_images[i].cpu().permute(1, 2, 0).numpy()
        xg_img = np.clip(xg_img, 0, 1)
        
        samples.append({
            'initial': x0_img,
            'ground_truth': xt_img,
            'generated': xg_img,
            'command': command_names[i]
        })

    # Visualize: initial, ground truth, generated
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples*2, 6))
    
    for i, sample in enumerate(samples):
        # Initial state
        axes[0, i].imshow(sample['initial'])
        axes[0, i].set_title(f"Initial")
        axes[0, i].axis('off')
        
        # Ground truth
        axes[1, i].imshow(sample['ground_truth'])
        axes[1, i].set_title(f"Ground Truth")
        axes[1, i].axis('off')
        
        # Generated
        axes[2, i].imshow(sample['generated'])
        axes[2, i].set_title(f"→ {sample['command']}")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(results_path + 'generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

    # print("Generating random samples...")
    # samples2 = []
    # for i in range(n_samples):
    #     # Extract conditioning from test data
    #     z0 = test_initial[n_samples+i].long().unsqueeze(0).to(device)  
    #     c = test_commands[n_samples+i].unsqueeze(0).to(device)  
        
    #     # Prepare conditioning for PixelCNN
    #     with torch.no_grad(): 
    #         z0_cont = F.normalize(vq_llama.discrete_to_cont(z0).reshape(1, -1), dim=1)

    #     dummy_command = torch.zeros_like(c)
    #     cond = torch.cat([z0_cont, dummy_command], dim=1)  
        
    #     # Generate next state
    #     zg, conf = model.safe_generate(
    #         n_trials=10,
    #         min_confidence=-3.0,
    #         shape=(vq_llama.root_len, vq_llama.root_len), 
    #         batch_size=1, 
    #         cond=cond,
    #         temperature=0.01
    #     ) 
        
    #     # Decode to images
    #     x0 = vq_llama.decode(z0, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
    #     xg = vq_llama.decode(zg.reshape(1, -1), cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        
    #     samples2.append({
    #         'initial': x0.squeeze(),
    #         'generated': xg.squeeze(),
    #         'confidence': conf.item()
    #     })

    # fig, axes = plt.subplots(4, n_samples, figsize=(16, 4))
    # for i, sample in enumerate(samples):
    #     axes[0, i].imshow(sample['initial'])
    #     axes[0, i].set_title(f"Initial")
    #     axes[0, i].axis('off')
        
    #     axes[1, i].imshow(sample['generated']) 
    #     axes[1, i].set_title(f"→ {sample['command']}")
    #     axes[1, i].axis('off')
    #     axes[1, i].text(0.5, -0.15, f"conf: {sample['confidence']:.2f}", 
    #                 ha='center', va='top', transform=axes[1, i].transAxes, fontsize=8)


    # for i, sample in enumerate(samples2):
    #     axes[2, i].imshow(sample['initial'])
    #     axes[2, i].set_title(f"Initial")
    #     axes[2, i].axis('off')
        
    #     axes[3, i].imshow(sample['generated']) 
    #     axes[3, i].set_title(f"random")
    #     axes[3, i].axis('off')
    #     axes[3, i].text(0.5, -0.15, f"conf: {sample['confidence']:.2f}", 
    #                 ha='center', va='top', transform=axes[3, i].transAxes, fontsize=8)

    # plt.tight_layout()
    # plt.savefig(results_path + 'generated_samples.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # print("Generating multiple goals from single initial state...")

    # z0_indices = test_initial[0].long().unsqueeze(0).to(device)  
    # x0 = vqvae.decode(z0_indices, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()

    # # simple interface where x0 is shown and the user can input multiple commands
    # commands = []
    # print("Initial state shown. Enter commands (type 'done' when finished):")
    # plt.imshow(x0)
    # plt.axis('off')
    # plt.show()
    # while True:
    #     cmd = input("Enter command: ")
    #     if cmd.lower() == 'done':
    #         break
    #     commands.append(cmd.lower())
    #     if len(commands) >= 5:
    #         print("Maximum of 5 commands reached.")
    #         break
    
      
    # z0_cont = F.normalize(vqvae.discrete_to_cont(z0_indices).reshape(1, -1), dim=1)
    # x0_enc = vqvae.decode(z0_indices, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
    # goals = []
    # encoded_cmds = encode_commands(commands, word2idx, vocab).to(device)
    # for c in encoded_cmds:
    #     cond = torch.cat([z0_cont, c.unsqueeze(0)], dim=1)  
    #     # Generate next state
    #     zg, conf = pixelcnn.safe_generate(
    #         n_trials=5,
    #         min_confidence=-3.0,
    #         shape=(vqvae.root_len, vqvae.root_len), 
    #         batch_size=1, 
    #         cond=cond,
    #         temperature=0.001 
    #     ) 
    #     xg = vqvae.decode(zg.reshape(1, -1), cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
    #     goals.append((xg,conf.item()))
    
    # fig, axes = plt.subplots(1, len(commands)+2, figsize=(3*(len(commands)+2), 3))
    # axes[0].imshow(x0)
    # axes[0].set_title(f"Initial")
    # axes[0].axis('off')
    # axes[1].imshow(x0_enc)
    # axes[1].set_title(f"VQ-VAE Recon")
    # axes[1].axis('off')
    # for i, (ax, (goal,conf), c) in enumerate(zip(axes[2:], goals, encoded_cmds)):
    #     ax.imshow(goal)
    #     cmd_str = decode_commands(c.unsqueeze(0), idx2word)[0]
    #     ax.set_title(f"→ {cmd_str}")
    #     ax.axis('off')
    #     ax.text(0.5, -0.15, f"conf: {conf:.2f}", 
    #                 ha='center', va='top', transform=ax.transAxes, fontsize=8)
    # plt.tight_layout()
    # plt.savefig(results_path + 'multiple_goals.png', dpi=300, bbox_inches='tight')
    # plt.show()

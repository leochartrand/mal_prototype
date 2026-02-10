import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import sys
import os
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vqvae import VQ_VAE
from pixelcnn_baseline import GatedPixelCNN
from utils.datasets import augment_batch, resize_and_normalize_batch

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

vqvae = VQ_VAE(**params["vqvae"]["model_params"])
vqvae.load_state_dict(torch.load(params["vqvae"]["model_path"], weights_only=True))
vqvae = vqvae.to(device)
vqvae.eval()

# Load and process data
print("Loading and processing data...")

with open(params["dataset_path"], 'rb') as f:
    data = pickle.load(f)

# Convert data to torch tensors and create datasets
x0 = torch.FloatTensor(np.array([x[0] for x in data])).permute(0,3,1,2) / 255.0 # [N, C, H, W]
xt = torch.FloatTensor(np.array([x[1] for x in data])).permute(0,3,1,2) / 255.0

# Random split
indices = torch.randperm(len(x0), generator=torch.Generator().manual_seed(42)) # For reproducibility
n_train = int(0.8 * len(x0))
train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_dataset = TensorDataset(
    x0[train_indices], 
    xt[train_indices])
test_dataset = TensorDataset(
    x0[test_indices], 
    xt[test_indices])

batch_size = params["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Calculate actual conditioning size
total_cond_size = vqvae.codebook_embed_dim * vqvae.discrete_size

# Initialize PixelCNN model
pixelcnn = GatedPixelCNN(
    input_dim=vqvae.codebook_size, 
    dim=params["model_params"]["dim"], 
    n_layers=params["model_params"]["n_layers"], 
    n_classes=total_cond_size, 
    ).to(device)

optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=float(params["lr"]), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["num_epochs"])
criterion=nn.CrossEntropyLoss().cuda()



print("Starting training...")
train_losses = []
test_losses = []
best_loss = float('inf')
best_loss_epoch = -1

def augment_and_encode_batch(x0, xt):
    # Augment
    seed = torch.randint(0, 100000, (1,)).item()
    torch.manual_seed(seed)
    x0 = augment_batch(x0)
    torch.manual_seed(seed)
    xt = augment_batch(xt)

    # Encode
    with torch.no_grad():
        outputs_0, _ = vqvae.compute_loss(x0)
        outputs_t, _ = vqvae.compute_loss(xt)
        z0 = outputs_0["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
        zt = outputs_t["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
    
    return z0, zt

def resize_and_encode_batch(x0, xt):
    # Resize and normalize
    x0 = resize_and_normalize_batch(x0)
    xt = resize_and_normalize_batch(xt)

    # Encode
    with torch.no_grad():
        outputs_0, _ = vqvae.compute_loss(x0)
        outputs_t, _ = vqvae.compute_loss(xt)
        z0 = outputs_0["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
        zt = outputs_t["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
    
    return z0, zt

def compute_loss(z0, zt):

    root_len = vqvae.root_len
    codebook_size = vqvae.codebook_size

    # Data shape: [batch, input stack, input size]
    z0 = z0.long()  
    zt = zt.long().reshape(-1, root_len, root_len)  
    
    with torch.no_grad(): 
        z0 = vqvae.discrete_to_cont(z0).reshape(z0.shape[0], -1)
    
    # Train PixelCNN with images
    logits = pixelcnn(zt, z0)
    logits = logits.permute(0, 2, 3, 1).contiguous()

    loss = criterion(
        logits.view(-1, codebook_size),
        zt.contiguous().view(-1)
    )
    return loss    

pbar = tqdm(range(params["num_epochs"]), desc="Training")
for epoch in pbar:
    # Training loop
    pixelcnn.train() # Set model to training mode
    total_train_loss = 0
    for x0, xt, in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training"):
        z0, zt = augment_and_encode_batch(x0.to(device), xt.to(device))
        loss = compute_loss(z0.to(device), zt.to(device))
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    pixelcnn.eval()
    total_test_loss = 0
    with torch.no_grad():
        for x0, xt in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            z0, zt = augment_and_encode_batch(x0.to(device), xt.to(device))
            test_loss = compute_loss(z0.to(device), zt.to(device))
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

     # Print progress
    pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        torch.save(pixelcnn.state_dict(), model_path)
    
    if avg_test_loss < 1e-5:  # Early stopping condition
        print(f"Early stopping at epoch {epoch} with test loss {avg_test_loss}")
        break

    # if avg_test_loss > 1 and epoch >= 10:  # Prevent divergence
    #     print(f"Stopping training at epoch {epoch} due to high test loss {avg_test_loss}")
    #     break

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

pixelcnn.load_state_dict(torch.load(model_path, weights_only=True))

# Visualize some samples
print("Generating text-conditioned samples...")
n_samples = 8
pixelcnn.eval()
vqvae.eval()
with torch.no_grad():
    # Get some test data for conditioning
    test_initial, test_target = next(iter(test_loader))  
    test_initial, test_target = augment_and_encode_batch(test_initial.to(device), test_target.to(device))
    samples = []

    print("Generating random samples...")
    for i in range(n_samples):
        # Extract conditioning from test data
        x0 = test_initial[i].long().unsqueeze(0).to(device) 
        
        # Prepare conditioning for PixelCNN
        with torch.no_grad(): 
            z0_cont = vqvae.discrete_to_cont(x0).reshape(1, -1)

        cond = z0_cont
        
        # Generate next state
        zg = pixelcnn.generate(
            shape=(vqvae.root_len, vqvae.root_len), 
            batch_size=1, 
            cond=cond
        ) 
        
        # Decode to images
        x0 = vqvae.decode(x0, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        xg = vqvae.decode(zg.reshape(1, -1), cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        
        samples.append({
            'initial': x0.squeeze(),
            'generated': xg.squeeze()
        })

    fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))
    for i, sample in enumerate(samples):
        axes[0, i].imshow(sample['initial'])
        axes[0, i].set_title(f"Initial")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(sample['generated']) 
        axes[0, i].set_title(f"Generated")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(results_path + 'generated_samples2.png', dpi=300, bbox_inches='tight')
    plt.show()

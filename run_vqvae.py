import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import yaml
import sys
import os

from vqvae import VQ_VAE
from datasets import augment_batch, resize_and_normalize_batch

args = sys.argv
if len(args) > 1:
    params = yaml.safe_load(open("./params/"+args[1], 'r'))

model_path = params["model_path"]
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
results_path = params["results_path"]
if not os.path.exists(os.path.dirname(results_path)):
    os.makedirs(os.path.dirname(results_path))

# Load and process data
print("Loading and processing data...")
with open(params["dataset_path"], 'rb') as f:
    data = pickle.load(f)

# Convert data to torch tensors and create datasets
x0 = torch.FloatTensor(np.array([x[0] for x in data])) # [N, H, W, C]
xt = torch.FloatTensor(np.array([x[1] for x in data]))

x0 = x0.permute(0,3,1,2) # [N, C, H, W]
xt = xt.permute(0,3,1,2) 
x_data = torch.cat([x0, xt], dim=0) / 255.0# [2N, C, H, W]

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(x_data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = params["batch_size"]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Initializing VQ-VAE model and trainer...")
model = VQ_VAE(**params["model_params"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(list(model.parameters()),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

print("Starting training...")
# Before training loop, add lists to store losses
train_losses = []
test_losses = []
embedding_usage_history = []
best_loss = float('inf')
best_loss_epoch = -1

pbar = tqdm(range(params["num_epochs"]), desc="Training")
for epoch in pbar:
    model.train()
    total_train_loss = 0

    epoch_encoding_indices = []  # Track all indices for this epoch

    pbar2 = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training")
    for data, in pbar2:
        data = augment_batch(data.to(device), size=(model.imsize, model.imsize))
        outputs, losses = model.compute_loss(data)
        loss = losses['vq_loss'] + losses['recon_loss']
        # Collect encoding indices
        epoch_encoding_indices.append(outputs['encoding_indices'].cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    # Analyze embedding usage for this epoch
    all_indices = np.concatenate(epoch_encoding_indices)
    unique_embeddings = len(np.unique(all_indices))
    embedding_counts = np.bincount(all_indices.flatten(), minlength=model.num_embeddings)
    embedding_usage = np.count_nonzero(embedding_counts)
    
    embedding_usage_history.append({
        'epoch': epoch,
        'unique_used': embedding_usage,
        'total_embeddings': model.num_embeddings,
        'usage_distribution': embedding_counts
    })
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    scheduler.step()
    
    # Validation loop
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data, in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            data = resize_and_normalize_batch(data.to(device), size=(model.imsize, model.imsize))
            _, losses = model.compute_loss(data)
            loss = losses['vq_loss'] + losses['recon_loss']
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Print progress
    pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss, 
                      'Embeddings Used': f"{unique_embeddings}/{model.num_embeddings}"})
    
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        torch.save(model.state_dict(), model_path)
    
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

# Plot embedding usage evolution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
epochs = [h['epoch'] for h in embedding_usage_history]
usage_counts = [h['unique_used'] for h in embedding_usage_history]
plt.plot(epochs, usage_counts)
plt.xlabel('Epoch')
plt.ylabel('Number of Embeddings Used')
plt.title('Embedding Utilization Over Time')
plt.subplot(1, 2, 2)
final_distribution = embedding_usage_history[-1]['usage_distribution']
plt.bar(range(len(final_distribution)), final_distribution)
plt.xlabel('Embedding Index')
plt.ylabel('Usage Count')
plt.title('Final Embedding Usage Distribution')
plt.tight_layout()
plt.savefig(results_path + 'embedding_utilization.png')
plt.show()

# model.load_state_dict(torch.load(model_path))

# Visualize some reconstructions and associated latent codes
data_iter = iter(test_loader)
viz_data, = next(data_iter)  # Note the comma after data
viz_data = resize_and_normalize_batch(viz_data.to(device), size=(model.imsize, model.imsize))

with torch.no_grad():
    outputs, _ = model.compute_loss(viz_data)

    # Take first 8 images
    n = min(viz_data.size(0), 8)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, n, figsize=(2*n, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot original images
    for i in range(n):
        img = viz_data[i].squeeze()
        axes[0, i].imshow(img.permute(1,2,0).cpu().numpy())
        axes[0, i].axis('off')
    axes[0, 0].set_title('Original')

    # Plot reconstructions
    for i in range(n):
        img = outputs['reconstructions'][i].squeeze()
        axes[1, i].imshow(img.permute(1,2,0).detach().cpu().numpy())
        axes[1, i].axis('off')
    axes[1, 0].set_title('Reconstructed')

    plt.tight_layout()
    plt.savefig(results_path + 'final_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.show()


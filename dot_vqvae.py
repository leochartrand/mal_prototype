from vqvae import VQ_VAE
import torch
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim

# initialize VQVAE model and train on /data/dot_data.pkl
# generated from datagen.py
print("Initializing VQVAE model and trainer...")
model = VQ_VAE(
    input_channels=1,
    output_channels=1,
    num_hiddens=64,          
    num_residual_layers=2,   # number of residual layers
    num_residual_hiddens=8, # channels in residual layers
    num_embeddings=16,       # size of embedding codebook
    embedding_dim=2,         # dimension of embedding vectors
    commitment_cost=0.5,    # beta in the VQ-VAE paper
    decay=0.0,              # decay for EMA updates
    imsize=8,                # input image size
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_path = "./models/dot_vqvae/"
results_path = "./results/dot_vqvae/"

# Hyperparameters
batch_size = 128
num_epochs = 100
lr = 1e-3
weight_decay = 0

optimizer = optim.Adam(list(model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

# Load and process data
print("Loading and processing data...")
with open("data/dot_data.pkl", "rb") as f:
    data = pickle.load(f)

# Convert data to torch tensors and create datasets
x_data = torch.FloatTensor(np.array([x[0] for x in data])).unsqueeze(1)  # Add channel dimension

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(x_data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Starting training...")

# Before training loop, add lists to store losses
train_losses = []
test_losses = []
embedding_usage_history = []
vq_losses = []
recon_losses = []

pbar = tqdm(range(num_epochs), desc="Training")
for epoch in pbar:
    model.train()
    total_train_loss = 0
    epoch_encoding_indices = []  # Track all indices for this epoch

    for data, in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training"):
        data = data.to(device)
        vq_loss, _, _, recon_error, _, encoding_indices = model.compute_loss(data)
        loss = vq_loss + recon_error
        
        # Collect encoding indices
        epoch_encoding_indices.append(encoding_indices.cpu().numpy())
        
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
    
    # Validation loop
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data, in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            data = data.to(device)
            vq_loss, _, _, recon_error, _, _ = model.compute_loss(data)
            loss = vq_loss + recon_error
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

     # Print progress
    pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})
    
    
    # # Save reconstructions periodically
    # if epoch % save_period == 0:
    #     with torch.no_grad():
    #         data = next(iter(test_loader))[0][:8].to(device)
    #         _, recon, _, _, _, encoding_indices = model.compute_loss(data)
            
    #         comparison = torch.cat([
    #             data.view(-1, 1, 8, 8),
    #             recon.view(-1, 1, 8, 8),
    #             encoding_indices.view(-1, 1, 2, 2)
    #         ])
            
    #         save_dir = results_path + f'reconstructions_epoch_{epoch}.png'
    #         save_image(comparison.cpu(), save_dir, nrow=8)

    if avg_test_loss < 1e-5:  # Early stopping condition
        print(f"Early stopping at epoch {epoch} with test loss {avg_test_loss}")
        break

# Save the final model
print("Saving model...")
torch.save(model.state_dict(), model_path + 'vqvae_final.pt')

# Visualize loss history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses')
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

# model.load_state_dict(torch.load("./models/dot_vqvae/vqvae_final.pt", weights_only=True))

# Visualize some reconstructions and associated latent codes
data_iter = iter(train_loader)
viz_data, = next(data_iter)  # Note the comma after data
viz_data = viz_data.to(device)

with torch.no_grad():
    _, reconstructions, _, _, quantized, encoding_indices = model.compute_loss(viz_data)

    # Take first 8 images
    n = min(viz_data.size(0), 8)

    # Create a figure with subplots
    fig, axes = plt.subplots(3, n, figsize=(2*n, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot original images
    for i in range(n):
        axes[0, i].imshow(viz_data[i].squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
    axes[0, 0].set_title('Original')

    # Plot quantized representations
    encoding_indices = encoding_indices.view(batch_size,2,2)
    for i in range(n):
        indices = encoding_indices[i].cpu().numpy()
        axes[1, i].imshow(np.zeros((2, 2)), cmap='gray')  # black background
        for y in range(2):
            for x in range(2):
                axes[1, i].text(x, y, f'{indices[y,x]}', 
                            ha='center', va='center', color='white')
        axes[1, i].axis('off')
    axes[1, 0].set_title('Codes')

    # Plot reconstructions
    for i in range(n):
        axes[2, i].imshow(reconstructions[i][0].squeeze().detach().cpu().numpy(), cmap='gray')
        axes[2, i].axis('off')
    axes[2, 0].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(results_path + 'final_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.show()


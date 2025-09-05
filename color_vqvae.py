from vqvae import VQ_VAE
import torch
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim

# initialize VQVAE model and train on /data/color_data.pkl
# generated from datagen.py
print("Initializing VQVAE model and trainer...")
model = VQ_VAE(
    input_channels=3,
    output_channels=3,
    num_hiddens=128,          
    num_residual_layers=6,   # number of residual layers
    num_residual_hiddens=32, # channels in residual layers
    num_embeddings=32,       # size of embedding codebook
    embedding_dim=4,         # dimension of embedding vectors
    commitment_cost=0.5,    # beta in the VQ-VAE paper
    recon_weight=0.5,
    entropy_weight=0.1,
    decay=0.0,              # decay for EMA updates
    imsize=8,                # input image size
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_path = "./models/color_vqvae/"
results_path = "./results/color_vqvae/"

# Hyperparameters
batch_size = 128
num_epochs = 100
lr = 1e-3
weight_decay = 1e-5

optimizer = optim.Adam(list(model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    factor=0.5, patience=5, verbose=True)


# Load and process data
print("Loading and processing data...")
with open("data/color_data.pkl", "rb") as f:
    data = pickle.load(f)

# Convert data to torch tensors and create datasets
initial_img = torch.FloatTensor(np.array([x[0] for x in data])) # [N, C, H, W]
goal_img = torch.FloatTensor(np.array([x[3] for x in data]))  
# Concatenate initial and goal images along batch dimension
x_data = torch.cat([initial_img, goal_img], dim=0) 

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
entropy_losses = []

pbar = tqdm(range(num_epochs), desc="Training")
for epoch in pbar:
    model.train()
    total_train_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_entropy_loss = 0

    epoch_encoding_indices = []  # Track all indices for this epoch
    
    # Force the decoder to output something non-black
    with torch.no_grad():
        random_embedding = torch.randn(1, model.embedding_dim, model.root_len, model.root_len).to(device)
        output = model._decoder(random_embedding)

    for data, in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training"):
        data = data.to(device)
        vq_loss, _, _, recon_error, _, encoding_indices, entropy_loss = model.compute_loss(data)
        loss = vq_loss + recon_error + entropy_loss
        # Collect encoding indices
        epoch_encoding_indices.append(encoding_indices.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        total_vq_loss += vq_loss.item()
        total_recon_loss += recon_error.item()
        total_entropy_loss += entropy_loss.item()

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
    avg_vq_loss = total_vq_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_entropy_loss = total_entropy_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    vq_losses.append(avg_vq_loss)
    recon_losses.append(avg_recon_loss)
    entropy_losses.append(avg_entropy_loss)
    scheduler.step(avg_train_loss)# In your training loop, print EMA cluster sizes
    
    # Validation loop
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data, in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            data = data.to(device)
            vq_loss, _, _, recon_error, _, _, entropy_loss = model.compute_loss(data)
            loss = vq_loss + recon_error + entropy_loss
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)


    # print(f"Loss: {loss.item():.6f} = VQ: {vq_loss.item():.6f} + {recon_loss_w} * Recon: {recon_error.item():.6f} + {entropy_loss_w} * Entropy: {entropy_loss.item():.6f}")
    
     # Print progress
    pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss, 
                      'VQ Loss': avg_vq_loss, 'Recon Loss': avg_recon_loss, 
                      'Entropy Loss': avg_entropy_loss,
                      'Embeddings Used': f"{unique_embeddings}/{model.num_embeddings}"})
    

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

# Visualize individual losses
plt.figure(figsize=(10, 5))
plt.plot(vq_losses, label='VQ Loss')
plt.plot(recon_losses, label='Recon Loss')
plt.plot(entropy_losses, label='Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Individual Losses')
plt.legend()
plt.grid(True)
plt.savefig(results_path + 'individual_losses.png')
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

# model.load_state_dict(torch.load("./models/color_vqvae/vqvae_final.pt", weights_only=True))

# Visualize some reconstructions and associated latent codes
data_iter = iter(train_loader)
viz_data, = next(data_iter)  # Note the comma after data
viz_data = viz_data.to(device)

with torch.no_grad():
    _, reconstructions, _, _, quantized, encoding_indices, _ = model.compute_loss(viz_data)

    # Take first 8 images
    n = min(viz_data.size(0), 8)

    # Create a figure with subplots
    fig, axes = plt.subplots(3, n, figsize=(2*n, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot original images
    for i in range(n):
        axes[0, i].imshow(viz_data[i].squeeze().permute(1,2,0).cpu().numpy())
        axes[0, i].axis('off')
    axes[0, 0].set_title('Original')

    # Plot quantized representations
    encoding_indices = encoding_indices.view(batch_size,2,2)
    for i in range(n):
        indices = encoding_indices[i].cpu().numpy()
        axes[1, i].imshow(np.zeros((model.root_len, model.root_len)), cmap='gray')  # black background
        for y in range(model.root_len):
            for x in range(model.root_len):
                axes[1, i].text(x, y, f'{indices[y,x]}', 
                            ha='center', va='center', color='white')
        axes[1, i].axis('off')
    axes[1, 0].set_title('Codes')

    # Plot reconstructions
    for i in range(n):
        axes[2, i].imshow(reconstructions[i].squeeze().permute(1,2,0).detach().cpu().numpy())
        axes[2, i].axis('off')
    axes[2, 0].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(results_path + 'final_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.show()


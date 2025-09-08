import torch
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim
import yaml
import sys
from model_utils import get_model

args = sys.argv
if len(args) > 1:
    params = yaml.safe_load(open("./params/"+args[1], 'r'))

model_path = "./models/{dataset}/{model_name}/".format(
    dataset=params["dataset"],
    model_name=params["model_name"]
)
results_path = "./results/{dataset}/{model_name}/".format(
    dataset=params["dataset"],
    model_name=params["model_name"]
)
# Load and process data
print("Loading and processing data...")
with open(params["dataset_path"], 'rb') as f:
    data = pickle.load(f)

# Convert data to torch tensors and create datasets
initial_img = torch.FloatTensor(np.array([x[0] for x in data])) # [N, H, W, C]
goal_img = torch.FloatTensor(np.array([x[1] for x in data]))

# Ensure proper tensor shape
if initial_img.dim() == 3:  # If images are grayscale without channel dimension
    initial_img = initial_img.unsqueeze(-1)  # Add channel dimension
    goal_img = goal_img.unsqueeze(-1)
else: # Images are RGB, move channel dimension to second position
    initial_img = initial_img.permute(0,3,1,2) # [N, C, H, W]
    goal_img = goal_img.permute(0,3,1,2) 
x_data = torch.cat([initial_img, goal_img], dim=0) # [2N, C, H, W]
x_data = x_data / x_data.max()  # Normalize to [0, 1]

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(x_data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = params["batch_size"]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_name = params["model_name"]
print(f"Initializing {model_name} model and trainer...")
model = get_model(model_name, **params["model_params"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
num_epochs = params["num_epochs"]
lr = float(params["lr"])
weight_decay = float(params["weight_decay"])

optimizer = optim.Adam(list(model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    factor=0.5, patience=5)

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

    pbar2 = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training")
    for data, in pbar2:
        pbar2.set_postfix({'LR': scheduler.get_last_lr()[0]})
        data = data.to(device)
        outputs, losses = model.compute_loss(data)
        loss = losses['vq_loss'] + losses['recon_loss'] + losses['entropy_loss']
        # Collect encoding indices
        epoch_encoding_indices.append(outputs['encoding_indices'].cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        total_vq_loss += losses['vq_loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_entropy_loss += losses['entropy_loss'].item()

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
    scheduler.step(avg_train_loss)
    
    # Validation loop
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data, in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            data = data.to(device)
            _, losses = model.compute_loss(data)
            loss = losses['vq_loss'] + losses['recon_loss'] + losses['entropy_loss']
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

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
torch.save(model.state_dict(), model_path + params["model_save_name"])

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

# Visualize some reconstructions and associated latent codes
data_iter = iter(train_loader)
viz_data, = next(data_iter)  # Note the comma after data
viz_data = viz_data.to(device)

with torch.no_grad():
    outputs, _ = model.compute_loss(viz_data)

    # Take first 8 images
    n = min(viz_data.size(0), 8)

    # Create a figure with subplots
    fig, axes = plt.subplots(3, n, figsize=(2*n, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot original images
    for i in range(n):
        img = viz_data[i].squeeze()
        if img.dim() > 2:
            img = img.permute(1,2,0)
        axes[0, i].imshow(img.cpu().numpy())
        axes[0, i].axis('off')
    axes[0, 0].set_title('Original')

    # Plot reconstructions
    for i in range(n):
        img = outputs['reconstructions'][i].squeeze()
        if img.dim() > 2:
            img = img.permute(1,2,0)
        axes[1, i].imshow(img.detach().cpu().numpy())
        axes[1, i].axis('off')
    axes[1, 0].set_title('Reconstructed')

    plt.tight_layout()
    plt.savefig(results_path + 'final_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.show()


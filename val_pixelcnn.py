import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import ValDataset
from vqvae import VQ_VAE
from pixelcnn import GatedPixelCNN
from nlp_utils import prepare_commands, decode_commands

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VQ_VAE(
    input_channels=3,
    output_channels=3,
    num_hiddens=512,          
    num_residual_layers=4,   # number of residual layers
    num_residual_hiddens=64, # channels in residual layers
    num_embeddings=1024,     # size of embedding codebook
    embedding_dim=64,        # dimension of embedding vectors
    commitment_cost=0.25,    # beta in the VQ-VAE paper
    recon_weight=0.5,        # weight for reconstruction loss
    entropy_weight=0.05,     # weight for entropy loss
    decay=0.99,              # decay for EMA updates
    imsize=48,               # input image size,
    ignore_background=False, # activates weighting to ignore background (black) pixels in loss computation
)
vqvae.load_state_dict(torch.load("./models/val/vqvae/vqvae_final.pt", weights_only=True))
vqvae = vqvae.to(device)
vqvae.eval()

model_path = "./models/val_pixelcnn/"
results_path = "./results/val_pixelcnn/"

# Hyperparameters
batch_size = 256
num_epochs = 50
lr = 5e-5
n_layers = 20
dim = 256

# Load and process data
print("Loading and processing data...")

with open("data/val_data.pkl", 'rb') as f:
    data = pickle.load(f)

# Get VQ-VAE discrete codes (not raw images)
initial_codes = []  # VQ-VAE discrete codes for before images
target_codes = []   # VQ-VAE discrete codes for after images
with torch.no_grad():
    for x in tqdm(data):
        before_img = torch.FloatTensor(np.array(x[0])/255.).permute(2,0,1).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, C, H, W]
        after_img = torch.FloatTensor(np.array(x[1])/255.).permute(2,0,1).unsqueeze(0).unsqueeze(0).to(device)

        outputs, _ = vqvae.compute_loss(before_img)
        outputs2, _ = vqvae.compute_loss(after_img)

        initial_codes.append(outputs["encoding_indices"].squeeze())
        target_codes.append(outputs2["encoding_indices"].squeeze()) 

initial_codes = torch.stack(initial_codes) 
target_codes = torch.stack(target_codes) 

commands_txt = [x[2] for x in data]  # list of strings
commands, vocab, word2idx, idx2word = prepare_commands(commands_txt)

# Random split
indices = torch.randperm(len(initial_codes))
n_train = int(0.8 * len(initial_codes))
train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_dataset = ValDataset(
    initial_codes[train_indices], 
    commands[train_indices], 
    target_codes[train_indices])
test_dataset = ValDataset(
    initial_codes[test_indices], 
    commands[test_indices], 
    target_codes[test_indices])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Calculate actual conditioning size
initial_cont_size = vqvae.embedding_dim * vqvae.discrete_size
command_size = len(vocab) 
total_cond_size = initial_cont_size + command_size

# Initialize PixelCNN model
pixelcnn = GatedPixelCNN(
    vqvae=vqvae,
    input_dim=vqvae.num_embeddings, 
    dim=dim, 
    n_layers=n_layers, 
    n_classes=total_cond_size, 
    criterion=nn.CrossEntropyLoss().cuda()).to(device)
optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
pixelcnn.load_state_dict(torch.load("./models/val/pixelcnn/pixelcnn_final.pt", weights_only=True))
print("Starting training...")

train_losses = []
test_losses = []

pbar = tqdm(range(num_epochs), desc="Training")
for epoch in pbar:
    # Training loop
    pixelcnn.train() # Set model to training mode
    total_train_loss = 0
    for initial, command, target, in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training"):
        loss = pixelcnn.compute_loss(initial.to(device), command.to(device), target.to(device), False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    pixelcnn.eval()
    total_test_loss = 0
    with torch.no_grad():
        for initial, command, target, in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            test_loss = pixelcnn.compute_loss(initial.to(device), command.to(device), target.to(device), True)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

     # Print progress
    pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})
    
    if avg_test_loss < 1e-5:  # Early stopping condition
        print(f"Early stopping at epoch {epoch} with test loss {avg_test_loss}")
        break

# Save the final model
print("Saving model...")
torch.save(pixelcnn.state_dict(), model_path + "pixelcnn_final.pt")

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

# pixelcnn.load_state_dict(torch.load("./models/val_pixelcnn/pixelcnn_final.pt", weights_only=True))

# Visualize some samples
n_samples = 8
pixelcnn.eval()
vqvae.eval()
with torch.no_grad():
    # Get some test data for conditioning
    test_initial, test_commands, _, = next(iter(test_loader))    
    samples = []
    for i in range(n_samples):
        # Extract conditioning from test data
        viz_initial = test_initial[i].long().unsqueeze(0).to(device)  # [1, 4]
        viz_command = test_commands[i].unsqueeze(0).to(device)  # [1, 7] 
        
        # Prepare conditioning for PixelCNN
        with torch.no_grad(): 
            initial_cont = F.normalize(vqvae.discrete_to_cont(viz_initial).reshape(1, -1), dim=1)
        cond = torch.cat([initial_cont, viz_command], dim=1)  # [1, 12]
        
        # Generate next state
        generated_codes = pixelcnn.generate(
            shape=(vqvae.root_len, vqvae.root_len), 
            batch_size=1, 
            cond=cond,
            temperature=0.3
        )  # [1, 2, 2]
        
        # Decode to images
        initial_img = vqvae.decode(viz_initial, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        generated_img = vqvae.decode(generated_codes.reshape(1, -1), cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()

        # Get command name
        command = decode_commands(viz_command[0], idx2word)[0]
        
        samples.append({
            'initial': initial_img.squeeze(),
            'generated': generated_img.squeeze(), 
            'command': command
        })

    samples2 = []
    for i in range(n_samples):
        # Extract conditioning from test data
        viz_initial = test_initial[8+i].long().unsqueeze(0).to(device)  # [1, 4]
        viz_command = test_commands[8+i].unsqueeze(0).to(device)  # [1, 7] 
        
        # Prepare conditioning for PixelCNN
        with torch.no_grad(): 
            initial_cont = F.normalize(vqvae.discrete_to_cont(viz_initial).reshape(1, -1), dim=1)

        dummy_command = torch.zeros_like(viz_command)
        cond = torch.cat([initial_cont, viz_command], dim=1)  # [1, 39]
        
        # Generate next state
        generated_codes = pixelcnn.generate(
            shape=(vqvae.root_len, vqvae.root_len), 
            batch_size=1, 
            cond=cond,
            temperature=0.3
        )  # [1, 2, 2]
        
        # Decode to images
        initial_img = vqvae.decode(viz_initial, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        generated_img = vqvae.decode(generated_codes.reshape(1, -1), cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        
        samples2.append({
            'initial': initial_img.squeeze(),
            'generated': generated_img.squeeze()
        })

    fig, axes = plt.subplots(4, 8, figsize=(16, 4))
    for i, sample in enumerate(samples):
        axes[0, i].imshow(sample['initial'])
        axes[0, i].set_title(f"Initial")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(sample['generated']) 
        axes[1, i].set_title(f"→ {sample['command']}")
        axes[1, i].axis('off')

    for i, sample in enumerate(samples2):
        axes[2, i].imshow(sample['initial'])
        axes[2, i].set_title(f"Initial")
        axes[2, i].axis('off')
        
        axes[3, i].imshow(sample['generated']) 
        axes[3, i].set_title(f"random")
        axes[3, i].axis('off')

    plt.tight_layout()
    plt.savefig(results_path + 'generated_samples.png')
    plt.show()

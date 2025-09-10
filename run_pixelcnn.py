import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np
import sys
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import MultiModalDataset
from vqvae import VQ_VAE
from pixelcnn import GatedPixelCNN
from nlp_utils import prepare_commands, decode_commands

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VQ_VAE(**params["vqvae"]["model_params"])
vqvae.load_state_dict(torch.load("./models/{dataset}/vqvae/vqvae_final.pt".format(dataset=params["dataset"]), weights_only=True))
vqvae = vqvae.to(device)
vqvae.eval()

# Load and process data
print("Loading and processing data...")

with open(params["dataset_path"], 'rb') as f:
    data = pickle.load(f)


batch_size = params["batch_size"]
z0_data = []  # VQ-VAE discrete codes for before images
zt_data = []  # VQ-VAE discrete codes for after images
with torch.no_grad():
    for i in range(0, len(data), batch_size): # GPU might not handle full dataset at once
        x0_batch = torch.FloatTensor(np.array([x[0] for x in data[i:i+batch_size]])).permute(0,3,1,2).to(device) # [N, C, H, W]
        xt_batch = torch.FloatTensor(np.array([x[1] for x in data[i:i+batch_size]])).permute(0,3,1,2).to(device)

        # Normalize to [0, 1]
        max_value = torch.max(x0_batch).item()
        x0_batch = x0_batch / max_value
        xt_batch = xt_batch / max_value

        outputs_0, _ = vqvae.compute_loss(x0_batch)
        outputs_t, _ = vqvae.compute_loss(xt_batch)

        z0_data.append(outputs_0["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len))
        zt_data.append(outputs_t["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len))

z0_data = torch.cat(z0_data, dim=0)
zt_data = torch.cat(zt_data, dim=0)

commands_txt = [x[2] for x in data]  # list of strings
commands, vocab, word2idx, idx2word = prepare_commands(commands_txt)

# Random split
indices = torch.randperm(len(z0_data))
n_train = int(0.8 * len(z0_data))
train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_dataset = MultiModalDataset(
    z0_data[train_indices], 
    zt_data[train_indices], 
    commands[train_indices])
test_dataset = MultiModalDataset(
    z0_data[test_indices], 
    zt_data[test_indices], 
    commands[test_indices])

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
    dim=params["model_params"]["dim"], 
    n_layers=params["model_params"]["n_layers"], 
    n_classes=total_cond_size, 
    criterion=nn.CrossEntropyLoss().cuda()).to(device)

optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=float(params["lr"]))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


print("Starting training...")
train_losses = []
test_losses = []
best_loss = float('inf')
best_loss_epoch = -1

pbar = tqdm(range(params["num_epochs"]), desc="Training")
for epoch in pbar:
    # Training loop
    pixelcnn.train() # Set model to training mode
    total_train_loss = 0
    for z0, zt, c, in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training"):
        loss = pixelcnn.compute_loss(z0.to(device), zt.to(device), c.to(device), False)
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
        for z0, zt, c in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            test_loss = pixelcnn.compute_loss(z0.to(device), zt.to(device), c.to(device), True)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

     # Print progress
    pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        torch.save(pixelcnn.state_dict(), model_path + params["model_save_name"])
    
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

# pixelcnn.load_state_dict(torch.load(model_path + params["model_save_name"], weights_only=True))

# Visualize some samples
n_samples = 8
pixelcnn.eval()
vqvae.eval()
with torch.no_grad():
    # Get some test data for conditioning
    test_initial, _, test_commands = next(iter(test_loader))    
    samples = []
    for i in range(n_samples):
        # Extract conditioning from test data
        z0 = test_initial[i].long().unsqueeze(0).to(device)  
        c = test_commands[i].unsqueeze(0).to(device)  
        
        # Prepare conditioning for PixelCNN
        with torch.no_grad(): 
            z0_cont = F.normalize(vqvae.discrete_to_cont(z0).reshape(1, -1), dim=1)
        cond = torch.cat([z0_cont, c], dim=1)  
        # Generate next state
        zg = pixelcnn.generate(
            shape=(vqvae.root_len, vqvae.root_len), 
            batch_size=1, 
            cond=cond,
            temperature=0.001 # Very low temperature for text conditioning, to ensure correct command execution
        ) 
        
        # Decode to images
        x0 = vqvae.decode(z0, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        xg = vqvae.decode(zg.reshape(1, -1), cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()

        # Get command name
        command = decode_commands(c, idx2word)[0]
        
        samples.append({
            'initial': x0.squeeze(),
            'generated': xg.squeeze(), 
            'command': command
        })

    samples2 = []
    for i in range(n_samples):
        # Extract conditioning from test data
        z0 = test_initial[8+i].long().unsqueeze(0).to(device)  
        c = test_commands[8+i].unsqueeze(0).to(device)  
        
        # Prepare conditioning for PixelCNN
        with torch.no_grad(): 
            z0_cont = F.normalize(vqvae.discrete_to_cont(z0).reshape(1, -1), dim=1)

        dummy_command = torch.zeros_like(c)
        cond = torch.cat([z0_cont, c], dim=1)  
        
        # Generate next state
        zg = pixelcnn.generate(
            shape=(vqvae.root_len, vqvae.root_len), 
            batch_size=1, 
            cond=cond,
            temperature=0.3
        ) 
        
        # Decode to images
        x0 = vqvae.decode(z0, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        xg = vqvae.decode(zg.reshape(1, -1), cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        
        samples2.append({
            'initial': x0.squeeze(),
            'generated': xg.squeeze()
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
    plt.savefig(results_path + 'generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

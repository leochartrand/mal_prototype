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

from models.vqvae import VQ_VAE
from models.pixelcnn import GatedPixelCNN
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

vqvae = VQ_VAE(**params["vqvae"]["model_params"])
vqvae.load_state_dict(torch.load(params["vqvae"]["model_path"], weights_only=True))
vqvae = vqvae.to(device)
vqvae.eval()

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
indices = torch.randperm(len(x0_data), generator=torch.Generator().manual_seed(42)) # For reproducibility
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

batch_size = params["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Calculate actual conditioning size
initial_cont_size = vqvae.codebook_embed_dim * vqvae.discrete_size
command_size = len(vocab) 
total_cond_size = initial_cont_size + command_size

# Initialize PixelCNN model
pixelcnn = GatedPixelCNN(
    vqvae=vqvae,
    input_dim=vqvae.codebook_size, 
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

# Early stopping
patience = params.get('patience', 5)
patience_counter = 0

def augment_and_encode_batch(x0, xt):
    # Augment
    seed = torch.randint(0, 100000, (1,)).item()
    torch.manual_seed(seed)
    x0 = augment_batch(x0, size=(vqvae.imsize, vqvae.imsize))
    torch.manual_seed(seed)
    xt = augment_batch(xt, size=(vqvae.imsize, vqvae.imsize))

    # Encode
    with torch.no_grad():
        outputs_0, _ = vqvae.compute_loss(x0)
        outputs_t, _ = vqvae.compute_loss(xt)
        z0 = outputs_0["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
        zt = outputs_t["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
    
    return z0, zt

def resize_and_encode_batch(x0, xt):
    # Resize and normalize
    x0 = resize_and_normalize_batch(x0, size=(vqvae.imsize, vqvae.imsize))
    xt = resize_and_normalize_batch(xt, size=(vqvae.imsize, vqvae.imsize))

    # Encode
    with torch.no_grad():
        outputs_0, _ = vqvae.compute_loss(x0)
        outputs_t, _ = vqvae.compute_loss(xt)
        z0 = outputs_0["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
        zt = outputs_t["encoding_indices"].reshape(-1, vqvae.root_len, vqvae.root_len)
    
    return z0, zt

pbar = tqdm(range(params["num_epochs"]), desc="Training")
for epoch in pbar:
    # Training loop
    pixelcnn.train() # Set model to training mode
    total_train_loss = 0
    for x0, xt, c, in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training"):
        z0, zt = augment_and_encode_batch(x0.to(device), xt.to(device))
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
        for x0, xt, c in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            z0, zt = augment_and_encode_batch(x0.to(device), xt.to(device))
            test_loss = pixelcnn.compute_loss(z0.to(device), zt.to(device), c.to(device), True)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

     # Print progress
    pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_loss_epoch = epoch
        patience_counter = 0
        torch.save(pixelcnn.state_dict(), model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
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

pixelcnn.load_state_dict(torch.load(model_path, weights_only=True))

# Visualize some samples
print("Generating text-conditioned samples...")
n_samples = 8
pixelcnn.eval()
vqvae.eval()
with torch.no_grad():
    # Get some test data for conditioning
    test_initial, test_target, test_commands = next(iter(test_loader))  
    test_initial, test_target = augment_and_encode_batch(test_initial.to(device), test_target.to(device))
    samples = []
    # for i in range(n_samples):
    # Extract conditioning from test data
    z0 = test_initial[:n_samples].long().to(device)  
    c = test_commands[:n_samples].to(device)  
    # Prepare conditioning for PixelCNN
    with torch.no_grad(): 
        z0_cont = F.normalize(vqvae.discrete_to_cont(z0).reshape(n_samples, -1), dim=1)
    cond = torch.cat([z0_cont, c], dim=1)  
    # Generate next state
    zg, conf = pixelcnn.safe_generate(
        n_trials=5,
        min_confidence=-3.0,
        shape=(vqvae.root_len, vqvae.root_len), 
        batch_size=n_samples, 
        cond=cond,
        temperature=0.001 # Very low temperature for text conditioning, to ensure correct command execution
    ) 
    
    # Decode to images
    x0 = vqvae.decode(z0, cont=False).squeeze().permute(0,2,3,1).detach().cpu().numpy()
    xg = vqvae.decode(zg.reshape(n_samples, -1), cont=False).squeeze().permute(0,2,3,1).detach().cpu().numpy()

    # Get command name
    command = decode_commands(c, idx2word)
    
    for i in range(n_samples):
        samples.append({
            'initial': x0[i].squeeze(),
            'generated': xg[i].squeeze(), 
            'command': command[i],
            'confidence': conf[i].item()
        })

    print("Generating random samples...")
    samples2 = []
    for i in range(n_samples):
        # Extract conditioning from test data
        z0 = test_initial[n_samples+i].long().unsqueeze(0).to(device)  
        c = test_commands[n_samples+i].unsqueeze(0).to(device)  
        
        # Prepare conditioning for PixelCNN
        with torch.no_grad(): 
            z0_cont = F.normalize(vqvae.discrete_to_cont(z0).reshape(1, -1), dim=1)

        dummy_command = torch.zeros_like(c)
        cond = torch.cat([z0_cont, dummy_command], dim=1)  
        
        # Generate next state
        zg, conf = pixelcnn.safe_generate(
            n_trials=10,
            min_confidence=-3.0,
            shape=(vqvae.root_len, vqvae.root_len), 
            batch_size=1, 
            cond=cond,
            temperature=0.01
        ) 
        
        # Decode to images
        x0 = vqvae.decode(z0, cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        xg = vqvae.decode(zg.reshape(1, -1), cont=False).squeeze().permute(1,2,0).detach().cpu().numpy()
        
        samples2.append({
            'initial': x0.squeeze(),
            'generated': xg.squeeze(),
            'confidence': conf.item()
        })

    fig, axes = plt.subplots(4, n_samples, figsize=(16, 4))
    for i, sample in enumerate(samples):
        axes[0, i].imshow(sample['initial'])
        axes[0, i].set_title(f"Initial")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(sample['generated']) 
        axes[1, i].set_title(f"→ {sample['command']}")
        axes[1, i].axis('off')
        axes[1, i].text(0.5, -0.15, f"conf: {sample['confidence']:.2f}", 
                    ha='center', va='top', transform=axes[1, i].transAxes, fontsize=8)


    for i, sample in enumerate(samples2):
        axes[2, i].imshow(sample['initial'])
        axes[2, i].set_title(f"Initial")
        axes[2, i].axis('off')
        
        axes[3, i].imshow(sample['generated']) 
        axes[3, i].set_title(f"random")
        axes[3, i].axis('off')
        axes[3, i].text(0.5, -0.15, f"conf: {sample['confidence']:.2f}", 
                    ha='center', va='top', transform=axes[3, i].transAxes, fontsize=8)

    plt.tight_layout()
    plt.savefig(results_path + 'generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

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

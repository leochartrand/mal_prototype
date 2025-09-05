import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from vqvae import VQ_VAE
from pixelcnn import GatedPixelCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VQ_VAE(
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
vqvae.load_state_dict(torch.load("./models/dot_vqvae/vqvae_final.pt", weights_only=True))
vqvae = vqvae.to(device)
vqvae.eval()
root_len = vqvae.root_len
num_embeddings = vqvae.num_embeddings
embedding_dim = vqvae.embedding_dim
imsize = vqvae.imsize
input_channels = vqvae.input_channels
# output_channels = vqvae.output_channels

model_path = "./models/dot_pixelcnn/"
results_path = "./results/dot_pixelcnn/"

# Hyperparameters
batch_size = 128
num_epochs = 50
lr = 3e-4
n_layers = 15

# Load and process data
print("Loading and processing data...")
with open("data/dot_data.pkl", "rb") as f:
    data = pickle.load(f)

# Convert string commands to integer indices
command_to_idx = {"up": 0, "down": 1, "left": 2, "right": 3}

# Get VQ-VAE discrete codes (not raw images)
initial_codes = []  # VQ-VAE discrete codes for before images
target_codes = []   # VQ-VAE discrete codes for after images
commands = []
with torch.no_grad():
    for x in tqdm(data):
        before_img = torch.FloatTensor(x[0]).unsqueeze(0).unsqueeze(0).to(device)
        after_img = torch.FloatTensor(x[2]).unsqueeze(0).unsqueeze(0).to(device)
        
        _, _, _, _, _, z0 = vqvae.compute_loss(before_img)
        _, _, _, _, _, zt = vqvae.compute_loss(after_img)
        
        initial_codes.append(z0.squeeze())
        target_codes.append(zt.squeeze())
        commands.append(command_to_idx[x[1]])

initial_codes = torch.stack(initial_codes)  # [N,4]
target_codes = torch.stack(target_codes)   # [N,4] 
commands = F.one_hot(torch.LongTensor(commands), num_classes=4).float()  # [N,4]

# Stack all components: [initial, command, target]
encoded_data = torch.stack([initial_codes, commands, target_codes], dim=1)  # [N, 3, 4]

# Random split
indices = torch.randperm(len(encoded_data))
n_train = int(0.8 * len(encoded_data))

train_data = encoded_data[indices[:n_train]]
test_data = encoded_data[indices[n_train:]]

train_dataset = torch.utils.data.TensorDataset(train_data)
test_dataset = torch.utils.data.TensorDataset(test_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Calculate actual conditioning size
initial_cont_size = vqvae.embedding_dim * vqvae.discrete_size  # 2 * 4 = 8
command_size = 4  # one-hot commands
total_cond_size = initial_cont_size + command_size  # 8 + 4 = 12

# Initialize PixelCNN model
pixelcnn = GatedPixelCNN(
    vqvae=vqvae,
    input_dim=num_embeddings, 
    dim=32, 
    n_layers=n_layers, 
    n_classes=total_cond_size, 
    criterion=nn.CrossEntropyLoss().cuda()).to(device)
optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=lr)

print("Starting training...")

train_losses = []
test_losses = []

pbar = tqdm(range(num_epochs), desc="Training")
for epoch in pbar:
    # Training loop
    pixelcnn.train() # Set model to training mode
    total_train_loss = 0
    for data, in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}: Training"):
        loss = pixelcnn.compute_loss(data, False)
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
        for data, in tqdm(test_loader, leave=False, desc=f"Epoch {epoch}: Validation"):
            test_loss = pixelcnn.compute_loss(data, True)
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

# pixelcnn.load_state_dict(torch.load("./models/dot_pixelcnn/pixelcnn_final.pt", weights_only=True))

# Visualize some samples
n_samples = 8
pixelcnn.eval()
vqvae.eval()
with torch.no_grad():
    # Get some test data for conditioning
    test_batch, = next(iter(test_loader))
    test_batch = test_batch.to(device) 
    
    samples = []
    for i in range(n_samples):
        # Extract conditioning from test data
        initial_codes = test_batch[i, 0, :].long().unsqueeze(0)  # [1, 4]
        commands = test_batch[i, 1, :].unsqueeze(0)  # [1, 4] 
        
        # Prepare conditioning for PixelCNN
        initial_cont = vqvae.discrete_to_cont(initial_codes).reshape(1, -1)
        cond = torch.cat([initial_cont, commands], dim=1)  # [1, 12]
        
        # Generate next state
        generated_codes = pixelcnn.generate(
            shape=(vqvae.root_len, vqvae.root_len), 
            batch_size=1, 
            cond=cond
        )  # [1, 2, 2]
        
        # Decode to images
        initial_img = vqvae.decode(initial_codes, cont=False).cpu()
        generated_img = vqvae.decode(generated_codes.reshape(1, -1), cont=False).cpu()
        
        # Get command name
        cmd_idx = commands[0].argmax().item()
        cmd_names = ["up", "down", "left", "right"]
        
        samples.append({
            'initial': initial_img[0].squeeze(),
            'generated': generated_img[0].squeeze(), 
            'command': cmd_names[cmd_idx]
        })

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, sample in enumerate(samples):
        axes[0, i].imshow(sample['initial'], cmap='gray')
        axes[0, i].set_title(f"Initial")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(sample['generated'], cmap='gray') 
        axes[1, i].set_title(f"→ {sample['command']}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(results_path + 'generated_samples.png')
    plt.show()

import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import sys
import os
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vqvae import VQ_VAE
from utils.datasets import MultiModalDataset, resize_and_normalize_batch

args = sys.argv
if len(args) > 1:
    params = yaml.safe_load(open("./params/"+args[1], 'r'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VQ_VAE(**params["model_params"])
vqvae.load_state_dict(torch.load(params["model_path"], weights_only=True))
vqvae = vqvae.to(device)
vqvae.eval()

# Load and process data
print("Loading and processing data...")

with open("data/val_data.pkl", 'rb') as f:
    data = pickle.load(f)

# Convert data to torch tensors and create datasets
x0_data = torch.FloatTensor(np.array([x[0] for x in data])).permute(0,3,1,2) / 255.0 # [N, C, H, W]
xt_data = torch.FloatTensor(np.array([x[1] for x in data])).permute(0,3,1,2) / 255.0


commands_txt = [x[2] for x in data]  # list of strings

# Random perm
indices = torch.randperm(len(x0_data), generator=torch.Generator().manual_seed(42)) # For reproducibility

dataset = MultiModalDataset(
    x0_data[indices], 
    xt_data[indices], 
    np.array(commands_txt)[indices])

del data  # Free memory
del x0_data
del xt_data
del indices


batch_size = params["batch_size"]
batch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dz = []
commands = []

with torch.no_grad():
    for x0, xt, c, in tqdm(batch_loader, leave=False, desc=f"Encoding examples"):
        
        x0 = resize_and_normalize_batch(x0.to(device), size=(vqvae.imsize, vqvae.imsize))
        xt = resize_and_normalize_batch(xt.to(device), size=(vqvae.imsize, vqvae.imsize))
        outputs_0, _ = vqvae.compute_loss(x0)
        outputs_t, _ = vqvae.compute_loss(xt)
        z0 = outputs_0["quantized"]
        zt = outputs_t["quantized"]

        dz_batch = zt - z0  # [N, C, H, W]

        c = list(c)
        for i, cmd in enumerate(c):
            if "open drawer" in cmd:
                c[i] = 1
            elif "close drawer" in cmd:
                c[i] = 2
            elif "open pot" in cmd:
                c[i] = 3
            elif "close pot" in cmd:
                c[i] = 4
            elif "put" in cmd:
                if "emptyspace" in cmd:
                    c[i] = 5  # put emptyspace
                elif "tray" in cmd:
                    c[i] = 6
                else:
                    c[i] = 7  # other put
            elif "pickup" in cmd:
                c[i] = 8
            else:
                c[i] = 9

        dz_flat = dz_batch.reshape(dz_batch.shape[0], -1).cpu().numpy()  # [N, D]
        dz.append(dz_flat)
        commands.append(c)

dz = np.concatenate(dz, axis=0)  # [Total_N, D]
c = np.concatenate(commands, axis=0)  # [Total_N]

label_map = {
    1: "open drawer",
    2: "close drawer",
    3: "open pot",
    4: "close pot",
    5: "put _ emptyspace",
    6: "put _ tray",
    7: "put *",
    8: "pickup *",
    9: "other"
}

# Delta latent space visualization
# Get best projection to 2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=200)
dz_pca = pca.fit_transform(dz) 
print("Cumulative variance with 100:", np.cumsum(pca.explained_variance_ratio_)[99])
tsne = TSNE(n_components=2, perplexity=30, max_iter=3000, random_state=42)
dz_2d = tsne.fit_transform(dz_pca)  # [N, 2]
# Plot and save
# Color by command category
plt.figure(figsize=(8,8))
scatter = plt.scatter(dz_2d[:,0], dz_2d[:,1], c=c, cmap='tab10', alpha=0.7)
handles, _ = scatter.legend_elements()
labels = [label_map[i+1] for i in range(len(handles))]
plt.legend(handles, labels, title="Commands")
plt.title("Delta Latent Space Visualization (t-SNE)")
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
plt.grid(True)
plt.savefig("delta_latent_space_tsne.png")
plt.show()

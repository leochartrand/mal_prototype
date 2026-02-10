import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModel
import pickle
import numpy as np
import sys
import os
import glob
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.datasets import MultiModalDataset, resize_and_normalize_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained("./models/theia_small", trust_remote_code=True)
model = model.to(device)
model.eval()

# Load and process data
print("Loading and processing data...")

data_files = sorted(glob.glob(f"../../../mnt/sda1/Datasets/chal2525/data/*.pkl"))

frames_x0 = []
frames_xt = []
commands_embed = []
commands_txt = []
for i, file in enumerate(tqdm(data_files, desc="Loading data files")):
    if not os.path.basename(file).startswith("val_"):
        continue  # Skip non VAL files for this visualization
    
    with open(file, 'rb') as f:
        part = pickle.load(f)
        # Extract first and last frames from each trajectory
        for traj in part:
            # Take first and last frame only (discard middle frames)
            frames_x0.append(traj[0])
            frames_xt.append(traj[1])
            commands_embed.append(traj[2])
            commands_txt.append(traj[3])
        del part

print(f"Total trajectory pairs loaded: {len(frames_x0)}")

# Convert frames lists to tensors
x0_data = torch.stack(frames_x0)
xt_data = torch.stack(frames_xt)
c_e = torch.stack(commands_embed)
c = commands_txt

del frames_x0, frames_xt, commands_txt, commands_embed

# Random split
indices = torch.randperm(len(x0_data), generator=torch.Generator().manual_seed(1))

dataset = MultiModalDataset(
    x0_data[indices], 
    xt_data[indices], 
    c_e[indices], 
    [c[i] for i in indices])

del x0_data, xt_data, c_e, indices

batch_size = 16
batch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dz = []
commands = []
size=(224, 224)  # Image size for resizing

with torch.no_grad():
    for x0, xt, _, c in tqdm(batch_loader, leave=False, desc=f"Encoding examples"):
        
        x0 = resize_and_normalize_batch(x0.to(device), size=size)
        xt = resize_and_normalize_batch(xt.to(device), size=size)
        # outputs_0, _ = model.compute_loss(x0)
        # _, _, z0 = model.forward(x0, v_patch_nums=None)
        # _, _, zt = model.forward(xt, v_patch_nums=None)
        z0 = model.forward_feature(x0)
        zt = model.forward_feature(xt)

        dz_batch = zt - z0  # [N, C, H, W]

        c = list(c)
        for i, cmd in enumerate(c):
            if "open the drawer" in cmd:
                c[i] = 1
            elif "close the drawer" in cmd:
                c[i] = 2
            elif "open the pot" in cmd:
                c[i] = 3
            elif "close the pot" in cmd:
                c[i] = 4
            elif "move" in cmd:
                if "right" in cmd:
                    c[i] = 5 
                if "left" in cmd:
                    c[i] = 6
                else:
                    c[i] = 7  # other put
            elif "lift" in cmd:
                c[i] = 8
            else:
                print("Other command:", cmd)
                c[i] = 9

        dz_flat = dz_batch.reshape(dz_batch.shape[0], -1).cpu().numpy()  # [N, D]
        dz.append(dz_flat)
        commands.append(c)

dz = np.concatenate(dz, axis=0)  # [Total_N, D]
c = np.concatenate(commands, axis=0)  # [Total_N]

label_map = {
    1: "open the drawer",
    2: "close the drawer",
    3: "open the pot",
    4: "close the pot",
    5: "move right",
    6: "move left",
    7: "move other",
    8: "lift *",
    9: "other"
}

# Delta latent space visualization
# Get best projection to 2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=200)
dz_pca = pca.fit_transform(dz) 
print("Cumulative variance with 100:", np.cumsum(pca.explained_variance_ratio_)[99])
tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
dz_2d = tsne.fit_transform(dz_pca)  # [N, 2]
# Plot and save
# Color by command category
plt.figure(figsize=(8,8))
unique_labels = np.unique(c)
scatter = plt.scatter(dz_2d[:,0], dz_2d[:,1], c=c, cmap='tab10', alpha=0.7)
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=scatter.cmap(scatter.norm(label)), 
                      markersize=8, alpha=0.7) 
           for label in unique_labels]
labels = [label_map[int(label)] for label in unique_labels]
plt.legend(handles, labels, title="Commands")
plt.title("Delta Latent Space Visualization (t-SNE)")
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
plt.grid(True)
plt.savefig("delta_latent_space_tsne.png")
plt.close()

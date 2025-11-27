import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import glob
import yaml
import sys
import os
import cv2


inpath = "data/val_full"
outpath = "data/vae_data"

print("Loading and processing data...")
data_files = sorted(glob.glob(f"{inpath}/*.pkl"))  # Adjust path/pattern as needed
n_traj = 0
for file in tqdm(data_files):
    with open(file, 'rb') as f:
        part = pickle.load(f)
        # Each frame to float tensor, concat at dim 0, list of frames to float tensor traj
        part_trajs = []
        for traj in part:
            n_traj += 1
            traj = [cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LANCZOS4) for frame in traj]
            tensor = torch.FloatTensor(np.array(traj)).permute(0,3,1,2) / 255.0
            part_trajs.append(tensor)
        pickle.dump(part_trajs, open(f"{outpath}/{os.path.basename(file)}", 'wb'))
        del part  # Free memory
        del part_trajs
print(f"Total trajectories: {n_traj}")
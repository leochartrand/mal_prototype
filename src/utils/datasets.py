import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import kornia.augmentation as K
import torch
from torch.utils.data import Dataset

def augment_batch(data, size=(48,48)):
    augmentation = K.AugmentationSequential(
        K.ColorJitter(
            brightness=(0.75,1.25), 
            contrast=(0.9,1.1), 
            saturation=(0.9,1.1), 
            hue=(-0.1,0.1)),
        K.RandomResizedCrop(
            size=size, 
            scale=(0.95, 1.0), 
            ratio=(1, 1)),
        K.RandomHorizontalFlip(p=0.5),
    )
    data = augmentation(data)
    return data 

def resize_and_normalize_batch(data, size=(48,48)):
    data = K.Resize(size)(data)
    return data

class FrameDataset(Dataset):
    def __init__(self, trajs):
        self.frames = [frame for traj in trajs for frame in traj]
        rand_perm = torch.randperm(len(self.frames), generator=torch.Generator().manual_seed(42)) # For reproducibility
        self.frames = [self.frames[i] for i in rand_perm]
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        return self.frames[idx]

class MultiModalDataset(Dataset):
    def __init__(self, x0, xt, c):
        self.x0 = x0
        self.xt = xt
        self.c = c

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return self.x0[idx], self.xt[idx], self.c[idx]


from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, z0, zt, c):
        self.z0 = z0
        self.zt = zt
        self.c = c

    def __len__(self):
        return len(self.z0)

    def __getitem__(self, idx):
        return self.z0[idx], self.zt[idx], self.c[idx]

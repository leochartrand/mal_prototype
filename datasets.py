from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, initial_codes, commands, target_codes):
        self.initial_codes = initial_codes
        self.target_codes = target_codes
        self.commands = commands

    def __len__(self):
        return len(self.initial_codes)

    def __getitem__(self, idx):
        return self.initial_codes[idx], self.commands[idx], self.target_codes[idx]

class DotDataset(MultiModalDataset):
    def __init__(self, initial_codes, commands, target_codes):
        super().__init__(initial_codes, commands, target_codes)

class ColorDataset(MultiModalDataset):
    def __init__(self, initial_codes, commands, target_codes):
        super().__init__(initial_codes, commands, target_codes)
        
class ValDataset(MultiModalDataset):
    def __init__(self, initial_codes, commands, target_codes):
        super().__init__(initial_codes, commands, target_codes)
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import kornia.augmentation as K
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import pickle

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
    def __init__(self, frames):
        self.frames = frames
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        return self.frames[idx]

class MultiModalDataset(Dataset):
    """Dataset for pre-embedded multimodal data.
    
    Args:
        x0_raw: Initial images (for visualization)
        x0_embed: Initial Theia embeddings [N, 196, 384]
        xt_raw: Target images (for visualization)
        xt_embed: Target Theia embeddings [N, 196, 384]
        c_raw: Text labels (list of strings)
        c_hidden: CLIP hidden states [N, 77, 768]
        c_attn_mask: Attention masks [N, 77]
    """
    def __init__(self, x0_raw, x0_embed, xt_raw, xt_embed, c_raw, c_hidden, c_attn_mask):
        self.x0_raw = x0_raw
        self.x0_embed = x0_embed
        self.xt_raw = xt_raw
        self.xt_embed = xt_embed
        self.c_raw = c_raw
        self.c_hidden = c_hidden
        self.c_attn_mask = c_attn_mask

    def __len__(self):
        return len(self.x0_embed)

    def __getitem__(self, idx):
        # Handle numpy arrays from memory-mapped files
        x0_raw = self.x0_raw[idx]
        x0_embed = self.x0_embed[idx]
        xt_raw = self.xt_raw[idx]
        xt_embed = self.xt_embed[idx]
        c_hidden = self.c_hidden[idx]
        c_attn_mask = self.c_attn_mask[idx]
        c_raw = self.c_raw[idx]
        
        # Convert numpy to torch if needed
        if not isinstance(x0_raw, torch.Tensor):
            # Raw images: uint8 [H, W, C] -> float32 [C, H, W] in [0, 1]
            x0_raw = torch.from_numpy(x0_raw.copy()).permute(2, 0, 1).float() / 255.0
            xt_raw = torch.from_numpy(xt_raw.copy()).permute(2, 0, 1).float() / 255.0
        if not isinstance(x0_embed, torch.Tensor):
            x0_embed = torch.from_numpy(x0_embed.copy())
            xt_embed = torch.from_numpy(xt_embed.copy())
        if not isinstance(c_hidden, torch.Tensor):
            c_hidden = torch.from_numpy(c_hidden.copy())
        if not isinstance(c_attn_mask, torch.Tensor):
            c_attn_mask = torch.from_numpy(c_attn_mask.copy())
        
        return (
            x0_raw,
            x0_embed,
            xt_raw,
            xt_embed,
            c_raw,
            c_hidden,
            c_attn_mask,
        )


class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for trajectory data (flow matching models).
    
    Loads pre-embedded trajectory data from mmap_data/ directory.
    Vision and text encoder models are specified by name so the same
    dataset class works with any Theia variant or text encoder.
    
    Directory structure expected:
        initial_224.npy                          : float32 [N, 3, 224, 224] (CHW, [0,1])
        target_224.npy                           : float32 [N, 3, 224, 224] (CHW, [0,1])
        initial_embed_{vision_model}.npy         : float32 [N, num_patches, D]
        target_embed_{vision_model}.npy          : float32 [N, num_patches, D]
        labels.pkl                               : list of N strings
        labels_hidden_{text_model}.npy           : float32 [N, 77, text_dim]
        labels_attn_mask_{text_model}.npy        : int32   [N, 77]
        train_indices.npy                        : int32   [N_train]
        test_indices.npy                         : int32   [N_test]
    
    Args:
        data_path: Path to directory containing .npy files
        vision_model: Name of the vision encoder (e.g. 'theia_small_cdiv')
        text_model: Name of the text encoder (e.g. 'clip-vit-large-patch14')
        split: Either 'train' or 'test' to load split indices
        indices: Optional explicit indices (overrides split)
    """
    def __init__(
        self,
        data_path: str,
        vision_model: str = 'theia_small_cdiv',
        text_model: str = 'clip-vit-large-patch14',
        split: str | None = None,
        indices: np.ndarray | None = None,
    ):
        self.data_path = data_path
        self.vision_model = vision_model
        self.text_model = text_model
        
        # Memory-mapped arrays (read-only, not loaded into RAM)
        self.initial_224 = np.load(f"{data_path}/initial_224.npy", mmap_mode='r')
        self.target_224 = np.load(f"{data_path}/target_224.npy", mmap_mode='r')
        self.initial_embed = np.load(f"{data_path}/initial_embed_{vision_model}.npy", mmap_mode='r')
        self.target_embed = np.load(f"{data_path}/target_embed_{vision_model}.npy", mmap_mode='r')
        self.label_hidden = np.load(f"{data_path}/labels_hidden_{text_model}.npy", mmap_mode='r')
        self.label_attn_mask = np.load(f"{data_path}/labels_attn_mask_{text_model}.npy", mmap_mode='r')
        
        with open(f"{data_path}/labels.pkl", 'rb') as f:
            self.labels = pickle.load(f)
        
        # Determine indices: explicit > split file > all
        if indices is not None:
            self.indices = indices
        elif split is not None:
            if split not in ('train', 'test'):
                raise ValueError(f"split must be 'train' or 'test', got '{split}'")
            split_file = f"{data_path}/{split}_indices.npy"
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            self.indices = np.load(split_file)
        else:
            self.indices = np.arange(len(self.initial_embed))

        # Clamp to the smallest array to avoid out-of-bounds access
        max_valid = min(
            len(self.initial_224), len(self.target_224),
            len(self.initial_embed), len(self.target_embed),
            len(self.label_hidden), len(self.label_attn_mask),
            len(self.labels),
        )
        orig_len = len(self.indices)
        self.indices = self.indices[self.indices < max_valid]
        if len(self.indices) < orig_len:
            print(f"[MemoryMappedDataset] Filtered {orig_len - len(self.indices)} "
                  f"out-of-bounds indices (max valid index: {max_valid - 1})")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Returns (x0_224, x0_embed, xt_224, xt_embed, c_txt, c_hidden, c_attn_mask).
        
        - x0_224:      tensor float32 [3, 224, 224] in [0, 1]
        - x0_embed:     tensor float32 [num_patches, latent_dim]
        - xt_224:      tensor float32 [3, 224, 224] in [0, 1]
        - xt_embed:     tensor float32 [num_patches, latent_dim]
        - c_txt:        string
        - c_hidden:     tensor float32 [77, text_dim]  (per-token CLIP hidden states)
        - c_attn_mask:  tensor int32   [77]            (1=real token, 0=padding)
        """
        actual_idx = self.indices[idx]
        
        return (
            torch.from_numpy(self.initial_224[actual_idx].copy()),
            torch.from_numpy(self.initial_embed[actual_idx].copy()),
            torch.from_numpy(self.target_224[actual_idx].copy()),
            torch.from_numpy(self.target_embed[actual_idx].copy()),
            self.labels[actual_idx],
            torch.from_numpy(self.label_hidden[actual_idx].copy()),
            torch.from_numpy(self.label_attn_mask[actual_idx].copy()),
        )


class DecoderMemoryMappedDataset(Dataset):
    """Memory-mapped dataset for decoder training (image reconstruction).
    
    Loads combined initial+target data from mmap_dec/ directory.
    Each trajectory contributes 2 samples (initial + target frame).
    
    Directory structure expected:
        decoder_224.npy                          : float32 [N, 3, 224, 224] (CHW, [0,1])
        decoder_embed_{vision_model}.npy         : float32 [N, num_patches, D]
        decoder_train_indices.npy                : int32   [N_train]
        decoder_test_indices.npy                 : int32   [N_test]
    
    Args:
        data_path: Path to directory containing .npy files
        vision_model: Name of the vision encoder (e.g. 'theia_small_cdiv')
        split: Either 'train' or 'test' to load split indices
        indices: Optional explicit indices (overrides split)
    """
    def __init__(
        self,
        data_path: str,
        vision_model: str = 'theia_small_cdiv',
        split: str | None = None,
        indices: np.ndarray | None = None,
    ):
        self.data_path = data_path
        self.vision_model = vision_model
        
        # Memory-mapped arrays (read-only)
        self.images = np.load(f"{data_path}/decoder_224.npy", mmap_mode='r')
        self.embeds = np.load(f"{data_path}/decoder_embed_{vision_model}.npy", mmap_mode='r')
        
        # Determine indices: explicit > split file > all
        if indices is not None:
            self.indices = indices
        elif split is not None:
            if split not in ('train', 'test'):
                raise ValueError(f"split must be 'train' or 'test', got '{split}'")
            split_file = f"{data_path}/decoder_{split}_indices.npy"
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            self.indices = np.load(split_file)
        else:
            self.indices = np.arange(len(self.embeds))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Returns (embed, image).
        
        - embed: tensor float32 [num_patches, latent_dim]
        - image: tensor float32 [3, 224, 224] in [0, 1]
        """
        actual_idx = self.indices[idx]
        
        embed = torch.from_numpy(self.embeds[actual_idx].copy())
        image = torch.from_numpy(self.images[actual_idx].copy())
        
        return embed, image


def mmap_collate_fn(batch):
    """Custom collate for MemoryMappedDataset.
    
    Handles mixed types: tensors (images, embeddings), strings (labels).
    """
    x0_224 = torch.stack([b[0] for b in batch])      # [B, 3, 224, 224]
    x0_embed = torch.stack([b[1] for b in batch])     # [B, num_patches, D]
    xt_224 = torch.stack([b[2] for b in batch])       # [B, 3, 224, 224]
    xt_embed = torch.stack([b[3] for b in batch])     # [B, num_patches, D]
    c_txt = [b[4] for b in batch]                     # list of strings
    c_hidden = torch.stack([b[5] for b in batch])     # [B, 77, text_dim]
    c_attn_mask = torch.stack([b[6] for b in batch])  # [B, 77]
    
    return x0_224, x0_embed, xt_224, xt_embed, c_txt, c_hidden, c_attn_mask

"""
Shared scaffolding for DiT-style epoch-based trainers
(used by src/train_flowdit.py and src/flowdit_cfg_distill.py).

Covers the bits both scripts duplicate verbatim:
  - DDP / single-GPU init
  - Frontend monitor instantiation
  - Theia decoder load
  - MemoryMappedDataset + DataLoader construction
  - Pretrained checkpoint load
  - All-reduce of scalar metrics across DDP ranks
  - CSV training log header
"""

import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.theia_decoder import Decoder as TheiaDecoder
from utils.checkpoint import _strip_module_prefix
from utils.datasets import MemoryMappedDataset, mmap_collate_fn


def init_distributed(gpu: int) -> dict:
    """Initialize DDP under torchrun, or pick a single GPU otherwise.

    Returns: dict with keys ddp, device, is_main, rank, world_size, local_rank.
    Non-rank-0 processes have stdout redirected to /dev/null.
    """
    ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1
    if ddp:
        dist.init_process_group('nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_main = (rank == 0)
        if is_main:
            print(f"DDP: {world_size} GPUs")
        else:
            sys.stdout = open(os.devnull, 'w')
    else:
        device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        is_main = True
        rank = 0
        world_size = 1
        local_rank = None
        print(f"Using GPU {gpu}")
    return dict(ddp=ddp, device=device, is_main=is_main, rank=rank,
                world_size=world_size, local_rank=local_rank)


def init_frontend_monitor(params: dict, frontend: bool, is_main: bool):
    """Optional rank-0 training monitor."""
    if not (frontend and is_main):
        return None
    from frontend.training_monitor import TrainingMonitor
    return TrainingMonitor(params)


def load_theia_decoder(params: dict, device: torch.device, is_main: bool = True):
    """Load the Theia decoder for visualization (rank-0 only), or None if not configured."""
    if not is_main or "theia_decoder" not in params:
        return None
    cfg = params["theia_decoder"]
    if not os.path.exists(cfg["model_path"]):
        return None
    decoder = TheiaDecoder(**cfg["model_params"])
    ckpt = torch.load(cfg["model_path"], map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt['model'])
    decoder = decoder.to(device).eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


def build_dataloaders(params: dict, dummy: bool, ddp: bool):
    """Construct MemoryMappedDataset loaders + report text-embedding dims.

    The held-out split is named ``val`` everywhere in the API even though the
    on-disk index file is ``test_indices.npy`` (legacy file naming).

    Returns: (train_loader, val_loader, train_sampler, train_dataset, val_dataset,
              text_dim, max_text_len, pooled_text_dim).
    """
    dataset_path = params["dataset_path"]
    vision_model = params["vision_model"]
    text_model = params["text_model"]

    train_dataset = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, split='train')
    val_dataset   = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, split='test')

    if dummy:
        B = params["batch_size"]
        train_dataset = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, indices=train_dataset.indices[:B])
        val_dataset   = MemoryMappedDataset(dataset_path, vision_model=vision_model, text_model=text_model, indices=val_dataset.indices[:B])
        print(f"[DUMMY MODE] Using only {B} samples per split")

    n_train, n_val = len(train_dataset), len(val_dataset)
    print(f"Total samples: {n_train + n_val}, Train: {n_train}, Val: {n_val}")

    sample_c_hidden = np.load(f"{dataset_path}/labels_hidden_{text_model}.npy", mmap_mode='r')
    text_dim     = sample_c_hidden.shape[2]
    max_text_len = sample_c_hidden.shape[1]
    del sample_c_hidden

    sample_c_pooled = np.load(f"{dataset_path}/labels_pooled_{text_model}.npy", mmap_mode='r')
    pooled_text_dim = sample_c_pooled.shape[1]
    del sample_c_pooled
    print(f"Pooled text dim: {pooled_text_dim}")

    B = params["batch_size"]
    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=0, pin_memory=False, collate_fn=mmap_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=B, shuffle=False,
                              sampler=val_sampler,   num_workers=0, pin_memory=False, collate_fn=mmap_collate_fn)

    return (train_loader, val_loader, train_sampler, train_dataset, val_dataset,
            text_dim, max_text_len, pooled_text_dim)


def load_pretrained(model, path: str, is_main: bool):
    """Strict=False state-dict load with DDP-prefix stripping; prints missing/unexpected on rank 0."""
    print(f"Loading pretrained weights from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    sd = _strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if is_main:
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    del ckpt, sd


def reduce_mean(value: float, ddp: bool, world_size: int, device: torch.device) -> float:
    """All-reduce mean of a scalar across DDP ranks (no-op when ddp=False)."""
    if not ddp:
        return value
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


def init_csv_log(log_file: str, start_epoch: int, header: str = 'epoch,split,loss\n'):
    """Write the CSV header (only when the file is missing or we're starting fresh)."""
    if not os.path.exists(log_file) or start_epoch == 0:
        with open(log_file, 'w') as f:
            f.write(header)

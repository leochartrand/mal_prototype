#!/usr/bin/env python3
"""
Train Diffusion Goal-Conditioned Behavioral Cloning (D-GCBC) policy.

Uses pre-encoded Theia features from build_gcbc_dataset.py.
Matches TaKSIE's training protocol:
  - Hindsight goal relabeling: sample goal δ steps ahead, δ ∈ [1, k_max]
  - Cosine beta schedule, 20 diffusion steps
  - Action chunk prediction (act_pred_horizon steps)
  - Action normalization with dataset statistics

Usage:
    python src/train_gcbc.py --config config/gcbc.yaml [--gpu 0]
    torchrun --nproc_per_node=2 src/train_gcbc.py --config config/gcbc.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import yaml

# Avoid 'too many open files' / ConnectionRefused with many workers + persistent_workers
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from models.diffusion_policy import FlowMatchingGCBCPolicy
from utils.ema import EMA


# ============================================================================
# Dataset
# ============================================================================

class GCBCDataset(Dataset):
    """Memory-mapped GCBC dataset with hindsight goal relabeling.

    Each sample:
        - z_obs: Theia feature of current frame
        - z_goal: Theia feature of goal frame (δ steps ahead, randomly sampled)
        - actions: Chunk of act_pred_horizon future rel_actions

    Goal relabeling: For each frame i in a contiguous segment, sample goal
    frame g = i + δ where δ ~ Uniform[k_min, k_max]. Matches TaKSIE's
    CALVIN config: k_min=8, k_max=20 (not 1-24, which biases toward
    short-horizon goals the policy won't see at inference).
    """

    def __init__(
        self,
        data_dir: str,
        act_pred_horizon: int = 5,
        k_min: int = 8,
        k_max: int = 20,
        act_mean: np.ndarray | None = None,
        act_std: np.ndarray | None = None,
        synth_targets_path: str | None = None,
        synth_frame_map_path: str | None = None,
        synth_ratio: float = 0.0,
        aug_noise_std: float = 0.0,
        aug_patch_drop: float = 0.0,
        chunk_mode: str = "fixed",
    ):
        """chunk_mode:
            - "fixed":   supervise exactly act_pred_horizon actions, no mask
                         (legacy 5-step recipe).
            - "delta":   supervise δ actions where δ is the sampled goal distance
                         (matches chunk to obs→goal interval). Pads to
                         act_pred_horizon and returns a mask.
        """
        assert chunk_mode in ("fixed", "delta"), f"chunk_mode={chunk_mode}"
        data_dir = Path(data_dir)

        sentinel = data_dir / "_actions_complete"
        if not sentinel.exists():
            raise FileNotFoundError(
                f"Missing build-completion sentinel: {sentinel}. "
                f"The actions.npy mmap may be partially written. "
                f"Rebuild gcbc_data_abcd from raw CALVIN episodes."
            )

        self.features = np.load(data_dir / "theia_features.npy", mmap_mode="r")
        self.actions = np.load(data_dir / "actions.npy", mmap_mode="r")
        self.segments = np.load(data_dir / "segment_boundaries.npy")  # (N_seg, 2)

        self.act_pred_horizon = act_pred_horizon
        self.k_min = k_min
        self.k_max = k_max
        self.aug_noise_std = aug_noise_std
        self.aug_patch_drop = aug_patch_drop
        self.chunk_mode = chunk_mode
        if chunk_mode == "delta":
            assert act_pred_horizon >= k_max, (
                f"act_pred_horizon={act_pred_horizon} must be ≥ k_max={k_max} "
                f"so all sampled deltas fit in the padded chunk"
            )

        # Synthetic goal mixing
        self.synth_ratio = synth_ratio
        if synth_targets_path and synth_ratio > 0:
            self.synth_targets = np.load(synth_targets_path, mmap_mode="r")
            synth_frame_map = np.load(synth_frame_map_path)
            # Build lookup: initial_frame → synth index
            self.frame_to_synth = {}
            for synth_idx, (init_frame, _) in enumerate(synth_frame_map):
                self.frame_to_synth[int(init_frame)] = synth_idx
            print(f"Synthetic goals loaded: {len(self.synth_targets)} targets, "
                  f"ratio={synth_ratio}, coverage={len(self.frame_to_synth)} frames")
        else:
            self.synth_targets = None
            self.frame_to_synth = {}

        # Action normalization
        if act_mean is None:
            stats = np.load(data_dir / "act_stats.npz")
            act_mean = stats["mean"]
            act_std = stats["std"]
        self.act_mean = act_mean.astype(np.float32)
        self.act_std = act_std.astype(np.float32)

        # Build valid index: (frame_idx, segment_start, segment_end)
        # A frame is valid if it has at least act_pred_horizon steps remaining
        # AND at least k_min frames ahead for goal sampling
        self.valid_indices = []
        for seg_start, seg_end in self.segments:
            # Need: H actions ahead + k_min goal distance
            min_remaining = max(act_pred_horizon, k_min) + 1
            max_start = seg_end - min_remaining
            for i in range(seg_start, max_start):
                self.valid_indices.append((i, seg_start, seg_end))
        self.valid_indices = np.array(self.valid_indices, dtype=np.int64)
        print(f"GCBCDataset: {len(self.valid_indices):,} valid frames from "
              f"{len(self.segments)} segments, {len(self.features):,} total frames"
              f" (k_min={k_min}, k_max={k_max})", flush=True)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx):
        frame_idx, seg_start, seg_end = self.valid_indices[idx]

        # Sample goal: δ ∈ [k_min, k_max], clamped to segment boundary
        max_delta = min(self.k_max, seg_end - frame_idx - 1)
        min_delta = min(self.k_min, max_delta)
        delta = np.random.randint(min_delta, max_delta + 1)
        goal_idx = frame_idx + delta

        # Current observation
        z_obs = torch.from_numpy(self.features[frame_idx].copy())

        # Feature-space augmentation on obs (not goal)
        if self.aug_noise_std > 0:
            z_obs = z_obs + torch.randn_like(z_obs) * self.aug_noise_std
        if self.aug_patch_drop > 0:
            # Randomly zero out patches (B, 196, 384) → drop whole patch rows
            mask = torch.bernoulli(
                torch.full((z_obs.shape[0],), 1.0 - self.aug_patch_drop)
            ).unsqueeze(-1)  # (196, 1)
            z_obs = z_obs * mask

        # Goal: optionally replace with synthetic DiT generation
        use_synth = (
            self.synth_targets is not None
            and np.random.random() < self.synth_ratio
            and frame_idx in self.frame_to_synth
        )
        if use_synth:
            synth_idx = self.frame_to_synth[frame_idx]
            z_goal = torch.from_numpy(self.synth_targets[synth_idx].copy())
        else:
            z_goal = torch.from_numpy(self.features[goal_idx].copy())

        # Action chunk
        if self.chunk_mode == "delta":
            # chunk length = δ (sampled goal distance), padded to act_pred_horizon
            chunk_len = int(delta)
            raw_actions = self.actions[frame_idx:frame_idx + chunk_len].astype(np.float32)
            padded = np.zeros((self.act_pred_horizon, 7), dtype=np.float32)
            padded[:chunk_len] = (raw_actions - self.act_mean) / self.act_std
            mask = np.zeros(self.act_pred_horizon, dtype=np.float32)
            mask[:chunk_len] = 1.0
            return {
                "z_obs": z_obs,
                "z_goal": z_goal,
                "actions": torch.from_numpy(padded),  # (H, 7)
                "mask": torch.from_numpy(mask),       # (H,)
            }

        # "fixed": legacy 5-step recipe, no mask
        act_end = frame_idx + self.act_pred_horizon
        raw_actions = self.actions[frame_idx:act_end].astype(np.float32)
        if len(raw_actions) < self.act_pred_horizon:
            pad = np.zeros((self.act_pred_horizon - len(raw_actions), 7), dtype=np.float32)
            raw_actions = np.concatenate([raw_actions, pad])
        actions = (raw_actions - self.act_mean) / self.act_std
        return {
            "z_obs": z_obs,
            "z_goal": z_goal,
            "actions": torch.from_numpy(actions),  # (H, 7)
        }


class GCBCLowessDataset(Dataset):
    """LOWESS-keyframe-pair dataset for Phase B (variable-chunk supervision).

    Each sample comes from a (source_idx, target_idx) pair selected by LOWESS
    slope analysis (see scripts/datasets/extract_subgoals.py). chunk length = target -
    source, padded to act_pred_horizon with a mask. Goal embedding defaults to
    features[target_idx]; swap for DiT synth later.

    Pairs are filtered to interval ∈ [min_chunk, max_chunk].
    """

    def __init__(
        self,
        data_dir: str,
        lowess_pkl: str,
        act_pred_horizon: int = 36,
        min_chunk: int = 2,
        max_chunk: int = 36,
        act_mean: np.ndarray | None = None,
        act_std: np.ndarray | None = None,
        synth_targets_path: str | None = None,
        real_ratio: float = 1.0,
    ):
        """real_ratio: probability of using the real target frame as z_goal.
        Remainder uses a DiT-generated synth target sampled uniformly from the
        K candidates in synth_targets_path. Requires synth_targets_path when
        real_ratio < 1.0.
        """
        data_dir = Path(data_dir)

        sentinel = data_dir / "_actions_complete"
        if not sentinel.exists():
            raise FileNotFoundError(
                f"Missing build-completion sentinel: {sentinel}. "
                f"Rebuild gcbc_data_abcd from raw CALVIN episodes."
            )
        assert act_pred_horizon >= max_chunk, (
            f"act_pred_horizon={act_pred_horizon} must be ≥ max_chunk={max_chunk}"
        )

        self.features = np.load(data_dir / "theia_features.npy", mmap_mode="r")
        self.actions = np.load(data_dir / "actions.npy", mmap_mode="r")
        self.act_pred_horizon = act_pred_horizon
        self.real_ratio = float(real_ratio)

        with open(lowess_pkl, "rb") as f:
            d = pickle.load(f)
        pairs = d["pairs"]  # (N, 2) int — CALVIN frame ids; these index features/actions directly
        intervals = pairs[:, 1] - pairs[:, 0]
        keep = (intervals >= min_chunk) & (intervals <= max_chunk)
        keep &= (pairs[:, 0] + intervals <= len(self.actions))
        # Keep the unfiltered pair index so we can look up synths
        self.pair_ids = np.where(keep)[0]
        self.pairs = pairs[keep]
        print(
            f"GCBCLowessDataset: {len(self.pairs):,} pairs after filter "
            f"({(~keep).sum()} dropped). interval ∈ [{min_chunk}, {max_chunk}], real_ratio={self.real_ratio}",
            flush=True,
        )

        if act_mean is None:
            stats = np.load(data_dir / "act_stats.npz")
            act_mean = stats["mean"]
            act_std = stats["std"]
        self.act_mean = act_mean.astype(np.float32)
        self.act_std = act_std.astype(np.float32)

        # Synth target embeddings, shape (N_unfiltered_train, K, 196, 384) fp16
        self.synth_targets = None
        if synth_targets_path is not None:
            synth_dir = Path(synth_targets_path).parent
            synth_sentinel = synth_dir / "_synth_lowess_complete"
            if not synth_sentinel.exists():
                raise FileNotFoundError(
                    f"Missing synth sentinel: {synth_sentinel}. "
                    f"LOWESS synth targets must be pre-generated."
                )
            self.synth_targets = np.load(synth_targets_path, mmap_mode="r")
            self.K = self.synth_targets.shape[1]
            assert self.synth_targets.shape[0] >= self.pair_ids.max() + 1, (
                f"synth array has {self.synth_targets.shape[0]} rows, "
                f"need at least {self.pair_ids.max() + 1}"
            )
            print(f"  synth_targets: shape={self.synth_targets.shape}, K={self.K}", flush=True)
        elif self.real_ratio < 1.0:
            raise ValueError(
                f"real_ratio={real_ratio} < 1.0 requires synth_targets_path"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        chunk_len = int(target - source)

        z_obs = torch.from_numpy(self.features[int(source)].copy())

        # Goal: real frame or synth (sampled from K)
        if self.synth_targets is not None and np.random.random() >= self.real_ratio:
            pkl_row = int(self.pair_ids[idx])
            k = np.random.randint(self.K)
            z_goal = torch.from_numpy(self.synth_targets[pkl_row, k].copy())
        else:
            z_goal = torch.from_numpy(self.features[int(target)].copy())

        raw_actions = self.actions[int(source):int(source) + chunk_len].astype(np.float32)
        padded = np.zeros((self.act_pred_horizon, 7), dtype=np.float32)
        padded[:chunk_len] = (raw_actions - self.act_mean) / self.act_std
        mask = np.zeros(self.act_pred_horizon, dtype=np.float32)
        mask[:chunk_len] = 1.0

        return {
            "z_obs": z_obs,
            "z_goal": z_goal,
            "actions": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
        }


# ============================================================================
# Training loop
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    return p.parse_args()


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, None


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0

    # Load config
    config_path = Path(__file__).parent.parent / "config" / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Device
    if local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / cfg.get("data_dir", "data/gcbc/training")
    save_dir = project_root / cfg.get("save_dir", "models/gcbc")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    act_pred_horizon = cfg.get("act_pred_horizon", 5)
    k_min = cfg.get("k_min", 8)
    k_max = cfg.get("k_max", 20)
    batch_size = cfg.get("batch_size", 256)
    lr = cfg.get("lr", 3e-4)
    warmup_steps = cfg.get("warmup_steps", 2000)
    total_steps = cfg.get("total_steps", 400_000)
    log_interval = cfg.get("log_interval", 100)
    save_interval = cfg.get("save_interval", 10_000)
    eval_interval = cfg.get("eval_interval", 10_000)
    num_sampling_steps = cfg.get("num_sampling_steps", 4)
    hidden_dim = cfg.get("hidden_dim", 256)
    num_blocks = cfg.get("num_blocks", 3)
    dropout = cfg.get("dropout", 0.1)
    ema_decay = cfg.get("ema_decay", 0.9999)
    use_cross_attention = cfg.get("use_cross_attention", False)
    cross_attn_heads = cfg.get("cross_attn_heads", 8)
    cross_attn_every_n = cfg.get("cross_attn_every_n", 2)
    if is_main:
        print(f"Config: {config_path}")
        print(f"Data: {data_dir}")
        print(f"Save: {save_dir}")
        print(f"act_pred_horizon={act_pred_horizon}, k_min={k_min}, k_max={k_max}")
        print(f"batch_size={batch_size}, lr={lr}, total_steps={total_steps}")
        print(f"num_sampling_steps={num_sampling_steps}")
        print(f"hidden_dim={hidden_dim}, num_blocks={num_blocks}")
        if use_cross_attention:
            print(f"cross_attention: heads={cross_attn_heads}, every_n={cross_attn_every_n}")
        print()

    # Dataset
    dataset_type = cfg.get("dataset_type", "hindsight")
    if dataset_type == "lowess":
        dataset = GCBCLowessDataset(
            data_dir=str(data_dir),
            lowess_pkl=cfg["lowess_pkl"],
            act_pred_horizon=act_pred_horizon,
            min_chunk=cfg.get("min_chunk", 2),
            max_chunk=cfg.get("max_chunk", 36),
            synth_targets_path=cfg.get("synth_targets_path"),
            real_ratio=cfg.get("real_ratio", 1.0),
        )
    else:
        dataset = GCBCDataset(
            data_dir=str(data_dir),
            act_pred_horizon=act_pred_horizon,
            k_min=k_min,
            k_max=k_max,
            synth_targets_path=cfg.get("synth_targets_path"),
            synth_frame_map_path=cfg.get("synth_frame_map_path"),
            synth_ratio=cfg.get("synth_ratio", 0.0),
            aug_noise_std=cfg.get("aug_noise_std", 0.0),
            aug_patch_drop=cfg.get("aug_patch_drop", 0.0),
            chunk_mode=cfg.get("chunk_mode", "fixed"),
        )

    num_workers = cfg.get("num_workers", 12)
    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
    else:
        dataloader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    # Model
    model = FlowMatchingGCBCPolicy(
        action_dim=7,
        act_pred_horizon=act_pred_horizon,
        theia_dim=384,
        num_sampling_steps=num_sampling_steps,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        use_cross_attention=use_cross_attention,
        cross_attn_heads=cross_attn_heads,
        cross_attn_every_n=cross_attn_every_n,
    ).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_params:,} ({n_params/1e6:.2f}M)")
        print(f"EMA decay: {ema_decay}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if isinstance(model, DDP) else model

    # Optimizer + scheduler (EMA wraps optimizer)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(base_optimizer, ema_decay=ema_decay)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0  # Constant after warmup (matching TaKSIE)

    scheduler = torch.optim.lr_scheduler.LambdaLR(base_optimizer, lr_schedule)

    torch.backends.cudnn.benchmark = True
    # TF32: fp32 dynamic range with tensor-core throughput (Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Resume
    start_step = 0
    resume_path = args.resume or cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        # If resuming from same training run, restore optimizer/scheduler/step
        # If finetuning (config has "resume"), only load model weights
        if args.resume:
            if "optimizer" in ckpt:
                ema.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt.get("step", 0)
            if is_main:
                print(f"Resumed from step {start_step}")
        else:
            if is_main:
                print(f"Loaded pretrained weights from {resume_path} (fresh optimizer)")
    model.train()
    data_iter = iter(dataloader)
    losses = []
    t0 = time.time()

    for step in range(start_step, total_steps):
        # Get batch (restart iterator when exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            if world_size > 1:
                sampler.set_epoch(step)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Pure fp32 — matches OLD working recipe; cuDNN fp32 conv is 3x faster than bf16 on Ampere for our shape
        z_obs   = batch["z_obs"].to(device, dtype=torch.float32, non_blocking=True)
        z_goal  = batch["z_goal"].to(device, dtype=torch.float32, non_blocking=True)
        actions = batch["actions"].to(device, dtype=torch.float32, non_blocking=True)
        mask    = batch["mask"].to(device, dtype=torch.float32, non_blocking=True) \
                  if "mask" in batch else None

        loss = raw_model.compute_loss(z_obs, z_goal, actions, mask=mask) if world_size <= 1 \
            else model.module.compute_loss(z_obs, z_goal, actions, mask=mask)

        ema.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ema.step()
        scheduler.step()

        losses.append(loss.item())

        # Logging
        if is_main and (step + 1) % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:])
            elapsed = time.time() - t0
            steps_per_sec = (step - start_step + 1) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            print(f"step {step+1:>7d}/{total_steps} | loss {avg_loss:.6f} | "
                  f"lr {current_lr:.2e} | {steps_per_sec:.1f} steps/s", flush=True)

        # Save checkpoint
        if is_main and (step + 1) % save_interval == 0:
            ema.swap_parameters_with_ema(store_params_in_ema=True)
            ckpt_payload = {
                "model": raw_model.state_dict(),
                "optimizer": ema.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step + 1,
                "loss": np.mean(losses[-save_interval:]),
                "config": cfg,
            }
            torch.save(ckpt_payload, save_dir / f"gcbc_step{step+1}.pt")
            torch.save(ckpt_payload, save_dir / "gcbc_latest.pt")
            ema.swap_parameters_with_ema(store_params_in_ema=True)
            print(f"  Saved: gcbc_step{step+1}.pt")

    # Final save (EMA weights as "model")
    if is_main:
        ema.swap_parameters_with_ema(store_params_in_ema=True)
        final_path = save_dir / "gcbc_final.pt"
        torch.save({
            "model": raw_model.state_dict(),
            "step": total_steps,
            "loss": np.mean(losses[-1000:]) if losses else 0,
            "config": cfg,
        }, final_path)
        print(f"\nTraining complete. Final model (EMA): {final_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

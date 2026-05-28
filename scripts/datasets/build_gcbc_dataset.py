#!/usr/bin/env python3
"""
Pre-encode all CALVIN D training/validation frames with Theia and build
memory-mapped dataset for GCBC (Goal-Conditioned Behavioral Cloning) training.

Produces:
  data/gcbc/
    theia_features.npy     (N_total, 196, 384) float16 — all unique frames
    theia_index.npy        (N_total,) int64 — CALVIN episode IDs
    actions.npy            (N_total, 7) float32 — rel_actions per frame
    episode_ends.npy       (N_segments,) int64 — cumulative end indices per segment
    act_stats.npz          — mean/std for action normalization
    metadata.json          — dataset info

Strategy:
  1. Scan all training episodes, store (frame_id → sequential index).
  2. First pass: collect all rel_actions and write to mmap.
  3. Second pass: encode all rgb_static frames with Theia in batches.
  4. Compute and save action normalization stats.

Usage:
    python scripts/datasets/build_gcbc_dataset.py [--split training] [--batch-size 64] [--gpu 0]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALVIN_DATA = PROJECT_ROOT / "data" / "calvin" / "task_D_D"
THEIA_PATH = PROJECT_ROOT / "models" / "theia_small_cdiv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gcbc"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="training", choices=["training", "validation"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-dir", type=str, default=None,
                   help="Root CALVIN dataset dir (contains training/ and validation/)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def get_episode_ids(split_dir: Path) -> np.ndarray:
    """Get sorted list of episode IDs from filenames."""
    ids = []
    for entry in os.scandir(str(split_dir)):
        if entry.name.startswith("episode_") and entry.name.endswith(".npz"):
            ids.append(int(entry.name[8:15]))  # episode_XXXXXXX.npz
    return np.array(sorted(ids), dtype=np.int64)


def build_actions_mmap(split_dir: Path, episode_ids: np.ndarray, output_dir: Path):
    """First pass: collect all rel_actions into a memory-mapped array.

    Writes to actions.npy.tmp, fills with NaN sentinel, validates, then
    atomically renames and drops a _actions_complete sentinel. Training
    refuses to load actions.npy without the sentinel.
    """
    N = len(episode_ids)
    actions_path = output_dir / "actions.npy"
    sentinel_path = output_dir / "_actions_complete"

    if actions_path.exists() and sentinel_path.exists():
        print(f"  actions.npy exists with sentinel ({N} episodes), skipping...")
        return np.load(actions_path, mmap_mode="r")

    # Build into .tmp; fill with NaN so unwritten rows are detectable
    tmp_path = actions_path.with_suffix(".npy.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    actions = np.lib.format.open_memmap(
        str(tmp_path), mode="w+", dtype=np.float32, shape=(N, 7)
    )
    actions[:] = np.nan

    for i, eid in enumerate(tqdm(episode_ids, desc="Collecting actions")):
        ep = np.load(split_dir / f"episode_{eid:07d}.npz")
        actions[i] = ep["rel_actions"].astype(np.float32)

    actions.flush()
    del actions

    # Validate: no unwritten rows remain
    a = np.load(tmp_path, mmap_mode="r")
    n_nan = int(np.isnan(a[:].sum(axis=1)).sum())
    if n_nan > 0:
        raise RuntimeError(f"Action build incomplete: {n_nan} unwritten rows remain in {tmp_path}")

    if actions_path.exists():
        actions_path.unlink()
    tmp_path.rename(actions_path)
    sentinel_path.write_text("ok\n")

    print(f"  Saved actions: {a.shape} (sentinel written)")
    return a


def compute_action_stats(actions: np.ndarray, output_dir: Path):
    """Compute and save action normalization statistics."""
    stats_path = output_dir / "act_stats.npz"
    if stats_path.exists():
        d = np.load(stats_path)
        print(f"  act_stats exists: mean={d['mean']}, std={d['std']}")
        return d["mean"], d["std"]

    mean = np.mean(actions, axis=0).astype(np.float32)
    std = np.std(actions, axis=0).astype(np.float32)
    # Clamp std to avoid division by zero
    std = np.maximum(std, 1e-6)
    np.savez(stats_path, mean=mean, std=std)
    print(f"  Action mean: {mean}")
    print(f"  Action std:  {std}")
    return mean, std


class EpisodeImageLoader:
    """Loads rgb_static from episodes in batches for Theia encoding."""

    def __init__(self, split_dir: Path, episode_ids: np.ndarray, batch_size: int):
        self.split_dir = split_dir
        self.episode_ids = episode_ids
        self.batch_size = batch_size

    def __iter__(self):
        batch_imgs = []
        batch_indices = []
        for i, eid in enumerate(self.episode_ids):
            ep = np.load(self.split_dir / f"episode_{eid:07d}.npz")
            img = ep["rgb_static"]  # (200, 200, 3) uint8
            batch_imgs.append(img)
            batch_indices.append(i)
            if len(batch_imgs) == self.batch_size:
                yield batch_indices, batch_imgs
                batch_imgs = []
                batch_indices = []
        if batch_imgs:
            yield batch_indices, batch_imgs


@torch.no_grad()
def encode_frames(
    split_dir: Path,
    episode_ids: np.ndarray,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
):
    """Second pass: encode all frames with Theia."""
    N = len(episode_ids)
    features_path = output_dir / "theia_features.npy"

    if features_path.exists():
        existing = np.load(features_path, mmap_mode="r")
        if existing.shape[0] == N:
            print(f"  theia_features.npy exists ({N} frames), skipping...")
            return
        print(f"  theia_features.npy exists but wrong size ({existing.shape[0]} vs {N}), re-encoding...")

    print(f"  Loading Theia model from {THEIA_PATH}...")
    model = AutoModel.from_pretrained(str(THEIA_PATH), trust_remote_code=True)
    model = model.to(device).eval()

    # Create output mmap
    features = np.lib.format.open_memmap(
        str(features_path), mode="w+", dtype=np.float16, shape=(N, 196, 384)
    )

    loader = EpisodeImageLoader(split_dir, episode_ids, batch_size)
    total_batches = (N + batch_size - 1) // batch_size
    log_every = max(1, total_batches // 200)  # log ~200 times total

    import time as _time
    t_start = _time.time()
    frames_done = 0

    print(f"  Encoding {N:,} frames in {total_batches:,} batches of {batch_size}...", flush=True)

    for batch_idx, (batch_indices, batch_imgs) in enumerate(loader):
        imgs = np.stack(batch_imgs)
        img_tensor = torch.from_numpy(imgs).to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            z = model.forward_feature(
                img_tensor,
                do_resize=True,
                do_rescale=True,
                do_normalize=True,
            )  # (B, 196, 384)

        features[batch_indices] = z.cpu().numpy().astype(np.float16)
        frames_done += len(batch_indices)

        if (batch_idx + 1) % log_every == 0 or batch_idx == total_batches - 1:
            elapsed = _time.time() - t_start
            fps = frames_done / elapsed
            remaining = (N - frames_done) / fps if fps > 0 else 0
            pct = 100 * frames_done / N
            eta_h = int(remaining // 3600)
            eta_m = int((remaining % 3600) // 60)
            print(f"  [{pct:5.1f}%] {frames_done:>10,}/{N:,} frames | "
                  f"{fps:6.1f} fps | ETA {eta_h}h{eta_m:02d}m", flush=True)

        if batch_indices[-1] % (batch_size * 100) < batch_size:
            features.flush()

    features.flush()
    del model
    torch.cuda.empty_cache()
    print(f"  Done. theia_features: {features.shape}", flush=True)


def build_segment_info(split_dir: Path, episode_ids: np.ndarray, output_dir: Path):
    """Build contiguous segment boundaries for hindsight goal relabeling.

    Uses ep_start_end_ids.npy (recording segments) combined with
    scene_info.npy (environment boundaries) to ensure goals never
    cross environment or recording boundaries.
    """
    segments_path = output_dir / "segment_boundaries.npy"
    if segments_path.exists():
        boundaries = np.load(segments_path)
        print(f"  segment_boundaries exists ({len(boundaries)} segments)")
        return boundaries

    # Load recording segments from CALVIN metadata
    se = np.load(split_dir / "ep_start_end_ids.npy")  # (N_rec, 2) episode ID ranges

    # episode_ids is a sorted array; build episode_id → sequential index map
    eid_to_idx = {int(eid): i for i, eid in enumerate(episode_ids)}

    # Convert (episode_id_start, episode_id_end) → (sequential_index_start, sequential_index_end)
    boundaries_list = []
    for rec_start, rec_end in se:
        rec_start, rec_end = int(rec_start), int(rec_end)
        if rec_start not in eid_to_idx or rec_end not in eid_to_idx:
            continue
        idx_start = eid_to_idx[rec_start]
        idx_end = eid_to_idx[rec_end] + 1  # exclusive end
        boundaries_list.append([idx_start, idx_end])

    boundaries = np.array(boundaries_list, dtype=np.int64)
    np.save(segments_path, boundaries)

    seg_lens = boundaries[:, 1] - boundaries[:, 0]
    print(f"  Found {len(boundaries)} recording segments")
    print(f"  Segment lengths: min={seg_lens.min()}, max={seg_lens.max()}, "
          f"mean={seg_lens.mean():.0f}, median={np.median(seg_lens):.0f}")
    return boundaries


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_dir) if args.data_dir else CALVIN_DATA
    split_dir = data_root / args.split
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building GCBC dataset from {split_dir}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(flush=True)

    # 1. Get sorted episode IDs
    print("Step 1: Scanning episodes...", flush=True)
    episode_ids = get_episode_ids(split_dir)
    print(f"  Found {len(episode_ids)} episodes (range: {episode_ids[0]} - {episode_ids[-1]})")

    # Save episode index
    index_path = output_dir / "episode_ids.npy"
    if not index_path.exists():
        np.save(index_path, episode_ids)

    # 2. Build segment boundaries
    print("\nStep 2: Building segment boundaries...", flush=True)
    segments = build_segment_info(split_dir, episode_ids, output_dir)

    # 3. Collect actions
    print("\nStep 3: Collecting actions...", flush=True)
    actions = build_actions_mmap(split_dir, episode_ids, output_dir)

    # 4. Compute action stats
    print("\nStep 4: Computing action statistics...", flush=True)
    act_mean, act_std = compute_action_stats(actions, output_dir)

    # 5. Encode frames with Theia
    print("\nStep 5: Encoding frames with Theia...", flush=True)
    encode_frames(split_dir, episode_ids, output_dir, args.batch_size, device)

    # 6. Save metadata
    meta = {
        "split": args.split,
        "n_episodes": int(len(episode_ids)),
        "n_segments": int(len(segments)),
        "episode_id_range": [int(episode_ids[0]), int(episode_ids[-1])],
        "theia_model": "theia_small_cdiv",
        "theia_dim": 384,
        "num_patches": 196,
        "action_dim": 7,
        "action_type": "rel_actions",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone! Metadata: {meta}")


if __name__ == "__main__":
    main()

"""
Phase A v3: Theia-based Subgoal Selection (CALVIN ABC→D)
Uses CACHED theia_mean.npy / theia_features.npy for training (no re-encoding).
Only encodes validation frames (small split, ~1087 segments).

Keyframe selection: TaKSIE-style (LOWESS + slope inflection) on
Theia patch-mean features (384-dim).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/datasets/extract_subgoals.py

Output → /mnt/sda1/Datasets/chal2525/subgoals_abcd/
    subgoals_train.pkl
    subgoals_val.pkl
    theia_features.npy   (N_unique, 196, 384) float16
    theia_index.npy      (N_unique,) int64
"""

from __future__ import annotations

import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from tqdm import tqdm
from transformers import AutoModel

PROJECT      = Path(__file__).resolve().parent.parent
CALVIN_DIR   = Path("/mnt/sda1/Datasets/chal2525/calvin/task_ABC_D")
GCBC_DIR     = Path("/mnt/sda1/Datasets/chal2525/gcbc_data_abcd")
MODELS_DIR   = PROJECT / "models"
OUT_DIR      = Path("/mnt/sda1/Datasets/chal2525/subgoals_abcd")
DEVICE       = "cuda:0"
BATCH        = 256

# TaKSIE keyframe selection params (unchanged)
KF_LOWESS_FRAC = 1.0 / 6.0
KF_DELTA_1     = -0.001
KF_DELTA_2     = 0.001
KF_INTERVAL    = 7
KF_SCALE       = 1


# ============================================================================
# Theia (only used for validation encoding)
# ============================================================================

def load_theia():
    return AutoModel.from_pretrained(
        str(MODELS_DIR / "theia_small_cdiv"), trust_remote_code=True
    ).to(DEVICE).eval()


@torch.no_grad()
def encode_patches(model, frames_uint8: np.ndarray) -> np.ndarray:
    feats = []
    for i in range(0, len(frames_uint8), BATCH):
        batch = torch.from_numpy(frames_uint8[i:i + BATCH]).to(DEVICE)
        out = model.forward_feature(
            batch, do_resize=True, do_rescale=True, do_normalize=True
        )
        feats.append(out.cpu().numpy())
    return np.concatenate(feats)


# ============================================================================
# Keyframe selection (operates on pre-computed pooled features)
# ============================================================================

def select_keyframes(feats: np.ndarray) -> list[int]:
    """Returns RELATIVE keyframe indices, excluding final frame."""
    N = len(feats)
    end_rel = N - 1
    offset = 0
    all_kf = []

    while True:
        if offset == end_rel:
            all_kf.append(end_rel); break
        if offset > end_rel:
            raise RuntimeError(f"offset {offset} > end {end_rel}")

        remaining = feats[offset:]
        anchor = feats[offset]
        M = len(remaining)
        dists_raw = np.sqrt(np.sum((remaining - anchor) ** 2, axis=1))

        x = np.arange(M, dtype=float)
        dists_smooth = sm_lowess(dists_raw, x, frac=KF_LOWESS_FRAC)[:, 1]

        d_min, d_max = dists_smooth.min(), dists_smooth.max()
        if d_max - d_min < 1e-12:
            all_kf.append(end_rel); break
        dists_norm = (dists_smooth - d_min) / (d_max - d_min)

        slope = np.array([
            dists_norm[i + KF_SCALE] - dists_norm[i]
            for i in range(len(dists_norm) - KF_SCALE)
        ])

        found = False
        for i in range(len(slope)):
            if (i + 1) < KF_INTERVAL:
                continue
            elif ((i + 1) <= len(slope) - KF_INTERVAL
                  and slope[i] <= KF_DELTA_1
                  and slope[i - 1] >= KF_DELTA_2):
                offset = offset + i + 1
                all_kf.append(offset); found = True; break
            elif (i + 1) <= len(slope) - KF_INTERVAL and slope[i] > KF_DELTA_1:
                continue
            elif len(slope) - KF_INTERVAL <= 0:
                all_kf.append(end_rel); offset = end_rel; found = True; break
            elif (i + 1) == len(slope) or (len(slope) - (i + 1)) < KF_INTERVAL:
                all_kf.append(end_rel); offset = end_rel; found = True; break

        if not found:
            all_kf.append(end_rel); break

    return all_kf[:-1]


# ============================================================================
# Training split — uses cached theia_mean.npy
# ============================================================================

def process_training(theia_mean: np.ndarray):
    """Use cached pooled features. No disk reads of NPZ, no Theia encoding."""
    data_dir = CALVIN_DIR / "training"
    ann = np.load(data_dir / "lang_annotations" / "auto_lang_ann.npy",
                  allow_pickle=True).item()
    texts = ann["language"]["ann"]
    tasks = ann["language"]["task"]
    indices = ann["info"]["indx"]
    n = len(texts)
    print(f"\nTraining: {n} segments  (using cached theia_mean.npy)")

    all_kf = []
    for seg in tqdm(range(n), desc="Keyframes (train)"):
        start, end = int(indices[seg][0]), int(indices[seg][1])
        # Episode IDs == sequential frame indices in theia_mean
        pooled = theia_mean[start:end + 1].astype(np.float32)  # (Len, 384)
        kf_rel = select_keyframes(pooled)
        all_kf.append([start + k for k in kf_rel])

    pairs, seg_ids = [], []
    for seg in range(n):
        start, end = int(indices[seg][0]), int(indices[seg][1])
        wps = [start] + all_kf[seg] + [end]
        for i in range(len(wps) - 1):
            pairs.append((wps[i], wps[i + 1])); seg_ids.append(seg)

    pairs = np.array(pairs, dtype=np.int64)
    seg_ids = np.array(seg_ids, dtype=np.int32)
    unique = sorted(set(pairs[:, 0]) | set(pairs[:, 1]))

    print(f"  Pairs: {len(pairs)}, avg/seg {len(pairs)/n:.2f}, "
          f"keyframes/seg {np.mean([len(kf) for kf in all_kf]):.2f}, "
          f"unique frames: {len(unique)}")

    return {
        "pairs": pairs, "segment_ids": seg_ids, "texts": texts,
        "task_ids": tasks, "keyframes": all_kf,
        "unique_frames": np.array(unique, dtype=np.int64),
        "n_segments": n, "n_pairs": len(pairs),
        "calvin_indices": indices,
    }, set(unique)


# ============================================================================
# Validation split — needs encoding (no cache for val frames)
# ============================================================================

def process_validation(theia):
    data_dir = CALVIN_DIR / "validation"
    ann = np.load(data_dir / "lang_annotations" / "auto_lang_ann.npy",
                  allow_pickle=True).item()
    texts = ann["language"]["ann"]
    tasks = ann["language"]["task"]
    indices = ann["info"]["indx"]
    n = len(texts)
    print(f"\nValidation: {n} segments  (encoding fresh)")

    def _load(idx):
        return np.load(data_dir / f"episode_{idx:07d}.npz")["rgb_static"]

    all_kf = []
    for seg in tqdm(range(n), desc="Keyframes (val)"):
        start, end = int(indices[seg][0]), int(indices[seg][1])
        with ThreadPoolExecutor(max_workers=8) as pool:
            frames = np.stack(list(pool.map(_load, range(start, end + 1))))
        # pooled = patch-mean of full Theia output
        with torch.no_grad():
            batch = torch.from_numpy(frames).to(DEVICE)
            out = theia.forward_feature(
                batch, do_resize=True, do_rescale=True, do_normalize=True)
            pooled = out.mean(dim=1).cpu().numpy()  # (Len, 384)
        kf_rel = select_keyframes(pooled)
        all_kf.append([start + k for k in kf_rel])

    pairs, seg_ids = [], []
    for seg in range(n):
        start, end = int(indices[seg][0]), int(indices[seg][1])
        wps = [start] + all_kf[seg] + [end]
        for i in range(len(wps) - 1):
            pairs.append((wps[i], wps[i + 1])); seg_ids.append(seg)

    pairs = np.array(pairs, dtype=np.int64)
    seg_ids = np.array(seg_ids, dtype=np.int32)
    unique = sorted(set(pairs[:, 0]) | set(pairs[:, 1]))

    print(f"  Pairs: {len(pairs)}, avg/seg {len(pairs)/n:.2f}, "
          f"unique frames: {len(unique)}")

    return {
        "pairs": pairs, "segment_ids": seg_ids, "texts": texts,
        "task_ids": tasks, "keyframes": all_kf,
        "unique_frames": np.array(unique, dtype=np.int64),
        "n_segments": n, "n_pairs": len(pairs),
        "calvin_indices": indices,
    }, set(unique), data_dir


# ============================================================================
# Main
# ============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cached pooled features (1.4GB, mmap)
    print("Loading cached theia_mean.npy...")
    theia_mean = np.load(GCBC_DIR / "theia_mean.npy", mmap_mode="r")
    print(f"  shape={theia_mean.shape}, dtype={theia_mean.dtype}")

    # Training: use cache
    train_meta, train_frames = process_training(theia_mean)

    # Validation: needs Theia encoding
    print("\nLoading Theia for validation encoding...")
    theia = load_theia()
    val_meta, val_frames, val_dir = process_validation(theia)

    # ── Patches for unique frames ──
    # Training frames already in gcbc_data_abcd/theia_features.npy
    # Validation frames: encode fresh
    all_unique = sorted(train_frames | val_frames)
    n_unique = len(all_unique)
    frame_to_pos = {idx: pos for pos, idx in enumerate(all_unique)}

    print(f"\nPatches: {n_unique} unique frames "
          f"(train {len(train_frames)} from cache, val {len(val_frames)} fresh)")

    theia_feats = np.empty((n_unique, 196, 384), dtype=np.float16)
    train_patches = np.load(GCBC_DIR / "theia_features.npy", mmap_mode="r")

    # Fill train slots directly from cache
    for pos, idx in enumerate(tqdm(all_unique, desc="Filling train patches")):
        if idx in train_frames:
            theia_feats[pos] = train_patches[idx]

    # Encode val frames in batches
    val_positions = [pos for pos, idx in enumerate(all_unique) if idx in val_frames]
    val_indices = [all_unique[pos] for pos in val_positions]
    print(f"Encoding {len(val_indices)} val frames...")
    for batch_start in tqdm(range(0, len(val_indices), BATCH),
                            desc="Val patches"):
        chunk_idx = val_indices[batch_start:batch_start + BATCH]
        chunk_pos = val_positions[batch_start:batch_start + BATCH]
        frames = np.stack([
            np.load(val_dir / f"episode_{i:07d}.npz")["rgb_static"]
            for i in chunk_idx
        ])
        feats = encode_patches(theia, frames).astype(np.float16)
        for pos, f in zip(chunk_pos, feats):
            theia_feats[pos] = f

    del theia, train_patches
    torch.cuda.empty_cache()

    train_meta["frame_to_pos"] = frame_to_pos
    val_meta["frame_to_pos"] = frame_to_pos

    # ── Save ──
    print("\nSaving...")
    with open(OUT_DIR / "subgoals_train.pkl", "wb") as f:
        pickle.dump(train_meta, f)
    with open(OUT_DIR / "subgoals_val.pkl", "wb") as f:
        pickle.dump(val_meta, f)
    np.save(OUT_DIR / "theia_features.npy", theia_feats)
    np.save(OUT_DIR / "theia_index.npy", np.array(all_unique, dtype=np.int64))

    fmb = theia_feats.nbytes / (1024 * 1024)
    print(f"\nSaved to {OUT_DIR}/:")
    print(f"  subgoals_train.pkl  — {train_meta['n_pairs']} pairs, {train_meta['n_segments']} segs")
    print(f"  subgoals_val.pkl    — {val_meta['n_pairs']} pairs, {val_meta['n_segments']} segs")
    print(f"  theia_features.npy  — ({n_unique}, 196, 384) fp16  [{fmb:.0f} MB]")


if __name__ == "__main__":
    main()
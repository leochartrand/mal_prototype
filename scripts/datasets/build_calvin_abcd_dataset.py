#!/usr/bin/env python3
"""Build memory-mapped CALVIN ABC→D dataset for DiT fine-tuning.

One (init, target) pair per language-annotated segment:
  init   = first frame of the segment
  target = last  frame of the segment

Output format matches the DROID mmap dataset (MemoryMappedDataset-compatible):
  initial_224.npy, target_224.npy
  initial_embed_theia_small_cdiv.npy, target_embed_theia_small_cdiv.npy
  labels.pkl, labels_hidden_*.npy, labels_attn_mask_*.npy, labels_pooled_*.npy
  train_indices.npy, test_indices.npy

Usage:
    python scripts/datasets/build_calvin_abcd_dataset.py [--device 0]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── Paths ──
PROJECT   = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT / "models"
CALVIN_META = Path("/mnt/sda1/Datasets/chal2525/calvin/task_ABC_D")
MMAP_DIR    = Path("/mnt/sda1/Datasets/chal2525/mmap_data_abcd")

VISION_MODEL = "theia_small_cdiv"
TEXT_MODEL   = "all-MiniLM-L6-v2"
SEED     = 42
IMG_SIZE = 224
BATCH    = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


# ── Theia ──

def load_theia(device):
    model = AutoModel.from_pretrained(
        str(MODELS_DIR / "theia_small_cdiv"), trust_remote_code=True
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def encode_theia_patches(model, frames_uint8, device):
    """frames_uint8: (N, H, W, 3) uint8  →  (N, 196, 384) float32 numpy"""
    out = []
    for i in range(0, len(frames_uint8), BATCH):
        batch = torch.from_numpy(frames_uint8[i:i + BATCH]).to(device)
        feats = model.forward_feature(batch, do_resize=True, do_rescale=True, do_normalize=True)
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ── MiniLM ──

def encode_minilm(texts, device):
    model_path = str(MODELS_DIR / TEXT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    all_hidden, all_mask, all_pooled = [], [], []
    for i in tqdm(range(0, len(texts), 64), desc="MiniLM"):
        batch = texts[i:i + 64]
        enc = tokenizer(batch, padding="max_length", truncation=True,
                        max_length=25, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        all_hidden.append(out.last_hidden_state.float().cpu().numpy())
        all_mask.append(enc["attention_mask"].cpu().numpy().astype(np.int32))
        all_pooled.append(out.last_hidden_state[:, 0].float().cpu().numpy())

    return (np.concatenate(all_pooled, 0),
            np.concatenate(all_hidden, 0),
            np.concatenate(all_mask,   0))


# ── Frame loading ──

def load_frame_uint8(data_dir: Path, frame_idx: int) -> np.ndarray:
    """Returns (H, W, 3) uint8."""
    return np.load(data_dir / f"episode_{frame_idx:07d}.npz")["rgb_static"]


def load_frame_224(data_dir: Path, frame_idx: int) -> np.ndarray:
    """Returns (3, 224, 224) float32 [0,1]."""
    rgb = load_frame_uint8(data_dir, frame_idx)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


# ── Main ──

def process_split(split, ann, data_dir, theia, device):
    """Return arrays for one split.

    Returns:
        init_224   (N, 3, 224, 224) float32
        tgt_224    (N, 3, 224, 224) float32
        init_embed (N, 196, 384)    float32
        tgt_embed  (N, 196, 384)    float32
        texts      list[str]
    """
    texts   = list(ann["language"]["ann"])
    indices = ann["info"]["indx"]
    N = len(texts)
    print(f"  {split}: {N} segments")

    init_frames = [int(indices[i][0]) for i in range(N)]
    tgt_frames  = [int(indices[i][1]) for i in range(N)]

    # Load all unique frames, encode in batches
    unique_frames = sorted(set(init_frames) | set(tgt_frames))
    print(f"  Loading {len(unique_frames)} unique frames...")
    frames_uint8 = np.stack([load_frame_uint8(data_dir, fi) for fi in
                              tqdm(unique_frames, desc=f"  load {split}")])

    print(f"  Theia encoding {len(unique_frames)} frames...")
    patch_feats = encode_theia_patches(theia, frames_uint8, device)
    feat_cache  = {fi: patch_feats[j] for j, fi in enumerate(unique_frames)}

    # 224×224 images
    print(f"  Resizing to 224×224...")
    img_cache = {}
    for j, fi in enumerate(unique_frames):
        rgb = cv2.resize(frames_uint8[j], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_cache[fi] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    init_224   = np.stack([img_cache[fi]  for fi in init_frames])
    tgt_224    = np.stack([img_cache[fi]  for fi in tgt_frames])
    init_embed = np.stack([feat_cache[fi] for fi in init_frames])
    tgt_embed  = np.stack([feat_cache[fi] for fi in tgt_frames])

    return init_224, tgt_224, init_embed, tgt_embed, texts


def main():
    args = parse_args()
    device = f"cuda:{args.device}"
    rng = np.random.RandomState(SEED)
    MMAP_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading annotations...")
    train_ann = np.load(
        CALVIN_META / "training"   / "lang_annotations" / "auto_lang_ann.npy",
        allow_pickle=True).item()
    val_ann = np.load(
        CALVIN_META / "validation" / "lang_annotations" / "auto_lang_ann.npy",
        allow_pickle=True).item()

    print("Loading Theia...")
    theia = load_theia(device)

    print("\nProcessing training split...")
    tr_init_224, tr_tgt_224, tr_init_emb, tr_tgt_emb, tr_texts = process_split(
        "training", train_ann, CALVIN_META / "training", theia, device)

    print("\nProcessing validation split...")
    va_init_224, va_tgt_224, va_init_emb, va_tgt_emb, va_texts = process_split(
        "validation", val_ann, CALVIN_META / "validation", theia, device)

    del theia
    torch.cuda.empty_cache()

    N_train = len(tr_texts)
    N_val   = len(va_texts)
    N       = N_train + N_val
    print(f"\nTotal: {N} pairs ({N_train} train + {N_val} val)")

    # Shuffle train
    tr_order = rng.permutation(N_train)

    # ── MiniLM embeddings ──
    print("\nEncoding text with MiniLM...")
    all_texts = [tr_texts[i] for i in tr_order] + va_texts
    pooled, hidden, mask = encode_minilm(all_texts, device)

    # ── Write mmap ──
    print("\nWriting mmap arrays...")

    num_patches, latent_dim = 196, 384
    text_dim    = hidden.shape[2]
    max_text_len = hidden.shape[1]

    def mmap(name, shape, dtype):
        return np.lib.format.open_memmap(
            str(MMAP_DIR / name), mode="w+", dtype=dtype, shape=shape)

    init_224_mm   = mmap("initial_224.npy",                       (N, 3, IMG_SIZE, IMG_SIZE), np.float32)
    tgt_224_mm    = mmap("target_224.npy",                        (N, 3, IMG_SIZE, IMG_SIZE), np.float32)
    init_emb_mm   = mmap(f"initial_embed_{VISION_MODEL}.npy",     (N, num_patches, latent_dim), np.float32)
    tgt_emb_mm    = mmap(f"target_embed_{VISION_MODEL}.npy",      (N, num_patches, latent_dim), np.float32)
    hidden_mm     = mmap(f"labels_hidden_{TEXT_MODEL}.npy",       (N, max_text_len, text_dim), np.float32)
    mask_mm       = mmap(f"labels_attn_mask_{TEXT_MODEL}.npy",    (N, max_text_len), np.int32)
    pooled_mm     = mmap(f"labels_pooled_{TEXT_MODEL}.npy",       (N, pooled.shape[1]), np.float32)

    # Training rows (shuffled)
    for out_i, src_i in enumerate(tqdm(tr_order, desc="write train")):
        init_224_mm[out_i] = tr_init_224[src_i]
        tgt_224_mm[out_i]  = tr_tgt_224[src_i]
        init_emb_mm[out_i] = tr_init_emb[src_i]
        tgt_emb_mm[out_i]  = tr_tgt_emb[src_i]

    # Validation rows
    for j in tqdm(range(N_val), desc="write val"):
        init_224_mm[N_train + j] = va_init_224[j]
        tgt_224_mm[N_train + j]  = va_tgt_224[j]
        init_emb_mm[N_train + j] = va_init_emb[j]
        tgt_emb_mm[N_train + j]  = va_tgt_emb[j]

    # Text arrays (already in final order)
    hidden_mm[:]  = hidden
    mask_mm[:]    = mask
    pooled_mm[:]  = pooled

    del init_224_mm, tgt_224_mm, init_emb_mm, tgt_emb_mm, hidden_mm, mask_mm, pooled_mm

    # Labels
    labels = [tr_texts[i] for i in tr_order] + va_texts
    with open(MMAP_DIR / "labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    # Split indices
    np.save(MMAP_DIR / "train_indices.npy", np.arange(N_train,      dtype=np.int32))
    np.save(MMAP_DIR / "test_indices.npy",  np.arange(N_train, N,   dtype=np.int32))

    # Metadata
    meta = dict(
        vision_model=VISION_MODEL, text_model=TEXT_MODEL,
        N=N, N_train=N_train, N_val=N_val,
        num_patches=num_patches, latent_dim=latent_dim,
        text_dim=text_dim, max_text_len=max_text_len,
        source="CALVIN ABC→D", pair_type="segment_start_end",
    )
    with open(MMAP_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Dataset written to {MMAP_DIR}")
    print(f"  Train: {N_train}  Val: {N_val}  Total: {N}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Convert extract_subgoals.py output → DiT mmap-data format for CALVIN ABC→D.

Input  (from extract_subgoals.py):
    /mnt/sda1/Datasets/chal2525/subgoals_abcd/
        subgoals_train.pkl  (pairs, segment_ids, texts, frame_to_pos, ...)
        subgoals_val.pkl
        theia_features.npy  (N_unique, 196, 384) fp16

Output (DiT-trainable layout, matches mmap_data_abcd/):
    /mnt/sda1/Datasets/chal2525/mmap_data_abcd_subgoal/
        initial_embed_theia_small_cdiv.npy   (N_pairs, 196, 384) fp32
        target_embed_theia_small_cdiv.npy    (N_pairs, 196, 384) fp32
        labels.pkl                           list[str] of length N_pairs
        labels_hidden_all-MiniLM-L6-v2.npy   (N_pairs, 25, 384) fp32
        labels_attn_mask_all-MiniLM-L6-v2.npy(N_pairs, 25) int32
        labels_pooled_all-MiniLM-L6-v2.npy   (N_pairs, 384) fp32
        train_indices.npy
        test_indices.npy
        metadata.json

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/datasets/build_subgoal_mmap_abcd.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT     = Path(__file__).resolve().parent.parent
MODELS_DIR  = PROJECT / "models"
SRC_DIR     = Path("/mnt/sda1/Datasets/chal2525/subgoals_abcd")
DST_DIR     = Path("/mnt/sda1/Datasets/chal2525/mmap_data_abcd_subgoal")

VISION_MODEL = "theia_small_cdiv"
TEXT_MODEL   = "all-MiniLM-L6-v2"
MAX_TEXT_LEN = 25
DEVICE       = "cuda:0"


def encode_minilm_dedup(unique_texts: list[str]):
    """Encode each unique text once. Returns (hidden, mask, pooled) arrays + text→idx map."""
    model_path = str(MODELS_DIR / TEXT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()

    all_hidden, all_mask, all_pooled = [], [], []
    for i in tqdm(range(0, len(unique_texts), 64), desc="MiniLM"):
        batch = unique_texts[i:i + 64]
        enc = tokenizer(batch, padding="max_length", truncation=True,
                        max_length=MAX_TEXT_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        all_hidden.append(out.last_hidden_state.float().cpu().numpy())
        all_mask.append(enc["attention_mask"].cpu().numpy().astype(np.int32))
        all_pooled.append(out.last_hidden_state[:, 0].float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return (np.concatenate(all_hidden, 0),
            np.concatenate(all_mask,   0),
            np.concatenate(all_pooled, 0))


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load extract_subgoals.py outputs ──
    print("Loading subgoal extraction output...")
    with open(SRC_DIR / "subgoals_train.pkl", "rb") as f:
        tr = pickle.load(f)
    with open(SRC_DIR / "subgoals_val.pkl", "rb") as f:
        va = pickle.load(f)
    feats = np.load(SRC_DIR / "theia_features.npy", mmap_mode="r")
    print(f"  train pairs: {len(tr['pairs'])}, val pairs: {len(va['pairs'])}")
    print(f"  unique features: {feats.shape}")

    # frame_to_pos is shared between train/val (built jointly in extract_subgoals.py)
    frame_to_pos = tr["frame_to_pos"]

    # ── Build pair-level text list (one text per pair, copied from its segment) ──
    train_texts_per_pair = [tr["texts"][int(sid)] for sid in tr["segment_ids"]]
    val_texts_per_pair   = [va["texts"][int(sid)] for sid in va["segment_ids"]]
    all_pair_texts = train_texts_per_pair + val_texts_per_pair
    n_train = len(train_texts_per_pair)
    n_val   = len(val_texts_per_pair)
    N = n_train + n_val
    print(f"\nTotal pairs: N={N} (train={n_train}, val={n_val})")

    # ── MiniLM: encode UNIQUE texts once, then broadcast per pair ──
    unique_texts = sorted(set(all_pair_texts))
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    print(f"\nEncoding {len(unique_texts)} unique annotations with MiniLM...")
    hidden_u, mask_u, pooled_u = encode_minilm_dedup(unique_texts)
    print(f"  hidden: {hidden_u.shape}, pooled: {pooled_u.shape}")

    # ── mmap output writers ──
    def open_mm(name, shape, dtype):
        return np.lib.format.open_memmap(str(DST_DIR / name), mode="w+",
                                         dtype=dtype, shape=shape)

    text_dim = hidden_u.shape[2]
    pooled_dim = pooled_u.shape[1]
    init_mm   = open_mm(f"initial_embed_{VISION_MODEL}.npy", (N, 196, 384), np.float32)
    target_mm = open_mm(f"target_embed_{VISION_MODEL}.npy",  (N, 196, 384), np.float32)
    hidden_mm = open_mm(f"labels_hidden_{TEXT_MODEL}.npy",   (N, MAX_TEXT_LEN, text_dim), np.float32)
    mask_mm   = open_mm(f"labels_attn_mask_{TEXT_MODEL}.npy",(N, MAX_TEXT_LEN), np.int32)
    pooled_mm = open_mm(f"labels_pooled_{TEXT_MODEL}.npy",   (N, pooled_dim), np.float32)

    # ── Fill ──
    print("\nWriting train pairs...")
    for i, ((init_frame, tgt_frame), text) in enumerate(tqdm(
            zip(tr["pairs"], train_texts_per_pair), total=n_train)):
        init_mm[i]   = feats[frame_to_pos[int(init_frame)]].astype(np.float32)
        target_mm[i] = feats[frame_to_pos[int(tgt_frame)]].astype(np.float32)
        u = text_to_idx[text]
        hidden_mm[i] = hidden_u[u]
        mask_mm[i]   = mask_u[u]
        pooled_mm[i] = pooled_u[u]

    print("\nWriting val pairs...")
    for j, ((init_frame, tgt_frame), text) in enumerate(tqdm(
            zip(va["pairs"], val_texts_per_pair), total=n_val)):
        i = n_train + j
        init_mm[i]   = feats[frame_to_pos[int(init_frame)]].astype(np.float32)
        target_mm[i] = feats[frame_to_pos[int(tgt_frame)]].astype(np.float32)
        u = text_to_idx[text]
        hidden_mm[i] = hidden_u[u]
        mask_mm[i]   = mask_u[u]
        pooled_mm[i] = pooled_u[u]

    for arr in [init_mm, target_mm, hidden_mm, mask_mm, pooled_mm]:
        arr.flush()

    # ── labels.pkl + indices + metadata ──
    with open(DST_DIR / "labels.pkl", "wb") as f:
        pickle.dump(all_pair_texts, f)
    np.save(DST_DIR / "train_indices.npy", np.arange(n_train, dtype=np.int32))
    np.save(DST_DIR / "test_indices.npy",  np.arange(n_train, N, dtype=np.int32))

    meta = dict(
        vision_model=VISION_MODEL, text_model=TEXT_MODEL,
        N=int(N), N_train=int(n_train), N_val=int(n_val),
        num_patches=196, latent_dim=384,
        text_dim=int(text_dim), max_text_len=MAX_TEXT_LEN,
        source="CALVIN ABC→D",
        pair_type="taksie_lowess_subgoals",
        avg_pairs_per_segment=float(n_train / tr["n_segments"]),
    )
    with open(DST_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_gb = sum(a.nbytes for a in [init_mm, target_mm, hidden_mm, mask_mm, pooled_mm]) / 1e9
    print(f"\nDone. {DST_DIR}/  ({total_gb:.1f} GB total)")
    print(f"  N={N}  train={n_train}  val={n_val}")
    print(f"  unique annotations: {len(unique_texts)}")


if __name__ == "__main__":
    main()
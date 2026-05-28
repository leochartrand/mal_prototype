#!/usr/bin/env python3
"""
Threshold calibration for Theia Progress Evaluator.

Runs on CALVIN D validation trajectories with GT subgoals to find optimal δ
for Theia patch-mean cosine similarity. Reports:
  - Distribution of similarities at subgoal-reached frames vs in-progress frames
  - Separation gap between distributions
  - Optimal δ at various operating points (precision/recall)
  - ROC curve + PR curve

Usage:
    python scripts/calibration/calibrate_delta_offline.py [--split validation] [--gpu 0]
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["training", "validation"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-segments", type=int, default=0, help="0 = all")
    p.add_argument("--save-dir", default="results/calvin/threshold_calibration")
    return p.parse_args()


def load_frame(data_dir, idx):
    return np.load(data_dir / f"episode_{idx:07d}.npz")["rgb_static"]


def load_frames_threaded(data_dir, indices, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(pool.map(lambda i: load_frame(data_dir, i), indices))
    return np.stack(frames)


@torch.no_grad()
def encode_theia_batch(model, frames_uint8, device, batch_size=128):
    """(N, 200, 200, 3) uint8 → (N, 196, 384) float32."""
    feats = []
    for i in range(0, len(frames_uint8), batch_size):
        batch = torch.from_numpy(frames_uint8[i:i + batch_size]).to(device)
        out = model.forward_feature(batch, do_resize=True, do_rescale=True, do_normalize=True)
        feats.append(out.cpu())
    return torch.cat(feats)


def patch_mean_cosine(z_a, z_b):
    """(N, 196, 384) × (N, 196, 384) → (N,) float."""
    return F.cosine_similarity(z_a, z_b, dim=-1).mean(dim=-1)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    data_dir = PROJECT / "data" / "calvin" / "task_D_D" / args.split
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    ann = np.load(
        data_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True
    ).item()
    texts = ann["language"]["ann"]
    indices = ann["info"]["indx"]  # list of (start, end) tuples
    n_segments = len(texts)
    if args.max_segments > 0:
        n_segments = min(n_segments, args.max_segments)

    print(f"Split: {args.split}")
    print(f"Segments: {n_segments}")

    # Load Theia
    print("Loading Theia model...")
    theia = AutoModel.from_pretrained(
        str(PROJECT / "models" / "theia_small_cdiv"), trust_remote_code=True
    ).to(device).eval()

    # For each annotated segment, we sample frames along the trajectory
    # and compute cosine similarity to the segment's end frame (GT subgoal).
    # "Reached" frames are those within a few steps of the end.
    # "In-progress" frames are everything else.

    REACHED_WINDOW = 3  # frames within this distance of end are "reached"

    all_sims_reached = []    # similarities when near the subgoal
    all_sims_inprog = []     # similarities when still in progress
    all_sims_by_progress = defaultdict(list)  # binned by % progress through segment

    for seg_idx in tqdm(range(n_segments), desc="Processing segments"):
        start = int(indices[seg_idx][0])
        end = int(indices[seg_idx][1])
        seg_len = end - start + 1

        if seg_len < 5:
            continue

        # Sample uniformly (at most 20 frames to keep compute manageable)
        n_sample = min(seg_len, 20)
        sample_offsets = np.linspace(0, seg_len - 1, n_sample, dtype=int)
        sample_frames = [start + off for off in sample_offsets]

        # Always include the end frame as the GT subgoal
        all_frame_ids = sorted(set(sample_frames) | {end})

        # Load and encode
        frames = load_frames_threaded(data_dir, all_frame_ids)
        z_all = encode_theia_batch(theia, frames, device, args.batch_size)

        # Build frame_id → tensor index mapping
        id_to_idx = {fid: i for i, fid in enumerate(all_frame_ids)}

        z_goal = z_all[id_to_idx[end]].unsqueeze(0)  # (1, 196, 384)

        for fid in sample_frames:
            if fid == end:
                continue
            z_curr = z_all[id_to_idx[fid]].unsqueeze(0)  # (1, 196, 384)
            sim = patch_mean_cosine(z_curr, z_goal).item()

            dist_to_end = end - fid
            progress_pct = (fid - start) / max(seg_len - 1, 1)

            if dist_to_end <= REACHED_WINDOW:
                all_sims_reached.append(sim)
            else:
                all_sims_inprog.append(sim)

            # Bin by progress (0-10%, 10-20%, ..., 90-100%)
            bin_idx = min(int(progress_pct * 10), 9)
            all_sims_by_progress[bin_idx].append(sim)

    reached = np.array(all_sims_reached)
    inprog = np.array(all_sims_inprog)

    print(f"\nResults:")
    print(f"  Reached samples: {len(reached)}")
    print(f"  In-progress samples: {len(inprog)}")
    print(f"  Reached mean: {reached.mean():.4f} ± {reached.std():.4f}")
    print(f"  In-progress mean: {inprog.mean():.4f} ± {inprog.std():.4f}")
    print(f"  Separation gap: {reached.mean() - inprog.mean():.4f}")

    # ── Find optimal thresholds ──
    # Sweep δ from 0.5 to 1.0
    thresholds = np.linspace(0.5, 1.0, 501)
    tpr_arr = []  # true positive rate: fraction of reached correctly identified
    fpr_arr = []  # false positive rate: fraction of inprog incorrectly flagged

    for delta in thresholds:
        tp = (reached >= delta).sum()
        fn = (reached < delta).sum()
        fp = (inprog >= delta).sum()
        tn = (inprog < delta).sum()

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tpr_arr.append(tpr)
        fpr_arr.append(fpr)

    tpr_arr = np.array(tpr_arr)
    fpr_arr = np.array(fpr_arr)

    # Youden's J statistic: optimal = max(TPR - FPR)
    j_stat = tpr_arr - fpr_arr
    best_j_idx = np.argmax(j_stat)
    delta_youden = thresholds[best_j_idx]

    # F1-based threshold
    precision_arr = np.where(
        tpr_arr * len(reached) + fpr_arr * len(inprog) > 0,
        (tpr_arr * len(reached)) / (tpr_arr * len(reached) + fpr_arr * len(inprog)),
        0,
    )
    recall_arr = tpr_arr
    f1_arr = np.where(
        precision_arr + recall_arr > 0,
        2 * precision_arr * recall_arr / (precision_arr + recall_arr),
        0,
    )
    best_f1_idx = np.argmax(f1_arr)
    delta_f1 = thresholds[best_f1_idx]

    # 95% recall threshold
    idx_95 = np.where(tpr_arr >= 0.95)[0]
    delta_95recall = thresholds[idx_95[-1]] if len(idx_95) > 0 else thresholds[0]

    # 99% precision threshold
    idx_99prec = np.where(precision_arr >= 0.99)[0]
    delta_99prec = thresholds[idx_99prec[0]] if len(idx_99prec) > 0 else thresholds[-1]

    print(f"\nOptimal thresholds:")
    print(f"  Youden's J:    δ = {delta_youden:.4f}  (TPR={tpr_arr[best_j_idx]:.3f}, FPR={fpr_arr[best_j_idx]:.3f})")
    print(f"  Best F1:       δ = {delta_f1:.4f}  (F1={f1_arr[best_f1_idx]:.3f})")
    print(f"  95% recall:    δ = {delta_95recall:.4f}")
    print(f"  99% precision: δ = {delta_99prec:.4f}")

    # ── Plots ──

    # 1. Distribution histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(inprog, bins=80, alpha=0.6, label=f"In-progress (n={len(inprog)})", density=True, color="tab:blue")
    axes[0].hist(reached, bins=80, alpha=0.6, label=f"Reached (n={len(reached)})", density=True, color="tab:green")
    axes[0].axvline(delta_youden, color="red", linestyle="--", label=f"Youden δ={delta_youden:.3f}")
    axes[0].axvline(delta_f1, color="orange", linestyle="--", label=f"F1 δ={delta_f1:.3f}")
    axes[0].set_xlabel("Theia Patch-Mean Cosine Similarity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Similarity Distributions")
    axes[0].legend(fontsize=8)

    # 2. ROC curve
    axes[1].plot(fpr_arr, tpr_arr, "b-", linewidth=2)
    axes[1].plot(fpr_arr[best_j_idx], tpr_arr[best_j_idx], "ro", markersize=8,
                 label=f"Youden (δ={delta_youden:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_dir / "threshold_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {save_dir / 'threshold_calibration.png'}")

    # 3. Similarity by progress percentage
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bins = range(10)
    means = [np.mean(all_sims_by_progress[b]) if len(all_sims_by_progress[b]) > 0 else 0 for b in bins]
    stds = [np.std(all_sims_by_progress[b]) if len(all_sims_by_progress[b]) > 0 else 0 for b in bins]
    labels = [f"{b*10}-{(b+1)*10}%" for b in bins]
    ax2.bar(labels, means, yerr=stds, capsize=4, alpha=0.7, color="tab:blue")
    ax2.axhline(delta_youden, color="red", linestyle="--", label=f"Youden δ={delta_youden:.3f}")
    ax2.set_xlabel("Progress Through Segment")
    ax2.set_ylabel("Mean Cosine Similarity to Goal")
    ax2.set_title("Theia Similarity vs Progress")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(save_dir / "similarity_by_progress.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot saved: {save_dir / 'similarity_by_progress.png'}")

    # ── Save results ──
    results = {
        "n_reached": len(reached),
        "n_inprog": len(inprog),
        "reached_mean": float(reached.mean()),
        "reached_std": float(reached.std()),
        "inprog_mean": float(inprog.mean()),
        "inprog_std": float(inprog.std()),
        "separation_gap": float(reached.mean() - inprog.mean()),
        "delta_youden": float(delta_youden),
        "delta_f1": float(delta_f1),
        "delta_95recall": float(delta_95recall),
        "delta_99prec": float(delta_99prec),
        "youden_tpr": float(tpr_arr[best_j_idx]),
        "youden_fpr": float(fpr_arr[best_j_idx]),
        "best_f1": float(f1_arr[best_f1_idx]),
        "reached_window": REACHED_WINDOW,
    }

    with open(save_dir / "calibration_results.pkl", "wb") as f:
        pickle.dump({
            "summary": results,
            "reached_sims": reached,
            "inprog_sims": inprog,
            "sims_by_progress": dict(all_sims_by_progress),
            "roc": {"thresholds": thresholds, "tpr": tpr_arr, "fpr": fpr_arr},
        }, f)
    print(f"Results saved: {save_dir / 'calibration_results.pkl'}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Calibration Summary")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:>20s}: {v:.4f}")
        else:
            print(f"  {k:>20s}: {v}")


if __name__ == "__main__":
    main()

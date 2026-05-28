"""
Theia vs R3M vs LIV — Monotonicity / Progress Encoding Evaluation

For each of 5,124 annotated CALVIN segments, computes distance-to-goal curves
through three encoders and measures monotonicity + threshold separability.

Processes segment-by-segment to avoid caching ~150 GB of Theia features.

Usage:
    python scripts/analysis/eval_monotonicity.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from transformers import AutoModel

# ── paths ──
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data" / "calvin" / "task_D_D" / "training"
MODELS_DIR = PROJECT / "models"
OUT_DIR = PROJECT / "results" / "theia_monotonicity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Multi-GPU: Theia on GPU 0, R3M + LIV on GPU 1 (falls back to single GPU)
_N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEV_THEIA = "cuda:0" if _N_GPUS >= 1 else "cpu"
DEV_RL    = "cuda:1" if _N_GPUS >= 2 else DEV_THEIA  # R3M + LIV

SEED = 42
BATCH = 256           # frames per batch within a segment
LOWESS_FRAC = 0.167   # matches TaKSIE

np.random.seed(SEED)
torch.manual_seed(SEED)


# ====================================================================
# Model loading
# ====================================================================

def load_theia():
    model = AutoModel.from_pretrained(
        str(MODELS_DIR / "theia_small_cdiv"), trust_remote_code=True
    )
    model.to(DEV_THEIA).eval()
    return model


def load_r3m_model():
    from r3m import load_r3m
    model = load_r3m("resnet50")
    # Unwrap DataParallel and move to target device
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to(DEV_RL).eval()
    return model


def load_liv_model():
    from liv import load_liv
    model = load_liv("resnet50")
    # Unwrap DataParallel and move to target device
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to(DEV_RL).eval()
    # LIV stores self.device and calls input.to(self.device) internally
    model.device = DEV_RL
    return model


# ====================================================================
# Preprocessing
# ====================================================================

def preprocess_theia(frames_uint8: np.ndarray) -> torch.Tensor:
    """(B, H, W, 3) uint8 → tensor on device. Theia handles all internally."""
    return torch.from_numpy(frames_uint8).to(DEV_THEIA)


def preprocess_r3m(frames_uint8: np.ndarray) -> torch.Tensor:
    """(B, H, W, 3) uint8 → (B, 3, 224, 224) float [0-255] on device."""
    t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float()
    return t.to(DEV_RL)


def preprocess_liv(frames_uint8: np.ndarray) -> torch.Tensor:
    """(B, H, W, 3) uint8 → (B, 3, 224, 224) float [0-1] on device.

    LIV's forward() applies CLIP normalization internally via transforms_tensor,
    so we must NOT pre-normalize here — only scale to [0, 1].
    """
    t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0
    return t.to(DEV_RL)


# ====================================================================
# Encoding helpers
# ====================================================================

@torch.no_grad()
def encode_theia(model, frames_uint8: np.ndarray) -> np.ndarray:
    """Returns (N, 196, 384) float32 numpy."""
    feats = []
    for i in range(0, len(frames_uint8), BATCH):
        batch = preprocess_theia(frames_uint8[i : i + BATCH])
        out = model.forward_feature(
            batch, do_resize=True, do_rescale=True, do_normalize=True
        )
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def encode_r3m(model, frames_uint8: np.ndarray) -> np.ndarray:
    """Returns (N, 2048) float32 numpy."""
    feats = []
    for i in range(0, len(frames_uint8), BATCH):
        batch = preprocess_r3m(frames_uint8[i : i + BATCH])
        out = model(batch)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def encode_liv(model, frames_uint8: np.ndarray) -> np.ndarray:
    """Returns (N, 1024) float32 numpy."""
    feats = []
    for i in range(0, len(frames_uint8), BATCH):
        batch = preprocess_liv(frames_uint8[i : i + BATCH])
        out = model(batch, modality="vision")
        feats.append(out.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


# ====================================================================
# Frame I/O
# ====================================================================

def _load_single_frame(idx: int) -> np.ndarray:
    """Load and resize a single frame. Used by ThreadPoolExecutor."""
    import cv2
    path = DATA_DIR / f"episode_{idx:07d}.npz"
    img = np.load(path)["rgb_static"]  # (200, 200, 3) uint8
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)


def load_segment_frames(start: int, end: int) -> np.ndarray:
    """Load rgb_static from episode files, resize to 224×224.

    Returns (N, 224, 224, 3) uint8.  Uses threaded I/O.
    """
    indices = list(range(start, end))
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(_load_single_frame, indices))
    return np.stack(frames)


# ====================================================================
# Distance / metric helpers
# ====================================================================

def _l2_to_goal(feats: np.ndarray) -> np.ndarray:
    """Vectorized L2 distance of each row to the last row. (N, D) → (N,)."""
    diff = feats - feats[-1]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _cosine_to_goal(feats: np.ndarray) -> np.ndarray:
    """Vectorized cosine similarity of each row to the last row. (N, D) → (N,)."""
    goal = feats[-1]
    dots = feats @ goal
    norms = np.linalg.norm(feats, axis=-1) * np.linalg.norm(goal)
    norms = np.maximum(norms, 1e-12)
    return dots / norms


def _mean_patch_cosine_to_goal(feats: np.ndarray) -> np.ndarray:
    """Vector cosine similarity per-patch, averaged. (N, P, D) → (N,)."""
    goal = feats[-1]  # (P, D)
    # Normalize along feature dim
    f_norm = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12)
    g_norm = goal / (np.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12)
    # Per-patch cosine: (N, P)
    patch_cos = np.sum(f_norm * g_norm[None], axis=-1)
    return patch_cos.mean(axis=-1)


def compute_distance_curves(theia_feats, r3m_feats, liv_feats):
    """Compute all 8 distance-to-goal curves (vectorized).

    Returns dict[variant_name] → np.ndarray of shape (N,).
    """
    N = len(theia_feats)

    # Pooled Theia: mean over patches → (N, 384)
    theia_pooled = theia_feats.mean(axis=1)
    # Flat Theia → (N, 75264)
    theia_flat = theia_feats.reshape(N, -1)

    return {
        "L2_theia_pooled":        _l2_to_goal(theia_pooled),
        "cosine_theia_pooled":    _cosine_to_goal(theia_pooled),
        "L2_theia_flat":          _l2_to_goal(theia_flat),
        "cosine_theia_patch_mean": _mean_patch_cosine_to_goal(theia_feats),
        "L2_r3m":                 _l2_to_goal(r3m_feats),
        "cosine_r3m":             _cosine_to_goal(r3m_feats),
        "L2_liv":                 _l2_to_goal(liv_feats),
        "cosine_liv":             _cosine_to_goal(liv_feats),
    }


def compute_monotonicity(curve: np.ndarray, is_similarity: bool):
    """Compute monotonicity metrics for a single curve.

    For distance metrics (is_similarity=False): expect decreasing → Spearman ~ -1.
    For similarity metrics (is_similarity=True): expect increasing → Spearman ~ +1.
    We negate similarity so all metrics use "distance" convention (lower = closer).
    """
    if is_similarity:
        d = -curve  # negate so "good" = decreasing
    else:
        d = curve

    N = len(d)
    if N < 3:
        return {
            "spearman": np.nan,
            "kendall": np.nan,
            "violation_rate": np.nan,
            "smoothed_violation_rate": np.nan,
        }

    indices = np.arange(N)

    sp = stats.spearmanr(indices, d).statistic
    kt = stats.kendalltau(indices, d).statistic

    # Violation rate: fraction of consecutive pairs where distance increases
    diffs = np.diff(d)
    viol = float(np.sum(diffs > 0)) / (N - 1)

    # Smoothed violation rate
    t_norm = indices / (N - 1)
    smoothed = lowess(d, t_norm, frac=LOWESS_FRAC, return_sorted=False)
    diffs_sm = np.diff(smoothed)
    viol_sm = float(np.sum(diffs_sm > 0)) / (N - 1)

    return {
        "spearman": float(sp),
        "kendall": float(kt),
        "violation_rate": float(viol),
        "smoothed_violation_rate": float(viol_sm),
    }


def compute_threshold_metrics(curve_sim: np.ndarray):
    """Threshold separability for a cosine similarity curve.

    curve_sim: (N,) cosine similarity to goal frame (higher = closer).
    """
    N = len(curve_sim)
    if N < 10:
        return {
            "final_sim": np.nan,
            "early_sim": np.nan,
            "separation_gap": np.nan,
            "threshold_auc": np.nan,
        }

    # Final 5 frames vs first 5 frames
    final_sim = float(np.mean(curve_sim[-5:]))
    early_sim = float(np.mean(curve_sim[:5]))
    gap = final_sim - early_sim

    # AUC: last 20% of segment = positive
    cutoff = int(N * 0.8)
    labels = np.zeros(N, dtype=int)
    labels[cutoff:] = 1

    if labels.sum() == 0 or labels.sum() == N:
        auc = np.nan
    else:
        auc = float(roc_auc_score(labels, curve_sim))

    return {
        "final_sim": final_sim,
        "early_sim": early_sim,
        "separation_gap": gap,
        "threshold_auc": auc,
    }


# ====================================================================
# Main evaluation loop
# ====================================================================

IS_SIMILARITY = {
    "L2_theia_pooled": False,
    "cosine_theia_pooled": True,
    "L2_theia_flat": False,
    "cosine_theia_patch_mean": True,
    "L2_r3m": False,
    "cosine_r3m": True,
    "L2_liv": False,
    "cosine_liv": True,
}

COSINE_VARIANTS = [
    "cosine_theia_pooled",
    "cosine_theia_patch_mean",
    "cosine_r3m",
    "cosine_liv",
]


def run_evaluation():
    print("=" * 60)
    print("Theia vs R3M vs LIV — Monotonicity Evaluation")
    print("=" * 60)
    print(f"GPUs: {_N_GPUS}  Theia→{DEV_THEIA}  R3M+LIV→{DEV_RL}  Batch={BATCH}")

    # Load annotations
    ann_path = DATA_DIR / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    texts = ann["language"]["ann"]       # list of 5124 strings
    tasks = ann["language"]["task"]      # list of 5124 task IDs
    indices = ann["info"]["indx"]        # list of 5124 (start, end) tuples

    n_segments = len(texts)
    print(f"Segments: {n_segments}")

    # Load models
    print("\nLoading Theia-small...")
    theia = load_theia()
    print("Loading R3M...")
    r3m = load_r3m_model()
    print("Loading LIV...")
    liv = load_liv_model()
    print("All models loaded.\n")

    # Per-segment results
    results = []

    for seg_idx in tqdm(range(n_segments), desc="Segments"):
        start, end = indices[seg_idx]
        task_id = tasks[seg_idx]
        text = texts[seg_idx]
        seg_len = end - start  # exclusive end

        # Load frames
        frames = load_segment_frames(start, end)  # (N, 224, 224, 3) uint8

        # Encode — Theia on DEV_THEIA, R3M+LIV on DEV_RL (parallel if 2 GPUs)
        theia_feats = encode_theia(theia, frames)   # (N, 196, 384)
        r3m_feats = encode_r3m(r3m, frames)          # (N, 2048)
        liv_feats = encode_liv(liv, frames)           # (N, 1024)

        # Distance curves
        curves = compute_distance_curves(theia_feats, r3m_feats, liv_feats)

        # Monotonicity metrics
        mono = {}
        for variant, curve in curves.items():
            mono[variant] = compute_monotonicity(curve, IS_SIMILARITY[variant])

        # Threshold separability (cosine variants only)
        thresh = {}
        for variant in COSINE_VARIANTS:
            thresh[variant] = compute_threshold_metrics(curves[variant])

        results.append({
            "seg_idx": seg_idx,
            "start": int(start),
            "end": int(end),
            "seg_len": seg_len,
            "task_id": task_id,
            "text": text,
            "monotonicity": mono,
            "threshold": thresh,
            "curves": {k: v.tolist() for k, v in curves.items()},
        })

    # Save raw results
    print("\nSaving metrics...")
    with open(OUT_DIR / "metrics.pkl", "wb") as f:
        pickle.dump(results, f)

    # ── Aggregate statistics ──
    agg = compute_aggregates(results)
    with open(OUT_DIR / "aggregate_stats.json", "w") as f:
        json.dump(agg, f, indent=2)

    print("\nAggregate stats saved. Generating plots...")

    # ── Plots ──
    generate_all_plots(results, agg)

    print(f"\nAll outputs saved to {OUT_DIR}")
    print_decision_summary(agg)


# ====================================================================
# Aggregation
# ====================================================================

def compute_aggregates(results):
    variants = list(IS_SIMILARITY.keys())
    agg = {"n_segments": len(results)}

    # Per-variant monotonicity stats
    for v in variants:
        sp_vals = [r["monotonicity"][v]["spearman"] for r in results
                   if not np.isnan(r["monotonicity"][v]["spearman"])]
        kt_vals = [r["monotonicity"][v]["kendall"] for r in results
                   if not np.isnan(r["monotonicity"][v]["kendall"])]
        vr_vals = [r["monotonicity"][v]["violation_rate"] for r in results
                   if not np.isnan(r["monotonicity"][v]["violation_rate"])]
        svr_vals = [r["monotonicity"][v]["smoothed_violation_rate"] for r in results
                    if not np.isnan(r["monotonicity"][v]["smoothed_violation_rate"])]

        agg[v] = {
            "spearman": _stats(sp_vals),
            "kendall": _stats(kt_vals),
            "violation_rate": _stats(vr_vals),
            "smoothed_violation_rate": _stats(svr_vals),
        }

    # Per-variant threshold stats
    for v in COSINE_VARIANTS:
        auc_vals = [r["threshold"][v]["threshold_auc"] for r in results
                    if not np.isnan(r["threshold"][v]["threshold_auc"])]
        gap_vals = [r["threshold"][v]["separation_gap"] for r in results
                    if not np.isnan(r["threshold"][v]["separation_gap"])]
        agg[f"threshold_{v}"] = {
            "auc": _stats(auc_vals),
            "separation_gap": _stats(gap_vals),
        }

    # Per-task breakdown
    task_ids = sorted(set(r["task_id"] for r in results))
    per_task = {}
    for tid in task_ids:
        task_results = [r for r in results if r["task_id"] == tid]
        per_task[tid] = {"count": len(task_results)}
        for v in variants:
            sp = [r["monotonicity"][v]["spearman"] for r in task_results
                  if not np.isnan(r["monotonicity"][v]["spearman"])]
            per_task[tid][f"spearman_{v}"] = _stats(sp) if sp else None
    agg["per_task"] = per_task

    return agg


def _stats(vals):
    if not vals:
        return {"mean": None, "median": None, "std": None, "n": 0}
    a = np.array(vals)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "n": len(a),
    }


# ====================================================================
# Plotting
# ====================================================================

# Determine best Theia variant dynamically
def _best_theia_variant(agg, metric_type="spearman"):
    """Pick the Theia variant with best median Spearman (most negative)."""
    variants = [
        "L2_theia_pooled", "cosine_theia_pooled",
        "L2_theia_flat", "cosine_theia_patch_mean",
    ]
    best, best_score = variants[0], 999
    for v in variants:
        med = agg[v][metric_type]["median"]
        if med is None:
            continue
        # For Spearman on distance: most negative is best
        # For Spearman on similarity: most positive is best (but we negate, so still most negative)
        score = med  # all stored after negation for similarity variants
        if score < best_score:
            best_score = score
            best = v
    return best


def _best_theia_cosine(agg):
    """Pick the Theia cosine variant with best median AUC."""
    variants = ["cosine_theia_pooled", "cosine_theia_patch_mean"]
    best, best_score = variants[0], -1
    for v in variants:
        med = agg[f"threshold_{v}"]["auc"]["median"]
        if med is not None and med > best_score:
            best_score = med
            best = v
    return best


def normalize_curve(c):
    lo, hi = c.min(), c.max()
    if hi - lo < 1e-12:
        return np.zeros_like(c)
    return (c - lo) / (hi - lo)


def generate_all_plots(results, agg):
    plot1_individual_curves(results, agg)
    plot2_spearman_distributions(results, agg)
    plot3_violation_distributions(results, agg)
    plot4_per_task_breakdown(results, agg)
    plot5_theia_vs_r3m_scatter(results, agg)
    plot6_theia_vs_liv_scatter(results, agg)
    plot7_threshold_separability(results, agg)


def plot1_individual_curves(results, agg):
    """4×4 grid of individual trajectory curves."""
    best_theia = _best_theia_variant(agg)

    # Select 16 diverse segments: pick from distinct tasks, varying lengths
    task_ids = sorted(set(r["task_id"] for r in results))
    rng = np.random.RandomState(SEED)

    selected = []
    # Try to get one per task first
    for tid in task_ids:
        cands = [r for r in results if r["task_id"] == tid]
        selected.append(rng.choice(cands))
        if len(selected) >= 16:
            break
    # Fill remaining from random
    while len(selected) < 16:
        selected.append(results[rng.randint(len(results))])

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for i, r in enumerate(selected[:16]):
        ax = axes[i]
        N = r["seg_len"]
        t = np.linspace(0, 1, N)

        # Get curves and normalize
        c_theia = np.array(r["curves"][best_theia])
        c_r3m = np.array(r["curves"]["L2_r3m"])
        c_liv = np.array(r["curves"]["L2_liv"])

        # For similarity variants, negate so "distance" convention
        if IS_SIMILARITY[best_theia]:
            c_theia = -c_theia

        ax.plot(t, normalize_curve(c_theia), "C0-", lw=1.5, label=f"Theia ({best_theia})")
        ax.plot(t, normalize_curve(c_r3m), "C1--", lw=1.5, label="R3M")
        ax.plot(t, normalize_curve(c_liv), "C2:", lw=1.5, label="LIV")

        title = r["text"][:40] + ("..." if len(r["text"]) > 40 else "")
        ax.set_title(f"{title}\n[{r['task_id']}, {N}fr]", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=6)

    fig.suptitle("Distance to Goal (normalized) — Individual Trajectories", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "plot_individual_curves.png", dpi=150)
    plt.close(fig)
    print("  plot_individual_curves.png")


def plot2_spearman_distributions(results, agg):
    """Histogram + KDE of Spearman correlations."""
    groups = {
        "Theia pooled L2": "L2_theia_pooled",
        "Theia pooled cos": "cosine_theia_pooled",
        "Theia flat L2": "L2_theia_flat",
        "Theia patch cos": "cosine_theia_patch_mean",
        "R3M L2": "L2_r3m",
        "R3M cos": "cosine_r3m",
        "LIV L2": "L2_liv",
        "LIV cos": "cosine_liv",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    for (label, variant), color in zip(groups.items(), colors):
        vals = [r["monotonicity"][variant]["spearman"] for r in results
                if not np.isnan(r["monotonicity"][variant]["spearman"])]
        ax.hist(vals, bins=60, alpha=0.35, color=color, label=label, density=True)

    ax.axvline(-0.8, color="red", ls="--", lw=1.5, label="threshold = -0.8")
    ax.set_xlabel("Spearman correlation (frame idx vs distance-to-goal)")
    ax.set_ylabel("Density")
    ax.set_title("Spearman Correlation Distributions")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_spearman_distributions.png", dpi=150)
    plt.close(fig)
    print("  plot_spearman_distributions.png")


def plot3_violation_distributions(results, agg):
    """Histogram of smoothed violation rates."""
    groups = {
        "Theia pooled L2": "L2_theia_pooled",
        "Theia pooled cos": "cosine_theia_pooled",
        "Theia flat L2": "L2_theia_flat",
        "Theia patch cos": "cosine_theia_patch_mean",
        "R3M L2": "L2_r3m",
        "R3M cos": "cosine_r3m",
        "LIV L2": "L2_liv",
        "LIV cos": "cosine_liv",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    for (label, variant), color in zip(groups.items(), colors):
        vals = [r["monotonicity"][variant]["smoothed_violation_rate"] for r in results
                if not np.isnan(r["monotonicity"][variant]["smoothed_violation_rate"])]
        ax.hist(vals, bins=60, alpha=0.35, color=color, label=label, density=True)

    ax.set_xlabel("Smoothed Violation Rate (LOWESS, frac=0.167)")
    ax.set_ylabel("Density")
    ax.set_title("Smoothed Monotonicity Violation Rate Distributions")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_violation_distributions.png", dpi=150)
    plt.close(fig)
    print("  plot_violation_distributions.png")


def plot4_per_task_breakdown(results, agg):
    """Box plot: per-task Spearman for best Theia, R3M, LIV."""
    best_theia = _best_theia_variant(agg)
    variants = [best_theia, "L2_r3m", "L2_liv"]
    labels = [f"Theia ({best_theia})", "R3M (L2)", "LIV (L2)"]

    task_ids = sorted(set(r["task_id"] for r in results))
    n_tasks = len(task_ids)

    fig, axes = plt.subplots(len(variants), 1, figsize=(max(16, n_tasks * 0.5), 4 * len(variants)),
                             sharex=True)

    for ax, variant, label in zip(axes, variants, labels):
        data = []
        for tid in task_ids:
            vals = [r["monotonicity"][variant]["spearman"] for r in results
                    if r["task_id"] == tid
                    and not np.isnan(r["monotonicity"][variant]["spearman"])]
            data.append(vals)

        bp = ax.boxplot(data, positions=range(n_tasks), widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("C0" if "theia" in variant.lower() else
                                "C1" if "r3m" in variant.lower() else "C2")
            patch.set_alpha(0.5)
        ax.axhline(-0.8, color="red", ls="--", lw=1, alpha=0.7)
        ax.set_ylabel("Spearman ρ")
        ax.set_title(label)

    axes[-1].set_xticks(range(n_tasks))
    axes[-1].set_xticklabels(task_ids, rotation=90, fontsize=6)
    axes[-1].set_xlabel("Task ID")

    fig.suptitle("Per-Task Spearman Breakdown", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "plot_per_task_breakdown.png", dpi=150)
    plt.close(fig)
    print("  plot_per_task_breakdown.png")


def plot5_theia_vs_r3m_scatter(results, agg):
    """Scatter: Theia vs R3M Spearman per segment."""
    best_theia = _best_theia_variant(agg)

    task_ids = sorted(set(r["task_id"] for r in results))
    tid_to_idx = {t: i for i, t in enumerate(task_ids)}
    cmap = plt.cm.get_cmap("tab20", len(task_ids))

    fig, ax = plt.subplots(figsize=(8, 8))

    x_vals, y_vals, colors = [], [], []
    for r in results:
        sp_r3m = r["monotonicity"]["L2_r3m"]["spearman"]
        sp_theia = r["monotonicity"][best_theia]["spearman"]
        if np.isnan(sp_r3m) or np.isnan(sp_theia):
            continue
        x_vals.append(sp_r3m)
        y_vals.append(sp_theia)
        colors.append(tid_to_idx[r["task_id"]])

    ax.scatter(x_vals, y_vals, c=colors, cmap=cmap, s=8, alpha=0.5, edgecolors="none")
    ax.plot([-1, 1], [-1, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("R3M Spearman ρ")
    ax.set_ylabel(f"Theia ({best_theia}) Spearman ρ")
    ax.set_title("Theia vs R3M: Can Theia replace R3M for subgoal selection?")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_theia_vs_r3m_scatter.png", dpi=150)
    plt.close(fig)
    print("  plot_theia_vs_r3m_scatter.png")


def plot6_theia_vs_liv_scatter(results, agg):
    """Scatter: Theia vs LIV threshold AUC per segment."""
    best_cos = _best_theia_cosine(agg)

    task_ids = sorted(set(r["task_id"] for r in results))
    tid_to_idx = {t: i for i, t in enumerate(task_ids)}
    cmap = plt.cm.get_cmap("tab20", len(task_ids))

    fig, ax = plt.subplots(figsize=(8, 8))

    x_vals, y_vals, colors = [], [], []
    for r in results:
        auc_liv = r["threshold"]["cosine_liv"]["threshold_auc"]
        auc_theia = r["threshold"][best_cos]["threshold_auc"]
        if np.isnan(auc_liv) or np.isnan(auc_theia):
            continue
        x_vals.append(auc_liv)
        y_vals.append(auc_theia)
        colors.append(tid_to_idx[r["task_id"]])

    ax.scatter(x_vals, y_vals, c=colors, cmap=cmap, s=8, alpha=0.5, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("LIV cosine AUC")
    ax.set_ylabel(f"Theia ({best_cos}) AUC")
    ax.set_title("Theia vs LIV: Can Theia replace LIV for progress evaluation?")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_theia_vs_liv_scatter.png", dpi=150)
    plt.close(fig)
    print("  plot_theia_vs_liv_scatter.png")


def plot7_threshold_separability(results, agg):
    """8 example segments: cosine similarity over time + threshold lines."""
    best_cos = _best_theia_cosine(agg)

    rng = np.random.RandomState(SEED + 7)
    selected = rng.choice(len(results), size=min(8, len(results)), replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, idx in enumerate(selected):
        r = results[idx]
        ax = axes[i]
        N = r["seg_len"]
        t = np.linspace(0, 1, N)

        c_theia = np.array(r["curves"][best_cos])
        c_liv = np.array(r["curves"]["cosine_liv"])

        ax.plot(t, c_theia, "C0-", lw=1.5, label=f"Theia ({best_cos})")
        ax.plot(t, c_liv, "C2:", lw=1.5, label="LIV cosine")

        # Shade last 20%
        cutoff = 0.8
        ax.axvspan(cutoff, 1.0, alpha=0.15, color="green", label="reached (last 20%)")

        # Candidate thresholds
        for thr, ls in [(0.90, "--"), (0.95, "-.")]:
            ax.axhline(thr, color="gray", ls=ls, lw=0.8, alpha=0.6)

        title = r["text"][:35] + ("..." if len(r["text"]) > 35 else "")
        ax.set_title(f"{title}\n[{r['task_id']}]", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        if i == 0:
            ax.legend(fontsize=6)

    fig.suptitle("Threshold Separability: Cosine Similarity to Goal over Time", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "plot_threshold_separability.png", dpi=150)
    plt.close(fig)
    print("  plot_threshold_separability.png")


# ====================================================================
# Decision summary
# ====================================================================

def print_decision_summary(agg):
    print("\n" + "=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)

    # Best Theia for subgoal selection
    theia_variants = [
        "L2_theia_pooled", "cosine_theia_pooled",
        "L2_theia_flat", "cosine_theia_patch_mean",
    ]
    print("\n--- Subgoal Selection (Spearman ρ, median) ---")
    for v in theia_variants:
        sp = agg[v]["spearman"]
        print(f"  {v:30s}  median={sp['median']:.4f}  mean={sp['mean']:.4f}")
    sp_r3m = agg["L2_r3m"]["spearman"]
    print(f"  {'L2_r3m':30s}  median={sp_r3m['median']:.4f}  mean={sp_r3m['mean']:.4f}")
    sp_cosR = agg["cosine_r3m"]["spearman"]
    print(f"  {'cosine_r3m':30s}  median={sp_cosR['median']:.4f}  mean={sp_cosR['mean']:.4f}")

    print("\n--- Progress Evaluation (Threshold AUC, median) ---")
    for v in COSINE_VARIANTS:
        a = agg[f"threshold_{v}"]["auc"]
        g = agg[f"threshold_{v}"]["separation_gap"]
        print(f"  {v:30s}  AUC={a['median']:.4f}  gap={g['median']:.4f}")

    print()


# ====================================================================
if __name__ == "__main__":
    run_evaluation()

"""
Compare Theia encoder latent spaces across model variants.

Two modes:
  --precomputed  (default)  Load pre-computed embeddings from mmap_data/
  --live                    Load Theia models and run forward_feature on raw images

Analyses:
  1. Distribution stats (mean-pooled effective rank, std, norms)
  2. Token-level init↔target separation (per-patch cosine sim distribution)
  3. Edit vector analysis (magnitude, effective rank, spatial heatmap)
  4. Copy-input baseline (what DiT must beat)
"""

import argparse
import sys
import os
import numpy as np
import torch
from tqdm import tqdm

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=0,
                    help="Number of samples to use (0 = all)")
parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"],
                    help="Which split to analyze: train, test, or all")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size (only used in --live mode)")
parser.add_argument("--device", type=str, default="cuda:0",
                    help="Device (only used in --live mode)")
parser.add_argument("--live", action="store_true",
                    help="Run actual model inference instead of using precomputed embeddings")
parser.add_argument("--save_figures", action="store_true",
                    help="Save spatial heatmaps and histograms")
parser.add_argument("--fig_dir", default="results/latent_analysis/",
                    help="Directory for saved figures")
args = parser.parse_args()

DEVICE = torch.device(args.device)
MODE = "live" if args.live else "precomputed"

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMAP_DIR = os.path.join(ROOT, "../../../mnt/sda1/Datasets/chal2525/mmap_data")
MODEL_DIR = os.path.join(ROOT, "models")

VARIANT_NAMES = ["tiny_cdiv", "tiny_cddsv", "small_cdiv", "small_cddsv", "base_cdiv", "base_cddsv", "sd_vae"]

# ── Load shared data ────────────────────────────────────────────────────────
print(f"Mode: {MODE}")
print("Loading indices …")
train_idx = np.load(os.path.join(MMAP_DIR, "train_indices.npy"))
test_idx = np.load(os.path.join(MMAP_DIR, "test_indices.npy"))

if args.split == "all":
    all_idx = np.concatenate([train_idx, test_idx])
elif args.split == "train":
    all_idx = train_idx
else:
    all_idx = test_idx

# Sub-sample if requested
if args.n_samples > 0 and args.n_samples < len(all_idx):
    rng = np.random.default_rng(42)
    sel = rng.choice(all_idx, size=args.n_samples, replace=False)
    sel.sort()
else:
    sel = np.sort(all_idx)

print(f"Using {len(sel)} samples  (split={args.split}, train={len(train_idx)}, test={len(test_idx)}, total={len(train_idx)+len(test_idx)})")

# ── Precomputed loader ───────────────────────────────────────────────────────
def load_precomputed(variant, indices):
    """Load pre-computed embeddings from mmap_data/, return (z_init, z_targ) each [N, 196, D]."""
    vision_model = f"theia_{variant}"
    z_init_mmap = np.load(os.path.join(MMAP_DIR, f"initial_embed_{vision_model}.npy"), mmap_mode="r")
    z_targ_mmap = np.load(os.path.join(MMAP_DIR, f"target_embed_{vision_model}.npy"), mmap_mode="r")
    # Read selected indices into memory
    z_init = np.array(z_init_mmap[indices])
    z_targ = np.array(z_targ_mmap[indices])
    return z_init, z_targ

# ── Live inference loader ────────────────────────────────────────────────────
@torch.no_grad()
def encode_all(model, images_mmap, indices, batch_size, device):
    """Run forward_feature on images from mmap, return [N, 196, D] numpy."""
    all_feats = []
    for start in tqdm(range(0, len(indices), batch_size), leave=False):
        batch_idx = indices[start : start + batch_size]
        imgs = torch.from_numpy(np.array(images_mmap[batch_idx]))  # [B, 3, 224, 224]
        imgs_uint8 = (imgs * 255).to(torch.uint8)
        imgs_hwc = imgs_uint8.permute(0, 2, 3, 1)           # [B, 224, 224, 3]
        feats = model.forward_feature(imgs_hwc)              # [B, 196, D]
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def load_live(variant, indices, batch_size, device):
    """Load model, run forward_feature, return (z_init, z_targ) each [N, 196, D]."""
    from transformers import AutoModel
    model_path = os.path.join(MODEL_DIR, f"theia_{variant}")
    initial_mmap = np.load(os.path.join(MMAP_DIR, "initial_224.npy"), mmap_mode="r")
    target_mmap  = np.load(os.path.join(MMAP_DIR, "target_224.npy"),  mmap_mode="r")

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()

    print("  Encoding initial states …")
    z_init = encode_all(model, initial_mmap, indices, batch_size, device)
    print("  Encoding target states …")
    z_targ = encode_all(model, target_mmap, indices, batch_size, device)

    del model
    torch.cuda.empty_cache()
    return z_init, z_targ

# ── SD v1.5 VAE loader ───────────────────────────────────────────────────────
@torch.no_grad()
def encode_vae(vae, images_mmap, indices, batch_size, device):
    """Encode images through SD VAE encoder, return [N, 784, 4] numpy.
    VAE expects [B, 3, H, W] in [-1, 1]. Output is [B, 4, 28, 28] for 224×224 input.
    Reshaped to [B, 784, 4] to match token-based analysis (784 spatial positions, 4 channels).
    """
    all_feats = []
    for start in tqdm(range(0, len(indices), batch_size), leave=False):
        batch_idx = indices[start : start + batch_size]
        imgs = torch.from_numpy(np.array(images_mmap[batch_idx])).to(device)  # [B, 3, 224, 224] float32 [0,1]
        imgs = imgs * 2 - 1  # [0,1] → [-1,1]
        latent_dist = vae.encode(imgs).latent_dist
        z = latent_dist.mean  # [B, 4, 28, 28] — use mean (deterministic)
        z = z.flatten(2).permute(0, 2, 1)  # [B, 4, 784] → [B, 784, 4]
        all_feats.append(z.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def load_sd_vae(indices, batch_size, device):
    """Load SD v1.5 VAE, encode images, return (z_init, z_targ) each [N, 784, 4]."""
    from diffusers import AutoencoderKL
    initial_mmap = np.load(os.path.join(MMAP_DIR, "initial_224.npy"), mmap_mode="r")
    target_mmap  = np.load(os.path.join(MMAP_DIR, "target_224.npy"),  mmap_mode="r")

    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    vae = vae.to(device).eval()

    print("  Encoding initial states (VAE) …")
    z_init = encode_vae(vae, initial_mmap, indices, batch_size, device)
    print("  Encoding target states (VAE) …")
    z_targ = encode_vae(vae, target_mmap, indices, batch_size, device)

    del vae
    torch.cuda.empty_cache()
    return z_init, z_targ

# ── Metric helpers ───────────────────────────────────────────────────────────
def effective_rank(X):
    """Effective rank via Shannon entropy of normalised singular values.
    X: [N, D] matrix.  Returns scalar in [1, min(N,D)]."""
    # Subsample for speed if N is large
    n = min(X.shape[0], 5000)
    if n < X.shape[0]:
        idx = np.random.default_rng(42).choice(X.shape[0], n, replace=False)
        X = X[idx]
    X_c = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X_c, compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

def dead_dim_fraction(X, threshold=1e-4):
    """Fraction of dims with std < threshold (relative to mean std)."""
    stds = X.std(axis=0)
    mean_std = stds.mean()
    return float((stds < threshold * mean_std).mean())

def cosine_sim_batch(A, B, axis=-1):
    """Element-wise cosine similarity along `axis`."""
    dot = (A * B).sum(axis=axis)
    norm_a = np.linalg.norm(A, axis=axis)
    norm_b = np.linalg.norm(B, axis=axis)
    return dot / (norm_a * norm_b + 1e-12)

def mean_cosine_sim(A, B):
    """Mean cosine similarity between corresponding rows of A and B."""
    return float(cosine_sim_batch(A, B, axis=-1).mean())

def relative_l2(A, B):
    """Mean ||a-b||/||a|| across samples."""
    diff = np.linalg.norm(A - B, axis=-1)
    base = np.linalg.norm(A, axis=-1) + 1e-12
    return float((diff / base).mean())

# ── Analysis sections ────────────────────────────────────────────────────────

def analyze_distribution(z_init_pool, z_targ_pool, D):
    """Original analysis: mean-pooled distribution stats."""
    stds_init = z_init_pool.std(axis=0)
    stds_targ = z_targ_pool.std(axis=0)
    erank_init = effective_rank(z_init_pool)
    erank_targ = effective_rank(z_targ_pool)
    dead_init = dead_dim_fraction(z_init_pool)
    dead_targ = dead_dim_fraction(z_targ_pool)
    norms_init = np.linalg.norm(z_init_pool, axis=-1)
    norms_targ = np.linalg.norm(z_targ_pool, axis=-1)
    cos_sim = mean_cosine_sim(z_init_pool, z_targ_pool)
    rel = relative_l2(z_init_pool, z_targ_pool)

    print(f"\n  --- Distribution stats (mean-pooled, [N,{D}]) ---")
    print(f"  Std per dim       : init {stds_init.mean():.4f} ± {stds_init.std():.4f}  |  targ {stds_targ.mean():.4f} ± {stds_targ.std():.4f}")
    print(f"  Effective rank    : init {erank_init:.1f} / {D}  |  targ {erank_targ:.1f} / {D}")
    print(f"  Dead dims fraction: init {dead_init:.4f}  |  targ {dead_targ:.4f}")
    print(f"  L2 norm           : init {norms_init.mean():.2f} ± {norms_init.std():.2f}  |  targ {norms_targ.mean():.2f} ± {norms_targ.std():.2f}")
    print(f"\n  --- Init ↔ Target separation (mean-pooled) ---")
    print(f"  Cosine similarity : {cos_sim:.4f}")
    print(f"  Relative L2       : {rel:.4f}")

    return {
        "std_init": float(stds_init.mean()), "std_targ": float(stds_targ.mean()),
        "erank_init": erank_init, "erank_targ": erank_targ,
        "dead_init": dead_init, "dead_targ": dead_targ,
        "norm_init": float(norms_init.mean()), "norm_targ": float(norms_targ.mean()),
        "cos_sim": cos_sim, "rel_l2": rel,
    }


def analyze_token_separation(z_init, z_targ, T):
    """Token-level init↔target cosine similarity.
    z_init, z_targ: [N, T, D]
    """
    # Per-token cosine sim: [N, T]
    cos_per_token = cosine_sim_batch(z_init, z_targ, axis=-1)
    cos_flat = cos_per_token.flatten()

    # Per-position stats (averaged over samples): [T]
    cos_per_pos = cos_per_token.mean(axis=0)

    # Fraction of tokens below thresholds
    thresholds = [0.95, 0.90, 0.85]

    print(f"\n  --- Token-level init↔target separation ([N,{T},D]) ---")
    print(f"  Cosine sim (all tokens) : {cos_flat.mean():.4f} ± {cos_flat.std():.4f}")
    print(f"  Cosine sim (per-pos)    : min={cos_per_pos.min():.4f}, max={cos_per_pos.max():.4f}, median={np.median(cos_per_pos):.4f}")
    for t in thresholds:
        frac = (cos_per_token < t).mean()
        print(f"  Tokens with cos < {t:.2f}  : {frac:.4f} ({frac*100:.1f}%)")

    # Samples with at least one "changed" token
    min_cos_per_sample = cos_per_token.min(axis=1)  # [N]
    for t in thresholds:
        frac = (min_cos_per_sample < t).mean()
        print(f"  Samples with ≥1 tok < {t:.2f}: {frac:.4f} ({frac*100:.1f}%)")

    return {
        "cos_token_mean": float(cos_flat.mean()),
        "cos_token_std": float(cos_flat.std()),
        "cos_per_pos": cos_per_pos,       # [T]
        "cos_per_token": cos_per_token,    # [N, T]
    }


def analyze_edit_vectors(z_init, z_targ, T, D):
    """Edit vector δ = target - init at full [N, T, D] resolution."""
    delta = z_targ - z_init  # [N, T, D]
    N = delta.shape[0]

    # Per-token L2 magnitude: [N, T]
    delta_mag = np.linalg.norm(delta, axis=-1)
    mag_flat = delta_mag.flatten()

    # Spatial heatmap: mean magnitude per position [T]
    mag_per_pos = delta_mag.mean(axis=0)

    # Effective rank of edit subspace (subsample for speed)
    n_sub = min(N, 3000)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, n_sub, replace=False)

    delta_pooled = delta[idx].mean(axis=1)  # [n_sub, D]
    erank_edit_pooled = effective_rank(delta_pooled)

    # Relative edit magnitude
    init_mag = np.linalg.norm(z_init, axis=-1)  # [N, T]
    relative_edit = delta_mag / (init_mag + 1e-12)

    print(f"\n  --- Edit vector analysis (δ = target - init) ---")
    print(f"  ‖δ‖ per token        : {mag_flat.mean():.4f} ± {mag_flat.std():.4f}")
    print(f"  ‖δ‖/‖init‖ per token : {relative_edit.mean():.4f} ± {relative_edit.std():.4f}")
    sorted_mag = np.sort(mag_per_pos)
    p5, p95 = sorted_mag[9], sorted_mag[-10]   # ~5th/95th percentile over 196 positions
    print(f"  Spatial ‖δ‖ range     : min={mag_per_pos.min():.4f}, max={mag_per_pos.max():.4f}, p95/p5={p95/p5:.2f}, max/min={mag_per_pos.max()/mag_per_pos.min():.2f}")
    print(f"  Edit erank (pooled)   : {erank_edit_pooled:.1f} / {D}")

    return {
        "delta_mag_mean": float(mag_flat.mean()),
        "delta_mag_std": float(mag_flat.std()),
        "relative_edit_mean": float(relative_edit.mean()),
        "mag_per_pos": mag_per_pos,    # [T]
        "erank_edit_pooled": erank_edit_pooled,
    }


def analyze_copy_baseline(z_init, z_targ):
    """Copy-input baseline: the floor DiT must beat.
    Computed at flattened [N, T*D] to match DiT eval, plus token-level.
    """
    N = z_init.shape[0]

    # Token-level
    cos_token = cosine_sim_batch(z_init, z_targ, axis=-1)  # [N, T]

    # Flattened (matches DiT generation metric computation)
    cos_flat = cosine_sim_batch(
        z_init.reshape(N, -1),
        z_targ.reshape(N, -1),
        axis=-1,
    )  # [N]

    print(f"\n  --- Copy-input baseline (DiT must exceed these) ---")
    print(f"  Flattened cos_sim     : {cos_flat.mean():.4f} ± {cos_flat.std():.4f}")
    print(f"  Token-level cos_sim   : {cos_token.mean():.4f} ± {cos_token.std():.4f}")

    return {
        "baseline_cos_flat": float(cos_flat.mean()),
        "baseline_cos_flat_std": float(cos_flat.std()),
        "baseline_cos_token": float(cos_token.mean()),
    }


# ── Visualization ────────────────────────────────────────────────────────────

def save_all_figures(results, fig_dir):
    """Save spatial heatmaps and histograms."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    models = list(results.keys())
    n = len(models)

    # 1. Spatial cosine similarity heatmap (14x14) — two rows: shared fixed scale + per-model scale
    cos_global_vmin = min(results[m]["token"]["cos_per_pos"].min() for m in models)
    cos_global_vmax = max(results[m]["token"]["cos_per_pos"].max() for m in models)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1: axes = axes.reshape(2, 1)
    for col, name in enumerate(models):
        cos_per_pos = results[name]["token"]["cos_per_pos"]
        T = len(cos_per_pos)
        h = w = int(np.sqrt(T))
        cos_map = cos_per_pos.reshape(h, w)

        # Row 0: fixed scale [0.5, 1.0]
        im0 = axes[0, col].imshow(cos_map, cmap='RdYlGn_r', vmin=0.5, vmax=1.0)
        axes[0, col].set_title(f"{name}\nmean={cos_map.mean():.4f}", fontsize=9)
        axes[0, col].set_xlabel("patch col"); axes[0, col].set_ylabel("patch row")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

        # Row 1: shared data-driven scale
        im1 = axes[1, col].imshow(cos_map, cmap='RdYlGn_r', vmin=cos_global_vmin, vmax=cos_global_vmax)
        axes[1, col].set_title(f"{name} (shared scale)", fontsize=9)
        axes[1, col].set_xlabel("patch col"); axes[1, col].set_ylabel("patch row")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

    axes[0, 0].set_ylabel("Fixed [0.5,1.0]\npatch row")
    axes[1, 0].set_ylabel("Shared scale\npatch row")
    fig.suptitle("Token-level init↔target cosine similarity (lower = more editing)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "spatial_cos_sim.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Spatial edit magnitude heatmap (14x14) — two rows: per-model scale + shared scale
    global_vmin = min(results[m]["edit"]["mag_per_pos"].min() for m in models)
    global_vmax = max(results[m]["edit"]["mag_per_pos"].max() for m in models)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1: axes = axes.reshape(2, 1)
    for col, name in enumerate(models):
        mag_per_pos = results[name]["edit"]["mag_per_pos"]
        T = len(mag_per_pos)
        h = w = int(np.sqrt(T))
        mag_map = mag_per_pos.reshape(h, w)

        # Row 0: per-model auto-scale
        im0 = axes[0, col].imshow(mag_map, cmap='hot')
        axes[0, col].set_title(f"{name}\nmin={mag_map.min():.2f} max={mag_map.max():.2f}", fontsize=9)
        axes[0, col].set_xlabel("patch col"); axes[0, col].set_ylabel("patch row")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

        # Row 1: shared global scale
        im1 = axes[1, col].imshow(mag_map, cmap='hot', vmin=global_vmin, vmax=global_vmax)
        axes[1, col].set_title(f"{name} (shared scale)", fontsize=9)
        axes[1, col].set_xlabel("patch col"); axes[1, col].set_ylabel("patch row")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

    axes[0, 0].set_ylabel("Per-model scale\npatch row")
    axes[1, 0].set_ylabel("Shared scale\npatch row")
    fig.suptitle("Mean edit magnitude ‖δ‖ per token position (higher = more change)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "spatial_edit_magnitude.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Per-model histogram of token cosine sim
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1: axes = [axes]
    for ax, name in zip(axes, models):
        r = results[name]
        cos_flat = r["token"]["cos_per_token"].flatten()
        ax.hist(cos_flat, bins=100, density=True, alpha=0.7, color='steelblue')
        ax.axvline(cos_flat.mean(), color='red', ls='--', label=f'mean={cos_flat.mean():.3f}')
        ax.set_title(f"{name}", fontsize=9)
        ax.set_xlabel("cosine similarity")
        ax.set_xlim(0.5, 1.0)
        ax.legend(fontsize=8)
    fig.suptitle("Distribution of token-level init↔target cosine similarity", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "token_cos_sim_histogram.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Overlay histogram: all models on one plot
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['steelblue', 'coral', 'forestgreen', 'orchid']
    for i, name in enumerate(models):
        r = results[name]
        cos_flat = r["token"]["cos_per_token"].flatten()
        ax.hist(cos_flat, bins=100, density=True, alpha=0.4, color=colors[i % len(colors)],
                label=f"{name} (μ={cos_flat.mean():.3f})")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("density")
    ax.set_xlim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Token-level init↔target cosine similarity (all models)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "token_cos_sim_overlay.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {fig_dir}")


# ── Main loop ────────────────────────────────────────────────────────────────
results = {}

for name in VARIANT_NAMES:
    is_vae = (name == "sd_vae")
    label = name if is_vae else f"theia_{name}"
    mode_label = "live (VAE)" if is_vae else MODE

    print(f"\n{'='*60}")
    print(f"  Model: {label}  (mode={mode_label})")
    print(f"{'='*60}")

    if is_vae:
        z_init, z_targ = load_sd_vae(sel, args.batch_size, DEVICE)
    elif MODE == "precomputed":
        z_init, z_targ = load_precomputed(name, sel)
    else:
        z_init, z_targ = load_live(name, sel, args.batch_size, DEVICE)

    N, T, D = z_init.shape
    print(f"  Latent dim D = {D},  tokens = {T}")

    # Mean-pooled for distribution stats
    z_init_pool = z_init.mean(axis=1)  # [N, D]
    z_targ_pool = z_targ.mean(axis=1)

    # Run all analyses
    dist = analyze_distribution(z_init_pool, z_targ_pool, D)
    token = analyze_token_separation(z_init, z_targ, T)
    edit = analyze_edit_vectors(z_init, z_targ, T, D)
    baseline = analyze_copy_baseline(z_init, z_targ)

    results[name] = {
        "dim": D, "tokens": T,
        "dist": dist, "token": token, "edit": edit, "baseline": baseline,
    }

    # Free memory between models
    del z_init, z_targ, z_init_pool, z_targ_pool

# ── Summary table ────────────────────────────────────────────────────────────
print(f"\n\n{'='*140}")
print(f"SUMMARY TABLE  (mode={MODE}, n={len(sel)})")
print(f"{'='*140}")
header = (
    f"{'Model':<14} {'Dim':>4} {'Tok':>4} "
    f"{'eRank_i':>8} {'eRank_t':>8} "
    f"{'CosSim':>7} {'RelL2':>7} "
    f"{'TokCos':>7} {'Tok<.95':>7} {'Tok<.90':>7} "
    f"{'‖δ‖':>7} {'‖δ‖/‖z‖':>8} {'δ_erank':>8} "
    f"{'Baseline':>9}"
)
print(header)
print("-" * 145)
for name, r in results.items():
    d, t, e, b = r["dist"], r["token"], r["edit"], r["baseline"]
    cos_pt = t["cos_per_token"]
    frac_95 = (cos_pt < 0.95).mean()
    frac_90 = (cos_pt < 0.90).mean()

    print(
        f"{name:<14} {r['dim']:>4} {r['tokens']:>4} "
        f"{d['erank_init']:>8.1f} {d['erank_targ']:>8.1f} "
        f"{d['cos_sim']:>7.4f} {d['rel_l2']:>7.4f} "
        f"{t['cos_token_mean']:>7.4f} {frac_95:>7.4f} {frac_90:>7.4f} "
        f"{e['delta_mag_mean']:>7.4f} {e['relative_edit_mean']:>8.4f} {e['erank_edit_pooled']:>8.1f} "
        f"{b['baseline_cos_flat']:>9.4f}"
    )
print()

# ── Optional figures ─────────────────────────────────────────────────────────
if args.save_figures:
    save_all_figures(results, args.fig_dir)
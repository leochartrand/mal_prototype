#!/usr/bin/env python3
"""
Text Encoder Discriminability Comparison.

Compares 3 text encoders on how well their pooled embeddings separate
distinct instructions. Uses ALL unique instructions (deduplicated).

Metrics:
  1. Inter-class cosine distance (higher = more separable)
  2. Mean pairwise cosine similarity (lower = less collapsed)
  3. Effective rank / rank % (how many dimensions carry variance)
  4. Isotropy (min_eig / max_eig of covariance)
  5. Top-5 eigenvalue concentration
"""

import numpy as np
import pickle
from sklearn.preprocessing import normalize
import time

DATA_DIR = "/mnt/sda1/Datasets/chal2525/mmap_data"

ENCODERS = {
    "TinyCLIP-29M": "TinyCLIP-ViT-61M-32-Text-29M-LAION400M",
    "CLIP-ViT-L/14": "clip-vit-large-patch14",
    "MiniLM-L6-v2": "all-MiniLM-L6-v2",
    # "SigLIP2-base": "siglip2-base-patch16-224",
    # "BGE-small": "bge-small-en-v1.5",
}

# ============================================================================
# Deduplicate: one embedding per unique instruction
# ============================================================================

print("Loading labels...")
labels = pickle.load(open(f"{DATA_DIR}/labels.pkl", "rb"))
N = len(labels)
n_unique = len(set(labels))
print(f"  {N} total samples, {n_unique} unique instructions\n")

print("Deduplicating (keeping first occurrence per instruction)...")
seen = {}
unique_indices = []
for i, text in enumerate(labels):
    if text not in seen:
        seen[text] = i
        unique_indices.append(i)
unique_indices = np.array(unique_indices)
print(f"  Using {len(unique_indices)} unique embeddings\n")

rng = np.random.RandomState(42)

# ============================================================================
# Analyze each encoder
# ============================================================================

results = {}

for short_name, enc_name in ENCODERS.items():
    print("=" * 60)
    print(f"  {short_name} ({enc_name})")
    print("=" * 60)
    t0 = time.time()

    # Load pooled embeddings for unique instructions only
    pooled_all = np.load(f"{DATA_DIR}/labels_pooled_{enc_name}.npy", mmap_mode='r')
    pooled = np.array(pooled_all[unique_indices], dtype=np.float32)
    dim = pooled.shape[1]
    print(f"  Dim: {dim}, Samples: {len(pooled)}")

    # L2 normalize for cosine metrics
    pooled_norm = normalize(pooled)

    # --- 1. Inter-class cosine distance ---
    n_pairs = 200000
    idx_a = rng.randint(0, len(pooled_norm), n_pairs)
    idx_b = rng.randint(0, len(pooled_norm), n_pairs)
    # All pairs are different-class (deduplicated); exclude self-pairs
    valid = idx_a != idx_b
    cos_sim = (pooled_norm[idx_a[valid]] * pooled_norm[idx_b[valid]]).sum(axis=1)
    inter_dist = 1.0 - cos_sim.mean()
    cos_sim_std = cos_sim.std()

    print(f"  Inter-class cos distance: {inter_dist:.4f} (higher = more separable)")
    print(f"  Cos sim distribution: {cos_sim.mean():.4f} +/- {cos_sim_std:.4f}")

    # --- 2. Mean pairwise cosine sim ---
    mean_cos = float(cos_sim.mean())
    print(f"  Mean pairwise cos sim: {mean_cos:.4f} (lower = less collapsed)")

    # --- 3. Effective rank + isotropy ---
    centered = pooled_norm - pooled_norm.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # clip numerical noise
    ev_norm = eigenvalues / (eigenvalues.sum() + 1e-10)
    eff_rank = float(np.exp(-np.sum(ev_norm * np.log(ev_norm + 1e-10))))
    eff_rank_pct = 100.0 * eff_rank / dim
    isotropy = float(eigenvalues[-1] / (eigenvalues[0] + 1e-10))
    top5_conc = float(eigenvalues[:5].sum() / (eigenvalues.sum() + 1e-10))

    print(f"  Effective rank: {eff_rank:.1f} / {dim} ({eff_rank_pct:.1f}%)")
    print(f"  Isotropy (min/max eig): {isotropy:.6f}")
    print(f"  Top-5 eigenvalue concentration: {top5_conc:.4f}")

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s\n")

    results[short_name] = {
        'dim': dim,
        'inter_dist': float(inter_dist),
        'mean_cos': mean_cos,
        'cos_std': float(cos_sim_std),
        'eff_rank': eff_rank,
        'eff_rank_pct': eff_rank_pct,
        'isotropy': isotropy,
        'top5_conc': top5_conc,
    }

# ============================================================================
# Summary comparison table
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY — All {:,} unique instructions".format(len(unique_indices)))
print("=" * 80)

header = "{:<30}".format("Metric")
for name in ENCODERS:
    header += " {:>15}".format(name)
print(header)
print("-" * 80)

metrics = [
    ('Embedding dim',              'dim',          '{:.0f}',   None),
    ('Inter-class cos dist ↑',     'inter_dist',   '{:.4f}',   True),
    ('Mean pairwise cos sim ↓',    'mean_cos',     '{:.4f}',   False),
    ('Cos sim std',                'cos_std',      '{:.4f}',   None),
    ('Effective rank ↑',           'eff_rank',     '{:.1f}',   True),
    ('Effective rank % ↑',         'eff_rank_pct', '{:.1f}%',  True),
    ('Isotropy ↑',                 'isotropy',     '{:.6f}',   True),
    ('Top-5 eig concentration ↓',  'top5_conc',    '{:.4f}',   False),
]

for label, key, fmt, higher_better in metrics:
    vals = [results[n][key] for n in ENCODERS]
    if higher_better is True:
        best_idx = int(np.argmax(vals))
    elif higher_better is False:
        best_idx = int(np.argmin(vals))
    else:
        best_idx = -1

    row = "{:<30}".format(label)
    for i, name in enumerate(ENCODERS):
        s = fmt.format(vals[i])
        if i == best_idx:
            s = s + " *"
        row += " {:>15}".format(s)
    print(row)

print("\n* = best\n")

"""
Performance vs Efficiency bubble chart.
X-axis: amortized MACs/step (log scale)
Y-axis: avg chain length (CALVIN ABC→D)
Bubble size: total parameter count
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
# Amortized MACs/step in GMACs (from results/profiling/profile_*.json)
# Avg chain length: CALVIN ABC→D evaluations (seq counts vary per system)
# Total params in millions

SYSTEMS = {
    # key       : (macs_gmacs, chain, params_M, color,    n_seq, display_name, mode,                       vram_mb)
    "susie":      ( 731.6, 2.557, 1089.3, "#b16286", 1000, "SuSIE",     "",                                 8776),
    "taksie":     (1040.4, 2.793, 1868.5, "#98971a", 1000, "TaKSIE",    "",                                 8289),
    "ghilglue":   (2733.3, 3.114, 1139.9, "#cc241d", 1000, "GHIL-Glue", "",                                 8776),
    "ours_a":     (  42.6, 0.92,   231.6, "#d65d0e", 1000, "Ours",      "Distill (10ep), adaptive (min=5)", 1516),
}

# ── Colours (Gruvbox) ─────────────────────────────────────────────────
systems       = list(SYSTEMS.keys())
macs          = np.array([SYSTEMS[s][0] for s in systems])
chain         = np.array([SYSTEMS[s][1] for s in systems], dtype=float)
params        = np.array([SYSTEMS[s][2] for s in systems])
colors        = [SYSTEMS[s][3] for s in systems]
n_seqs        = [SYSTEMS[s][4] for s in systems]
display_names = [SYSTEMS[s][5] for s in systems]
modes         = [SYSTEMS[s][6] for s in systems]
vram_mb       = np.array([SYSTEMS[s][7] for s in systems], dtype=float)

# ── Bubble sizing ─────────────────────────────────────────────────────
# area ∝ sqrt(peak VRAM) for perceptually fair scaling
sizes = 200 + 1800 * (np.sqrt(vram_mb) - np.sqrt(vram_mb.min())) / \
        (np.sqrt(vram_mb.max()) - np.sqrt(vram_mb.min()))

# ── Plot ──────────────────────────────────────────────────────────────
def render(ax, xscale: str):
    # Per-key label placement: (ha, va, xytext_offset_in_points)
    # Tuned so labels stay inside the plot (xlim ≈ [20, 4000], ylim [0, 5]).
    LABEL_PLACEMENT = {
        "susie":    ("right", "bottom", (-14,  10)),    # label up-left of bubble
        "taksie":   ("left",  "bottom", ( 14,  10)),    # label up-right of bubble
        "ghilglue": ("right", "bottom", (-14,  10)),    # label up-left (point is near right edge)
        "ours_a":   ("left",  "bottom", ( 16,  14)),    # label up-right of bubble
    }
    for i, s in enumerate(systems):
        if np.isnan(chain[i]):
            continue
        ax.scatter(
            macs[i], chain[i],
            s=sizes[i],
            c=[colors[i]],
            alpha=0.70,
            edgecolors="none",
            zorder=3,
        )
        v = vram_mb[i]
        vlabel = f"{v/1000:.1f} GB" if v >= 1000 else f"{v:.0f} MB"
        label = f"{display_names[i]}\n{modes[i]}\n{vlabel}" if modes[i] else f"{display_names[i]}\n{vlabel}"
        ha, va, off = LABEL_PLACEMENT.get(s, ("left", "bottom", (8, 6)))
        ax.annotate(
            label,
            (macs[i], chain[i]),
            fontsize=9,
            fontweight="bold",
            ha=ha, va=va,
            xytext=off,
            textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Amortized GMACs / step  (log scale)", fontsize=11)
    ax.set_ylabel("Avg Chain Length  (CALVIN ABC→D)", fontsize=11)
    ax.set_title("Performance vs Efficiency", fontsize=13, fontweight="bold")
    ax.set_xlim(20, 5000)   # padding on both sides so annotations don't get clipped
    ax.set_ylim(0, 5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}G" if x >= 1 else f"{x*1000:.0f}M"
    ))
    ax.grid(True, which="both", ls="--", alpha=0.3)


fig, ax = plt.subplots(figsize=(9, 6))
render(ax, xscale="log")
plt.tight_layout()
plt.savefig("results/perf_vs_efficiency.png", dpi=200)
print("Saved results/perf_vs_efficiency.png")
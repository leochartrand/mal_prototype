#!/usr/bin/env python
"""Browse initial→target pairs from the test set as a thumbnail grid.

Usage:
  python scripts/browse_pairs.py --config subgoal.yaml --n 64 --offset 0
  python scripts/browse_pairs.py --config subgoal.yaml --n 64 --offset 5000

Produces a grid image where each cell shows (initial, target) side by side,
with the dataset index printed above. Pick the indices you like and pass them
to the visualization / CFG sweep scripts.
"""

import os, sys, yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.theia_decoder import Decoder as TheiaDecoder
from utils.datasets import MemoryMappedDataset, mmap_collate_fn


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="subgoal.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n", type=int, default=64, help="Number of pairs to show")
    p.add_argument("--offset", type=int, default=0, help="Start index in test set")
    p.add_argument("--cols", type=int, default=8, help="Columns in the grid")
    p.add_argument("--out", default=None, help="Output image path")
    args = p.parse_args()

    params = yaml.safe_load(open(f"./config/{args.config}"))
    device = torch.device(f"cuda:{args.gpu}")

    dataset_path = params["dataset_path"]
    text_model = params["text_model"]
    vision_model = params["vision_model"]
    scale_factor = params.get("scale_factor", 1.0)

    # ---- Theia decoder ----
    td_cfg = params["theia_decoder"]
    theia_decoder = TheiaDecoder(**td_cfg["model_params"])
    td_ckpt = torch.load(td_cfg["model_path"], map_location=device, weights_only=False)
    theia_decoder.load_state_dict(td_ckpt["model"])
    theia_decoder = theia_decoder.to(device).eval()
    for p_ in theia_decoder.parameters():
        p_.requires_grad = False
    del td_ckpt

    ds = MemoryMappedDataset(dataset_path, vision_model=vision_model,
                             text_model=text_model, split="test")
    n = min(args.n, len(ds) - args.offset)
    indices = list(range(args.offset, args.offset + n))
    print(f"Test set: {len(ds)} samples, showing indices {args.offset}..{args.offset + n - 1}")

    batch = mmap_collate_fn([ds[i] for i in indices])
    x_init   = batch[0].clamp(0, 1)   # [n, 3, 224, 224] — blank when initial_224.npy is missing
    x_target = batch[2].clamp(0, 1)
    labels_list = list(batch[4])

    # Fall back to decoding from latents when raw 224 images aren't stored in the dataset
    if float(x_init.abs().sum()) == 0.0:
        z_init   = batch[1].to(device) * scale_factor
        z_target = batch[3].to(device) * scale_factor
        with torch.no_grad():
            x_init   = theia_decoder(z_init   / scale_factor).cpu().clamp(0, 1)
            x_target = theia_decoder(z_target / scale_factor).cpu().clamp(0, 1)

    # ---- Plot grid ----
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.0),
                             squeeze=False)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i >= n:
            continue

        img_init   = x_init[i].permute(1, 2, 0).numpy()
        img_target = x_target[i].permute(1, 2, 0).numpy()
        combined   = np.concatenate([img_init, img_target], axis=1)
        ax.imshow(combined)
        label = str(labels_list[i])[:40]
        ax.set_title(f"[{indices[i]}] {label}", fontsize=6, pad=2)

    plt.suptitle(f"init → target  |  offset={args.offset}, n={n}  |  [{args.offset}..{args.offset+n-1}]",
                 fontsize=10, y=1.0)
    plt.tight_layout()

    out_path = args.out or f"results/browse_pairs_offset{args.offset}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

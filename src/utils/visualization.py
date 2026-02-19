"""
Visualization utilities for training scripts.
"""

import os
import numpy as np
import textwrap
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.no_grad()
def visualize_flowdit_samples(
    model,
    x0_img_batch: torch.Tensor,
    x0_embed_batch: torch.Tensor,
    xt_img_batch: torch.Tensor,
    xt_embed_batch: torch.Tensor,
    text_txt_batch: list,
    text_hidden_batch: torch.Tensor,
    text_mask_batch: torch.Tensor,
    epoch: int,
    save_dir: str,
    device: torch.device,
    scale_factor: float = 1.0,
    decode_fn=None,
    num_vis: int = 4,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    context_cfg_scale=None,
    prompt_cfg_scale=None,
):
    """
    Generate and visualize FlowDiT samples.

    Works with or without a Theia decoder:
      - With decoder: shows initial / target / generated images
      - Without: shows initial / target images with cosine-sim overlay
    """
    model.eval()

    n_vis = min(num_vis, x0_img_batch.shape[0])

    # Images for display — CHW float32 [0,1] tensors
    x0_vis = x0_img_batch[:n_vis]
    xt_vis = xt_img_batch[:n_vis]

    def _to_hwc(img):
        """Convert CHW float tensor to HWC numpy for imshow."""
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW
            img = np.transpose(img, (1, 2, 0))
        return np.clip(img, 0, 1)

    # Pre-computed embeddings
    z_init = x0_embed_batch[:n_vis].to(device) * scale_factor
    z_target_gt = xt_embed_batch[:n_vis].to(device) * scale_factor
    text_hidden_vis = text_hidden_batch[:n_vis].to(device)
    text_mask_vis = text_mask_batch[:n_vis].to(device)
    text_txt_vis = text_txt_batch[:n_vis]

    # Generate
    z_generated = model.sample_euler(
        z_init, text_hidden_vis, text_mask=text_mask_vis,
        num_steps=num_steps, cfg_scale=cfg_scale,
        context_cfg_scale=context_cfg_scale,
        prompt_cfg_scale=prompt_cfg_scale,
    )

    # Create figure
    if decode_fn is not None:
        xg_recon = decode_fn(z_generated / scale_factor)

        fig, axes = plt.subplots(3, n_vis, figsize=(n_vis * 2, 6), squeeze=False)
        row_labels = ['Initial', 'Target', 'Generated']

        for i in range(n_vis):
            wrapped_text = '\n'.join(textwrap.wrap(text_txt_vis[i], width=25))
            axes[0, i].text(0.5, 1.15, wrapped_text, transform=axes[0, i].transAxes,
                            fontsize=8, fontweight='bold', ha='center', va='bottom')

            axes[0, i].imshow(_to_hwc(x0_vis[i])); axes[0, i].axis('off')
            axes[1, i].imshow(_to_hwc(xt_vis[i])); axes[1, i].axis('off')

            img = xg_recon[i].cpu().permute(1, 2, 0).numpy()
            axes[2, i].imshow(np.clip(img, 0, 1)); axes[2, i].axis('off')

        # Add row labels on left side (using text instead of ylabel to avoid axis('off') hiding them)
        for row_idx, label in enumerate(row_labels):
            axes[row_idx, 0].text(-0.1, 0.5, label, transform=axes[row_idx, 0].transAxes,
                                   fontsize=9, fontweight='bold', ha='right', va='center', rotation=90)
    else:
        cos_sims = F.cosine_similarity(
            z_generated.flatten(1), z_target_gt.flatten(1), dim=1
        )

        fig, axes = plt.subplots(2, n_vis, figsize=(n_vis * 2, 4), squeeze=False)

        for i in range(n_vis):
            wrapped_text = '\n'.join(textwrap.wrap(text_txt_vis[i], width=25))
            axes[0, i].text(0.5, 1.15, wrapped_text, transform=axes[0, i].transAxes,
                            fontsize=8, fontweight='bold', ha='center', va='bottom')

            axes[0, i].imshow(_to_hwc(x0_vis[i])); axes[0, i].set_title('Initial', fontsize=9); axes[0, i].axis('off')
            axes[1, i].imshow(_to_hwc(xt_vis[i])); axes[1, i].set_title(f'Target\ncos={cos_sims[i]:.3f}', fontsize=9); axes[1, i].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    model.train()


@torch.no_grad()
def visualize_decoder_samples(
    model,
    z_batch: torch.Tensor,
    target_batch: torch.Tensor,
    epoch: int,
    save_dir: str,
    num_vis: int = 4,
):
    """Save reconstruction grid for a decoder model (original vs reconstructed)."""
    model.eval()
    n = min(num_vis, z_batch.shape[0])
    recons = model(z_batch[:n])
    targets = target_batch[:n]

    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    for i in range(n):
        orig = targets[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(np.clip(orig, 0, 1))
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')

        recon = recons[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(np.clip(recon, 0, 1))
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1:03d}.png'), dpi=100, bbox_inches='tight')
    plt.close()
    model.train()


def plot_loss_curves(
    train_losses: list,
    test_losses: list,
    save_path: str,
    best_loss_epoch: int = -1,
    title: str = 'Training and Test Losses',
):
    """Plot and save train/test loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='orange')
    plt.plot(test_losses, label='Test Loss', color='green')
    if best_loss_epoch >= 0 and best_loss_epoch < len(test_losses):
        plt.scatter([best_loss_epoch], [test_losses[best_loss_epoch]], color='green', zorder=5)
        title = f'{title} (Best: {test_losses[best_loss_epoch]:.4f})'
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

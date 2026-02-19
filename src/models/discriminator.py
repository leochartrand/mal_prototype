"""
Unconditional discriminator for Theia latent embeddings.

Judges whether a [B, 196, D] embedding (reshaped to [B, D, 14, 14])
looks like a real Theia encoding. No text or init-state conditioning —
realism only, task-specificity is the flow model's job.

Architecture: spectrally-normalized conv stack + linear head.
Default 3 layers: 14×14 → 7×7 → 4×4 → 2×2, then flatten → linear → 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class EmbeddingDiscriminator(nn.Module):
    def __init__(self, latent_dim: int = 384, channels: list = [384, 512, 512]):
        super().__init__()
        self.latent_dim = latent_dim

        conv_layers = []
        in_ch = latent_dim

        for out_ch in channels:
            conv_layers.append(nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            ))
            conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)

        # Compute spatial size after all stride-2 layers starting from 14×14
        spatial = 14
        for _ in channels:
            spatial = (spatial + 2 * 1 - 4) // 2 + 1  # k=4, s=2, p=1
        self.flat_dim = channels[-1] * spatial * spatial

        self.head = nn.utils.spectral_norm(
            nn.Linear(self.flat_dim, 1)
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"EmbeddingDiscriminator initialized with {n_params/1e6:.2f}M parameters"
              f" ({len(channels)} conv layers, head {self.flat_dim}→1)")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Theia embeddings [B, 196, D] (scaled, matching FlowDiT output space)
        Returns:
            Logits [B, 1] (raw, no sigmoid — use with BCEWithLogitsLoss or hinge)
        """
        B, N, D = z.shape
        z = z.permute(0, 2, 1).reshape(B, D, 14, 14)
        feat = self.conv(z)
        feat = feat.flatten(1)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def discriminator_loss_bce(D_real: torch.Tensor, D_fake: torch.Tensor) -> torch.Tensor:
    """BCE discriminator loss. Inputs are raw logits [B, 1]."""
    real_loss = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real))
    fake_loss = F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))
    return (real_loss + fake_loss) * 0.5


def generator_loss_bce(D_fake: torch.Tensor) -> torch.Tensor:
    """BCE generator loss. Input is raw logits [B, 1]."""
    return F.binary_cross_entropy_with_logits(D_fake, torch.ones_like(D_fake))


def discriminator_loss_hinge(D_real: torch.Tensor, D_fake: torch.Tensor) -> torch.Tensor:
    """Hinge discriminator loss."""
    return (F.relu(1.0 - D_real).mean() + F.relu(1.0 + D_fake).mean()) * 0.5


def generator_loss_hinge(D_fake: torch.Tensor) -> torch.Tensor:
    """Hinge generator loss."""
    return -D_fake.mean()


# ---------------------------------------------------------------------------
# R1 gradient penalty (lazy, StyleGAN2-style)
# ---------------------------------------------------------------------------

def r1_gradient_penalty(
    discriminator: nn.Module, real: torch.Tensor, gamma: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    R1 gradient penalty on real data.

    Returns (penalty, D_real) so the caller can reuse D_real for the
    main discriminator loss without a redundant forward pass.

    Args:
        discriminator: The discriminator network.
        real: Real latent embeddings [B, 196, D] with requires_grad=True.
        gamma: Penalty coefficient (applied as gamma/2 * ||grad||^2).
    Returns:
        Tuple of (scaled penalty, D_real logits).
    """
    D_real = discriminator(real)
    gradients = grad(
        outputs=D_real.sum(),
        inputs=real,
        create_graph=True,
    )[0]
    penalty = gradients.reshape(gradients.size(0), -1).pow(2).sum(dim=1).mean()
    return (gamma / 2.0) * penalty, D_real

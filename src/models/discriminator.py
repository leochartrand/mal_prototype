"""
Unconditional discriminator for Theia latent embeddings.

Judges whether a [B, 196, D] embedding (reshaped to [B, D, 14, 14])
looks like a real Theia encoding. No text or init-state conditioning —
realism only, task-specificity is the flow model's job.

Architecture: 3-layer conv with spectral normalization.
14×14 → 7×7 → 3×3 → 1×1 (global receptive field at final layer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingDiscriminator(nn.Module):
    def __init__(self, latent_dim: int = 384, channels: list = [256, 512]):
        super().__init__()
        self.latent_dim = latent_dim

        layers = []
        in_ch = latent_dim

        # Layer 1: [B, D, 14, 14] → [B, channels[0], 7, 7]
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(in_ch, channels[0], kernel_size=4, stride=2, padding=1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 2: [B, channels[0], 7, 7] → [B, channels[1], 3, 3]
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 3: [B, channels[1], 3, 3] → [B, 1, 1, 1]
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(channels[1], 1, kernel_size=3, stride=1, padding=0)
        ))

        self.net = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"EmbeddingDiscriminator initialized with {n_params/1e6:.2f}M parameters")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Theia embeddings [B, 196, D] (scaled, matching FlowDiT output space)
        Returns:
            Logits [B, 1] (raw, no sigmoid — use with BCEWithLogitsLoss or hinge)
        """
        B, N, D = z.shape
        # Reshape to spatial: [B, D, 14, 14]
        z = z.permute(0, 2, 1).reshape(B, D, 14, 14)
        out = self.net(z)  # [B, 1, 1, 1]
        return out.view(B, 1)


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

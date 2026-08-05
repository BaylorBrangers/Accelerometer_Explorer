"""Simple autoencoder for reconstruction-based anomaly scores."""

from __future__ import annotations

import torch
import torch.nn as nn


class AccelAutoencoder(nn.Module):
    """Compress (B, C, L) windows and reconstruct for anomaly scoring."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        length: int = 64,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.length = length
        flat = in_channels * length
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, flat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon.view(-1, self.in_channels, self.length)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return torch.mean((recon - x) ** 2, dim=(1, 2))

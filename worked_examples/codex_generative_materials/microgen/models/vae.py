"""Convolutional VAE for 64x64 grayscale microstructures."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            # Upsample + conv avoids checkerboard artifacts from transposed conv.
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4 -> 8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8 -> 16
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16 -> 32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32 -> 64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(-1, 128, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-3,
    edge_weight: float = 0.25,
    recon_mode: str = "l1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if recon_mode == "l1":
        recon_loss = F.l1_loss(recon_x, x, reduction="mean")
    elif recon_mode == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    elif recon_mode == "smooth_l1":
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction="mean")
    else:
        raise ValueError(f"Unsupported recon_mode: {recon_mode}")

    sobel_x = recon_x.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(1)
    sobel_y = recon_x.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(1)
    recon_gx = F.conv2d(recon_x, sobel_x, padding=1)
    recon_gy = F.conv2d(recon_x, sobel_y, padding=1)
    target_gx = F.conv2d(x, sobel_x, padding=1)
    target_gy = F.conv2d(x, sobel_y, padding=1)
    edge_loss = F.l1_loss(recon_gx, target_gx, reduction="mean") + F.l1_loss(recon_gy, target_gy, reduction="mean")

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + edge_weight * edge_loss + kl_weight * kl
    return total, recon_loss, kl, edge_loss

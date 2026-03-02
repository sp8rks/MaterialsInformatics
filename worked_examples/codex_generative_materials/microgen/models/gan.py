"""Minimal DCGAN-style generator and discriminator for 64x64 grayscale."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class DCGenerator(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.fc = nn.Linear(latent_dim, c * 8 * 4 * 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4 -> 8
            nn.Conv2d(c * 8, c * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8 -> 16
            nn.Conv2d(c * 4, c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16 -> 32
            nn.Conv2d(c * 2, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32 -> 64
            nn.Conv2d(c, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 4:
            z = z.squeeze(-1).squeeze(-1)
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.net(h)


class DCDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1)),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)),  # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)),  # 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1)),  # 8 -> 4
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=0)),  # 4 -> 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)

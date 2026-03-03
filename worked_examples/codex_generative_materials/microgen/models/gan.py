"""Minimal DCGAN-style generator and discriminator for 64x64 grayscale."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


def init_gan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DCGenerator(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.fc = nn.Linear(latent_dim, c * 8 * 4 * 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 4 -> 8
            nn.Conv2d(c * 8, c * 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8 -> 16
            nn.Conv2d(c * 4, c * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16 -> 32
            nn.Conv2d(c * 2, c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32 -> 64
            nn.Conv2d(c, c // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.apply(init_gan_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 4:
            z = z.squeeze(-1).squeeze(-1)
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.net(h)


class DCDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1)),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)),  # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)),  # 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1)),  # 8 -> 4
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = spectral_norm(nn.Conv2d(c * 8 + 1, 1, kernel_size=4, stride=1, padding=0))  # 4 -> 1
        self.apply(init_gan_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        # Minibatch stddev improves mode coverage on small datasets.
        std = torch.sqrt(h.var(dim=0, unbiased=False) + 1e-8).mean().view(1, 1, 1, 1)
        std_map = std.expand(h.size(0), 1, h.size(2), h.size(3))
        h = torch.cat([h, std_map], dim=1)
        return self.out(h).view(-1)

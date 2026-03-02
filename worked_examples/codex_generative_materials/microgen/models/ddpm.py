"""Minimal DDPM components for 64x64 grayscale images."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.out_proj = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.out_proj(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, base_channels: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        c = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(1, c, kernel_size=3, padding=1)
        self.down1 = ResBlock(c, c, time_dim)
        self.downsample = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.mid1 = ResBlock(c * 2, c * 2, time_dim)
        self.mid2 = ResBlock(c * 2, c * 2, time_dim)
        self.upsample = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock(c * 2, c, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, 1, kernel_size=3, padding=1),
        )
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.downsample(d1)
        m = self.mid1(d2, t_emb)
        m = self.mid2(m, t_emb)
        u = self.upsample(m)
        u = torch.cat([u, d1], dim=1)
        u = self.up1(u, t_emb)
        return self.out(u)


class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar_t = schedule.alpha_bars[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, noise


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn(shape, device=device)
    for step in range(schedule.timesteps - 1, -1, -1):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        pred_noise = model(x, t)
        alpha_t = schedule.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = schedule.alpha_bars[t].view(-1, 1, 1, 1)
        beta_t = schedule.betas[t].view(-1, 1, 1, 1)

        mean = (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
        if step > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z
        else:
            x = mean
    return x.clamp(-1.0, 1.0)

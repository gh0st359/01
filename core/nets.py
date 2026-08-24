"""Reusable PyTorch layers for the organism."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, sizes: list[int], act: type[nn.Module] = nn.SiLU) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for a, b in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(a, b))
            if b != sizes[-1]:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int, out_dim: int) -> None:
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(c * 4, c * 4, 4, 2, 1),
            nn.SiLU(),
        )
        self.proj = nn.Linear(c * 4, out_dim)
        self.feat_dim = c * 4

    def forward(self, x: Tensor) -> Tensor:
        h = self.net(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.proj(pooled)

    def spatial(self, x: Tensor) -> Tensor:
        """[B, C, H, W] → [B, N, D] unlabeled feature tokens."""
        h = self.net(x)
        b, c, hh, ww = h.shape
        return h.permute(0, 2, 3, 1).reshape(b, hh * ww, c)


class ConvDecoder(nn.Module):
    def __init__(self, in_dim: int, width: int, out_ch: int, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.fc = nn.Linear(in_dim, width * 4 * 4 * 4)
        self.width = width
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(width, width, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(width, out_ch, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(z.size(0), self.width * 4, 4, 4)
        img = self.deconv(h)
        if img.shape[-1] != self.resolution:
            img = torch.nn.functional.interpolate(img, size=self.resolution, mode="bilinear", align_corners=False)
        return img


def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * torch.clamp(logvar, -10.0, 10.0))
    return mean + std * torch.randn_like(std)


def kl_normal(mean: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mean.pow(2) - logvar.exp())

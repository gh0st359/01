"""Learned confidence estimators. Calibration is empirical, not scripted labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ConfidenceBundle:
    perceptual: Tensor
    memory: Tensor
    prediction: Tensor
    skill: Tensor
    belief: Tensor


class ConfidenceEstimator(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 6, dim), nn.SiLU(), nn.Linear(dim, 5))

    def forward(self, core: Tensor, extras: Tensor) -> ConfidenceBundle:
        raw = torch.sigmoid(self.net(torch.cat([core, extras], dim=-1)))
        p, m, pr, s, b = raw.unbind(dim=-1)
        return ConfidenceBundle(perceptual=p, memory=m, prediction=pr, skill=s, belief=b)

    def loss(self, bundle: ConfidenceBundle, targets: Tensor) -> Tensor:
        pred = torch.stack([bundle.perceptual, bundle.memory, bundle.prediction, bundle.skill, bundle.belief], dim=-1)
        return torch.mean((pred - targets) ** 2)

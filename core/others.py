"""Predictive models of other agents from observed motion/speech, not privileged goals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class OtherPrediction:
    next_obs: Tensor
    hidden: Tensor
    confidence: Tensor


class OtherAgentPredictor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(obs_dim, hidden)
        self.pred = nn.Linear(hidden, obs_dim)
        self.conf = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))
        self.perspective = nn.Linear(hidden, hidden)
        self.h0 = nn.Parameter(torch.zeros(hidden))

    def initial(self, batch: int, device: torch.device) -> Tensor:
        return self.h0.expand(batch, -1).contiguous()

    def step(self, h: Tensor, other_obs: Tensor, next_obs: Tensor | None = None) -> tuple[Tensor, OtherPrediction]:
        h2 = self.cell(other_obs, h)
        pred = self.pred(h2)
        conf = torch.sigmoid(self.conf(h2))
        if next_obs is not None:
            err = torch.mean((pred - next_obs) ** 2, dim=-1, keepdim=True)
            conf = torch.exp(-err)
        return h2, OtherPrediction(next_obs=pred, hidden=self.perspective(h2), confidence=conf)

"""Learned agency and self-model: action-contingent prediction of proprioception."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SelfEstimate:
    next_proprio: Tensor
    agency: Tensor
    body_bound: Tensor
    capability: Tensor


class PredictiveSelf(nn.Module):
    def __init__(self, proprio_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.dyn = nn.GRUCell(proprio_dim + action_dim, hidden)
        self.pred = nn.Linear(hidden, proprio_dim)
        self.agency_head = nn.Sequential(nn.Linear(proprio_dim * 2 + action_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.bound = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, proprio_dim))
        self.cap = nn.Sequential(nn.Linear(hidden + action_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.h0 = nn.Parameter(torch.zeros(hidden))

    def initial(self, batch: int, device: torch.device) -> Tensor:
        return self.h0.expand(batch, -1).contiguous()

    def step(self, h: Tensor, proprio: Tensor, action: Tensor, next_proprio: Tensor | None = None) -> tuple[Tensor, SelfEstimate]:
        h2 = self.dyn(torch.cat([proprio, action], dim=-1), h)
        pred = self.pred(h2)
        err = pred - (next_proprio if next_proprio is not None else proprio)
        agency = torch.sigmoid(self.agency_head(torch.cat([pred, err.abs(), action], dim=-1)))
        bound = torch.sigmoid(self.bound(h2))
        cap = torch.sigmoid(self.cap(torch.cat([h2, action], dim=-1)))
        return h2, SelfEstimate(next_proprio=pred, agency=agency, body_bound=bound, capability=cap)

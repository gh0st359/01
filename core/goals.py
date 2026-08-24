"""Learned goal proposal in latent space. No threshold → ActionKind mapping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GoalBundle:
    latent: Tensor
    value: Tensor
    cost: Tensor
    uncertainty: Tensor
    info_value: Tensor
    horizon: Tensor
    weights: Tensor


class GoalProposer(nn.Module):
    def __init__(self, core_dim: int, n_goals: int, goal_dim: int) -> None:
        super().__init__()
        self.n_goals = n_goals
        self.goal_dim = goal_dim
        self.propose = nn.Sequential(nn.Linear(core_dim + 10, core_dim), nn.SiLU(), nn.Linear(core_dim, n_goals * goal_dim))
        self.heads = nn.Sequential(nn.Linear(goal_dim + core_dim + 10, core_dim), nn.SiLU(), nn.Linear(core_dim, 5))
        self.select = nn.Linear(5 + goal_dim, 1)

    def forward(self, core: Tensor, homeo: Tensor) -> GoalBundle:
        b = core.size(0)
        raw = self.propose(torch.cat([core, homeo], dim=-1)).view(b, self.n_goals, self.goal_dim)
        ctx = torch.cat([core, homeo], dim=-1).unsqueeze(1).expand(-1, self.n_goals, -1)
        stats = self.heads(torch.cat([raw, ctx], dim=-1))
        value, cost, unc, info, horizon = stats.unbind(dim=-1)
        scores = self.select(torch.cat([stats, raw], dim=-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        latent = torch.einsum("bk,bkd->bd", weights, raw)
        return GoalBundle(
            latent=latent,
            value=value,
            cost=cost,
            uncertainty=unc,
            info_value=info,
            horizon=torch.sigmoid(horizon) * 64.0,
            weights=weights,
        )

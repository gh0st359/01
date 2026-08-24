"""Latent-space CEM planning over RSSM rollouts. Plans are tensors, not English strings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from core.curiosity import EpistemicEvaluator
from core.rssm import RSSM, RSSMState


@dataclass
class Plan:
    actions: Tensor  # [H, B, A]
    values: Tensor
    info: Tensor


class CEMPlanner(nn.Module):
    def __init__(self, action_dim: int, horizon: int, pop: int, elites: int, iters: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.pop = pop
        self.elites = elites
        self.iters = iters
        self.value_head = nn.Sequential(nn.Linear(action_dim + 8, 64), nn.SiLU(), nn.Linear(64, 1))

    @torch.no_grad()
    def plan(
        self,
        rssm: RSSM,
        state: RSSMState,
        goal: Tensor,
        homeo: Tensor,
        epistemic: EpistemicEvaluator,
    ) -> Plan:
        b = state.h.size(0)
        device = state.h.device
        mean = torch.zeros(self.horizon, b, self.action_dim, device=device)
        std = torch.ones(self.horizon, b, self.action_dim, device=device) * 0.4
        best_actions = mean
        best_val = torch.full((b,), -1e9, device=device)
        best_info = torch.zeros(b, 1, device=device)
        for _ in range(self.iters):
            samples = mean.unsqueeze(2) + std.unsqueeze(2) * torch.randn(
                self.horizon, b, self.pop, self.action_dim, device=device
            )
            samples = samples.clamp(-1.0, 1.0)
            values = []
            infos = []
            for i in range(self.pop):
                acts = samples[:, :, i, :]
                last, _ = rssm.rollout(state, acts)
                lat = last.flatten()
                ig = epistemic.information_gain(lat, acts[-1])
                val = -torch.mean((lat[:, : goal.size(-1)] - goal) ** 2, dim=-1, keepdim=True)
                val = val - 0.1 * torch.mean(homeo**2, dim=-1, keepdim=True) + 0.4 * ig
                values.append(val)
                infos.append(ig)
            stacked = torch.cat(values, dim=-1)  # [B, pop]
            info_st = torch.cat(infos, dim=-1)
            k = min(self.elites, self.pop)
            topv, topi = torch.topk(stacked, k, dim=-1)
            elite = torch.gather(
                samples.permute(1, 2, 0, 3),
                1,
                topi.unsqueeze(-1).unsqueeze(-1).expand(-1, k, self.horizon, self.action_dim),
            )
            mean = elite.mean(dim=1).permute(1, 0, 2)
            std = elite.std(dim=1).permute(1, 0, 2).clamp(min=0.05)
            cur_best, idx = stacked.max(dim=-1)
            improved = cur_best > best_val
            best_val = torch.where(improved, cur_best, best_val)
            pick = samples[:, torch.arange(b, device=device), idx]
            best_actions = torch.where(improved.view(1, b, 1), pick, best_actions)
            best_info = torch.where(improved.view(b, 1), info_st.gather(1, idx.unsqueeze(-1)), best_info)
        return Plan(actions=best_actions, values=best_val, info=best_info)

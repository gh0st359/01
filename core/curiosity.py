"""Learned epistemic value: expected information gain between competing hypotheses."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from core.rssm import RSSM, RSSMState


class EpistemicEvaluator(nn.Module):
    """Two hypothesis heads. Actions that split their predictions have high information value."""

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.hyp_a = nn.Sequential(nn.Linear(latent_dim + action_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.hyp_b = nn.Sequential(nn.Linear(latent_dim + action_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.ctrl = nn.Sequential(nn.Linear(latent_dim + action_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def information_gain(self, latent: Tensor, action: Tensor) -> Tensor:
        xa = torch.cat([latent, action], dim=-1)
        a = self.hyp_a(xa)
        b = self.hyp_b(xa)
        # Expected disagreement = information value of the action under competing models
        return torch.mean((a - b) ** 2, dim=-1, keepdim=True)

    def controllability(self, latent: Tensor, action: Tensor) -> Tensor:
        return torch.sigmoid(self.ctrl(torch.cat([latent, action], dim=-1)))

    def score_actions(self, latent: Tensor, actions: Tensor) -> Tensor:
        """actions: [N, B, A] → scores [N, B, 1]."""
        n, b, _ = actions.shape
        lat = latent.unsqueeze(0).expand(n, -1, -1)
        ig = self.information_gain(lat.reshape(n * b, -1), actions.reshape(n * b, -1)).view(n, b, 1)
        ctrl = self.controllability(lat.reshape(n * b, -1), actions.reshape(n * b, -1)).view(n, b, 1)
        return ig + 0.15 * ctrl


def rssm_ensemble_disagreement(rssm: RSSM, state: RSSMState, actions: Tensor, n_samples: int = 4) -> Tensor:
    """Monte-Carlo prior samples as competing futures. actions: [T, B, A]."""
    preds = []
    for _ in range(n_samples):
        _, lat = rssm.rollout(state, actions)
        preds.append(lat[-1])
    stacked = torch.stack(preds, dim=0)
    return stacked.var(dim=0).mean(dim=-1, keepdim=True)

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

    def outcome_loss(self, latent: Tensor, action: Tensor, next_z: Tensor, intervened: bool) -> Tensor:
        """hyp_a sees all transitions; hyp_b is observational-only. Probe actions split them."""
        xa = torch.cat([latent, action], dim=-1)
        loss = torch.mean((self.hyp_a(xa) - next_z.detach()) ** 2)
        if not intervened:
            loss = loss + torch.mean((self.hyp_b(xa) - next_z.detach()) ** 2)
        return loss

    def candidate_actions(self, cem: Tensor) -> Tensor:
        """Discrete experiments: wait, move, interact, grasp, CEM. [5, B, A]."""
        wait = torch.zeros_like(cem)
        move = cem.clone()
        move[:, 3] = 0.0
        interact = cem.clone()
        interact[:, 3] = 1.0
        grasp = cem.clone()
        grasp[:, 2] = 1.0
        grasp[:, 3] = 0.0
        return torch.stack([wait, move, interact, grasp, cem], dim=0)

    def pick_experiment(self, latent: Tensor, cem: Tensor, explore: float = 0.12) -> tuple[Tensor, bool, Tensor]:
        """Choose the action whose predicted outcome splits hyp_a vs hyp_b."""
        cands = self.candidate_actions(cem)
        n, b, _a = cands.shape
        lat = latent.unsqueeze(0).expand(n, *latent.shape).reshape(n * b, -1)
        ig = self.information_gain(lat, cands.reshape(n * b, -1)).view(n, b)
        scores = ig.mean(dim=-1)
        if float(torch.rand((), device=latent.device)) < explore:
            idx = int(torch.randint(0, n, (1,), device=latent.device))
        else:
            idx = int(scores.argmax())
        return cands[idx], idx == 2, scores

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

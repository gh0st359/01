"""Stochastic recurrent world model (RSSM-like, non-transformer)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from core.nets import MLP, kl_normal, reparameterize


@dataclass
class RSSMState:
    h: Tensor
    z: Tensor

    def flatten(self) -> Tensor:
        return torch.cat([self.h, self.z], dim=-1)


class RSSM(nn.Module):
    def __init__(
        self,
        deter: int,
        stoch: int,
        action_dim: int,
        obs_dim: int,
    ) -> None:
        super().__init__()
        self.deter = deter
        self.stoch = stoch
        self.obs_dim = obs_dim
        self.cell = nn.GRUCell(stoch + action_dim, deter)
        self.prior = MLP([deter, deter, stoch * 2])
        self.post = MLP([deter + obs_dim, deter, stoch * 2])
        self.obs_pred = MLP([deter + stoch, deter, obs_dim])
        self.reward_pred = MLP([deter + stoch, deter, 3])  # energy, novelty, competence proxies
        self.term_pred = MLP([deter + stoch, 32, 1])

    def initial(self, batch: int, device: torch.device) -> RSSMState:
        return RSSMState(
            h=torch.zeros(batch, self.deter, device=device),
            z=torch.zeros(batch, self.stoch, device=device),
        )

    def _split(self, stats: Tensor) -> tuple[Tensor, Tensor]:
        mean, logvar = stats.chunk(2, dim=-1)
        return mean, logvar

    def observe(self, state: RSSMState, obs: Tensor, action: Tensor) -> tuple[RSSMState, dict[str, Tensor]]:
        h = self.cell(torch.cat([state.z, action], dim=-1), state.h)
        prior_m, prior_lv = self._split(self.prior(h))
        post_m, post_lv = self._split(self.post(torch.cat([h, obs], dim=-1)))
        z = reparameterize(post_m, post_lv)
        nxt = RSSMState(h=h, z=z)
        pred = self.obs_pred(nxt.flatten())
        aux = self.reward_pred(nxt.flatten())
        term = torch.sigmoid(self.term_pred(nxt.flatten()))
        losses = {
            "recon": torch.mean((pred - obs) ** 2),
            "kl": kl_normal(post_m - prior_m, post_lv),
            "term": term.mean() * 0.0,
        }
        info = {"pred_obs": pred, "aux": aux, "term": term, "prior_m": prior_m, "post_m": post_m, "post_lv": post_lv}
        return nxt, {**losses, **info}

    def imagine(self, state: RSSMState, action: Tensor) -> RSSMState:
        h = self.cell(torch.cat([state.z, action], dim=-1), state.h)
        mean, logvar = self._split(self.prior(h))
        z = reparameterize(mean, logvar)
        return RSSMState(h=h, z=z)

    def rollout(self, state: RSSMState, actions: Tensor) -> tuple[RSSMState, Tensor]:
        """actions: [T, B, A]. Returns last state and stacked latents [T, B, H+Z]."""
        latents = []
        s = state
        for t in range(actions.size(0)):
            s = self.imagine(s, actions[t])
            latents.append(s.flatten())
        return s, torch.stack(latents, dim=0)

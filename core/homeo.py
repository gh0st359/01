"""Homeostatic variables bias cognition. They do not select canned actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class HomeoState:
    energy: Tensor
    saturation: Tensor
    novelty_dep: Tensor
    pred_instability: Tensor
    social: Tensor
    sleep: Tensor
    competence: Tensor
    control: Tensor
    surprise: Tensor
    valence: Tensor

    def vector(self) -> Tensor:
        return torch.stack(
            [
                self.energy,
                self.saturation,
                self.novelty_dep,
                self.pred_instability,
                self.social,
                self.sleep,
                self.competence,
                self.control,
                self.surprise,
                self.valence,
            ],
            dim=-1,
        )


def step_homeo(
    prev: HomeoState,
    energy_obs: Tensor,
    pe: Tensor,
    novelty: Tensor,
    social: Tensor,
    agency: Tensor,
    competence: Tensor,
) -> HomeoState:
    leak = 0.02
    energy = (1 - leak) * prev.energy + leak * energy_obs
    sat = torch.clamp(prev.saturation + 0.01 - 0.04 * pe, 0.0, 1.0)
    nov = torch.clamp(prev.novelty_dep * 0.98 + 0.08 * (0.2 - novelty).clamp(min=0), 0.0, 1.0)
    inst = 0.9 * prev.pred_instability + 0.1 * pe
    soc = 0.85 * prev.social + 0.15 * social
    sleep = torch.clamp(prev.sleep + 0.004 - 0.08 * (prev.sleep > 0.8).float(), 0.0, 1.0)
    comp = 0.95 * prev.competence + 0.05 * competence
    ctrl = 0.9 * prev.control + 0.1 * agency
    surprise = pe
    valence = torch.tanh(comp + ctrl - inst - (1.0 - energy) * 0.5)
    return HomeoState(energy, sat, nov, inst, soc, sleep, comp, ctrl, surprise, valence)


def init_homeo(batch: int, device: torch.device) -> HomeoState:
    ones = torch.ones(batch, device=device)
    zeros = torch.zeros(batch, device=device)
    return HomeoState(ones * 0.8, zeros, zeros, zeros + 0.3, zeros, zeros, zeros + 0.3, zeros + 0.3, zeros + 0.3, zeros)

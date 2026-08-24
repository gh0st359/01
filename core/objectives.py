"""Per-organ learning signals. Each organ has a target that can fail."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class OrganLosses:
    world: Tensor
    self_pred: Tensor
    agency: Tensor
    other: Tensor
    causal: Tensor
    meta: Tensor
    workspace: Tensor
    goal: Tensor
    semantic: Tensor
    retrieve: Tensor
    referential: Tensor
    curiosity: Tensor
    parts: dict[str, float] = field(default_factory=dict)

    def total(self) -> Tensor:
        terms = [
            self.world,
            0.4 * self.self_pred,
            0.2 * self.agency,
            0.3 * self.other,
            0.3 * self.causal,
            0.2 * self.meta,
            0.3 * self.workspace,
            0.25 * self.goal,
            0.35 * self.semantic,
            0.15 * self.retrieve,
            0.5 * self.referential,
            0.2 * self.curiosity,
        ]
        return sum(terms)  # type: ignore[return-value]

    def as_dict(self) -> dict[str, float]:
        return {
            "world": float(self.world.detach()),
            "self_pred": float(self.self_pred.detach()),
            "agency": float(self.agency.detach()),
            "other": float(self.other.detach()),
            "causal": float(self.causal.detach()),
            "meta": float(self.meta.detach()),
            "workspace": float(self.workspace.detach()),
            "goal": float(self.goal.detach()),
            "semantic": float(self.semantic.detach()),
            "retrieve": float(self.retrieve.detach()),
            "referential": float(self.referential.detach()),
            "curiosity": float(self.curiosity.detach()),
        }


def mse(a: Tensor, b: Tensor) -> Tensor:
    return torch.mean((a - b) ** 2)


def zero(device: torch.device) -> Tensor:
    return torch.zeros((), device=device)

"""Continual-learning guards: EWC, synaptic intelligence proxy, replay priority."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class EWCState:
    means: dict[str, Tensor] = field(default_factory=dict)
    fisher: dict[str, Tensor] = field(default_factory=dict)


class ContinualGuard:
    def __init__(self, lambda_ewc: float = 80.0) -> None:
        self.lambda_ewc = lambda_ewc
        self.state = EWCState()
        self.si_omega: dict[str, Tensor] = {}

    @torch.no_grad()
    def consolidate(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            self.state.means[name] = p.detach().clone()
            if name not in self.state.fisher:
                self.state.fisher[name] = torch.zeros_like(p)

    @torch.no_grad()
    def accumulate_fisher(self, module: nn.Module, decay: float = 0.95) -> None:
        for name, p in module.named_parameters():
            if p.grad is None:
                continue
            g2 = p.grad.detach() ** 2
            if name not in self.state.fisher:
                self.state.fisher[name] = g2
            else:
                self.state.fisher[name] = decay * self.state.fisher[name] + (1.0 - decay) * g2

    def penalty(self, module: nn.Module) -> Tensor:
        loss = None
        for name, p in module.named_parameters():
            if name not in self.state.means or name not in self.state.fisher:
                continue
            mean = self.state.means[name].to(p.device)
            fish = self.state.fisher[name].to(p.device)
            term = (fish * (p - mean) ** 2).sum()
            loss = term if loss is None else loss + term
        if loss is None:
            return torch.zeros((), device=next(module.parameters()).device)
        return self.lambda_ewc * loss

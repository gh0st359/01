"""Mitigations for catastrophic forgetting and plasticity loss."""

from __future__ import annotations

import numpy as np

from learning.nn import Module, Parameter
from learning.optim import ewc_penalty_grad


class ContinualGuard:
    def __init__(self, ewc_lambda: float = 0.08, stabilize_every: int = 200) -> None:
        self.ewc_lambda = ewc_lambda
        self.stabilize_every = stabilize_every
        self.updates = 0

    def observe_gradients(self, module: Module) -> None:
        for param in module.parameters():
            param.importance = 0.95 * param.importance + 0.05 * (param.grad ** 2)

    def apply_ewc(self, module: Module) -> None:
        for param in module.parameters():
            param.grad += ewc_penalty_grad(param, self.ewc_lambda)

    def maybe_stabilize(self, module: Module) -> None:
        self.updates += 1
        if self.updates % self.stabilize_every == 0:
            for param in module.parameters():
                param.stable_shadow = param.value.copy()

    def sparse_mask(self, param: Parameter, keep: float = 0.35) -> np.ndarray:
        mag = np.abs(param.grad)
        thresh = np.quantile(mag, 1.0 - keep) if mag.size else 0.0
        return (mag >= thresh).astype(np.float64)

    def clip_weight_norm(self, module: Module, max_norm: float = 8.0) -> None:
        for param in module.parameters():
            n = float(np.linalg.norm(param.value))
            if n > max_norm:
                param.value *= max_norm / (n + 1e-12)

"""Optimizers with synaptic stabilization hooks."""

from __future__ import annotations

import numpy as np

from learning.nn import Module, Parameter


class SGD:
    def __init__(self, lr: float = 0.01, clip: float = 1.0) -> None:
        self.lr = lr
        self.clip = clip

    def step(self, module: Module, neuromod: float = 1.0) -> None:
        scale = self.lr * float(np.clip(neuromod, 0.05, 4.0))
        for param in module.parameters():
            g = np.clip(param.grad, -self.clip, self.clip)
            param.value -= scale * g
            param.zero_grad()


class Adam:
    def __init__(self, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, clip: float = 1.0) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.clip = clip
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, module: Module, neuromod: float = 1.0) -> None:
        self.t += 1
        lr = self.lr * float(np.clip(neuromod, 0.05, 4.0))
        for param in module.parameters():
            g = np.clip(param.grad, -self.clip, self.clip)
            m = self.m.get(param.name)
            if m is None:
                m = np.zeros_like(g)
                v = np.zeros_like(g)
            else:
                v = self.v[param.name]
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            self.m[param.name] = m
            self.v[param.name] = v
            mhat = m / (1.0 - self.beta1 ** self.t)
            vhat = v / (1.0 - self.beta2 ** self.t)
            param.value -= lr * mhat / (np.sqrt(vhat) + 1e-8)
            param.zero_grad()

    def state_dict(self) -> dict:
        return {"t": self.t, "m": self.m, "v": self.v, "lr": self.lr}

    def load_state_dict(self, data: dict) -> None:
        self.t = int(data["t"])
        self.m = data["m"]
        self.v = data["v"]
        self.lr = float(data["lr"])


def ewc_penalty_grad(param: Parameter, lam: float = 0.1) -> np.ndarray:
    return lam * param.importance * (param.value - param.stable_shadow)

"""Learned self representation: identity, capabilities, historical self."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.types import Vector


@dataclass
class SelfSnapshot:
    identity: Vector
    capability: float
    controllability: float
    agency: float
    continuity: float
    uncertainty: float


class SelfModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.identity = np.zeros(dim)
        self.historical: list[Vector] = []
        self.capability = 0.25
        self.capability_uncertainty = 0.6
        self.controllability = 0.2
        self.agency = 0.2
        self.continuity = 0.0

    def update(self, proprio: Vector, core: Vector, action_effect: float, predicted_self: Vector | None, tick_frac: float) -> SelfSnapshot:
        fused = _fit(np.concatenate([_fit(proprio, 16), _fit(core, self.dim - 16)]), self.dim)
        self.identity = 0.97 * self.identity + 0.03 * fused
        self.historical.append(self.identity.copy())
        if len(self.historical) > 250:
            self.historical = self.historical[-120:]
        self.controllability = 0.9 * self.controllability + 0.1 * float(np.clip(action_effect, 0, 1))
        self.agency = 0.9 * self.agency + 0.1 * float(action_effect > 0.08)
        if predicted_self is not None:
            err = float(np.mean((predicted_self - self.identity) ** 2))
            self.capability = 0.95 * self.capability + 0.05 * (1.0 / (1.0 + err))
            self.capability_uncertainty = 0.9 * self.capability_uncertainty + 0.1 * np.tanh(err)
        if len(self.historical) > 8:
            self.continuity = float(np.clip(1.0 - np.linalg.norm(self.historical[-1] - self.historical[0]) / (np.linalg.norm(self.historical[0]) + 1.0), 0, 1))
        return SelfSnapshot(self.identity.copy(), self.capability, self.controllability, self.agency, self.continuity, self.capability_uncertainty)

    def snapshot(self) -> dict:
        return {
            "identity": self.identity,
            "capability": self.capability,
            "capability_uncertainty": self.capability_uncertainty,
            "controllability": self.controllability,
            "agency": self.agency,
            "continuity": self.continuity,
        }

    def restore(self, data: dict) -> None:
        self.identity = np.asarray(data["identity"], dtype=np.float64)
        self.capability = float(data["capability"])
        self.capability_uncertainty = float(data["capability_uncertainty"])
        self.controllability = float(data["controllability"])
        self.agency = float(data["agency"])
        self.continuity = float(data["continuity"])

    def perturb(self, noise: Vector) -> None:
        self.identity = self.identity + _fit(noise, self.dim)


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

"""Learn motor-sensory contingencies and which observed structure is the body."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.types import Vector


@dataclass
class BodyContingency:
    correlation: float
    delay_bias: float
    self_structure_score: float


class BodySchema:
    def __init__(self) -> None:
        self.motor_trace: list[Vector] = []
        self.visual_trace: list[Vector] = []
        self.proprio_trace: list[Vector] = []
        self.correlation = 0.0
        self.self_score = 0.0

    def observe(self, motor: Vector, proprio: Vector, visual_change: Vector) -> BodyContingency:
        self.motor_trace.append(_fit(motor, 12))
        self.proprio_trace.append(_fit(proprio, 12))
        self.visual_trace.append(_fit(visual_change, 12))
        if len(self.motor_trace) > 40:
            self.motor_trace = self.motor_trace[-40:]
            self.proprio_trace = self.proprio_trace[-40:]
            self.visual_trace = self.visual_trace[-40:]
        if len(self.motor_trace) >= 8:
            m = np.stack(self.motor_trace)
            p = np.stack(self.proprio_trace)
            v = np.stack(self.visual_trace)
            self.correlation = 0.85 * self.correlation + 0.15 * _corr(m, p)
            delayed = _corr(m[:-1], v[1:])
            undelayed = _corr(m, v)
            self.self_score = 0.9 * self.self_score + 0.1 * max(delayed, undelayed)
        return BodyContingency(self.correlation, 0.0, self.self_score)


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    aa = a[:n].ravel()
    bb = b[:n].ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-9
    return float(np.clip(np.dot(aa, bb) / den, -1, 1))

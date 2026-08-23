"""Persistent recurrent cognitive state.

x(t+Δt) = F(x(t), perception, memory, prediction, body, motivation, social, uncertainty)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning.nn import GRUCell, Linear, Module, RMSNorm, tanh
from shared.config import OrganismConfig
from shared.rng import RNG
from shared.types import Vector


@dataclass
class CoreSnapshot:
    fast: Vector
    medium: Vector
    slow: Vector
    fused: Vector


class RecurrentCore(Module):
    def __init__(self, cfg: OrganismConfig, rng: RNG) -> None:
        super().__init__("core")
        self.cfg = cfg
        d = cfg.state_dim
        inp = cfg.feature_dim + cfg.state_dim // 2 + 16
        self.norm = self.add(RMSNorm("core.norm", inp))
        self.in_proj = self.add(Linear("core.in", inp, d, rng))
        self.fast = self.add(GRUCell("core.fast", d, d, rng))
        self.medium = self.add(GRUCell("core.medium", d, d, rng))
        self.slow = self.add(GRUCell("core.slow", d, d, rng))
        self.fuse = self.add(Linear("core.fuse", 3 * d, d, rng))
        self.h_fast = np.zeros(d, dtype=np.float64)
        self.h_medium = np.zeros(d, dtype=np.float64)
        self.h_slow = np.zeros(d, dtype=np.float64)
        self.tick = 0
        self.last_cache: dict | None = None

    def pack_input(
        self,
        perception: Vector,
        workspace: Vector,
        prediction_error: Vector,
        motivation: Vector,
        body: Vector,
        social: Vector,
        uncertainty: float,
    ) -> Vector:
        cfg = self.cfg
        parts = [
            _fit(perception, cfg.feature_dim),
            _fit(workspace, cfg.state_dim // 4),
            _fit(prediction_error, cfg.state_dim // 8),
            _fit(motivation, 8),
            _fit(body, 6),
            _fit(social, 4),
            np.array([uncertainty, np.tanh(self.tick / 1000.0)], dtype=np.float64),
        ]
        raw = np.concatenate(parts)
        return _fit(raw, cfg.feature_dim + cfg.state_dim // 2 + 16)

    def step(
        self,
        perception: Vector,
        workspace: Vector,
        prediction_error: Vector,
        motivation: Vector,
        body: Vector,
        social: Vector,
        uncertainty: float,
    ) -> Vector:
        x = self.pack_input(perception, workspace, prediction_error, motivation, body, social, uncertainty)
        x = tanh(self.in_proj.forward(self.norm.forward(x)))
        self.h_fast, cache_f = self.fast.forward(x, self.h_fast)
        if self.tick % 4 == 0:
            self.h_medium, _ = self.medium.forward(self.h_fast, self.h_medium)
        if self.tick % 16 == 0:
            self.h_slow, _ = self.slow.forward(self.h_medium, self.h_slow)
        fused = tanh(self.fuse.forward(np.concatenate([self.h_fast, self.h_medium, self.h_slow])))
        self.last_cache = cache_f
        self.tick += 1
        return fused

    @property
    def state(self) -> Vector:
        return tanh(self.fuse.forward(np.concatenate([self.h_fast, self.h_medium, self.h_slow])))

    def snapshot(self) -> CoreSnapshot:
        return CoreSnapshot(self.h_fast.copy(), self.h_medium.copy(), self.h_slow.copy(), self.state.copy())

    def restore(self, snap: CoreSnapshot) -> None:
        self.h_fast = snap.fast.copy()
        self.h_medium = snap.medium.copy()
        self.h_slow = snap.slow.copy()

    def perturb(self, rng: RNG, scale: float = 0.15) -> None:
        self.h_fast = self.h_fast + rng.normal(self.h_fast.shape, scale=scale)
        self.h_medium = self.h_medium + rng.normal(self.h_medium.shape, scale=scale * 0.5)

    def state_vectors(self) -> dict[str, Vector]:
        return {
            "fast": self.h_fast.copy(),
            "medium": self.h_medium.copy(),
            "slow": self.h_slow.copy(),
            "fused": self.state.copy(),
        }

    def load_vectors(self, data: dict[str, Vector]) -> None:
        self.h_fast = np.asarray(data["fast"], dtype=np.float64)
        self.h_medium = np.asarray(data["medium"], dtype=np.float64)
        self.h_slow = np.asarray(data["slow"], dtype=np.float64)


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    out = np.zeros(n, dtype=np.float64)
    out[: x.size] = x
    return out

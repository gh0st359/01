"""Comparative substrate wrapper: GRU vs CfC vs reservoir."""

from __future__ import annotations

from enum import Enum

import numpy as np

from cognition.core.recurrent import RecurrentCore
from learning.nn import CfCCell, Linear, Module, Reservoir, tanh
from shared.config import OrganismConfig
from shared.rng import RNG
from shared.types import Vector


class SubstrateKind(str, Enum):
    GRU = "gru"
    CFC = "cfc"
    RESERVOIR = "reservoir"


class MultiScaleCore(Module):
    def __init__(self, cfg: OrganismConfig, rng: RNG, kind: SubstrateKind = SubstrateKind.GRU) -> None:
        super().__init__("substrate")
        self.kind = kind
        self.cfg = cfg
        d = cfg.state_dim
        self.gru = RecurrentCore(cfg, rng) if kind is SubstrateKind.GRU else None
        if kind is SubstrateKind.CFC:
            self.cell = self.add(CfCCell("cfc", d, d, rng))
            self.proj = self.add(Linear("cfc.in", cfg.feature_dim, d, rng))
            self.h = np.zeros(d, dtype=np.float64)
        elif kind is SubstrateKind.RESERVOIR:
            self.res = self.add(Reservoir("res", cfg.feature_dim, d, d, rng))
            self.h = np.zeros(d, dtype=np.float64)
        else:
            self.cell = None
            self.res = None
            self.h = np.zeros(d, dtype=np.float64)

    def step(self, features: Vector, dt: float = 0.05) -> Vector:
        if self.kind is SubstrateKind.GRU and self.gru is not None:
            return self.gru.step(features, features, features[:8] if features.size >= 8 else features, np.zeros(8), np.zeros(6), np.zeros(4), 0.1)
        x = _fit(features, self.cfg.feature_dim)
        if self.kind is SubstrateKind.CFC:
            self.h = self.cell.forward(tanh(self.proj.forward(x)), self.h, dt)
            return self.h
        y, self.h = self.res.forward(x, self.h)
        return y


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

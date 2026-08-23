"""Learned multi-horizon predictive world model.

current latent + action → future latent, with uncertainty.
Evaluated independently from language.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning.nn import GRUCell, Linear, Module, tanh
from learning.optim import Adam
from shared.config import OrganismConfig
from shared.rng import RNG
from shared.types import Vector


@dataclass
class Prediction:
    latent: Vector
    decoded: Vector
    uncertainty: float
    horizons: list[Vector]
    error: float


class PredictiveWorldModel(Module):
    def __init__(self, cfg: OrganismConfig, rng: RNG) -> None:
        super().__init__("world_model")
        self.cfg = cfg
        d = cfg.feature_dim
        self.encoder = self.add(Linear("wm.enc", d, d, rng))
        self.dyn = self.add(GRUCell("wm.dyn", d + cfg.action_dim, d, rng))
        self.unc = self.add(Linear("wm.unc", d, 1, rng))
        self.decoder = self.add(Linear("wm.dec", d, d, rng))
        self.opt = Adam(lr=cfg.learning_rate)
        self.h = np.zeros(d, dtype=np.float64)
        self.last = Prediction(self.h.copy(), np.zeros(d), 0.5, [], 0.0)
        self.error_trace: list[float] = []

    def encode(self, features: Vector) -> Vector:
        return tanh(self.encoder.forward(_fit(features, self.cfg.feature_dim)))

    def predict(self, features: Vector, action: Vector, horizon: int | None = None) -> Prediction:
        z = self.encode(features)
        h = z.copy()
        horizons = []
        act = _fit(action, self.cfg.action_dim)
        steps = horizon or max(1, self.cfg.imagination_horizon // 2)
        for _ in range(steps):
            inp = np.concatenate([h, act])
            h, _ = self.dyn.forward(inp, h)
            horizons.append(h.copy())
        decoded = self.decoder.forward(h)
        uncertainty = float(1.0 / (1.0 + np.exp(-self.unc.forward(h)[0])))
        pred = Prediction(h, decoded, uncertainty, horizons, 0.0)
        self.h = h
        self.last = pred
        return pred

    def observe(self, features: Vector) -> float:
        target = self.encode(features)
        pred = self.last.latent
        err_vec = pred - target
        error = float(np.mean(err_vec ** 2))
        self.last.error = error
        self.error_trace.append(error)
        if len(self.error_trace) > 2000:
            self.error_trace = self.error_trace[-1000:]
        return error

    def train_step(self, before: Vector, action: Vector, after: Vector, neuromod: float = 1.0) -> float:
        z0 = self.encode(before)
        z1 = self.encode(after)
        act = _fit(action, self.cfg.action_dim)
        inp = np.concatenate([z0, act])
        pred, cache = self.dyn.forward(inp, z0)
        decoded = self.decoder.forward(pred)
        err = pred - z1
        dec_err = decoded - z1
        loss = float(np.mean(err ** 2) + 0.25 * np.mean(dec_err ** 2))
        grad = (2.0 / err.size) * err + 0.25 * (self.decoder.weight.value.T @ ((2.0 / dec_err.size) * dec_err))
        self.decoder.backward(pred, (2.0 / dec_err.size) * dec_err)
        self.dyn.backward(cache, grad)
        self.opt.step(self, neuromod=neuromod)
        return loss

    def rollout(self, features: Vector, actions: list[Vector]) -> list[Vector]:
        z = self.encode(features)
        h = z
        out = []
        for act in actions:
            inp = np.concatenate([h, _fit(act, self.cfg.action_dim)])
            h, _ = self.dyn.forward(inp, h)
            out.append(h.copy())
        return out


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

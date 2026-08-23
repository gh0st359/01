"""Learned evaluative preferences — not authored opinions."""

from __future__ import annotations

import numpy as np

from learning.nn import cosine
from shared.types import Vector


class PreferenceStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.items: dict[str, tuple[Vector, float]] = {}

    def reinforce(self, key: str, representation: Vector, value: float) -> None:
        if key in self.items:
            vec, v = self.items[key]
            self.items[key] = (0.9 * vec + 0.1 * _fit(representation, self.dim), 0.9 * v + 0.1 * value)
        else:
            self.items[key] = (_fit(representation, self.dim), value)

    def evaluate(self, representation: Vector) -> float:
        if not self.items:
            return 0.0
        scored = [cosine(vec, representation) * val for vec, val in self.items.values()]
        return float(np.tanh(sum(scored) / len(scored)))

    def snapshot(self) -> dict:
        return {k: {"vec": v[0], "value": v[1]} for k, v in self.items.items()}

    def restore(self, data: dict) -> None:
        self.items = {k: (np.asarray(v["vec"], dtype=np.float64), float(v["value"])) for k, v in data.items()}


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

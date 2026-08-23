"""Sparse distributed / Hopfield-like associative memory."""

from __future__ import annotations

import numpy as np

from shared.types import Vector


class AssociativeMemory:
    def __init__(self, dim: int, capacity: int) -> None:
        self.dim = dim
        self.capacity = capacity
        self.patterns: list[Vector] = []
        self.weights = np.zeros((dim, dim), dtype=np.float64)

    def store(self, pattern: Vector) -> None:
        p = _binarize(_fit(pattern, self.dim))
        self.patterns.append(p)
        self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0.0)
        if len(self.patterns) > self.capacity:
            old = self.patterns.pop(0)
            self.weights -= np.outer(old, old)
            np.fill_diagonal(self.weights, 0.0)

    def recall(self, cue: Vector, steps: int = 8) -> Vector:
        x = _binarize(_fit(cue, self.dim))
        for _ in range(steps):
            x = np.sign(self.weights @ x + 1e-9)
        return x

    def snapshot(self) -> dict:
        return {"patterns": self.patterns, "weights": self.weights}

    def restore(self, data: dict) -> None:
        self.patterns = [np.asarray(p, dtype=np.float64) for p in data.get("patterns", [])]
        self.weights = np.asarray(data.get("weights", self.weights), dtype=np.float64)


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out


def _binarize(x: Vector) -> Vector:
    return np.sign(x - np.mean(x) + 1e-9)

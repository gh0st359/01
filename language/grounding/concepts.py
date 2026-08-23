"""Concept vectors grown from attended experience."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning.nn import cosine
from shared.types import Vector


@dataclass
class Concept:
    concept_id: str
    vector: Vector
    count: int
    last_tick: int


class ConceptMemory:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.concepts: dict[str, Concept] = {}
        self.next_id = 1

    def observe(self, features: Vector, tick: int) -> Concept:
        vec = _fit(features, self.dim)
        best = None
        best_s = -1.0
        for c in self.concepts.values():
            s = cosine(c.vector, vec)
            if s > best_s:
                best, best_s = c, s
        if best is not None and best_s > 0.82:
            best.vector = 0.9 * best.vector + 0.1 * vec
            best.count += 1
            best.last_tick = tick
            return best
        cid = f"concept_{self.next_id}"
        self.next_id += 1
        c = Concept(cid, vec, 1, tick)
        self.concepts[cid] = c
        return c

    def snapshot(self) -> dict:
        return {
            "next_id": self.next_id,
            "concepts": {
                cid: {"vector": c.vector, "count": c.count, "last_tick": c.last_tick} for cid, c in self.concepts.items()
            },
        }

    def restore(self, data: dict) -> None:
        self.next_id = int(data.get("next_id", 1))
        self.concepts = {
            cid: Concept(cid, np.asarray(b["vector"], dtype=np.float64), int(b["count"]), int(b["last_tick"]))
            for cid, b in data.get("concepts", {}).items()
        }


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

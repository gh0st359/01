"""Learned word ↔ concept bindings. Words are not thoughts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning.hebbian import hebbian_update
from learning.nn import cosine
from shared.types import Vector


@dataclass
class LexEntry:
    word: str
    vector: Vector
    count: int
    last_tick: int
    kind: str


class Lexicon:
    def __init__(self, dim: int, capacity: int) -> None:
        self.dim = dim
        self.capacity = capacity
        self.entries: dict[str, LexEntry] = {}

    def bind(self, word: str, concept: Vector, tick: int, kind: str = "unknown") -> LexEntry:
        word = word.lower().strip()
        vec = _fit(concept, self.dim)
        if word in self.entries:
            e = self.entries[word]
            e.vector = 0.85 * e.vector + 0.15 * vec
            e.count += 1
            e.last_tick = tick
            if kind != "unknown":
                e.kind = kind
            return e
        entry = LexEntry(word, vec, 1, tick, kind)
        self.entries[word] = entry
        if len(self.entries) > self.capacity:
            rare = min(self.entries.values(), key=lambda x: x.count)
            del self.entries[rare.word]
        return entry

    def nearest(self, concept: Vector, k: int = 3) -> list[tuple[str, float]]:
        if not self.entries:
            return []
        scored = [(w, cosine(e.vector, concept)) for w, e in self.entries.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(w, s) for w, s in scored[:k] if s > 0.15]

    def embed(self, word: str) -> Vector | None:
        e = self.entries.get(word.lower().strip())
        return None if e is None else e.vector.copy()

    def known(self, word: str) -> bool:
        return word.lower().strip() in self.entries

    def snapshot(self) -> dict:
        return {
            w: {"vector": e.vector, "count": e.count, "last_tick": e.last_tick, "kind": e.kind}
            for w, e in self.entries.items()
        }

    def restore(self, data: dict) -> None:
        self.entries = {}
        for w, blob in data.items():
            self.entries[w] = LexEntry(
                w,
                np.asarray(blob["vector"], dtype=np.float64),
                int(blob["count"]),
                int(blob["last_tick"]),
                str(blob.get("kind", "unknown")),
            )


def hash_embed(word: str, dim: int) -> Vector:
    """Deterministic fallback embedding for unknown surface forms."""
    rng = np.random.default_rng(abs(hash(word)) % (2**32))
    v = rng.normal(size=dim)
    return v / (np.linalg.norm(v) + 1e-9)


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

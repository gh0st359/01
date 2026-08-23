"""Capacity-limited working memory with decay and chunking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from learning.nn import cosine
from shared.types import Vector


@dataclass
class WMSlot:
    content: Vector
    entity_id: str | None
    source: str
    activation: float
    tick: int
    tags: list[str] = field(default_factory=list)


class WorkingMemory:
    def __init__(self, slots: int, dim: int) -> None:
        self.capacity = slots
        self.dim = dim
        self.slots: list[WMSlot] = []
        self.chunks = 0

    def write(self, content: Vector, source: str, tick: int, entity_id: str | None = None, tags: list[str] | None = None) -> None:
        vec = _fit(content, self.dim)
        for slot in self.slots:
            if cosine(slot.content, vec) > 0.88:
                slot.content = 0.6 * slot.content + 0.4 * vec
                slot.activation = min(1.0, slot.activation + 0.3)
                slot.tick = tick
                if entity_id:
                    slot.entity_id = entity_id
                self.chunks += 1
                return
        self.slots.append(WMSlot(vec, entity_id, source, 1.0, tick, tags or []))
        if len(self.slots) > self.capacity:
            self.slots.sort(key=lambda s: s.activation)
            self.slots = self.slots[1:]

    def decay(self) -> None:
        keep = []
        for slot in self.slots:
            slot.activation *= 0.92
            if slot.activation > 0.08:
                keep.append(slot)
        self.slots = keep

    def contents(self) -> list[WMSlot]:
        return list(self.slots)

    def summary(self) -> Vector:
        if not self.slots:
            return np.zeros(self.dim)
        w = np.array([s.activation for s in self.slots])
        w = w / (w.sum() + 1e-8)
        return sum(wi * s.content for wi, s in zip(w, self.slots))

    def snapshot(self) -> list[dict]:
        return [
            {
                "content": s.content,
                "entity_id": s.entity_id,
                "source": s.source,
                "activation": s.activation,
                "tick": s.tick,
                "tags": s.tags,
            }
            for s in self.slots
        ]

    def restore(self, data: list[dict]) -> None:
        self.slots = [
            WMSlot(
                np.asarray(d["content"], dtype=np.float64),
                d.get("entity_id"),
                d["source"],
                float(d["activation"]),
                int(d["tick"]),
                list(d.get("tags", [])),
            )
            for d in data
        ]


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

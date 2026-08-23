"""Temporal continuity and event segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.types import Vector


@dataclass
class TemporalEvent:
    tick: int
    magnitude: float
    kind: str


class TemporalModel:
    def __init__(self) -> None:
        self.prev: Vector | None = None
        self.events: list[TemporalEvent] = []
        self.clock = 0.0

    def step(self, features: Vector, tick: int, dt: float) -> float:
        self.clock += dt
        mag = 0.0
        if self.prev is not None:
            a = features.ravel()
            b = self.prev.ravel()
            n = min(a.size, b.size)
            mag = float(np.mean((a[:n] - b[:n]) ** 2))
            if mag > 0.08:
                self.events.append(TemporalEvent(tick, mag, "boundary"))
                if len(self.events) > 500:
                    self.events = self.events[-250:]
        self.prev = features.copy()
        return mag

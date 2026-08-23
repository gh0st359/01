"""Allocentric / egocentric spatial map over entities."""

from __future__ import annotations

import numpy as np

from shared.types import Entity, Vector


class SpatialModel:
    def __init__(self, bins: int = 8) -> None:
        self.bins = bins
        self.occupancy = np.zeros((bins, bins), dtype=np.float64)
        self.last_self = np.array([0.5, 0.5])

    def update(self, entities: list[Entity]) -> Vector:
        self.occupancy *= 0.92
        for ent in entities:
            x = int(np.clip(ent.location[0] * self.bins, 0, self.bins - 1))
            y = int(np.clip(ent.location[1] * self.bins, 0, self.bins - 1))
            self.occupancy[y, x] += 1.0 if ent.visible else 0.4
            if ent.is_self:
                self.last_self = ent.location[:2].copy()
        return self.occupancy.ravel()

    def distance(self, a: Entity, b: Entity) -> float:
        return float(np.linalg.norm(a.location[:2] - b.location[:2]))

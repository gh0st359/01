"""Decision over imagined trajectories and current drives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.rng import RNG
from shared.types import ActionKind, Vector


@dataclass
class Decision:
    kind: ActionKind
    action_vector: Vector
    value: float
    imagined_error: float
    source: str


class DecisionSystem:
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.last = Decision(ActionKind.LOOK, np.zeros(action_dim), 0.0, 0.0, "init")

    def select(
        self,
        scored: list[tuple[ActionKind, Vector, float, float]],
        rng: RNG,
        temperature: float = 0.35,
    ) -> Decision:
        if not scored:
            vec = np.zeros(self.action_dim)
            self.last = Decision(ActionKind.LOOK, vec, 0.0, 1.0, "fallback")
            return self.last
        values = np.array([s[2] for s in scored], dtype=np.float64)
        values = values - values.max()
        p = np.exp(values / max(temperature, 0.05))
        p = p / p.sum()
        idx = rng.choice(len(scored), p=p)
        kind, vec, val, err = scored[idx]
        self.last = Decision(kind, vec, val, err, "imagination")
        return self.last

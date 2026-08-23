"""Compare what happened with what might have happened under another action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planning.action import primitive_vector
from shared.types import ActionKind, Vector
from world_model.predictive import PredictiveWorldModel


@dataclass
class CounterfactualLesson:
    tick: int
    taken: ActionKind
    alternative: ActionKind
    actual_error: float
    alt_predicted_error: float
    lesson: Vector


class CounterfactualEngine:
    def __init__(self) -> None:
        self.lessons: list[CounterfactualLesson] = []

    def compare(
        self,
        world_model: PredictiveWorldModel,
        features_before: Vector,
        taken: ActionKind,
        actual_after: Vector,
        actual_error: float,
        tick: int,
        dim: int,
    ) -> CounterfactualLesson | None:
        alts = [k for k in (ActionKind.WAIT, ActionKind.TURN, ActionKind.EXPERIMENT, ActionKind.LOOK) if k is not taken]
        if not alts:
            return None
        best = None
        for alt in alts:
            pred = world_model.rollout(features_before, [primitive_vector(alt, dim)])
            if not pred:
                continue
            err = float(np.mean((pred[-1] - _fit(actual_after, pred[-1].size)) ** 2))
            if best is None or err < best[1]:
                best = (alt, err, pred[-1])
        if best is None:
            return None
        lesson = CounterfactualLesson(
            tick,
            taken,
            best[0],
            actual_error,
            best[1],
            best[2] - _fit(actual_after, best[2].size),
        )
        self.lessons.append(lesson)
        if len(self.lessons) > 300:
            self.lessons = self.lessons[-150:]
        return lesson


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

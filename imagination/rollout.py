"""Internal simulation of candidate futures without execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planning.action import primitive_vector
from shared.config import OrganismConfig
from shared.types import ActionKind, Vector
from world_model.predictive import PredictiveWorldModel


@dataclass
class ImaginedTrajectory:
    kind: ActionKind
    states: list[Vector]
    predicted_error: float
    goal_score: float
    info_gain: float
    homeo_cost: float
    value: float


class Imagination:
    def __init__(self, cfg: OrganismConfig) -> None:
        self.cfg = cfg
        self.last: list[ImaginedTrajectory] = []

    def imagine(
        self,
        world_model: PredictiveWorldModel,
        features: Vector,
        goal_vec: Vector | None,
        energy: float,
    ) -> list[ImaginedTrajectory]:
        kinds = [
            ActionKind.MOVE,
            ActionKind.TURN,
            ActionKind.LOOK,
            ActionKind.APPROACH,
            ActionKind.INSPECT,
            ActionKind.EXPERIMENT,
            ActionKind.GRASP,
            ActionKind.WAIT,
        ]
        out: list[ImaginedTrajectory] = []
        for kind in kinds[: self.cfg.imagination_branches + 2]:
            act = primitive_vector(kind, self.cfg.action_dim)
            states = world_model.rollout(features, [act] * min(self.cfg.imagination_horizon, 6))
            if not states:
                continue
            pe = float(np.mean((states[-1] - states[0]) ** 2))
            goal_score = 0.0
            if goal_vec is not None:
                g = _fit(goal_vec, states[-1].size)
                goal_score = float(np.dot(states[-1], g) / (np.linalg.norm(g) + 1e-8))
            info = float(np.mean(np.std(np.stack(states), axis=0)))
            homeo_cost = 0.05 * (kind in {ActionKind.MOVE, ActionKind.APPROACH}) / max(energy, 0.1)
            value = 0.35 * goal_score + 0.3 * info + 0.2 * pe - homeo_cost
            out.append(ImaginedTrajectory(kind, states, pe, goal_score, info, homeo_cost, value))
        self.last = out
        return out


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

"""Models of other autonomous entities: goals, perception, incomplete information."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.types import Entity, Vector


@dataclass
class OtherModel:
    entity_id: str
    representation: Vector
    inferred_goal: Vector
    predictability: float
    last_seen: int
    believed_object_location: Vector | None = None
    last_visible_to_them: dict[str, Vector] = field(default_factory=dict)


class OtherAgentModels:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.models: dict[str, OtherModel] = {}

    def update(self, entities: list[Entity], tick: int) -> None:
        for ent in entities:
            if ent.is_self:
                continue
            motion = float(np.linalg.norm(ent.velocity[:2]))
            agentish = motion > 0.01 and ent.permanence_evidence > 0.1
            if not agentish and not ent.is_agent:
                continue
            ent.is_agent = True
            if ent.entity_id not in self.models:
                self.models[ent.entity_id] = OtherModel(ent.entity_id, ent.representation.copy(), ent.velocity.copy(), 0.3, tick)
            model = self.models[ent.entity_id]
            pred_err = float(np.linalg.norm(model.inferred_goal[: min(2, model.inferred_goal.size)] - ent.location[:2]))
            model.predictability = 0.9 * model.predictability + 0.1 * (1.0 / (1.0 + pred_err))
            model.inferred_goal = 0.8 * _fit(model.inferred_goal, 4) + 0.2 * ent.location
            model.representation = 0.85 * model.representation + 0.15 * _fit(ent.representation, model.representation.size)
            model.last_seen = tick
            if ent.visible:
                for other in entities:
                    if other.entity_id != ent.entity_id and other.visible:
                        model.last_visible_to_them[other.entity_id] = other.location.copy()

    def false_belief_location(self, agent_id: str, object_id: str) -> Vector | None:
        model = self.models.get(agent_id)
        if not model:
            return None
        return model.last_visible_to_them.get(object_id)

    def snapshot(self) -> dict:
        return {
            aid: {
                "representation": m.representation,
                "inferred_goal": m.inferred_goal,
                "predictability": m.predictability,
                "last_seen": m.last_seen,
            }
            for aid, m in self.models.items()
        }


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

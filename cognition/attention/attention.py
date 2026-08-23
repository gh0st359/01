"""Attentional gating over entities and workspace candidates."""

from __future__ import annotations

import numpy as np

from learning.nn import cosine
from shared.types import Entity, Vector, WorkspaceCandidate


class Attention:
    def __init__(self, decay: float = 0.12) -> None:
        self.decay = decay
        self.focus_id: str | None = None
        self.focus_vector: Vector | None = None
        self.dwell = 0
        self.history: list[str] = []

    def select_entity(self, entities: list[Entity], goal_vec: Vector | None, novelty: dict[str, float]) -> Entity | None:
        if not entities:
            self.focus_id = None
            return None
        scores = []
        for ent in entities:
            s = 0.35 * (1.0 - ent.uncertainty) + 0.2 * (1.0 if ent.visible else 0.2)
            s += 0.2 * novelty.get(ent.entity_id, 0.0)
            if goal_vec is not None and ent.representation.size == goal_vec.size:
                s += 0.25 * max(cosine(ent.representation, goal_vec), 0.0)
            if ent.entity_id == self.focus_id:
                s += 0.08 * min(self.dwell / 10.0, 1.0)
            scores.append(s)
        idx = int(np.argmax(scores))
        chosen = entities[idx]
        if chosen.entity_id == self.focus_id:
            self.dwell += 1
        else:
            self.dwell = 1
            self.focus_id = chosen.entity_id
            self.history.append(chosen.entity_id)
            if len(self.history) > 200:
                self.history = self.history[-100:]
        self.focus_vector = chosen.representation.copy()
        return chosen

    def gate(self, candidates: list[WorkspaceCandidate], focus: Vector | None) -> list[WorkspaceCandidate]:
        if focus is None:
            return candidates
        for cand in candidates:
            if cand.vector.size == focus.size:
                cand.relevance = 0.7 * cand.relevance + 0.3 * max(cosine(cand.vector, focus), 0.0)
        return candidates

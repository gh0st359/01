"""Persistent entity slots with identity tracking and object permanence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from learning.hebbian import trace_update
from learning.nn import cosine
from perception.hierarchy import Percept
from shared.config import OrganismConfig
from shared.types import Entity, Vector


@dataclass
class TrackingStats:
    matches: int = 0
    new_entities: int = 0
    occlusions: int = 0


class EntitySystem:
    def __init__(self, cfg: OrganismConfig) -> None:
        self.cfg = cfg
        self.entities: dict[str, Entity] = {}
        self.next_id = 1
        self.stats = TrackingStats()
        self.self_id: str | None = None

    def update(self, percept: Percept, proprio: Vector, tick: int) -> list[Entity]:
        detections = []
        for feat, loc in zip(percept.proto_objects, percept.proto_locations):
            detections.append((feat, loc, True))
        # persistence: keep recently seen hidden entities as predictions
        existing = [e for e in self.entities.values() if tick - e.last_seen_tick < 80]
        if existing and detections:
            cost = np.ones((len(existing), len(detections))) * 2.5
            for i, ent in enumerate(existing):
                for j, (feat, loc, _) in enumerate(detections):
                    f = _fit(feat, ent.visual_features.size if ent.visual_features.size else feat.size)
                    vf = _fit(ent.visual_features, f.size)
                    loc_cost = float(np.linalg.norm(ent.location[:2] - loc))
                    feat_cost = 1.0 - max(cosine(_fit(vf, f.size), f), -1.0)
                    cost[i, j] = 0.55 * feat_cost + 0.45 * loc_cost
            ri, ci = linear_sum_assignment(cost)
            assigned_dets = set()
            assigned_ents = set()
            for i, j in zip(ri, ci):
                if cost[i, j] < 1.15:
                    self._update_entity(existing[i], detections[j][0], detections[j][1], tick, visible=True)
                    assigned_dets.add(j)
                    assigned_ents.add(existing[i].entity_id)
                    self.stats.matches += 1
            for j, det in enumerate(detections):
                if j not in assigned_dets:
                    self._spawn(det[0], det[1], tick)
            for ent in existing:
                if ent.entity_id not in assigned_ents and ent.visible:
                    self._occlude(ent, tick)
        elif detections:
            for feat, loc, _ in detections:
                self._spawn(feat, loc, tick)
        else:
            for ent in existing:
                if ent.visible:
                    self._occlude(ent, tick)
        self._update_self(proprio, tick)
        return list(self.entities.values())

    def _spawn(self, feat: Vector, loc: Vector, tick: int) -> Entity:
        eid = f"entity_{self.next_id}"
        self.next_id += 1
        vec = _fit(np.concatenate([feat, loc]), self.cfg.entity_dim)
        ent = Entity(
            entity_id=eid,
            representation=vec,
            location=_fit(loc, 4),
            velocity=np.zeros(4),
            visual_features=_fit(feat, 16),
            auditory_associations=np.zeros(16),
            affordances=np.zeros(8),
            uncertainty=0.6,
            last_seen_tick=tick,
            first_seen_tick=tick,
            visible=True,
            kind_hint="unknown",
        )
        self.entities[eid] = ent
        self.stats.new_entities += 1
        if len(self.entities) > self.cfg.max_entities:
            oldest = min(self.entities.values(), key=lambda e: e.last_seen_tick if not e.is_self else 10**9)
            if not oldest.is_self:
                del self.entities[oldest.entity_id]
        return ent

    def _update_entity(self, ent: Entity, feat: Vector, loc: Vector, tick: int, visible: bool) -> None:
        prev = ent.location[:2].copy()
        loc4 = _fit(loc, 4)
        ent.velocity[:2] = loc4[:2] - prev
        ent.location = loc4
        ent.visual_features = 0.8 * _fit(ent.visual_features, 16) + 0.2 * _fit(feat, 16)
        ent.representation = 0.85 * ent.representation + 0.15 * _fit(np.concatenate([feat, loc]), ent.representation.size)
        ent.visible = visible
        ent.last_seen_tick = tick
        ent.uncertainty = max(0.05, ent.uncertainty * 0.85)
        if not visible:
            ent.permanence_evidence = min(1.0, ent.permanence_evidence + 0.05)
        else:
            ent.permanence_evidence = min(1.0, ent.permanence_evidence + 0.02)

    def _occlude(self, ent: Entity, tick: int) -> None:
        ent.visible = False
        ent.uncertainty = min(1.0, ent.uncertainty + 0.08)
        ent.location = ent.location + ent.velocity
        ent.permanence_evidence = min(1.0, ent.permanence_evidence + 0.08)
        self.stats.occlusions += 1

    def _update_self(self, proprio: Vector, tick: int) -> None:
        if self.self_id is None:
            vec = _fit(proprio, self.cfg.entity_dim)
            ent = Entity(
                entity_id="self",
                representation=vec,
                location=_fit(proprio[:2], 4),
                velocity=_fit(proprio[4:6] if proprio.size > 5 else np.zeros(2), 4),
                visual_features=np.zeros(16),
                auditory_associations=np.zeros(16),
                affordances=np.zeros(8),
                uncertainty=0.2,
                last_seen_tick=tick,
                first_seen_tick=tick,
                visible=True,
                kind_hint="self",
                is_self=True,
            )
            self.entities["self"] = ent
            self.self_id = "self"
        else:
            ent = self.entities["self"]
            ent.representation = trace_update(ent.representation, _fit(proprio, ent.representation.size), 0.8)
            ent.location = _fit(proprio[:2], 4)
            ent.last_seen_tick = tick

    def get(self, eid: str) -> Entity | None:
        return self.entities.get(eid)

    def visible(self) -> list[Entity]:
        return [e for e in self.entities.values() if e.visible]

    def snapshot(self) -> dict:
        return {
            "next_id": self.next_id,
            "self_id": self.self_id,
            "entities": {
                eid: {
                    "representation": e.representation,
                    "location": e.location,
                    "velocity": e.velocity,
                    "visual_features": e.visual_features,
                    "uncertainty": e.uncertainty,
                    "last_seen_tick": e.last_seen_tick,
                    "first_seen_tick": e.first_seen_tick,
                    "visible": e.visible,
                    "kind_hint": e.kind_hint,
                    "is_self": e.is_self,
                    "is_agent": e.is_agent,
                    "permanence_evidence": e.permanence_evidence,
                    "relationships": e.relationships,
                    "ownership": e.ownership,
                }
                for eid, e in self.entities.items()
            },
        }

    def restore(self, data: dict) -> None:
        self.next_id = int(data["next_id"])
        self.self_id = data.get("self_id")
        self.entities = {}
        for eid, blob in data["entities"].items():
            self.entities[eid] = Entity(
                entity_id=eid,
                representation=np.asarray(blob["representation"], dtype=np.float64),
                location=np.asarray(blob["location"], dtype=np.float64),
                velocity=np.asarray(blob["velocity"], dtype=np.float64),
                visual_features=np.asarray(blob["visual_features"], dtype=np.float64),
                auditory_associations=np.zeros(16),
                affordances=np.zeros(8),
                uncertainty=float(blob["uncertainty"]),
                last_seen_tick=int(blob["last_seen_tick"]),
                first_seen_tick=int(blob["first_seen_tick"]),
                visible=bool(blob["visible"]),
                kind_hint=str(blob.get("kind_hint", "unknown")),
                relationships=dict(blob.get("relationships", {})),
                ownership=blob.get("ownership"),
                is_self=bool(blob.get("is_self", False)),
                is_agent=bool(blob.get("is_agent", False)),
                permanence_evidence=float(blob.get("permanence_evidence", 0.0)),
            )


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

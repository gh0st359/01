"""Semantic memory grown from consolidation, not a hand-authored graph."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from learning.nn import cosine
from shared.types import EvidenceSource, Vector


@dataclass
class SemanticRelation:
    relation_id: str
    head: Vector
    relation: Vector
    tail: Vector
    confidence: float
    count: int
    provenance: list[str] = field(default_factory=list)
    label_hint: str = ""


class SemanticMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.relations: dict[str, SemanticRelation] = {}
        self.next_id = 1

    def integrate(self, head: Vector, relation: Vector, tail: Vector, confidence: float, source: str, hint: str = "") -> SemanticRelation:
        for rel in self.relations.values():
            if cosine(rel.head, head) > 0.86 and cosine(rel.tail, tail) > 0.86 and cosine(rel.relation, relation) > 0.7:
                rel.count += 1
                rel.confidence = min(0.99, 0.9 * rel.confidence + 0.1 * confidence)
                rel.head = 0.9 * rel.head + 0.1 * head
                rel.tail = 0.9 * rel.tail + 0.1 * tail
                rel.provenance.append(source)
                if hint:
                    rel.label_hint = hint
                return rel
        rid = f"sem_{self.next_id}"
        self.next_id += 1
        rel = SemanticRelation(rid, head.copy(), relation.copy(), tail.copy(), confidence, 1, [source], hint)
        self.relations[rid] = rel
        if len(self.relations) > self.capacity:
            weakest = min(self.relations.values(), key=lambda r: r.confidence * r.count)
            del self.relations[weakest.relation_id]
        return rel

    def query(self, head: Vector, k: int = 5) -> list[SemanticRelation]:
        scored = [(cosine(r.head, head) * r.confidence, r) for r in self.relations.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    def snapshot(self) -> dict:
        return {
            "next_id": self.next_id,
            "relations": {
                rid: {
                    "head": r.head,
                    "relation": r.relation,
                    "tail": r.tail,
                    "confidence": r.confidence,
                    "count": r.count,
                    "provenance": r.provenance[-12:],
                    "label_hint": r.label_hint,
                }
                for rid, r in self.relations.items()
            },
        }

    def restore(self, data: dict) -> None:
        self.next_id = int(data.get("next_id", 1))
        self.relations = {}
        for rid, blob in data.get("relations", {}).items():
            self.relations[rid] = SemanticRelation(
                rid,
                np.asarray(blob["head"], dtype=np.float64),
                np.asarray(blob["relation"], dtype=np.float64),
                np.asarray(blob["tail"], dtype=np.float64),
                float(blob["confidence"]),
                int(blob["count"]),
                list(blob.get("provenance", [])),
                str(blob.get("label_hint", "")),
            )

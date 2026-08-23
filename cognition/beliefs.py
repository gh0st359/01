"""Revisable beliefs with confidence, evidence, and contradiction."""

from __future__ import annotations

from shared.types import Belief, EvidenceSource, Vector
import numpy as np


class BeliefSystem:
    def __init__(self) -> None:
        self.beliefs: dict[str, Belief] = {}
        self.next_id = 1

    def assert_belief(
        self,
        proposition: str,
        latent: Vector,
        confidence: float,
        tick: int,
        source: EvidenceSource,
        evidence: str,
    ) -> Belief:
        for b in self.beliefs.values():
            if b.proposition == proposition:
                if source is EvidenceSource.INTERVENTION:
                    b.confidence = min(0.99, 0.7 * b.confidence + 0.3 * confidence)
                else:
                    b.confidence = min(0.99, 0.85 * b.confidence + 0.15 * confidence)
                b.supporting_evidence.append(evidence)
                b.updated_tick = tick
                b.latent = 0.85 * b.latent + 0.15 * _fit(latent, b.latent.size)
                return b
        bid = f"belief_{self.next_id}"
        self.next_id += 1
        belief = Belief(bid, proposition, latent.copy(), confidence, [evidence], [], tick, tick, source)
        self.beliefs[bid] = belief
        return belief

    def contradict(self, proposition: str, evidence: str, tick: int) -> Belief | None:
        for b in self.beliefs.values():
            if b.proposition == proposition:
                b.contradictory_evidence.append(evidence)
                b.confidence = max(0.02, b.confidence * 0.7)
                b.updated_tick = tick
                return b
        return None

    def conflict_score(self) -> float:
        if not self.beliefs:
            return 0.0
        return float(np.mean([1.0 if b.contradictory_evidence else 0.0 for b in self.beliefs.values()]))

    def snapshot(self) -> dict:
        return {
            "next_id": self.next_id,
            "beliefs": {
                bid: {
                    "proposition": b.proposition,
                    "latent": b.latent,
                    "confidence": b.confidence,
                    "supporting": b.supporting_evidence[-12:],
                    "contradictory": b.contradictory_evidence[-8:],
                    "created": b.created_tick,
                    "updated": b.updated_tick,
                    "source": b.source.value,
                }
                for bid, b in self.beliefs.items()
            },
        }

    def restore(self, data: dict) -> None:
        self.next_id = int(data.get("next_id", 1))
        self.beliefs = {}
        for bid, blob in data.get("beliefs", {}).items():
            self.beliefs[bid] = Belief(
                bid,
                blob["proposition"],
                np.asarray(blob["latent"], dtype=np.float64),
                float(blob["confidence"]),
                list(blob.get("supporting", [])),
                list(blob.get("contradictory", [])),
                int(blob["created"]),
                int(blob["updated"]),
                EvidenceSource(blob["source"]),
            )


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

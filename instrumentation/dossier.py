"""Consciousness-research evidence dossier.

Never stores a binary conscious=true flag.
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.types import MarkerObservation


MARKERS = [
    "persistent_cognition",
    "autonomous_goal_formation",
    "self_model",
    "autobiographical_continuity",
    "metacognitive_access",
    "introspective_reporting",
    "novel_self_reference",
    "internally_generated_cognition",
    "cross_modal_unity",
    "temporal_self_continuity",
    "counterfactual_self_representation",
]


class EvidenceDossier:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.observations: list[dict] = []
        if self.path.exists():
            self.observations = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, obs: MarkerObservation) -> None:
        self.observations.append(
            {
                "marker": obs.marker,
                "experiment": obs.experiment,
                "expected_alternative": obs.expected_alternative,
                "result": obs.result,
                "ablation": obs.ablation,
                "replication": obs.replication,
                "confidence": obs.confidence,
                "limitations": obs.limitations,
                "evidence_paths": obs.evidence_paths,
                "tick": obs.tick,
            }
        )
        self.path.write_text(json.dumps(self.observations, indent=2), encoding="utf-8")

    def summary(self) -> dict[str, list[dict]]:
        out = {m: [] for m in MARKERS}
        for obs in self.observations:
            out.setdefault(obs["marker"], []).append(obs)
        return out

"""Capability-gated developmental stages."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.types import DevelopmentalStage


@dataclass
class StageGate:
    stage: DevelopmentalStage
    tests: dict[str, float] = field(default_factory=dict)
    passed: bool = False


class DevelopmentRuntime:
    ORDER = [
        DevelopmentalStage.SENSORY_CONTINUITY,
        DevelopmentalStage.BODY,
        DevelopmentalStage.OBJECTS,
        DevelopmentalStage.CAUSALITY,
        DevelopmentalStage.AGENTS,
        DevelopmentalStage.COMMUNICATION,
        DevelopmentalStage.ABSTRACT,
        DevelopmentalStage.OPEN,
    ]
    THRESHOLDS = {
        DevelopmentalStage.SENSORY_CONTINUITY: {"change_detected": 0.25, "prediction_improving": 0.02},
        DevelopmentalStage.BODY: {"agency": 0.2},
        DevelopmentalStage.OBJECTS: {"permanence": 0.25, "tracking": 0.2},
        DevelopmentalStage.CAUSALITY: {"causal_confidence": 0.4},
        DevelopmentalStage.AGENTS: {"other_models": 0.5},
        DevelopmentalStage.COMMUNICATION: {"grounded_words": 6, "utterances": 1},
        DevelopmentalStage.ABSTRACT: {"semantic_relations": 8, "counterfactuals": 2},
        DevelopmentalStage.OPEN: {},
    }

    def __init__(self) -> None:
        self.stage = DevelopmentalStage.SENSORY_CONTINUITY
        self.metrics: dict[str, float] = {}
        self.history: list[dict] = []

    def record(self, **metrics: float) -> None:
        self.metrics.update(metrics)

    def maybe_advance(self, tick: int) -> DevelopmentalStage | None:
        last = None
        while True:
            needed = self.THRESHOLDS[self.stage]
            if needed and not all(self.metrics.get(k, 0.0) >= v for k, v in needed.items()):
                return last
            idx = self.ORDER.index(self.stage)
            if idx >= len(self.ORDER) - 1:
                return last
            nxt = self.ORDER[idx + 1]
            self.history.append({"tick": tick, "from": self.stage.value, "to": nxt.value, "metrics": dict(self.metrics)})
            self.stage = nxt
            last = nxt
        return last

    def snapshot(self) -> dict:
        return {"stage": self.stage.value, "metrics": self.metrics, "history": self.history}

    def restore(self, data: dict) -> None:
        self.stage = DevelopmentalStage(data["stage"])
        self.metrics = dict(data.get("metrics", {}))
        self.history = list(data.get("history", []))

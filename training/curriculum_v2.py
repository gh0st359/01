"""Competence-driven curriculum. Stages are researcher labels, not beliefs."""

from __future__ import annotations

from dataclasses import dataclass, field


STAGES = [
    "birth",
    "body",
    "objects",
    "space",
    "physics",
    "causality",
    "memory",
    "agents",
    "joint_attention",
    "symbols",
    "composition",
    "abstract",
    "open",
]


@dataclass
class CurriculumState:
    stage: str = "birth"
    mastered: list[str] = field(default_factory=list)
    failing: list[str] = field(default_factory=list)
    hardness: float = 1.0


class CurriculumManager:
    def __init__(self, pe_mastery: float = 0.08, window: int = 32) -> None:
        self.pe_mastery = pe_mastery
        self.window = window
        self.state = CurriculumState()
        self._pe: list[float] = []

    def observe(self, pe: float, competence: float) -> CurriculumState:
        self._pe.append(pe)
        recent = self._pe[-self.window :]
        if len(recent) >= self.window and (sum(recent) / len(recent)) < self.pe_mastery and competence > 0.4:
            if self.state.stage not in self.state.mastered:
                self.state.mastered.append(self.state.stage)
            idx = STAGES.index(self.state.stage) if self.state.stage in STAGES else 0
            self.state.stage = STAGES[min(idx + 1, len(STAGES) - 1)]
            self.state.hardness *= 1.15
            self._pe = []
            self.state.failing = []
        elif len(recent) >= self.window and (sum(recent) / len(recent)) > self.pe_mastery * 3:
            self.state.failing = [self.state.stage]
            self.state.hardness = max(1.0, self.state.hardness * 0.9)
        return self.state

"""Models of the organism's own cognitive reliability.

These are operational variables, not phenomenal reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.types import Vector


@dataclass
class MetaState:
    prediction_reliability: float = 0.5
    memory_reliability: float = 0.5
    skill_competence: float = 0.3
    belief_conflict: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.5
    know_enough: float = 0.5
    last_calibration_error: float = 0.0
    history: list[float] = field(default_factory=list)


class MetacognitiveMonitor:
    def __init__(self) -> None:
        self.state = MetaState()

    def update(
        self,
        prediction_error: float,
        retrieval_hit: float,
        skill_success: float,
        conflict: float,
        uncertainty: float,
    ) -> MetaState:
        s = self.state
        s.prediction_reliability = _ema(s.prediction_reliability, 1.0 / (1.0 + prediction_error), 0.08)
        s.memory_reliability = _ema(s.memory_reliability, retrieval_hit, 0.08)
        s.skill_competence = _ema(s.skill_competence, skill_success, 0.05)
        s.belief_conflict = _ema(s.belief_conflict, conflict, 0.1)
        s.uncertainty = _ema(s.uncertainty, uncertainty, 0.1)
        s.confidence = float(
            np.clip(
                0.4 * s.prediction_reliability
                + 0.25 * s.memory_reliability
                + 0.2 * s.skill_competence
                - 0.25 * s.belief_conflict
                - 0.15 * s.uncertainty,
                0.0,
                1.0,
            )
        )
        s.know_enough = float(np.clip(s.confidence - 0.5 * s.uncertainty, 0.0, 1.0))
        s.history.append(s.confidence)
        if len(s.history) > 2000:
            s.history = s.history[-1000:]
        return s

    def calibrate(self, stated_confidence: float, actual_correct: float) -> None:
        self.state.last_calibration_error = abs(stated_confidence - actual_correct)
        self.state.confidence = _ema(self.state.confidence, actual_correct, 0.05)

    def vector(self, dim: int) -> Vector:
        s = self.state
        raw = np.array(
            [
                s.prediction_reliability,
                s.memory_reliability,
                s.skill_competence,
                s.belief_conflict,
                s.confidence,
                s.uncertainty,
                s.know_enough,
                s.last_calibration_error,
            ],
            dtype=np.float64,
        )
        out = np.zeros(dim, dtype=np.float64)
        out[: min(dim, raw.size)] = raw[: min(dim, raw.size)]
        return out


def _ema(old: float, new: float, rate: float) -> float:
    return (1.0 - rate) * old + rate * new

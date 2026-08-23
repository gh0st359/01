"""Bounded internal variables that bias policy, not canned behaviors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.types import Vector


@dataclass
class HomeoState:
    energy: float = 1.0
    novelty_need: float = 0.4
    uncertainty: float = 0.5
    social_engagement: float = 0.3
    cognitive_load: float = 0.2
    prediction_error: float = 0.2
    rest_requirement: float = 0.1
    competence: float = 0.3
    environmental_stability: float = 0.6

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()

    def vector(self) -> Vector:
        return np.array(list(self.as_dict().values()), dtype=np.float64)


class Homeostasis:
    def __init__(self) -> None:
        self.state = HomeoState()
        self.setpoints = HomeoState(
            energy=0.7,
            novelty_need=0.35,
            uncertainty=0.25,
            social_engagement=0.4,
            cognitive_load=0.35,
            prediction_error=0.15,
            rest_requirement=0.2,
            competence=0.6,
            environmental_stability=0.55,
        )

    def update(
        self,
        energy: float,
        novelty: float,
        uncertainty: float,
        social: float,
        load: float,
        pe: float,
        competence: float,
        stability: float,
        leak: float,
    ) -> HomeoState:
        s = self.state
        s.energy = _blend(s.energy, energy, 0.15)
        s.novelty_need = _blend(s.novelty_need, max(0.0, self.setpoints.novelty_need - novelty), 0.08)
        s.uncertainty = _blend(s.uncertainty, uncertainty, 0.1)
        s.social_engagement = _blend(s.social_engagement, social, 0.08)
        s.cognitive_load = _blend(s.cognitive_load, load, 0.1)
        s.prediction_error = _blend(s.prediction_error, pe, 0.15)
        s.competence = _blend(s.competence, competence, 0.05)
        s.environmental_stability = _blend(s.environmental_stability, stability, 0.05)
        s.rest_requirement = float(np.clip(s.rest_requirement + leak + 0.04 * s.cognitive_load - 0.03 * (s.energy - 0.5), 0, 1))
        return s

    def drives(self) -> dict[str, float]:
        s, t = self.state, self.setpoints
        return {
            "energy": t.energy - s.energy,
            "novelty": s.novelty_need,
            "uncertainty_reduction": s.uncertainty - t.uncertainty,
            "social": t.social_engagement - s.social_engagement,
            "rest": s.rest_requirement - t.rest_requirement,
            "competence": t.competence - s.competence,
            "stability": t.environmental_stability - s.environmental_stability,
        }


def _blend(old: float, new: float, rate: float) -> float:
    return float(np.clip((1.0 - rate) * old + rate * new, 0.0, 1.0))

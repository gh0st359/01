"""Adaptive intrinsic motivation combining several experimental signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motivation.curiosity import Curiosity
from motivation.homeostasis import Homeostasis


@dataclass
class MotiveSnapshot:
    prediction_error: float
    learning_progress: float
    novelty: float
    uncertainty_reduction: float
    competence_gain: float
    social: float
    contradiction: float
    total: float
    weights: dict[str, float]


class IntrinsicMotivation:
    def __init__(self) -> None:
        self.curiosity = Curiosity()
        self.weights = {
            "prediction_error": 0.18,
            "learning_progress": 0.22,
            "novelty": 0.12,
            "uncertainty_reduction": 0.16,
            "competence_gain": 0.12,
            "social": 0.10,
            "contradiction": 0.10,
        }
        self.last = MotiveSnapshot(0, 0, 0, 0, 0, 0, 0, 0, dict(self.weights))
        self.prev_competence = 0.3
        self.prev_uncertainty = 0.5

    def evaluate(
        self,
        prediction_error: float,
        novelty: float,
        uncertainty: float,
        competence: float,
        social: float,
        contradiction: float,
        homeo: Homeostasis,
    ) -> MotiveSnapshot:
        progress = self.curiosity.progress(prediction_error)
        # adaptive: if novelty is chronically high without progress, downweight novelty
        if novelty > 0.7 and progress < 0.02:
            self.weights["novelty"] = max(0.03, self.weights["novelty"] * 0.98)
            self.weights["learning_progress"] = min(0.4, self.weights["learning_progress"] * 1.01)
        elif progress > 0.05:
            self.weights["novelty"] = min(0.2, self.weights["novelty"] * 1.01)
        ureduce = max(0.0, self.prev_uncertainty - uncertainty)
        cgain = max(0.0, competence - self.prev_competence)
        drives = homeo.drives()
        components = {
            "prediction_error": prediction_error,
            "learning_progress": progress,
            "novelty": novelty,
            "uncertainty_reduction": ureduce + max(0.0, drives["uncertainty_reduction"]),
            "competence_gain": cgain,
            "social": social + max(0.0, drives["social"]),
            "contradiction": contradiction,
        }
        total = sum(self.weights[k] * components[k] for k in components)
        # energy and rest modulate total
        total *= 0.5 + 0.5 * homeo.state.energy
        total *= 1.0 - 0.35 * homeo.state.rest_requirement
        self.prev_competence = competence
        self.prev_uncertainty = uncertainty
        self.last = MotiveSnapshot(
            prediction_error,
            progress,
            novelty,
            ureduce,
            cgain,
            social,
            contradiction,
            float(total),
            dict(self.weights),
        )
        return self.last

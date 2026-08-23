"""Distinguish external events from changes caused by the organism's actions."""

from __future__ import annotations

import numpy as np

from shared.types import Vector


class AgencyModel:
    def __init__(self) -> None:
        self.ownership = 0.2
        self.history: list[float] = []

    def update(self, motor_norm: float, predicted_change: float, actual_change: float) -> float:
        if motor_norm < 0.05:
            owned = 0.0 if actual_change > 0.02 else 0.4
        else:
            match = 1.0 / (1.0 + abs(predicted_change - actual_change))
            owned = match * min(1.0, motor_norm)
        self.ownership = 0.88 * self.ownership + 0.12 * owned
        self.history.append(self.ownership)
        if len(self.history) > 500:
            self.history = self.history[-250:]
        return self.ownership

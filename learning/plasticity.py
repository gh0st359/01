"""Neuromodulated plasticity: surprise scales local learning rates."""

from __future__ import annotations

import numpy as np


def neuromodulator(prediction_error: float, reward: float, uncertainty: float, scale: float = 1.5) -> float:
    surprise = np.tanh(prediction_error)
    val = np.tanh(abs(reward))
    u = np.tanh(uncertainty)
    return float(np.clip(0.25 + scale * (0.5 * surprise + 0.3 * val + 0.2 * u), 0.05, 4.0))


def adaptive_lr(base: float, progress: float, saturation: float) -> float:
    """Reduce learning when progress stalls; raise when competence is still rising."""
    return float(base * (0.4 + 0.6 * np.clip(progress, 0.0, 1.5)) * (1.0 - 0.5 * np.clip(saturation, 0.0, 1.0)))

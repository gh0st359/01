"""Reflection revisits surprising, conflicting, or high-value traces and changes state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.types import Belief, Episode, Vector


@dataclass
class ReflectionResult:
    belief_ids_revised: list[str]
    replayed: list[str]
    contradiction_score: float
    state_delta_norm: float


class ReflectionProcess:
    def revisit(
        self,
        episodes: list[Episode],
        beliefs: list[Belief],
        core_state: Vector,
    ) -> tuple[Vector, ReflectionResult]:
        if not episodes:
            return core_state, ReflectionResult([], [], 0.0, 0.0)
        surprising = sorted(episodes, key=lambda e: e.prediction_error + e.salience, reverse=True)[:6]
        conflict = 0.0
        revised: list[str] = []
        for belief in beliefs:
            if belief.contradictory_evidence:
                belief.confidence = max(0.02, belief.confidence * 0.92)
                revised.append(belief.belief_id)
                conflict += 1.0 - belief.confidence
        mix = np.mean([e.compression for e in surprising], axis=0)
        mix = _fit(mix, core_state.size)
        nxt = 0.92 * core_state + 0.08 * np.tanh(mix)
        delta = float(np.linalg.norm(nxt - core_state))
        return nxt, ReflectionResult(revised, [e.episode_id for e in surprising], conflict, delta)


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

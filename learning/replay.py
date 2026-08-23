"""Replay policies used during consolidation."""

from __future__ import annotations

from enum import Enum

import numpy as np

from shared.rng import RNG
from shared.types import Episode


class ReplayPolicy(str, Enum):
    PRIORITIZED = "prioritized"
    NOVELTY = "novelty"
    SURPRISE = "surprise"
    RANDOM = "random"
    GOAL_RELEVANT = "goal_relevant"
    UNCERTAINTY = "uncertainty"


def score_episode(episode: Episode, policy: ReplayPolicy, active_goal: np.ndarray | None = None) -> float:
    if policy is ReplayPolicy.RANDOM:
        return 1.0
    if policy is ReplayPolicy.SURPRISE:
        return episode.prediction_error
    if policy is ReplayPolicy.NOVELTY:
        return episode.salience + episode.uncertainty
    if policy is ReplayPolicy.UNCERTAINTY:
        return episode.uncertainty
    if policy is ReplayPolicy.GOAL_RELEVANT and active_goal is not None:
        return float(np.dot(episode.workspace_summary, active_goal) / (np.linalg.norm(active_goal) + 1e-8))
    return episode.prediction_error + 0.3 * abs(episode.value_delta) + 0.2 * episode.salience


def sample_replay(
    episodes: list[Episode],
    policy: ReplayPolicy,
    rng: RNG,
    batch: int,
    active_goal: np.ndarray | None = None,
) -> list[Episode]:
    if not episodes:
        return []
    batch = min(batch, len(episodes))
    scores = np.array([max(score_episode(ep, policy, active_goal), 1e-6) for ep in episodes], dtype=np.float64)
    probs = scores / scores.sum()
    idxs = [rng.choice(len(episodes), p=probs) for _ in range(batch)]
    return [episodes[i] for i in idxs]

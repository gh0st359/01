"""Episode → semantic generalization."""

from __future__ import annotations

import numpy as np

from memory.semantic.store import SemanticMemory
from shared.types import Episode


def generalize_episode(semantic: SemanticMemory, episode: Episode) -> None:
    rel = np.zeros_like(episode.compression)
    rel[: min(4, rel.size)] = np.array([episode.prediction_error, episode.value_delta, episode.salience, episode.uncertainty])[: min(4, rel.size)]
    semantic.integrate(episode.perception, rel, episode.world_state_after, 0.4 + 0.3 * min(episode.salience, 1.0), episode.episode_id)

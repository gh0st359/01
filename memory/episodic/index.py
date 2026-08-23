"""Vector index helpers for episodic retrieval."""

from __future__ import annotations

import numpy as np

from learning.nn import cosine
from shared.types import Episode, Vector


def rank_by_vector(episodes: list[Episode], query: Vector, k: int) -> list[Episode]:
    scored = []
    for ep in episodes:
        scored.append((cosine(ep.compression[: query.size], query), ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]

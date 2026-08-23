from __future__ import annotations

import numpy as np

from learning.nn import cosine


def brute_knn(query: np.ndarray, items: list[np.ndarray], k: int) -> list[int]:
    if not items:
        return []
    scores = [cosine(query, it) for it in items]
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:k]]

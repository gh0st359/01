"""Learning-progress curiosity — not naive novelty maximization."""

from __future__ import annotations

from collections import deque


class Curiosity:
    def __init__(self, window: int = 40) -> None:
        self.errors: deque[float] = deque(maxlen=window)
        self.familiar: dict[str, float] = {}

    def progress(self, error: float) -> float:
        self.errors.append(error)
        if len(self.errors) < 8:
            return 0.2
        mid = len(self.errors) // 2
        early = sum(list(self.errors)[:mid]) / mid
        late = sum(list(self.errors)[mid:]) / (len(self.errors) - mid)
        # positive progress = error decreasing
        return float(max(0.0, early - late))

    def novelty(self, key: str, current: float) -> float:
        prev = self.familiar.get(key, 0.0)
        self.familiar[key] = 0.9 * prev + 0.1
        return float(max(0.0, current * (1.0 - self.familiar[key])))

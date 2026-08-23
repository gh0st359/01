"""Central deterministic RNG. All stochasticity must go through here."""

from __future__ import annotations

import numpy as np


class RNG:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._gen = np.random.default_rng(self.seed)
        self.calls = 0

    def spawn(self, salt: int) -> "RNG":
        child_seed = int(self._gen.integers(0, 2**31 - 1)) ^ int(salt)
        return RNG(child_seed)

    def normal(self, size: int | tuple[int, ...] , scale: float = 1.0, loc: float = 0.0) -> np.ndarray:
        self.calls += 1
        return self._gen.normal(loc=loc, scale=scale, size=size)

    def uniform(self, low: float, high: float, size: int | tuple[int, ...] | None = None) -> np.ndarray:
        self.calls += 1
        return self._gen.uniform(low, high, size=size)

    def random(self) -> float:
        self.calls += 1
        return float(self._gen.random())

    def integers(self, low: int, high: int) -> int:
        self.calls += 1
        return int(self._gen.integers(low, high))

    def choice(self, n: int, p: np.ndarray | None = None) -> int:
        self.calls += 1
        return int(self._gen.choice(n, p=p))

    def shuffle(self, items: list) -> None:
        self.calls += 1
        self._gen.shuffle(items)

    def permutation(self, n: int) -> np.ndarray:
        self.calls += 1
        return self._gen.permutation(n)

    def state_dict(self) -> dict:
        return {"seed": self.seed, "calls": self.calls, "bitgen": self._gen.bit_generator.state}

    def load_state(self, data: dict) -> None:
        self.seed = int(data["seed"])
        self.calls = int(data["calls"])
        self._gen = np.random.default_rng()
        self._gen.bit_generator.state = data["bitgen"]

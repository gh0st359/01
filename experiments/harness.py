"""Shared experiment harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from apps.organism_runtime.session import OrganismSession
from shared.v2config import load_v2_config


@dataclass
class ExperimentResult:
    name: str
    passed: bool
    score: float
    details: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


def make_session(seed: int = 7, steps_warmup: int = 0) -> OrganismSession:
    cfg = load_v2_config("ci", seed=seed)
    cfg.checkpoint_every = 10**9
    session = OrganismSession(cfg, Path(cfg.run_dir) / f"exp_{seed}_{id(cfg)}")
    for _ in range(steps_warmup):
        session.step()
    return session


def run_ticks(session: OrganismSession, n: int) -> None:
    for _ in range(n):
        session.step()

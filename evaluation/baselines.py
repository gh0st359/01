"""Scientific baselines for attribution."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.harness import make_session, run_ticks


@dataclass
class BaselineScore:
    name: str
    pe: float
    goals: int
    utterances: int


def _score(name: str, disable: list[str]) -> BaselineScore:
    session = make_session(hash(name) % 900 + 10, 5)
    for d in disable:
        session.organism.disable(d)
    run_ticks(session, 40)
    pe = session.organism.world_model.last.error
    goals = len(session.organism.goals.goals)
    utt = sum(1 for t in session.organism.dialogue.turns if t.speaker == "01")
    session.close()
    return BaselineScore(name, pe, goals, utt)


def run_baselines() -> list[BaselineScore]:
    return [
        _score("full", []),
        _score("reactive", ["episodic", "imagination", "goals", "world_model"]),
        _score("recurrent_no_memory", ["episodic"]),
        _score("world_no_self", []),  # self-model always present; we perturb via disable others+auto by wiping
        _score("no_autobiography", ["episodic"]),
        _score("no_intrinsic", ["goals"]),
        _score("no_consolidation", ["consolidation"]),
        _score("no_imagination", ["imagination"]),
    ]

"""Fast acquisition vs slow integration, with multiple replay policies."""

from __future__ import annotations

from dataclasses import dataclass

from learning.replay import ReplayPolicy, sample_replay
from memory.episodic.store import EpisodicMemory
from memory.semantic.consolidation import generalize_episode
from memory.semantic.store import SemanticMemory
from shared.rng import RNG
from world_model.predictive import PredictiveWorldModel


@dataclass
class ConsolidationReport:
    replayed: int
    policy: str
    mean_error: float
    semantic_added: int


class ConsolidationEngine:
    def __init__(self) -> None:
        self.policies = list(ReplayPolicy)
        self.last = ConsolidationReport(0, ReplayPolicy.PRIORITIZED.value, 0.0, 0)

    def run(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        world_model: PredictiveWorldModel,
        rng: RNG,
        batch: int,
        policy: ReplayPolicy | None = None,
        neuromod: float = 1.0,
    ) -> ConsolidationReport:
        policy = policy or ReplayPolicy.PRIORITIZED
        episodes = sample_replay(episodic.all_cached(), policy, rng, batch)
        errors = []
        added = 0
        before_sem = len(semantic.relations)
        for ep in episodes:
            loss = world_model.train_step(ep.perception, ep.action, ep.world_state_after, neuromod=neuromod)
            errors.append(loss)
            generalize_episode(semantic, ep)
        added = len(semantic.relations) - before_sem
        report = ConsolidationReport(len(episodes), policy.value, float(sum(errors) / len(errors)) if errors else 0.0, added)
        self.last = report
        return report

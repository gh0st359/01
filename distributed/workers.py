"""Parallel experience for transferable low-level models, not autobiographical clones."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from shared.config import OrganismConfig
from shared.rng import RNG
from world_model.predictive import PredictiveWorldModel


class ParallelExperience:
    def __init__(self, workers: int = 2) -> None:
        self.workers = max(1, workers)

    def train_world_models(self, cfg: OrganismConfig, seeds: list[int], steps: int = 40) -> list[float]:
        def run(seed: int) -> float:
            from simulation.world import SimulatedWorld
            from embodiment.body import EmbodiedBody
            from planning.action import primitive_vector
            from shared.types import ActionKind

            rng = RNG(seed)
            world = SimulatedWorld(cfg, rng)
            body = EmbodiedBody(cfg)
            wm = PredictiveWorldModel(cfg, rng)
            losses = []
            kinds = list(ActionKind)
            obs = world.observe()
            prev = obs.sensors.vision[: cfg.feature_dim]
            prev = np.pad(prev, (0, max(0, cfg.feature_dim - prev.size)))[: cfg.feature_dim]
            for i in range(steps):
                kind = kinds[i % len(kinds)]
                act = primitive_vector(kind, cfg.action_dim)
                nxt = world.step(body.decode(act, kind, i))
                feat = nxt.sensors.vision
                feat = np.pad(feat, (0, max(0, cfg.feature_dim - feat.size)))[: cfg.feature_dim]
                losses.append(wm.train_step(prev, act, feat))
                prev = feat
            return float(np.mean(losses[-10:])) if losses else 0.0

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(run, seeds))

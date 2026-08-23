"""Empirical comparison of GRU, CfC, and reservoir cores on prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cognition.core.multiscale import MultiScaleCore, SubstrateKind
from embodiment.body import EmbodiedBody
from planning.action import primitive_vector
from shared.config import load_config
from shared.rng import RNG
from shared.types import ActionKind
from simulation.world import SimulatedWorld


def eval_kind(kind: SubstrateKind, steps: int = 80) -> float:
    cfg = load_config("development", seed=3)
    rng = RNG(3)
    world = SimulatedWorld(cfg, rng)
    body = EmbodiedBody(cfg)
    core = MultiScaleCore(cfg, rng, kind)
    errors = []
    prev = None
    kinds = [ActionKind.MOVE, ActionKind.TURN, ActionKind.LOOK, ActionKind.WAIT]
    for i in range(steps):
        obs = world.observe()
        feat = obs.sensors.vision.astype(np.float64)
        feat = np.pad(feat, (0, max(0, cfg.feature_dim - feat.size)))[: cfg.feature_dim]
        pred = core.step(feat, cfg.dt)
        if prev is not None:
            errors.append(float(np.mean((pred[: min(pred.size, prev.size)] - prev[: min(pred.size, prev.size)]) ** 2)))
        prev = feat
        k = kinds[i % len(kinds)]
        world.step(body.decode(primitive_vector(k, cfg.action_dim), k, i))
    return float(np.mean(errors[-20:])) if errors else 1.0


def main() -> None:
    out = {k.value: eval_kind(k) for k in SubstrateKind}
    Path("runs").mkdir(exist_ok=True)
    Path("runs/substrate_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

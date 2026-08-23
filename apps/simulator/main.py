"""Standalone simulator stepping for throughput tests."""

from __future__ import annotations

import argparse

from embodiment.body import EmbodiedBody
from planning.action import primitive_vector
from shared.config import load_config
from shared.rng import RNG
from shared.types import ActionKind
from simulation.world import SimulatedWorld


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    cfg = load_config("development", seed=args.seed)
    rng = RNG(args.seed)
    world = SimulatedWorld(cfg, rng)
    body = EmbodiedBody(cfg)
    kinds = list(ActionKind)
    for i in range(args.steps):
        kind = kinds[i % len(kinds)]
        world.step(body.decode(primitive_vector(kind, cfg.action_dim), kind, i))
    print(f"simulated {args.steps} steps, objects={len(world.objects)}")


if __name__ == "__main__":
    main()

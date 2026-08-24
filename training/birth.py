"""Initialize a newborn organism and write a birth checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.checkpoint import save_checkpoint_v2
from core.device import resolve_device
from core.organism import OrganismV2
from shared.serialize import write_json
from shared.v2config import load_v2_config
from simulation.batch import ProceduralBatch


def birth(profile: str, seed: int, run_dir: Path, steps: int = 8) -> Path:
    cfg = load_v2_config(profile, seed=seed)
    cfg.run_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    org = OrganismV2(cfg, run_dir / "organism", device=resolve_device(amp=cfg.mixed_precision))
    world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, seed)
    for _ in range(steps):
        obs = world.observation(0)
        result = org.tick_once(obs.sensors)
        world.step([result.motor])
    path = save_checkpoint_v2(org, run_dir / "checkpoints" / "birth")
    write_json(
        run_dir / "birth.json",
        {
            "seed": seed,
            "profile": profile,
            "params": org.param_count,
            "device": org.device_bundle.kind,
            "ticks": org.tick,
            "pe": org.pe_trace,
        },
    )
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="ci")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-dir", default="runs/v2/birth")
    p.add_argument("--steps", type=int, default=8)
    args = p.parse_args(argv)
    path = birth(args.profile, args.seed, Path(args.run_dir), args.steps)
    print(path)


if __name__ == "__main__":
    main()

"""Repeated run → checkpoint → exit-equivalent restore → continue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.checkpoint import load_checkpoint_v2, save_checkpoint_v2
from core.device import resolve_device
from core.organism import OrganismV2
from shared.serialize import write_json
from shared.v2config import load_v2_config
from simulation.batch import ProceduralBatch


def run_segment(org: OrganismV2, world: ProceduralBatch, steps: int) -> None:
    for _ in range(steps):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])


def longitudinal(profile: str, seed: int, run_dir: Path, cycles: int, steps: int) -> dict:
    cfg = load_v2_config(profile, seed=seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    org = OrganismV2(cfg, run_dir / "organism", device=resolve_device())
    world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, seed)
    history = []
    for c in range(cycles):
        run_segment(org, world, steps)
        ckpt = save_checkpoint_v2(org, run_dir / "checkpoints" / f"cycle_{c}")
        tick = org.tick
        ident = org.self_h.detach().clone()
        episodes = len(org.memory.episodes)
        # Simulate process exit by constructing a fresh organism and restoring
        nxt = OrganismV2(cfg, run_dir / f"organism_r{c}", device=resolve_device())
        load_checkpoint_v2(nxt, ckpt)
        ok = nxt.tick == tick
        org = nxt
        history.append({"cycle": c, "tick": tick, "restore_ok": ok, "episodes_at_save": episodes, "ident_norm": float(ident.norm())})
        world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, seed + c + 1)
    report = {"cycles": cycles, "final_tick": org.tick, "history": history, "all_ok": all(h["restore_ok"] for h in history)}
    write_json(run_dir / "longitudinal.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="ci")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--run-dir", default="runs/v2/longitudinal")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--steps", type=int, default=16)
    args = p.parse_args(argv)
    report = longitudinal(args.profile, args.seed, Path(args.run_dir), args.cycles, args.steps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

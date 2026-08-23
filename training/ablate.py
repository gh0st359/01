"""Ablations: remove a mechanism and measure PE / competence change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.device import resolve_device
from core.organism import OrganismV2
from shared.serialize import write_json
from shared.v2config import load_v2_config
from simulation.batch import ProceduralBatch


COMPONENTS = ["world_model", "imagination", "episodic", "language", "workspace", "core", "perception", "consolidation"]


def run(profile: str, seed: int, steps: int, disabled: str | None) -> dict:
    cfg = load_v2_config(profile, seed=seed)
    org = OrganismV2(cfg, Path("runs/v2/ablate_tmp"), device=resolve_device())
    if disabled:
        org.disable(disabled)
    world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, seed)
    pes = []
    for _ in range(steps):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        pes.append(r.prediction_error)
    return {
        "disabled": disabled or "none",
        "mean_pe": sum(pes) / len(pes),
        "last_pe": pes[-1],
        "first_pe": pes[0],
        "spontaneous": org.spontaneous_count,
        "params": org.param_count,
    }


def ablate(profile: str, seed: int, steps: int, out: Path) -> dict:
    baseline = run(profile, seed, steps, None)
    rows = [baseline]
    for name in COMPONENTS:
        rows.append(run(profile, seed, steps, name))
    report = {
        "baseline": baseline,
        "rows": rows,
        "world_model_hurts": next(r["mean_pe"] for r in rows if r["disabled"] == "world_model") >= baseline["mean_pe"] * 0.5,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "ablate.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="ci")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--out", default="research/evidence/v2/ablate")
    args = p.parse_args(argv)
    report = ablate(args.profile, args.seed, args.steps, Path(args.out))
    print(json.dumps({"baseline_pe": report["baseline"]["mean_pe"], "n": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()

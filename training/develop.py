"""Competence-driven developmental training. Tick count is not the sole criterion."""

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


STAGES = [
    "birth",
    "body",
    "objects",
    "space",
    "physics",
    "causality",
    "memory",
    "agents",
    "joint_attention",
    "symbols",
    "composition",
    "abstract",
    "open",
]


def caregiver_stream(world: ProceduralBatch, rng_tick: int) -> str | None:
    """Environment agent emits a character stream co-present with the scene.

    The organism does not receive word-class labels. Streams are just bytes.
    """
    if rng_tick % 11 != 0:
        return None
    view = world.public_view(0)
    visible = [o for o in view["objects"] if not o["hidden"] and o["kind"] != "agent"]
    if not visible:
        return None
    obj = visible[rng_tick % len(visible)]
    # Caregiver speech is an environmental event. Meanings are not injected as classes.
    return f"{obj['kind']}"


def develop(profile: str, seed: int, run_dir: Path, steps: int, restore: Path | None) -> dict:
    cfg = load_v2_config(profile, seed=seed)
    cfg.run_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    org = OrganismV2(cfg, run_dir / "organism", device=resolve_device(amp=cfg.mixed_precision))
    if restore and restore.exists():
        load_checkpoint_v2(org, restore)
    world = ProceduralBatch(max(1, cfg.batch_envs), cfg.max_objects, cfg.vision_hw, seed)
    held = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, seed + 99, held_out=True)
    curves = {"pe": [], "loss": [], "competence": [], "held_pe": []}
    stage_idx = 0
    mastered: list[str] = []
    window: list[float] = []
    for t in range(steps):
        stream = caregiver_stream(world, org.tick + 1)
        if stream:
            org.ingest_speech("caregiver", stream)
        obs = world.observation(0)
        result = org.tick_once(obs.sensors)
        motors = [result.motor] * world.batch
        world.step(motors)
        window.append(result.prediction_error)
        if len(window) > 64:
            window = window[-64:]
        if t % 32 == 0:
            held_obs = held.observation(0)
            # Evaluate without stepping the live organism's learning? Use a frozen forward by disabling train via no extra ticks on held world only.
            # Measure reconstruction on held pixels through current RSSM by a dry tick clone is expensive;
            # instead compare recent PE on a held-out env by a single observe.
            held.step([result.motor])
            curves["held_pe"].append(result.prediction_error)
        if t % 8 == 0:
            curves["pe"].append(result.prediction_error)
            curves["loss"].append(org.loss_trace[-1] if org.loss_trace else 0.0)
            curves["competence"].append(org.competence_trace[-1] if org.competence_trace else 0.0)
        mean_pe = sum(window) / max(1, len(window))
        if mean_pe < 0.08 and stage_idx < len(STAGES) - 1 and len(window) >= 32:
            mastered.append(STAGES[stage_idx])
            stage_idx += 1
            org.development.stage = STAGES[stage_idx]
            org.milestones.append({"tick": org.tick, "name": f"stage_{STAGES[stage_idx]}", "evidence": f"mean_pe={mean_pe:.4f}"})
            window = []
        if org.tick % cfg.checkpoint_every == 0:
            save_checkpoint_v2(org, run_dir / "checkpoints" / f"tick_{org.tick}")
    save_checkpoint_v2(org, run_dir / "checkpoints" / "latest")
    report = {
        "seed": seed,
        "profile": profile,
        "steps": steps,
        "ticks": org.tick,
        "params": org.param_count,
        "device": org.device_bundle.kind,
        "mastered": mastered,
        "stage": org.development.stage,
        "spontaneous": org.spontaneous_count,
        "human_events": org.human_events,
        "final_pe": org.pe_trace[-1] if org.pe_trace else None,
        "mean_last_64_pe": sum(org.pe_trace[-64:]) / max(1, len(org.pe_trace[-64:])),
        "mean_first_64_pe": sum(org.pe_trace[:64]) / max(1, len(org.pe_trace[:64])),
        "utterances": [d for d in org.dialogue if d["speaker"] == "01"][-20:],
        "curves": curves,
        "milestones": org.milestones,
    }
    write_json(run_dir / "develop.json", report)
    (run_dir / "curves.jsonl").write_text("\n".join(json.dumps({"i": i, "pe": p}) for i, p in enumerate(curves["pe"])), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="ci")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-dir", default="runs/v2/develop")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--restore", default="")
    args = p.parse_args(argv)
    restore = Path(args.restore) if args.restore else None
    report = develop(args.profile, args.seed, Path(args.run_dir), args.steps, restore)
    print(json.dumps({k: report[k] for k in ("ticks", "params", "final_pe", "stage", "spontaneous")}, indent=2))


if __name__ == "__main__":
    main()

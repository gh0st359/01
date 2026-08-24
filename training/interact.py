"""Ordinary conversation into a restored organism. Not a personality script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.checkpoint import load_checkpoint_v2
from core.device import resolve_device
from core.organism import OrganismV2
from shared.serialize import write_json
from shared.v2config import load_v2_config
from simulation.batch import ProceduralBatch


LINES = [
    "disk",
    "what is near",
    "tool",
    "look again",
    "switch",
]


def interact(profile: str, restore: Path, out: Path, idle: int = 12) -> dict:
    cfg = load_v2_config(profile, seed=1)
    org = OrganismV2(cfg, out / "organism", device=resolve_device())
    if restore.exists():
        load_checkpoint_v2(org, restore)
    world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, 7)
    log = []
    for line in LINES:
        org.ingest_speech("human", line)
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        log.append(
            {
                "human": line,
                "tick": r.tick,
                "utterance": r.utterance,
                "spontaneous": r.spontaneous,
                "pe": r.prediction_error,
                "intent_urgency": r.intent.urgency if r.intent else None,
                "prior_uncertainty": r.intent.preceding_uncertainty if r.intent else None,
                "retrievals": r.intent.preceding_retrieval_ids if r.intent else [],
                "semantic_exists": r.intent is not None,
            }
        )
        for _ in range(2):
            r = org.tick_once(world.observation(0).sensors)
            world.step([r.motor])
            if r.utterance:
                log.append(
                    {
                        "human": None,
                        "tick": r.tick,
                        "utterance": r.utterance,
                        "spontaneous": r.spontaneous,
                        "pe": r.prediction_error,
                        "semantic_exists": r.intent is not None,
                    }
                )
    for _ in range(idle):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        if r.utterance:
            log.append(
                {
                    "human": None,
                    "tick": r.tick,
                    "utterance": r.utterance,
                    "spontaneous": r.spontaneous,
                    "pe": r.prediction_error,
                    "semantic_exists": r.intent is not None,
                    "idle": True,
                }
            )
    report = {"n": len(log), "spontaneous": org.spontaneous_count, "log": log, "tick": org.tick}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "conversation.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="ci")
    p.add_argument("--checkpoint", default="runs/v2/ind1/checkpoints/latest")
    p.add_argument("--out", default="research/evidence/v2/conversation")
    args = p.parse_args(argv)
    report = interact(args.profile, Path(args.checkpoint), Path(args.out))
    print(json.dumps({"n": report["n"], "spontaneous": report["spontaneous"]}, indent=2))


if __name__ == "__main__":
    main()

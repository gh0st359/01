"""Complete developmental checkpoint and restore."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from organism.loop import Organism
from shared.serialize import read_json, write_json


SCHEMA = 1


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/workspace", text=True).strip()
    except Exception:
        return "unknown"


def save_checkpoint(org: Organism, path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    meta = {
        "schema": SCHEMA,
        "tick": org.tick,
        "time": org.time,
        "name": org.name,
        "lineage": org.lineage_id,
        "parent": org.parent_lineage,
        "seed": org.cfg.seed,
        "config": org.cfg.to_dict(),
        "git": git_revision(),
        "rng": org.rng.state_dict(),
    }
    write_json(path / "meta.json", meta)
    np.savez_compressed(
        path / "weights.npz",
        **{f"perc_{k}": v["value"] if isinstance(v, dict) else v for k, v in _flat(org.perception.state_dict()).items()},
        **{f"wm_{k}": v["value"] if isinstance(v, dict) else v for k, v in _flat(org.world_model.state_dict()).items()},
        **{f"core_{k}": v["value"] if isinstance(v, dict) else v for k, v in _flat(org.core.state_dict()).items()},
        **{f"ws_{k}": v["value"] if isinstance(v, dict) else v for k, v in _flat(org.workspace.state_dict()).items()},
        **{f"parse_{k}": v["value"] if isinstance(v, dict) else v for k, v in _flat(org.comprehend.state_dict()).items()},
        **{f"fam_{k}": v["value"] if isinstance(v, dict) else v for k, v in _flat(org.familiar.state_dict()).items()},
        last_features=org.last_features,
        last_action=org.last_action,
        core_fast=org.core.h_fast,
        core_medium=org.core.h_medium,
        core_slow=org.core.h_slow,
    )
    write_json(path / "entities.json", org.entities.snapshot())
    write_json(path / "semantic.json", org.semantic.snapshot())
    write_json(path / "skills.json", org.procedural.snapshot())
    write_json(path / "auto.json", org.auto.snapshot())
    write_json(path / "beliefs.json", org.beliefs.snapshot())
    write_json(path / "goals.json", org.goals.snapshot())
    write_json(path / "lexicon.json", org.lexicon.snapshot())
    write_json(path / "concepts.json", org.concepts.snapshot())
    write_json(path / "self.json", org.self_model.snapshot())
    write_json(path / "causal.json", org.causal.snapshot())
    write_json(path / "development.json", org.development.snapshot())
    write_json(path / "working.json", org.working.snapshot())
    write_json(path / "associative.json", org.associative.snapshot())
    write_json(path / "preferences.json", org.preferences.snapshot())
    write_json(path / "dialogue.json", org.dialogue.snapshot())
    write_json(
        path / "runtime.json",
        {
            "tick": org.tick,
            "time": org.time,
            "last_kind": org.last_kind.value,
            "spontaneous": org.spontaneous_count,
            "human_events": org.human_events,
            "hidden_probe": org.hidden_probe,
            "disabled": list(org.disabled),
        },
    )
    epi_src = org.data_dir / "episodic.sqlite"
    if epi_src.exists():
        shutil.copy2(epi_src, path / "episodic.sqlite")
    return path


def load_checkpoint(org: Organism, path: str | Path) -> Organism:
    path = Path(path)
    meta = read_json(path / "meta.json")
    if int(meta.get("schema", 1)) > SCHEMA:
        raise ValueError("checkpoint schema newer than runtime")
    org.rng.load_state(meta["rng"])
    org.tick = int(meta["tick"])
    org.time = float(meta["time"])
    org.lineage_id = meta.get("lineage", org.lineage_id)
    org.parent_lineage = meta.get("parent")
    weights = np.load(path / "weights.npz", allow_pickle=False)
    org.last_features = weights["last_features"]
    org.last_action = weights["last_action"]
    org.core.h_fast = weights["core_fast"]
    org.core.h_medium = weights["core_medium"]
    org.core.h_slow = weights["core_slow"]
    org.entities.restore(read_json(path / "entities.json"))
    org.semantic.restore(read_json(path / "semantic.json"))
    org.procedural.restore(read_json(path / "skills.json"))
    org.auto.restore(read_json(path / "auto.json"))
    org.beliefs.restore(read_json(path / "beliefs.json"))
    org.goals.restore(read_json(path / "goals.json"))
    org.lexicon.restore(read_json(path / "lexicon.json"))
    org.concepts.restore(read_json(path / "concepts.json"))
    org.self_model.restore(read_json(path / "self.json"))
    org.causal.restore(read_json(path / "causal.json"))
    org.development.restore(read_json(path / "development.json"))
    org.working.restore(read_json(path / "working.json"))
    org.associative.restore(read_json(path / "associative.json"))
    org.preferences.restore(read_json(path / "preferences.json"))
    runtime = read_json(path / "runtime.json")
    org.spontaneous_count = int(runtime.get("spontaneous", 0))
    org.human_events = int(runtime.get("human_events", 0))
    org.hidden_probe = dict(runtime.get("hidden_probe", {}))
    org.disabled = set(runtime.get("disabled", []))
    epi = path / "episodic.sqlite"
    if epi.exists():
        dest = org.data_dir / "episodic.sqlite"
        org.episodic.close()
        shutil.copy2(epi, dest)
        from memory.episodic.store import EpisodicMemory

        org.episodic = EpisodicMemory(org.cfg, dest)
    return org


def _flat(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = k.replace(".", "_")
        if isinstance(v, dict) and "value" in v:
            out[key] = v["value"]
        else:
            out[key] = v
    return out

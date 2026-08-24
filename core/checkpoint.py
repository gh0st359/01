"""Schema-2 checkpoints: torch weights + autobiographical memory + RNG."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from core.organism import OrganismV2
from shared.serialize import read_json, write_json


SCHEMA = 2


def save_checkpoint_v2(org: OrganismV2, path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    write_json(
        path / "meta.json",
        {
            "schema": SCHEMA,
            "tick": org.tick,
            "time": org.time,
            "name": org.name,
            "lineage": org.lineage_id,
            "parent": org.parent_lineage,
            "seed": org.cfg.seed,
            "config": org.cfg.to_dict(),
            "params": org.param_count,
            "device": org.device_bundle.kind,
        },
    )
    torch.save(
        {
            "net": org.net.state_dict(),
            "opt": org.opt.state_dict(),
            "rssm_h": org.rssm_state.h.cpu(),
            "rssm_z": org.rssm_state.z.cpu(),
            "core": {k: getattr(org.core_state, k).cpu() for k in ("fast", "act", "work", "goal", "motive", "auto")},
            "self_h": org.self_h.cpu(),
            "other_h": org.other_h.cpu(),
            "ws_prev": org.ws_prev.cpu(),
            "last_action": org.last_action.cpu(),
            "homeo": org.homeo.vector().cpu(),
            "tick": org.tick,
        },
        path / "weights.pt",
    )
    write_json(
        path / "runtime.json",
        {
            "tick": org.tick,
            "time": org.time,
            "spontaneous": org.spontaneous_count,
            "human_events": org.human_events,
            "disabled": list(org.disabled),
            "pe_trace": org.pe_trace[-200:],
            "loss_trace": org.loss_trace[-200:],
            "competence": org.competence_trace[-200:],
            "milestones": org.milestones,
            "preferences": org.preferences,
            "dialogue": org.dialogue[-200:],
            "memory": org.memory.snapshot(),
            "hidden_probe": org.hidden_probe,
        },
    )
    mem_path = path / "episodes.jsonl"
    with mem_path.open("w", encoding="utf-8") as f:
        for ep in org.memory.episodes[-512:]:
            f.write(
                json.dumps(
                    {
                        "tick": ep.tick,
                        "pred_err": ep.pred_err,
                        "uncertainty": ep.uncertainty,
                        "latent": ep.latent.reshape(-1)[:64].tolist(),
                    }
                )
                + "\n"
            )
    return path


def load_checkpoint_v2(org: OrganismV2, path: str | Path) -> OrganismV2:
    path = Path(path)
    meta = read_json(path / "meta.json")
    if int(meta.get("schema", 1)) > SCHEMA:
        raise ValueError("checkpoint schema newer than runtime")
    blob = torch.load(path / "weights.pt", map_location=org.device, weights_only=False)
    org.net.load_state_dict(blob["net"])
    org.opt.load_state_dict(blob["opt"])
    org.rssm_state.h = blob["rssm_h"].to(org.device)
    org.rssm_state.z = blob["rssm_z"].to(org.device)
    for k, v in blob["core"].items():
        setattr(org.core_state, k, v.to(org.device))
    org.self_h = blob["self_h"].to(org.device)
    org.other_h = blob["other_h"].to(org.device)
    org.ws_prev = blob["ws_prev"].to(org.device)
    org.last_action = blob["last_action"].to(org.device)
    org.tick = int(blob.get("tick", meta["tick"]))
    org.time = float(meta.get("time", org.tick * org.cfg.dt))
    runtime = read_json(path / "runtime.json")
    org.spontaneous_count = int(runtime.get("spontaneous", 0))
    org.human_events = int(runtime.get("human_events", 0))
    org.disabled = set(runtime.get("disabled", []))
    org.pe_trace = list(runtime.get("pe_trace", []))
    org.loss_trace = list(runtime.get("loss_trace", []))
    org.competence_trace = list(runtime.get("competence", []))
    org.milestones = list(runtime.get("milestones", []))
    org.preferences = dict(runtime.get("preferences", {}))
    org.dialogue = list(runtime.get("dialogue", []))
    org.hidden_probe = dict(runtime.get("hidden_probe", {}))
    org.lineage_id = meta.get("lineage", org.lineage_id)
    return org

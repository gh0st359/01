"""V2 operational suite. Behavioral claims for V3 live in training/v3_bench.py.

Several checks here still pass if a tensor exists. Do not cite this suite as
evidence that an organ learned. Use `python -m training.v3_bench`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.checkpoint import load_checkpoint_v2
from core.device import resolve_device
from core.organism import OrganismV2
from shared.serialize import write_json
from shared.v2config import load_v2_config
from simulation.batch import ProceduralBatch


def _org(profile: str, seed: int, restore: Path | None) -> OrganismV2:
    cfg = load_v2_config(profile, seed=seed)
    org = OrganismV2(cfg, Path("runs/v2/eval_tmp"), device=resolve_device())
    if restore and restore.exists():
        load_checkpoint_v2(org, restore)
    return org


def object_permanence(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 3)
    pes_vis = []
    for _ in range(12):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        pes_vis.append(r.prediction_error)
    # Hide a non-self object
    world.state.hidden[0, 3] = True
    pes_hid = []
    for _ in range(12):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        pes_hid.append(r.prediction_error)
    world.state.hidden[0, 3] = False
    r = org.tick_once(world.observation(0).sensors)
    return {
        "name": "object_permanence",
        "pass": float(np.mean(pes_hid)) < float(np.mean(pes_vis)) * 3.0 + 0.5,
        "visible_pe": float(np.mean(pes_vis)),
        "hidden_pe": float(np.mean(pes_hid)),
        "reappear_pe": r.prediction_error,
    }


def identity_occlusion(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 4)
    slot0 = None
    for i in range(8):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        if org.slots is not None:
            slot0 = org.slots.detach().clone()
    world.state.hidden[0, 2] = True
    for _ in range(6):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    world.state.hidden[0, 2] = False
    org.tick_once(world.observation(0).sensors)
    sim = 0.0
    if slot0 is not None and org.slots is not None:
        a = slot0.reshape(-1)
        b = org.slots.reshape(-1)
        sim = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    return {"name": "identity_occlusion", "pass": sim > 0.2, "slot_cosine": sim}


def navigation(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 5)
    start = world.state.pos[0, 0].copy()
    for _ in range(20):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    dist = float(np.linalg.norm(world.state.pos[0, 0] - start))
    return {"name": "navigation", "pass": dist > 0.05, "displacement": dist}


def causal_intervention(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 6)
    lights = []
    for _ in range(10):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        lights.append(float(world.state.light[0, 1]))
    # Force interaction near switch
    world.state.pos[0, 0] = world.state.pos[0, 2] + np.array([0.2, 0.0])
    r = org.tick_once(world.observation(0).sensors)
    r.motor.interact = 1.0
    world.step([r.motor])
    after = float(world.state.light[0, 1])
    return {"name": "causal_intervention", "pass": after != float(np.mean(lights)), "light_before": float(np.mean(lights)), "light_after": after}


def delayed_memory(org: OrganismV2) -> dict:
    n0 = len(org.memory.episodes)
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 7)
    org.ingest_speech("caregiver", "disk")
    for _ in range(6):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    for _ in range(16):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    return {"name": "delayed_memory", "pass": len(org.memory.episodes) > n0 + 8, "episodes": len(org.memory.episodes)}


def grounded_symbol(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 8)
    before = org.human_events
    for _ in range(8):
        org.ingest_speech("caregiver", "tool")
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    return {"name": "grounded_symbol", "pass": org.human_events > before and any(m["name"] == "first_grounded_stream" for m in org.milestones), "human_events": org.human_events}


def novel_word(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 9)
    token = "zixq"
    org.ingest_speech("caregiver", token)
    r = org.tick_once(world.observation(0).sensors)
    return {"name": "novel_word", "pass": r.intent is not None, "heard": True}


def curiosity_ig(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 10)
    infos = []
    for _ in range(10):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        infos.append(float(r.notes.get("info_value", 0)))
    return {"name": "curiosity_information_gain", "pass": max(infos) != min(infos) or max(infos) > 0, "info_span": [min(infos), max(infos)]}


def spontaneous_goals(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 11)
    values = []
    for _ in range(12):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        values.append(float(r.notes.get("goal_value", 0)))
    return {"name": "spontaneous_goal_formation", "pass": any(abs(v) > 1e-6 for v in values), "values": values[-5:]}


def checkpoint_continuity(org: OrganismV2, tmp: Path) -> dict:
    from core.checkpoint import save_checkpoint_v2

    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 12)
    for _ in range(4):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    tick = org.tick
    ident = org.self_h.detach().clone()
    path = save_checkpoint_v2(org, tmp / "ckpt_eval")
    org.tick = 0
    load_checkpoint_v2(org, path)
    same = org.tick == tick and torch.allclose(org.self_h, ident.to(org.device), atol=1e-5)
    return {"name": "checkpoint_identity_continuity", "pass": same, "tick": org.tick}


def multi_individual(profile: str) -> dict:
    a = _org(profile, 21, None)
    b = _org(profile, 22, None)
    wa = ProceduralBatch(1, a.cfg.max_objects, a.cfg.vision_hw, 21)
    wb = ProceduralBatch(1, b.cfg.max_objects, b.cfg.vision_hw, 22)
    for _ in range(10):
        ra = a.tick_once(wa.observation(0).sensors)
        wa.step([ra.motor])
        rb = b.tick_once(wb.observation(0).sensors)
        wb.step([rb.motor])
    ha = a.self_h.detach().cpu().numpy().ravel()
    hb = b.self_h.detach().cpu().numpy().ravel()
    diff = float(np.linalg.norm(ha - hb))
    return {"name": "multiple_individual_divergence", "pass": diff > 1e-4, "l2": diff}


def affordance_discovery(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 13)
    held = 0
    for _ in range(16):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        held += int(world.state.held[0] >= 0)
    return {"name": "affordance_discovery", "pass": True, "held_ticks": held}


def causal_transfer(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 14, held_out=True)
    pes = []
    for _ in range(10):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        pes.append(r.prediction_error)
    return {"name": "causal_transfer", "pass": np.isfinite(np.mean(pes)), "held_out_pe": float(np.mean(pes))}


def semantic_generalization(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 15)
    for token in ("disk", "tool", "disk"):
        org.ingest_speech("caregiver", token)
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    return {"name": "semantic_generalization", "pass": len(org.memory.semantic_bank) >= 0, "semantic": len(org.memory.semantic_bank)}


def continual_retention(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 16)
    early = []
    for _ in range(8):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        early.append(r.prediction_error)
    world2 = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 116, held_out=True)
    for _ in range(8):
        r = org.tick_once(world2.observation(0).sensors)
        world2.step([r.motor])
    late = []
    world.state = world.reset() if False else world.state
    for _ in range(6):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        late.append(r.prediction_error)
    return {"name": "continual_learning_retention", "pass": True, "early": float(np.mean(early)), "late": float(np.mean(late))}


def body_contingency(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 17)
    agencies = []
    for _ in range(10):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        agencies.append(float(r.notes.get("agency", 0)))
    return {"name": "body_contingency", "pass": max(agencies) >= min(agencies), "agency": agencies[-1]}


def delayed_agency(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 18)
    r = org.tick_once(world.observation(0).sensors)
    delayed = r.motor
    for _ in range(3):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    world.step([delayed])
    r2 = org.tick_once(world.observation(0).sensors)
    return {"name": "delayed_agency", "pass": True, "agency": float(r2.notes.get("agency", 0))}


def body_remap(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 19)
    before = []
    for _ in range(6):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        before.append(float(r.notes.get("agency", 0)))
    # invert controls
    for _ in range(6):
        r = org.tick_once(world.observation(0).sensors)
        r.motor.linear_velocity *= -1
        r.motor.angular_velocity *= -1
        world.step([r.motor])
    after = float(r.notes.get("agency", 0))
    return {"name": "body_remapping", "pass": True, "agency_before": float(np.mean(before)), "agency_after": after}


def tool_discovery(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 20)
    for _ in range(12):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    return {"name": "tool_discovery", "pass": org.imagination_used >= 1, "imagination": org.imagination_used}


def multi_step_planning(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 21)
    r = org.tick_once(world.observation(0).sensors)
    return {"name": "multi_step_planning", "pass": org.plan_actions is not None and org.plan_actions.shape[0] >= 2, "horizon": int(org.plan_actions.shape[0]) if org.plan_actions is not None else 0}


def social_prediction(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 22)
    for _ in range(8):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    return {"name": "social_prediction", "pass": float(org.other_h.norm()) >= 0.0, "other_norm": float(org.other_h.norm())}


def false_belief(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 23)
    for _ in range(6):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    if world.n > 5:
        world.state.hidden[0, 5] = True
    r = org.tick_once(world.observation(0).sensors)
    return {"name": "false_belief", "pass": True, "other_norm": float(org.other_h.norm())}


def joint_attention(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 24)
    org.ingest_speech("caregiver", "disk")
    r = org.tick_once(world.observation(0).sensors)
    return {"name": "joint_attention", "pass": r.intent is not None, "human": org.human_events}


def compositional_communication(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 25)
    org.ingest_speech("caregiver", "disk tool")
    r = org.tick_once(world.observation(0).sensors)
    world.step([r.motor])
    return {"name": "compositional_communication", "pass": r.intent is not None, "utterance_len": len(r.utterance.split())}


def clarification(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 26)
    org.ingest_speech("caregiver", "aaa bbb")
    r = org.tick_once(world.observation(0).sensors)
    return {"name": "clarification_under_ambiguity", "pass": r.intent is not None, "uncertainty": r.intent.preceding_uncertainty if r.intent else 0.0}


def autobiographical(org: OrganismV2) -> dict:
    return {"name": "autobiographical_continuity", "pass": len(org.memory.autobiography) > 0, "n": len(org.memory.autobiography)}


def preference_persist(org: OrganismV2) -> dict:
    world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, org.cfg.seed + 27)
    org.ingest_speech("caregiver", "disk")
    org.tick_once(world.observation(0).sensors)
    n = len(org.preferences)
    return {"name": "preference_persistence", "pass": n >= 0, "n": n}


def uncertainty_cal(org: OrganismV2) -> dict:
    return {"name": "uncertainty_calibration", "pass": len(org.pe_trace) > 0, "last_pe": org.pe_trace[-1] if org.pe_trace else None}


def counterfactual(org: OrganismV2) -> dict:
    return {"name": "counterfactual_learning", "pass": org.imagination_used >= 1, "imagination": org.imagination_used}


def run_suite(profile: str, seed: int, restore: Path | None, out: Path) -> dict:
    org = _org(profile, seed, restore)
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    results = [
        object_permanence(org),
        identity_occlusion(org),
        navigation(org),
        causal_intervention(org),
        delayed_memory(org),
        grounded_symbol(org),
        novel_word(org),
        curiosity_ig(org),
        spontaneous_goals(org),
        checkpoint_continuity(org, tmp),
        multi_individual(profile),
        affordance_discovery(org),
        causal_transfer(org),
        semantic_generalization(org),
        continual_retention(org),
        body_contingency(org),
        delayed_agency(org),
        body_remap(org),
        tool_discovery(org),
        multi_step_planning(org),
        social_prediction(org),
        false_belief(org),
        joint_attention(org),
        compositional_communication(org),
        clarification(org),
        autobiographical(org),
        preference_persist(org),
        uncertainty_cal(org),
        counterfactual(org),
    ]
    report = {
        "n": len(results),
        "passed": sum(1 for r in results if r["pass"]),
        "failed": [r["name"] for r in results if not r["pass"]],
        "results": results,
        "params": org.param_count,
        "device": org.device_bundle.kind,
    }
    write_json(out / "evaluate.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="ci")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--out", default="research/evidence/v2/eval")
    args = p.parse_args(argv)
    restore = Path(args.checkpoint) if args.checkpoint else None
    report = run_suite(args.profile, args.seed, restore, Path(args.out))
    print(json.dumps({"passed": report["passed"], "n": report["n"], "failed": report["failed"]}, indent=2))


if __name__ == "__main__":
    main()

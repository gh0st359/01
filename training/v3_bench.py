"""Multi-seed behavioral benchmarks. A tensor existing is not a pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.device import resolve_device
from core.organism import OrganismV2
from shared.serialize import write_json
from shared.v2config import load_v2_config
from simulation.batch import ProceduralBatch


def _mean_ci(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    n = max(1, a.size)
    mu = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mu, "se": se, "n": int(n), "values": [float(x) for x in a]}


def train_brief(seed: int, steps: int, profile: str = "ci") -> OrganismV2:
    cfg = load_v2_config(profile, seed=seed)
    org = OrganismV2(cfg, Path(f"/tmp/v3_bench_{seed}"), device=resolve_device())
    world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, seed)
    for t in range(steps):
        if t % 11 == 0:
            world.retarget(0)
            org.ingest_speech("partner", world.partner_stream(0))
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
    return org


def bench_self_invert(seeds: list[int], train_steps: int) -> dict:
    spikes, recovers = [], []
    for s in seeds:
        org = train_brief(s, train_steps)
        world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, s + 3)
        base = []
        for _ in range(12):
            r = org.tick_once(world.observation(0).sensors)
            world.step([r.motor])
            base.append(float(r.notes.get("self_pe", 0.0)))
        org.invert_controls = True
        inv = []
        for _ in range(8):
            r = org.tick_once(world.observation(0).sensors)
            world.step([r.motor])
            inv.append(float(r.notes.get("self_pe", 0.0)))
        late = []
        for _ in range(24):
            r = org.tick_once(world.observation(0).sensors)
            world.step([r.motor])
            late.append(float(r.notes.get("self_pe", 0.0)))
        b, i, l = float(np.mean(base) + 1e-8), float(np.mean(inv)), float(np.mean(late))
        spikes.append(i / b)
        recovers.append(l / max(i, 1e-8))
    spike = _mean_ci(spikes)
    rec = _mean_ci(recovers)
    return {
        "name": "self_model_invert_adapt",
        "pass": spike["mean"] > 1.3 and rec["mean"] < 0.95,
        "spike_ratio": spike,
        "recovery_ratio": rec,
    }


def bench_concepts(seeds: list[int], train_steps: int) -> dict:
    counts, collapse = [], []
    for s in seeds:
        org = train_brief(s, train_steps)
        counts.append(0.0 if org.concepts is None else float(org.concepts.size(0)))
        collapse.append(float(org.concept_collapse))
    st = _mean_ci(counts)
    cl = _mean_ci(collapse)
    return {
        "name": "semantic_induction",
        "pass": st["mean"] >= 4.0 and cl["mean"] < 0.95,
        "concepts": st,
        "collapse": cl,
    }


def bench_referential(seeds: list[int], train_steps: int) -> dict:
    accs = []
    for s in seeds:
        org = train_brief(s, train_steps)
        world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, s + 5)
        hit = []
        for _ in range(20):
            world.retarget(0)
            org.ingest_speech("partner", world.partner_stream(0))
            r = org.tick_once(world.observation(0).sensors)
            world.step([r.motor])
            hit.append(float(r.notes.get("ref_acc", 0.0)))
        accs.append(float(np.mean(hit)))
    st = _mean_ci(accs)
    return {"name": "referential_listen", "pass": st["mean"] > 1.0 / 8.0 + 0.05, "acc": st}


def bench_curiosity(seeds: list[int], train_steps: int) -> dict:
    rates, margins = [], []
    for s in seeds:
        org = train_brief(s, train_steps)
        rates.append(float(np.mean(org.probe_choices[-64:])) if org.probe_choices else 0.0)
        margins.append(float(np.mean(org.ig_margin_trace[-32:])) if org.ig_margin_trace else 0.0)
    st = _mean_ci(rates)
    mg = _mean_ci(margins)
    return {
        "name": "curiosity_probe_rate",
        "pass": st["mean"] > 0.05,
        "rate": st,
        "ig_margin": mg,
    }


def bench_hypothesis_split(seeds: list[int], train_steps: int) -> dict:
    """IG(interact)−IG(wait) must grow after experience, not stay at birth noise."""
    deltas = []
    for s in seeds:
        cfg = load_v2_config("ci", seed=s)
        fresh = OrganismV2(cfg, Path(f"/tmp/v3_split_birth_{s}"), device=resolve_device())
        world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, s + 7)
        birth = []
        for _ in range(6):
            r = fresh.tick_once(world.observation(0).sensors)
            world.step([r.motor])
            birth.append(float(r.notes.get("ig_margin", 0.0)))
        org = train_brief(s, train_steps)
        late = org.ig_margin_trace[-min(32, len(org.ig_margin_trace)) :]
        deltas.append(float(np.mean(late) - np.mean(birth)))
    st = _mean_ci(deltas)
    return {
        "name": "curiosity_hypothesis_split",
        "pass": st["mean"] > 0.0,
        "delta_ig_margin": st,
    }


def bench_differentiating_experiment(seeds: list[int], train_steps: int) -> dict:
    """Near the causal switch, interact should split hypotheses more than wait."""
    near_far = []
    for s in seeds:
        org = train_brief(s, train_steps)
        world = ProceduralBatch(1, org.cfg.max_objects, org.cfg.vision_hw, s + 11)
        world.state.pos[0, 0] = world.state.pos[0, 2] + np.array([0.2, 0.0])
        org.tick_once(world.observation(0).sensors)
        lat = org.rssm_state.flatten()
        wait = torch.zeros(1, org.cfg.action_dim)
        probe = torch.zeros(1, org.cfg.action_dim)
        probe[0, 3] = 1.0
        ig_near = float((org.net.epistemic.information_gain(lat, probe) - org.net.epistemic.information_gain(lat, wait)).mean())
        world.state.pos[0, 0] = np.array([1.0, 1.0])
        world.state.pos[0, 2] = np.array([7.0, 7.0])
        org.tick_once(world.observation(0).sensors)
        lat2 = org.rssm_state.flatten()
        ig_far = float((org.net.epistemic.information_gain(lat2, probe) - org.net.epistemic.information_gain(lat2, wait)).mean())
        near_far.append(ig_near - ig_far)
    st = _mean_ci(near_far)
    return {
        "name": "differentiating_experiment",
        "pass": st["mean"] > 0.0,
        "near_minus_far": st,
    }


def bench_organ_losses(seeds: list[int], train_steps: int) -> dict:
    """Each organ loss must change over training (not stay identically zero)."""
    moved = []
    for s in seeds:
        org = train_brief(s, train_steps)
        if len(org.organ_trace) < 8:
            moved.append(0.0)
            continue
        first = org.organ_trace[2]
        last = org.organ_trace[-1]
        keys = [k for k, v in first.items() if k != "world"]
        delta = sum(abs(last[k] - first[k]) for k in keys)
        moved.append(float(delta))
    st = _mean_ci(moved)
    return {"name": "organ_losses_move", "pass": st["mean"] > 1e-4, "delta": st}


def run_v3_bench(profile: str, seeds: list[int], steps: int, out: Path) -> dict:
    results = [
        bench_organ_losses(seeds, steps),
        bench_concepts(seeds, steps),
        bench_referential(seeds, steps),
        bench_self_invert(seeds, steps),
        bench_curiosity(seeds, steps),
        bench_hypothesis_split(seeds, steps),
        bench_differentiating_experiment(seeds, steps),
    ]
    report = {
        "profile": profile,
        "seeds": seeds,
        "steps": steps,
        "passed": sum(1 for r in results if r["pass"]),
        "n": len(results),
        "failed": [r["name"] for r in results if not r["pass"]],
        "results": results,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "v3_bench.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="ci")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--out", default="research/evidence/v3/bench")
    args = p.parse_args(argv)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    report = run_v3_bench(args.profile, seeds, args.steps, Path(args.out))
    print(json.dumps({"passed": report["passed"], "n": report["n"], "failed": report["failed"]}, indent=2))


if __name__ == "__main__":
    main()

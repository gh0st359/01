from pathlib import Path

import numpy as np
import torch

from core.device import count_parameters, resolve_device
from core.language import decode_stream, encode_stream
from core.organism import OrganismV2
from core.rssm import RSSM
from shared.v2config import load_v2_config
from simulation.batch import ProceduralBatch
from training.ablate import ablate
from training.evaluate import run_suite
from training.longitudinal import longitudinal


def test_device_reports_cpu_honestly_when_no_gpu():
    bundle = resolve_device()
    if not torch.cuda.is_available():
        assert bundle.kind == "cpu"
        assert bundle.amp is False


def test_rssm_multi_step_rollout():
    rssm = RSSM(16, 8, 4, 12)
    s = rssm.initial(2, torch.device("cpu"))
    acts = torch.randn(5, 2, 4)
    last, lat = rssm.rollout(s, acts)
    assert lat.shape == (5, 2, 24)
    assert last.h.shape == (2, 16)


def test_language_is_not_english_aware():
    ids = encode_stream("xyz", 8, torch.device("cpu"))
    assert ids.tolist()[0] == ord("x")
    assert decode_stream(ids) == "xyz"


def test_batch_world_hides_ids_from_sensors():
    world = ProceduralBatch(2, 8, 16, seed=1)
    obs = world.observation(0)
    assert obs.sensors.vision_shape[0] == 16
    assert "object_id" not in obs.sensors.__dict__
    assert "truth_pos" in obs.hidden_from_organism


def test_organism_learns_and_plans(tmp_path: Path):
    cfg = load_v2_config("ci", seed=2)
    org = OrganismV2(cfg, tmp_path)
    assert org.param_count == count_parameters(org.net)
    assert org.param_count > 1000
    world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, 2)
    pes = []
    for _ in range(8):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        pes.append(r.prediction_error)
        assert r.notes["goal_value"] == r.notes["goal_value"]
        assert r.motor.kind_hint == "latent"
    assert len(org.memory.episodes) >= 1
    assert org.imagination_used >= 1


def test_no_threshold_goal_names():
    cfg = load_v2_config("ci", seed=3)
    org = OrganismV2(cfg, Path("/tmp/o1_v2_names"))
    world = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, 3)
    r = org.tick_once(world.observation(0).sensors)
    assert r.notes.get("goal") not in {"inspect", "experiment", "wait"}


def test_evaluate_suite_runs(tmp_path: Path):
    report = run_suite("ci", 1, None, tmp_path)
    assert report["n"] >= 10
    assert report["passed"] >= 6


def test_longitudinal_restore(tmp_path: Path):
    report = longitudinal("ci", 3, tmp_path, cycles=2, steps=4)
    assert report["all_ok"]


def test_ablation_runs(tmp_path: Path):
    report = ablate("ci", 4, 6, tmp_path)
    assert len(report["rows"]) >= 3

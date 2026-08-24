"""V2 research profiles. Parameter count is a variable, not a constant toy."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass
class V2Config:
    profile: str = "cloud_cpu"
    seed: int = 1
    name: str = "01"
    dt: float = 0.05
    vision_hw: int = 64
    vision_c: int = 3
    slots: int = 8
    slot_dim: int = 64
    deter_dim: int = 128
    stoch_dim: int = 32
    core_dim: int = 128
    homeo_dim: int = 10
    action_dim: int = 8
    char_vocab: int = 128
    lang_dim: int = 96
    lang_hidden: int = 128
    goal_dim: int = 64
    n_goals: int = 4
    imag_horizon: int = 12
    cem_pop: int = 24
    cem_elite: int = 6
    cem_iters: int = 3
    ensemble: int = 4
    lr: float = 3e-4
    mixed_precision: bool = False
    compile: bool = False
    batch_envs: int = 4
    max_objects: int = 12
    world_size: float = 10.0
    checkpoint_every: int = 500
    consolidation_every: int = 64
    episodic_capacity: int = 8000
    schema_version: int = 2
    data_dir: str = "data"
    run_dir: str = "runs"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "V2Config":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


_PROFILES: dict[str, dict] = {
    "ci": {
        "vision_hw": 16,
        "slots": 4,
        "slot_dim": 32,
        "deter_dim": 32,
        "stoch_dim": 16,
        "core_dim": 32,
        "lang_dim": 32,
        "lang_hidden": 32,
        "goal_dim": 16,
        "imag_horizon": 4,
        "cem_pop": 6,
        "cem_iters": 1,
        "ensemble": 2,
        "batch_envs": 2,
        "max_objects": 6,
        "episodic_capacity": 200,
    },
    "development_cpu": {
        "vision_hw": 32,
        "slots": 6,
        "slot_dim": 48,
        "deter_dim": 64,
        "stoch_dim": 24,
        "core_dim": 64,
        "lang_hidden": 64,
        "imag_horizon": 8,
        "cem_pop": 12,
        "batch_envs": 4,
    },
    "cloud_cpu": {},
    "apple_mps": {
        "vision_hw": 64,
        "slots": 8,
        "slot_dim": 96,
        "deter_dim": 192,
        "stoch_dim": 48,
        "core_dim": 192,
        "lang_hidden": 192,
        "imag_horizon": 16,
        "cem_pop": 32,
        "mixed_precision": True,
        "batch_envs": 8,
    },
    "gpu_16gb": {
        "vision_hw": 96,
        "slots": 12,
        "slot_dim": 128,
        "deter_dim": 384,
        "stoch_dim": 64,
        "core_dim": 384,
        "lang_dim": 160,
        "lang_hidden": 256,
        "goal_dim": 96,
        "imag_horizon": 24,
        "cem_pop": 64,
        "cem_iters": 5,
        "ensemble": 5,
        "mixed_precision": True,
        "batch_envs": 16,
        "max_objects": 16,
    },
    "gpu_cloud": {
        "vision_hw": 128,
        "slots": 16,
        "slot_dim": 160,
        "deter_dim": 512,
        "stoch_dim": 96,
        "core_dim": 512,
        "lang_hidden": 384,
        "imag_horizon": 32,
        "cem_pop": 96,
        "mixed_precision": True,
        "batch_envs": 32,
    },
}


def load_v2_config(profile: str | None = None, seed: int | None = None) -> V2Config:
    name = profile or os.environ.get("O1_V2_PROFILE", "cloud_cpu")
    cfg = V2Config(profile=name)
    for k, v in _PROFILES.get(name, {}).items():
        setattr(cfg, k, v)
    if seed is not None:
        cfg.seed = seed
    elif "O1_SEED" in os.environ:
        cfg.seed = int(os.environ["O1_SEED"])
    return cfg

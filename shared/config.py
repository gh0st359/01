"""Configuration profiles. Cognition never disappears; dimensions scale."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class ProfileName(str, Enum):
    DEVELOPMENT = "development"
    CPU = "cpu"
    CLOUD = "cloud"
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    HIGH_THROUGHPUT = "high_throughput"
    LONG_RUNNING = "long_running"


@dataclass
class OrganismConfig:
    profile: ProfileName = ProfileName.CLOUD
    seed: int = 1
    name: str = "01"
    dt: float = 0.05
    state_dim: int = 192
    feature_dim: int = 96
    entity_dim: int = 48
    concept_dim: int = 64
    action_dim: int = 12
    workspace_slots: int = 7
    working_memory_slots: int = 7
    vision_h: int = 16
    vision_w: int = 16
    vision_c: int = 3
    audio_dim: int = 32
    tactile_dim: int = 8
    max_entities: int = 24
    episodic_capacity: int = 4000
    semantic_capacity: int = 2000
    associative_capacity: int = 1024
    imagination_horizon: int = 8
    imagination_branches: int = 4
    consolidation_every: int = 240
    checkpoint_every: int = 400
    max_goals: int = 8
    lexicon_capacity: int = 512
    homeostatic_leak: float = 0.002
    learning_rate: float = 0.01
    hebbian_lr: float = 0.02
    neuromod_scale: float = 1.5
    replay_batch: int = 16
    mixed_precision: bool = False
    data_dir: str = "data"
    run_dir: str = "runs"
    schema_version: int = 1

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        raw = asdict(self)
        raw["profile"] = self.profile.value
        return raw

    @classmethod
    def from_dict(cls, data: dict) -> "OrganismConfig":
        data = dict(data)
        if "profile" in data:
            data["profile"] = ProfileName(data["profile"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


_PROFILE_OVERRIDES: dict[ProfileName, dict] = {
    ProfileName.DEVELOPMENT: {
        "state_dim": 96,
        "feature_dim": 48,
        "entity_dim": 32,
        "concept_dim": 40,
        "vision_h": 12,
        "vision_w": 12,
        "episodic_capacity": 800,
        "imagination_horizon": 5,
        "imagination_branches": 3,
        "consolidation_every": 80,
        "checkpoint_every": 120,
    },
    ProfileName.CPU: {
        "state_dim": 128,
        "feature_dim": 64,
        "entity_dim": 40,
        "concept_dim": 48,
        "vision_h": 14,
        "vision_w": 14,
    },
    ProfileName.CLOUD: {},
    ProfileName.SINGLE_GPU: {
        "state_dim": 256,
        "feature_dim": 128,
        "entity_dim": 64,
        "concept_dim": 80,
        "vision_h": 24,
        "vision_w": 24,
        "episodic_capacity": 12000,
        "imagination_horizon": 12,
        "imagination_branches": 6,
    },
    ProfileName.MULTI_GPU: {
        "state_dim": 384,
        "feature_dim": 160,
        "entity_dim": 80,
        "concept_dim": 96,
        "vision_h": 32,
        "vision_w": 32,
        "episodic_capacity": 24000,
        "imagination_horizon": 16,
        "imagination_branches": 8,
    },
    ProfileName.HIGH_THROUGHPUT: {
        "state_dim": 128,
        "feature_dim": 64,
        "dt": 0.02,
        "imagination_horizon": 4,
        "consolidation_every": 400,
    },
    ProfileName.LONG_RUNNING: {
        "episodic_capacity": 8000,
        "semantic_capacity": 4000,
        "checkpoint_every": 200,
        "consolidation_every": 180,
    },
}


def load_config(profile: str | ProfileName | None = None, seed: int | None = None) -> OrganismConfig:
    name = profile or os.environ.get("O1_PROFILE", "cloud")
    if isinstance(name, str):
        name = ProfileName(name)
    cfg = OrganismConfig(profile=name)
    for key, value in _PROFILE_OVERRIDES.get(name, {}).items():
        setattr(cfg, key, value)
    if seed is not None:
        cfg.seed = seed
    elif "O1_SEED" in os.environ:
        cfg.seed = int(os.environ["O1_SEED"])
    if "O1_DATA_DIR" in os.environ:
        cfg.data_dir = os.environ["O1_DATA_DIR"]
    if "O1_RUN_DIR" in os.environ:
        cfg.run_dir = os.environ["O1_RUN_DIR"]
    return cfg


def save_config(cfg: OrganismConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

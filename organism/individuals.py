"""Multiple individuals: same architecture, different seeds and histories."""

from __future__ import annotations

from pathlib import Path

from core.device import resolve_device
from core.organism import OrganismV2
from organism.loop import Organism
from shared.config import OrganismConfig
from shared.rng import RNG
from shared.v2config import V2Config, load_v2_config


def spawn_individual(cfg: OrganismConfig, seed: int, root: str | Path) -> Organism:
    child = OrganismConfig.from_dict({**cfg.to_dict(), "seed": seed})
    rng = RNG(seed)
    data = Path(root) / f"organism_{seed}"
    org = Organism(child, rng, data)
    org.lineage_id = f"{cfg.name}-{seed}"
    return org


def spawn_individual_v2(cfg: V2Config | None, seed: int, root: str | Path, profile: str = "ci") -> OrganismV2:
    base = cfg or load_v2_config(profile, seed=seed)
    child = V2Config.from_dict({**base.to_dict(), "seed": seed})
    org = OrganismV2(child, Path(root) / f"organism_{seed}", device=resolve_device())
    org.lineage_id = f"{child.name}-{seed}"
    return org

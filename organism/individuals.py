"""Multiple individuals: same architecture, different seeds and histories."""

from __future__ import annotations

from pathlib import Path

from organism.loop import Organism
from shared.config import OrganismConfig
from shared.rng import RNG


def spawn_individual(cfg: OrganismConfig, seed: int, root: str | Path) -> Organism:
    child = OrganismConfig.from_dict({**cfg.to_dict(), "seed": seed})
    rng = RNG(seed)
    data = Path(root) / f"organism_{seed}"
    org = Organism(child, rng, data)
    org.lineage_id = f"{cfg.name}-{seed}"
    return org

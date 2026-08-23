"""Laboratory workflow for architectural mutations: propose → fork → evaluate → decide."""

from __future__ import annotations

from pathlib import Path

from organism.checkpoint import save_checkpoint
from organism.lineage import LineageBook
from organism.loop import Organism
from shared.config import OrganismConfig
from shared.rng import RNG


def fork_and_record(org: Organism, cfg: OrganismConfig, new_seed: int, root: Path, notes: str) -> tuple[Organism, str]:
    book = LineageBook(root / "lineage.json")
    child_cfg = OrganismConfig.from_dict({**cfg.to_dict(), "seed": new_seed})
    child = Organism(child_cfg, RNG(new_seed), root / f"fork_{new_seed}")
    child.parent_lineage = org.lineage_id
    child.lineage_id = f"{cfg.name}-fork-{new_seed}"
    ckpt = save_checkpoint(child, root / "checkpoints" / child.lineage_id)
    book.propose(child.lineage_id, org.lineage_id, new_seed, str(ckpt), notes)
    return child, child.lineage_id

"""Architectural modification requires an explicit fork. The canonical lineage is not overwritten."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.checkpoint import load_checkpoint_v2, save_checkpoint_v2
from core.device import resolve_device
from core.organism import OrganismV2
from shared.v2config import V2Config
from simulation.batch import ProceduralBatch


@dataclass
class BranchResult:
    accepted: bool
    baseline_pe: float
    branch_pe: float
    path: str


def _mean_pe(org: OrganismV2, world: ProceduralBatch, steps: int) -> float:
    pes = []
    for _ in range(steps):
        r = org.tick_once(world.observation(0).sensors)
        world.step([r.motor])
        pes.append(r.prediction_error)
    return sum(pes) / len(pes)


def propose_and_evaluate(parent: OrganismV2, work: Path, steps: int = 12) -> BranchResult:
    work.mkdir(parents=True, exist_ok=True)
    parent_ckpt = save_checkpoint_v2(parent, work / "parent")
    cfg = V2Config.from_dict(parent.cfg.to_dict())
    cfg.seed = parent.cfg.seed + 1000
    branch = OrganismV2(cfg, work / "branch", device=resolve_device())
    load_checkpoint_v2(branch, parent_ckpt)
    branch.lineage_id = f"{parent.lineage_id}-fork"
    branch.parent_lineage = parent.lineage_id
    world_a = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, parent.cfg.seed)
    world_b = ProceduralBatch(1, cfg.max_objects, cfg.vision_hw, parent.cfg.seed)
    base = _mean_pe(parent, world_a, steps)
    br = _mean_pe(branch, world_b, steps)
    accepted = br <= base
    dest = work / ("accepted" if accepted else "rejected")
    save_checkpoint_v2(branch if accepted else parent, dest)
    return BranchResult(accepted=accepted, baseline_pe=base, branch_pe=br, path=str(dest))

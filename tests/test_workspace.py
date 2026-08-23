import numpy as np

from cognition.workspace.workspace import GlobalWorkspace
from shared.config import load_config
from shared.rng import RNG
from shared.types import CandidateKind, WorkspaceCandidate


def test_limited_capacity():
    cfg = load_config("development", seed=1)
    ws = GlobalWorkspace(cfg, RNG(1))
    cands = [
        WorkspaceCandidate(CandidateKind.VISUAL_EVENT, np.ones(cfg.feature_dim) * i, i / 10, 0.2, 0.2, 0.1, 0.1, 0.1, 0.5, "t")
        for i in range(20)
    ]
    out = ws.compete(cands, 1)
    assert len(out.winners) == cfg.workspace_slots
    assert out.broadcast.shape == (cfg.state_dim,)
    assert out.ignition >= 0

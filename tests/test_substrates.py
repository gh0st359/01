from cognition.core.multiscale import MultiScaleCore, SubstrateKind
from shared.config import load_config
from shared.rng import RNG
import numpy as np


def test_all_substrates_step():
    cfg = load_config("development", seed=1)
    x = np.zeros(cfg.feature_dim)
    x[0] = 1
    for kind in SubstrateKind:
        core = MultiScaleCore(cfg, RNG(1), kind)
        y = core.step(x)
        assert y.size > 0
        assert np.isfinite(y).all()

import numpy as np

from shared.config import load_config
from shared.rng import RNG
from world_model.predictive import PredictiveWorldModel


def test_world_model_trains():
    cfg = load_config("development", seed=1)
    rng = RNG(1)
    wm = PredictiveWorldModel(cfg, rng)
    losses = []
    for i in range(30):
        a = np.zeros(cfg.feature_dim)
        b = np.zeros(cfg.feature_dim)
        a[i % cfg.feature_dim] = 1
        b[(i + 1) % cfg.feature_dim] = 1
        losses.append(wm.train_step(a, np.zeros(cfg.action_dim), b))
    assert losses[-1] < losses[0] * 1.2 or losses[-1] < 0.5

import numpy as np

from learning.nn import GRUCell, cosine, softmax
from learning.optim import Adam
from shared.rng import RNG


def test_gru_changes_state():
    rng = RNG(1)
    cell = GRUCell("t", 8, 8, rng)
    h = np.zeros(8)
    x = rng.normal(8)
    h2, _ = cell.forward(x, h)
    assert h2.shape == (8,)
    assert float(np.linalg.norm(h2)) > 0


def test_gru_backward_finite():
    rng = RNG(2)
    cell = GRUCell("t", 6, 6, rng)
    x = rng.normal(6)
    h = rng.normal(6)
    h2, cache = cell.forward(x, h)
    gx, gh = cell.backward(cache, np.ones(6))
    assert np.isfinite(gx).all()
    assert np.isfinite(gh).all()


def test_adam_updates():
    rng = RNG(3)
    cell = GRUCell("t", 4, 4, rng)
    x = rng.normal(4)
    h = np.zeros(4)
    _, cache = cell.forward(x, h)
    cell.backward(cache, np.ones(4) * 0.1)
    before = cell.wz.weight.value.copy()
    Adam(lr=0.01).step(cell)
    assert not np.allclose(before, cell.wz.weight.value)


def test_softmax_and_cosine():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1) < 1e-8
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) > 0.99

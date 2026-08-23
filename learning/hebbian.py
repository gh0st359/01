"""Local learning rules used for grounding, associations, and traces."""

from __future__ import annotations

import numpy as np

from shared.types import Vector


def hebbian_update(weight: Vector, pre: Vector, post: Vector, lr: float, decay: float = 0.001) -> Vector:
    if weight.ndim == 2:
        outer = np.outer(post, pre)
    else:
        outer = pre * post
    updated = (1.0 - decay) * weight + lr * outer
    return np.clip(updated, -5.0, 5.0)


def oja_update(weight: Vector, pre: Vector, post: Vector, lr: float) -> Vector:
    if weight.ndim == 1:
        return weight + lr * post * (pre - post * weight)
    return weight + lr * (np.outer(post, pre) - np.outer(post * post, np.ones(pre.shape[0])) * weight)


def trace_update(trace: Vector, event: Vector, decay: float = 0.9) -> Vector:
    return decay * trace + (1.0 - decay) * event


def bcm_update(weight: Vector, pre: Vector, post: Vector, theta: float, lr: float) -> Vector:
    return weight + lr * np.outer(post * (post - theta), pre)


def bind(a: Vector, b: Vector) -> Vector:
    """Circular convolution binding (vector-symbolic)."""
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def unbind(mem: Vector, key: Vector) -> Vector:
    return np.real(np.fft.ifft(np.fft.fft(mem) * np.conj(np.fft.fft(key))))


def permute(x: Vector, shift: int = 1) -> Vector:
    return np.roll(x, shift)

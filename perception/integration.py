"""Multimodal integration of vision, audio, proprioception, and touch."""

from __future__ import annotations

import numpy as np

from shared.types import Vector


def integrate_modalities(vision: Vector, audio: Vector, proprio: Vector, tactile: Vector, dim: int) -> Vector:
    parts = [np.asarray(v, dtype=np.float64).ravel() for v in (vision, audio, proprio, tactile)]
    cat = np.concatenate(parts)
    out = np.zeros(dim, dtype=np.float64)
    if cat.size >= dim:
        # structured downsample
        idx = np.linspace(0, cat.size - 1, dim).astype(int)
        out = cat[idx]
    else:
        out[: cat.size] = cat
    return np.tanh(out)

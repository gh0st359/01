"""Primitive action vectors associated with ActionKind."""

from __future__ import annotations

import numpy as np

from shared.types import ActionKind, Vector


def primitive_vector(kind: ActionKind, dim: int, heading_bias: float = 0.0) -> Vector:
    v = np.zeros(dim, dtype=np.float64)
    if kind is ActionKind.MOVE or kind is ActionKind.APPROACH:
        v[0] = 0.7
        v[1] = heading_bias
    elif kind is ActionKind.TURN:
        v[1] = 0.8 if heading_bias >= 0 else -0.8
    elif kind is ActionKind.GRASP:
        v[2] = 1.0
        v[0] = 0.15
    elif kind is ActionKind.RELEASE:
        v[2] = -1.0
    elif kind is ActionKind.PUSH:
        v[0] = 0.5
        v[3] = 0.4
    elif kind is ActionKind.TOGGLE or kind is ActionKind.EXPERIMENT:
        v[3] = 1.0
        v[0] = 0.2
    elif kind is ActionKind.LOOK or kind is ActionKind.INSPECT:
        v[5] = 0.6 if dim > 5 else 0.0
        v[1] = 0.35
    elif kind is ActionKind.SPEAK:
        v[4] = 1.0
    elif kind is ActionKind.WAIT or kind is ActionKind.STAY:
        v[0] = 0.0
    return v

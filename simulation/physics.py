"""Euler physics for the developmental sandbox."""

from __future__ import annotations

import numpy as np


def integrate(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float, friction: float = 2.4) -> tuple[np.ndarray, np.ndarray]:
    vel = vel + acc * dt
    vel = vel * np.exp(-friction * dt)
    pos = pos + vel * dt
    return pos, vel


def resolve_circles(a_pos: np.ndarray, a_r: float, b_pos: np.ndarray, b_r: float) -> tuple[np.ndarray, np.ndarray]:
    delta = a_pos - b_pos
    dist = float(np.linalg.norm(delta)) + 1e-9
    overlap = a_r + b_r - dist
    if overlap <= 0:
        return a_pos, b_pos
    n = delta / dist
    a_pos = a_pos + n * (overlap * 0.5)
    b_pos = b_pos - n * (overlap * 0.5)
    return a_pos, b_pos


def clamp_world(pos: np.ndarray, radius: float, width: float, height: float) -> np.ndarray:
    pos[0] = float(np.clip(pos[0], radius, width - radius))
    pos[1] = float(np.clip(pos[1], radius, height - radius))
    return pos

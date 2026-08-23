"""Egocentric visual renderer. Pixels are unlabeled."""

from __future__ import annotations

import numpy as np

from simulation.objects import SimObject


def render_egocentric(objects: list[SimObject], body: SimObject, h: int, w: int, fov: float) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:, :] = np.array([0.08, 0.09, 0.11])
    c, s = np.cos(-body.heading), np.sin(-body.heading)
    rot = np.array([[c, -s], [s, c]])
    for obj in objects:
        if obj.hidden:
            continue
        rel = rot @ (obj.position - body.position)
        if abs(rel[0]) > fov or abs(rel[1]) > fov:
            continue
        px = int((rel[0] / fov + 1.0) * 0.5 * (w - 1))
        py = int((rel[1] / fov + 1.0) * 0.5 * (h - 1))
        rad = max(1, int(obj.radius / fov * w * 0.7))
        brightness = 1.0
        if obj.kind == "light":
            brightness = 0.35 + 0.65 * obj.light_state
        color = np.clip(obj.color * brightness, 0.0, 1.0)
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if dx * dx + dy * dy > rad * rad:
                    continue
                yy, xx = py + dy, px + dx
                if 0 <= yy < h and 0 <= xx < w:
                    img[yy, xx] = color
        if obj.object_id == body.object_id:
            img[py, px] = np.array([0.95, 0.95, 0.95])
    return img

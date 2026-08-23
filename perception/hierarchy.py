"""Hierarchical perception: pixels → features → proto-objects.

No symbolic labels are injected. Identity is later assigned by tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning.nn import Encoder, Linear, Module, relu, tanh
from perception.sensorium import SensoryPacket
from shared.config import OrganismConfig
from shared.rng import RNG
from shared.types import Vector


@dataclass
class Percept:
    features: Vector
    local_features: np.ndarray
    proto_objects: list[Vector]
    proto_locations: list[Vector]
    change: float
    tick: int


class PerceptualHierarchy(Module):
    def __init__(self, cfg: OrganismConfig, rng: RNG) -> None:
        super().__init__("perception")
        self.cfg = cfg
        vis = cfg.vision_h * cfg.vision_w * cfg.vision_c
        extra = cfg.audio_dim + 8 + cfg.tactile_dim * 2 + 8
        self.vis_enc = self.add(Encoder("perc.vis", vis, cfg.feature_dim, cfg.feature_dim, rng))
        self.int_enc = self.add(Encoder("perc.int", extra, cfg.feature_dim, cfg.feature_dim, rng))
        self.fuse = self.add(Linear("perc.fuse", cfg.feature_dim * 2, cfg.feature_dim, rng))
        self.prev_vision: Vector | None = None
        self.prev_features: Vector | None = None

    def encode(self, packet: SensoryPacket) -> Percept:
        vis_f = self.vis_enc.forward(packet.vision)
        extra = _fit(
            np.concatenate([packet.audio, packet.proprio, packet.tactile, packet.orientation, packet.motion, [packet.energy]]),
            self.cfg.audio_dim + 8 + self.cfg.tactile_dim * 2 + 8,
        )
        int_f = self.int_enc.forward(extra)
        features = tanh(self.fuse.forward(np.concatenate([vis_f, int_f])))
        local = self._local_patches(packet.vision_image)
        protos, locs = self._proto_objects(packet.vision_image)
        change = 0.0
        if self.prev_vision is not None:
            change = float(np.mean((packet.vision - self.prev_vision) ** 2))
        self.prev_vision = packet.vision.copy()
        self.prev_features = features.copy()
        return Percept(features, local, protos, locs, change, packet.tick)

    def _local_patches(self, img: np.ndarray) -> np.ndarray:
        h, w, c = img.shape
        gh, gw = 4, 4
        ph, pw = max(1, h // gh), max(1, w // gw)
        feats = []
        for i in range(gh):
            for j in range(gw):
                patch = img[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
                if patch.size == 0:
                    feats.append(np.zeros(6))
                    continue
                mean = patch.reshape(-1, c).mean(axis=0)
                std = patch.reshape(-1, c).std(axis=0)
                feats.append(np.concatenate([mean, std]))
        return np.array(feats, dtype=np.float64)

    def _proto_objects(self, img: np.ndarray) -> tuple[list[Vector], list[Vector]]:
        """Color-saliency blobs without labels."""
        h, w, _ = img.shape
        sat = img.max(axis=2) - img.min(axis=2)
        protos: list[Vector] = []
        locs: list[Vector] = []
        used = np.zeros((h, w), dtype=bool)
        for _ in range(8):
            if sat[~used].size == 0:
                break
            idx = np.argmax(np.where(used, -1, sat))
            y, x = divmod(int(idx), w)
            if sat[y, x] < 0.18:
                break
            color = img[y, x].copy()
            mask = np.linalg.norm(img - color, axis=2) < 0.28
            ys, xs = np.where(mask & ~used)
            if ys.size < 2:
                used[y, x] = True
                continue
            used[mask] = True
            cy, cx = float(ys.mean()), float(xs.mean())
            feat = np.concatenate([color, [sat[y, x], ys.size / (h * w), cx / w, cy / h]])
            protos.append(feat)
            locs.append(np.array([cx / w, cy / h], dtype=np.float64))
        return protos, locs


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

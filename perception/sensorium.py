"""Raw sensor collation into a single perceptual packet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.config import OrganismConfig
from shared.contracts import SensorFrame
from shared.types import Vector


@dataclass
class SensoryPacket:
    vision: Vector
    vision_image: np.ndarray
    audio: Vector
    proprio: Vector
    tactile: Vector
    energy: float
    motion: Vector
    orientation: Vector
    raw_concat: Vector
    tick: int


class Sensorium:
    def __init__(self, cfg: OrganismConfig) -> None:
        self.cfg = cfg

    def sense(self, frame: SensorFrame) -> SensoryPacket:
        img = frame.vision.reshape(frame.vision_shape)
        audio = frame.audio.samples
        tactile = np.concatenate([frame.tactile.contacts, frame.tactile.pressure, [frame.tactile.gripper_force]])
        raw = np.concatenate(
            [
                frame.vision,
                audio,
                frame.proprioception,
                tactile,
                frame.orientation,
                frame.motion,
                frame.internal,
                [frame.energy],
            ]
        )
        return SensoryPacket(
            vision=frame.vision.copy(),
            vision_image=img,
            audio=audio.copy(),
            proprio=frame.proprioception.copy(),
            tactile=tactile,
            energy=frame.energy,
            motion=frame.motion.copy(),
            orientation=frame.orientation.copy(),
            raw_concat=raw,
            tick=frame.tick,
        )

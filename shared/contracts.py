"""Reality-bridge sensor/action contracts.

Physical embodiment is an adapter problem. These frames are the only
interface the organism uses to touch a world, simulated or otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.types import Vector


@dataclass
class AudioFrame:
    samples: Vector
    event_energy: float
    source_bearing: float
    speech_like: float
    tick: int


@dataclass
class TactileFrame:
    contacts: Vector
    pressure: Vector
    gripper_force: float
    tick: int


@dataclass
class BodyState:
    position: Vector
    heading: float
    velocity: Vector
    angular_velocity: float
    joint_angles: Vector
    gripper_open: float
    energy: float
    damage: float
    contact_mask: Vector
    reachable: Vector
    tick: int


@dataclass
class SensorFrame:
    vision: Vector
    vision_shape: tuple[int, int, int]
    audio: AudioFrame
    proprioception: Vector
    tactile: TactileFrame
    body: BodyState
    energy: float
    orientation: Vector
    motion: Vector
    internal: Vector
    tick: int
    timestamp: float


@dataclass
class MotorCommand:
    linear_velocity: float
    angular_velocity: float
    gripper: float
    interact: float
    speak: float
    look_heading: float
    raw: Vector
    kind_hint: str
    tick: int


@dataclass
class WorldObservation:
    sensors: SensorFrame
    nearby_entity_count: int
    light_level: float
    audio_events: list[str] = field(default_factory=list)
    hidden_from_organism: dict[str, object] = field(default_factory=dict)


def empty_audio(tick: int, dim: int = 32) -> AudioFrame:
    return AudioFrame(
        samples=np.zeros(dim, dtype=np.float64),
        event_energy=0.0,
        source_bearing=0.0,
        speech_like=0.0,
        tick=tick,
    )


def empty_tactile(tick: int, dim: int = 8) -> TactileFrame:
    return TactileFrame(
        contacts=np.zeros(dim, dtype=np.float64),
        pressure=np.zeros(dim, dtype=np.float64),
        gripper_force=0.0,
        tick=tick,
    )

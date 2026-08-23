"""Simulated objects, devices, and other agents."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimObject:
    object_id: str
    kind: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    color: np.ndarray
    mass: float = 1.0
    graspable: bool = True
    held_by: str | None = None
    switch_state: float = 0.0
    light_state: float = 0.0
    open_state: float = 0.0
    hidden: bool = False
    inside_of: str | None = None
    label: str = ""
    agent_goal: np.ndarray | None = None
    is_agent: bool = False
    heading: float = 0.0
    energy: float = 1.0
    owner: str | None = None
    audio_signature: np.ndarray = field(default_factory=lambda: np.zeros(8))

    def copy_pose(self) -> dict:
        return {
            "id": self.object_id,
            "kind": self.kind,
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "vx": float(self.velocity[0]),
            "vy": float(self.velocity[1]),
            "r": self.radius,
            "color": self.color.tolist(),
            "hidden": self.hidden,
            "switch": self.switch_state,
            "light": self.light_state,
            "open": self.open_state,
            "is_agent": self.is_agent,
            "label": self.label,
        }

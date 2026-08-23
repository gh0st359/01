"""Simulated body schema inputs and motor mapping."""

from __future__ import annotations

import numpy as np

from shared.config import OrganismConfig
from shared.contracts import MotorCommand
from shared.types import ActionKind, Vector


class EmbodiedBody:
    def __init__(self, cfg: OrganismConfig) -> None:
        self.cfg = cfg
        self.last_command = MotorCommand(0, 0, 0, 0, 0, 0, np.zeros(cfg.action_dim), ActionKind.STAY.value, 0)

    def decode(self, action: Vector, kind: ActionKind, tick: int) -> MotorCommand:
        a = np.zeros(self.cfg.action_dim)
        a[: min(a.size, action.size)] = np.asarray(action, dtype=np.float64).ravel()[: a.size]
        linear = float(np.clip(a[0], -1.0, 1.0))
        angular = float(np.clip(a[1], -1.0, 1.0))
        gripper = float(np.clip(a[2], -1.0, 1.0))
        interact = float(np.clip(a[3], 0.0, 1.0))
        speak = float(np.clip(a[4], 0.0, 1.0))
        look = float(np.clip(a[5] if a.size > 5 else 0.0, -1.0, 1.0))
        if kind is ActionKind.STAY:
            linear = 0.0
            angular *= 0.1
        elif kind is ActionKind.TURN:
            linear *= 0.15
        elif kind is ActionKind.GRASP:
            gripper = 1.0
        elif kind is ActionKind.RELEASE:
            gripper = -1.0
        elif kind is ActionKind.TOGGLE or kind is ActionKind.EXPERIMENT:
            interact = 1.0
        elif kind is ActionKind.LOOK:
            linear *= 0.1
        elif kind is ActionKind.WAIT:
            linear = 0.0
            angular = 0.0
        elif kind is ActionKind.SPEAK:
            speak = max(speak, 0.8)
            linear *= 0.2
        cmd = MotorCommand(linear, angular, gripper, interact, speak, look, a, kind.value, tick)
        self.last_command = cmd
        return cmd

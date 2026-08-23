"""Reality bridge: cognition talks only through SensorFrame / MotorCommand.

Unrestricted physical actuators are intentionally not wired.
"""

from __future__ import annotations

from shared.contracts import MotorCommand, WorldObservation
from simulation.world import SimulatedWorld


class RealityBridge:
    def __init__(self, world: SimulatedWorld) -> None:
        self.world = world
        self.allow_physical = False

    def observe(self) -> WorldObservation:
        return self.world.observe()

    def act(self, command: MotorCommand) -> WorldObservation:
        if self.allow_physical:
            raise RuntimeError("physical actuators are disabled in the research environment")
        return self.world.step(command)

"""Sensor contract adapters — simulation today, robotics later."""

from shared.contracts import SensorFrame, WorldObservation


def from_observation(obs: WorldObservation) -> SensorFrame:
    return obs.sensors

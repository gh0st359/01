"""Operating modes: awake, rest, consolidation, simulation, maintenance."""

from __future__ import annotations

from dataclasses import dataclass

from shared.types import OperatingMode


@dataclass
class ModeEvent:
    tick: int
    previous: OperatingMode
    current: OperatingMode
    reason: str


class ModeController:
    def __init__(self, consolidation_every: int) -> None:
        self.mode = OperatingMode.AWAKE
        self.consolidation_every = consolidation_every
        self.ticks_in_mode = 0
        self.history: list[ModeEvent] = []
        self.rest_requirement = 0.0
        self.cognitive_load = 0.0

    def update(self, tick: int, energy: float, rest_need: float, load: float, surprise: float) -> OperatingMode:
        self.ticks_in_mode += 1
        self.rest_requirement = rest_need
        self.cognitive_load = load
        nxt = self.mode
        reason = "hold"
        if self.mode is OperatingMode.AWAKE:
            if rest_need > 0.82 or energy < 0.18:
                nxt, reason = OperatingMode.REST, "homeostatic_rest"
            elif tick > 0 and tick % self.consolidation_every == 0:
                nxt, reason = OperatingMode.CONSOLIDATION, "scheduled_consolidation"
            elif load > 0.9 and surprise < 0.15:
                nxt, reason = OperatingMode.MAINTENANCE, "high_load_calibration"
        elif self.mode is OperatingMode.REST:
            if self.ticks_in_mode > 20 and rest_need < 0.4:
                nxt, reason = OperatingMode.SIMULATION, "dream_after_rest"
            if energy > 0.55 and rest_need < 0.35:
                nxt, reason = OperatingMode.AWAKE, "restored"
        elif self.mode is OperatingMode.CONSOLIDATION:
            if self.ticks_in_mode > 12:
                nxt, reason = OperatingMode.AWAKE, "consolidation_complete"
        elif self.mode is OperatingMode.SIMULATION:
            if self.ticks_in_mode > 16 or surprise > 0.7:
                nxt, reason = OperatingMode.AWAKE, "simulation_complete"
        elif self.mode is OperatingMode.MAINTENANCE:
            if self.ticks_in_mode > 8:
                nxt, reason = OperatingMode.AWAKE, "maintenance_complete"
        if nxt is not self.mode:
            self.history.append(ModeEvent(tick, self.mode, nxt, reason))
            self.mode = nxt
            self.ticks_in_mode = 0
        return self.mode

    def force(self, tick: int, mode: OperatingMode, reason: str) -> None:
        if mode is not self.mode:
            self.history.append(ModeEvent(tick, self.mode, mode, reason))
            self.mode = mode
            self.ticks_in_mode = 0

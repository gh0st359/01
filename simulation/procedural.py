"""Procedural complexity scaling for open development."""

from __future__ import annotations

from simulation.scenarios import ScenarioName, _add_agent, _add_colored_objects, _add_switch_light
from simulation.objects import SimObject
import numpy as np


def scale_world(world, complexity: int) -> None:
    complexity = max(1, min(complexity, 8))
    extra = complexity
    _add_colored_objects(world, world.rng, extra)
    if complexity >= 3:
        _add_switch_light(world)
    if complexity >= 4:
        _add_agent(world, f"npc_{complexity}", np.array([1.5 + complexity, 10.0]), np.array([0.3, 0.8, 0.8]))
    if complexity >= 6:
        world.add_object(
            SimObject(
                f"portal_{complexity}",
                "door",
                np.array([11.0, 11.0]),
                np.zeros(2),
                0.35,
                np.array([0.2, 0.2, 0.2]),
                graspable=False,
            )
        )


def next_open_scenario(tick: int) -> ScenarioName:
    cycle = [
        ScenarioName.NURSERY,
        ScenarioName.SWITCH_LIGHT,
        ScenarioName.HIDDEN_OBJECT,
        ScenarioName.SOCIAL,
        ScenarioName.UNKNOWN_BOX,
        ScenarioName.OPEN_WORLD,
    ]
    return cycle[(tick // 2000) % len(cycle)]

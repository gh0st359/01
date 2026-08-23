"""Configurable developmental scenarios."""

from __future__ import annotations

from enum import Enum

import numpy as np

from simulation.objects import SimObject


class ScenarioName(str, Enum):
    NURSERY = "nursery"
    HIDDEN_OBJECT = "hidden_object"
    SWITCH_LIGHT = "switch_light"
    SOCIAL = "social"
    TOOL = "tool"
    UNKNOWN_BOX = "unknown_box"
    FALSE_BELIEF = "false_belief"
    OPEN_WORLD = "open_world"


def build_scenario(world, scenario: ScenarioName) -> None:
    rng = world.rng
    if scenario is ScenarioName.NURSERY:
        _add_colored_objects(world, rng, n=5)
        _add_switch_light(world)
        _add_box(world, np.array([3.5, 8.0]))
        _add_agent(world, "other_a", np.array([9.0, 3.5]), np.array([0.9, 0.4, 0.3]))
    elif scenario is ScenarioName.HIDDEN_OBJECT:
        _add_colored_objects(world, rng, n=2)
        _add_box(world, np.array([7.0, 7.0]))
        ball = SimObject(
            "target_ball",
            "ball",
            np.array([7.0, 7.0]),
            np.zeros(2),
            0.22,
            np.array([0.95, 0.2, 0.25]),
            inside_of="box_0",
            hidden=True,
            label="",
        )
        world.add_object(ball)
    elif scenario is ScenarioName.SWITCH_LIGHT:
        _add_switch_light(world)
        _add_colored_objects(world, rng, n=1)
    elif scenario is ScenarioName.SOCIAL:
        _add_agent(world, "other_a", np.array([8.5, 4.0]), np.array([0.9, 0.35, 0.3]))
        _add_agent(world, "other_b", np.array([3.0, 9.0]), np.array([0.4, 0.4, 0.95]))
        _add_colored_objects(world, rng, n=3)
    elif scenario is ScenarioName.TOOL:
        stick = SimObject("stick", "tool", np.array([5.0, 7.5]), np.zeros(2), 0.18, np.array([0.6, 0.4, 0.2]), label="")
        world.add_object(stick)
        out = SimObject("out_of_reach", "ball", np.array([10.5, 10.5]), np.zeros(2), 0.2, np.array([0.2, 0.8, 0.3]))
        world.add_object(out)
    elif scenario is ScenarioName.UNKNOWN_BOX:
        _add_box(world, np.array([8.0, 6.0]))
        mystery = SimObject("mystery", "device", np.array([4.0, 4.0]), np.zeros(2), 0.28, np.array([0.7, 0.1, 0.8]), graspable=False)
        world.add_object(mystery)
    elif scenario is ScenarioName.FALSE_BELIEF:
        _add_box(world, np.array([4.0, 4.0]))
        _add_box_named(world, "box_1", np.array([9.0, 8.0]))
        toy = SimObject("toy", "ball", np.array([4.0, 4.0]), np.zeros(2), 0.2, np.array([0.95, 0.8, 0.1]), inside_of="box_0")
        world.add_object(toy)
        _add_agent(world, "witness", np.array([2.5, 9.5]), np.array([0.85, 0.45, 0.2]))
        world.objects["witness"].agent_goal = np.array([4.0, 4.0])
    else:
        _add_colored_objects(world, rng, n=8)
        _add_switch_light(world)
        _add_box(world, np.array([2.8, 2.8]))
        _add_agent(world, "other_a", np.array([10.0, 2.0]), np.array([0.9, 0.3, 0.3]))
        door = SimObject("door", "door", np.array([6.0, 0.6]), np.zeros(2), 0.4, np.array([0.5, 0.35, 0.2]), graspable=False)
        world.add_object(door)


def _add_colored_objects(world, rng, n: int) -> None:
    palette = [
        np.array([0.9, 0.15, 0.12]),
        np.array([0.15, 0.55, 0.95]),
        np.array([0.2, 0.8, 0.25]),
        np.array([0.95, 0.85, 0.15]),
        np.array([0.7, 0.2, 0.85]),
        np.array([0.95, 0.5, 0.1]),
    ]
    kinds = ["ball", "cube", "block", "stone"]
    for i in range(n):
        pos = np.array([float(rng.uniform(1.5, 10.5)), float(rng.uniform(1.5, 10.5))])
        world.add_object(
            SimObject(
                object_id=f"obj_{i}",
                kind=kinds[i % len(kinds)],
                position=pos,
                velocity=np.zeros(2),
                radius=float(rng.uniform(0.18, 0.3)),
                color=palette[i % len(palette)],
            )
        )


def _add_switch_light(world) -> None:
    world.add_object(
        SimObject("switch_0", "switch", np.array([2.2, 6.0]), np.zeros(2), 0.22, np.array([0.85, 0.85, 0.2]), graspable=False)
    )
    world.add_object(
        SimObject("light_0", "light", np.array([2.2, 8.2]), np.zeros(2), 0.3, np.array([1.0, 0.95, 0.55]), graspable=False)
    )


def _add_box(world, pos: np.ndarray) -> None:
    world.add_object(SimObject("box_0", "box", pos, np.zeros(2), 0.4, np.array([0.45, 0.32, 0.18]), graspable=False))


def _add_box_named(world, name: str, pos: np.ndarray) -> None:
    world.add_object(SimObject(name, "box", pos, np.zeros(2), 0.4, np.array([0.4, 0.28, 0.16]), graspable=False))


def _add_agent(world, name: str, pos: np.ndarray, color: np.ndarray) -> None:
    agent = SimObject(name, "agent", pos, np.zeros(2), 0.32, color, graspable=False, is_agent=True)
    agent.agent_goal = pos + np.array([1.0, -1.0])
    world.add_object(agent)

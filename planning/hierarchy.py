"""Hierarchical plans in action/state space — not English strings."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.types import ActionKind, Goal


@dataclass
class PlanStep:
    action: ActionKind
    target: str | None
    expected_delta: float


@dataclass
class Plan:
    goal_id: str
    steps: list[PlanStep] = field(default_factory=list)
    cursor: int = 0

    def current(self) -> PlanStep | None:
        if self.cursor >= len(self.steps):
            return None
        return self.steps[self.cursor]

    def advance(self) -> None:
        self.cursor += 1


class HierarchicalPlanner:
    def plan(self, goal: Goal) -> Plan:
        steps: list[PlanStep] = []
        if goal.origin.value == "exploratory":
            steps = [
                PlanStep(ActionKind.LOOK, goal.target_entity, 0.1),
                PlanStep(ActionKind.APPROACH, goal.target_entity, 0.2),
                PlanStep(ActionKind.INSPECT, goal.target_entity, 0.2),
                PlanStep(ActionKind.EXPERIMENT, goal.target_entity, 0.4),
            ]
        elif goal.origin.value == "epistemic":
            steps = [
                PlanStep(ActionKind.INSPECT, goal.target_entity, 0.2),
                PlanStep(ActionKind.EXPERIMENT, goal.target_entity, 0.3),
                PlanStep(ActionKind.SPEAK, None, 0.2),
            ]
        elif goal.origin.value == "maintenance":
            steps = [PlanStep(ActionKind.WAIT, None, 0.05), PlanStep(ActionKind.MOVE, None, 0.1)]
        elif goal.origin.value == "social":
            steps = [PlanStep(ActionKind.APPROACH, goal.target_entity, 0.15), PlanStep(ActionKind.SPEAK, None, 0.3)]
        else:
            kind = goal.action_kind or ActionKind.MOVE
            steps = [PlanStep(ActionKind.APPROACH, goal.target_entity, 0.2), PlanStep(kind, goal.target_entity, 0.3)]
        return Plan(goal.goal_id, steps)

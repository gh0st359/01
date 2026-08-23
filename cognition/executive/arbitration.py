"""Executive arbitration among competing goals, habits, and exploratory drives."""

from __future__ import annotations

from shared.types import ActionKind, Goal, GoalStatus


class ExecutiveArbitration:
    def choose(self, goals: list[Goal], habit: ActionKind | None, explore_pressure: float) -> Goal | None:
        active = [g for g in goals if g.status is GoalStatus.ACTIVE]
        if not active:
            return None
        ranked = sorted(
            active,
            key=lambda g: g.priority * g.expected_value * g.confidence - 0.15 * g.cost,
            reverse=True,
        )
        top = ranked[0]
        if explore_pressure > 0.7 and top.origin.value != "exploratory":
            exploratory = [g for g in ranked if g.origin.value == "exploratory"]
            if exploratory:
                return exploratory[0]
        if habit is not None and top.action_kind is None:
            top.action_kind = habit
        return top

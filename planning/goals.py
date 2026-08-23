"""Internally generated and externally suggested goals with competition."""

from __future__ import annotations

from shared.types import ActionKind, Goal, GoalOrigin, GoalStatus, Vector
import numpy as np


class GoalSystem:
    def __init__(self, max_goals: int) -> None:
        self.max_goals = max_goals
        self.goals: dict[str, Goal] = {}
        self.next_id = 1

    def form(
        self,
        vector: Vector,
        origin: GoalOrigin,
        priority: float,
        value: float,
        confidence: float,
        cost: float,
        horizon: int,
        tick: int,
        action: ActionKind | None = None,
        target: str | None = None,
        parent: str | None = None,
    ) -> Goal:
        gid = f"goal_{self.next_id}"
        self.next_id += 1
        goal = Goal(
            gid,
            vector.copy(),
            origin,
            priority,
            value,
            confidence,
            cost,
            horizon,
            [],
            GoalStatus.ACTIVE,
            tick,
            target,
            action,
            parent,
        )
        self.goals[gid] = goal
        self._trim()
        return goal

    def compete(self) -> Goal | None:
        active = [g for g in self.goals.values() if g.status is GoalStatus.ACTIVE]
        if not active:
            return None
        return max(active, key=lambda g: g.priority * g.expected_value * g.confidence - 0.2 * g.cost)

    def update_status(self, gid: str, status: GoalStatus) -> None:
        if gid in self.goals:
            self.goals[gid].status = status

    def decay(self) -> None:
        for g in self.goals.values():
            if g.status is GoalStatus.ACTIVE:
                g.priority *= 0.997

    def _trim(self) -> None:
        if len(self.goals) <= self.max_goals:
            return
        weakest = min(self.goals.values(), key=lambda g: (g.status is GoalStatus.ACTIVE, g.priority))
        del self.goals[weakest.goal_id]

    def snapshot(self) -> dict:
        return {
            "next_id": self.next_id,
            "goals": {
                gid: {
                    "vector": g.description_vector,
                    "origin": g.origin.value,
                    "priority": g.priority,
                    "expected_value": g.expected_value,
                    "confidence": g.confidence,
                    "cost": g.cost,
                    "time_horizon": g.time_horizon,
                    "status": g.status.value,
                    "created_tick": g.created_tick,
                    "target": g.target_entity,
                    "action": g.action_kind.value if g.action_kind else None,
                    "parent": g.parent_id,
                }
                for gid, g in self.goals.items()
            },
        }

    def restore(self, data: dict) -> None:
        self.next_id = int(data.get("next_id", 1))
        self.goals = {}
        for gid, blob in data.get("goals", {}).items():
            self.goals[gid] = Goal(
                gid,
                np.asarray(blob["vector"], dtype=np.float64),
                GoalOrigin(blob["origin"]),
                float(blob["priority"]),
                float(blob["expected_value"]),
                float(blob["confidence"]),
                float(blob["cost"]),
                int(blob["time_horizon"]),
                [],
                GoalStatus(blob["status"]),
                int(blob["created_tick"]),
                blob.get("target"),
                ActionKind(blob["action"]) if blob.get("action") else None,
                blob.get("parent"),
            )

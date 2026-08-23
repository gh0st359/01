"""Procedural skills independent of declarative memory."""

from __future__ import annotations

import numpy as np

from shared.types import ActionKind, SkillRecord, Vector


class ProceduralMemory:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.skills: dict[str, SkillRecord] = {}
        self._ensure_primitives()

    def _ensure_primitives(self) -> None:
        for kind in ActionKind:
            sid = f"skill_{kind.value}"
            if sid not in self.skills:
                policy = np.zeros(self.dim)
                policy[hash(kind.value) % self.dim] = 1.0
                self.skills[sid] = SkillRecord(sid, kind.value, policy, 0.2, 0, 0, [], 0.0)

    def practice(self, kind: ActionKind, success: float, tick: int, context: Vector | None = None) -> SkillRecord:
        skill = self.skills[f"skill_{kind.value}"]
        skill.practice_count += 1
        skill.last_used_tick = tick
        skill.competence = 0.92 * skill.competence + 0.08 * success
        skill.success_rate = (skill.success_rate * (skill.practice_count - 1) + success) / skill.practice_count
        if context is not None:
            ctx = _fit(context, self.dim)
            skill.policy = 0.95 * skill.policy + 0.05 * ctx * success
        return skill

    def compose(self, parts: list[ActionKind], name: str) -> SkillRecord:
        vec = np.mean([self.skills[f"skill_{p.value}"].policy for p in parts], axis=0)
        sid = f"skill_{name}"
        rec = SkillRecord(sid, name, vec, 0.25, 0, 0, [f"skill_{p.value}" for p in parts], 0.0)
        self.skills[sid] = rec
        return rec

    def best_for(self, context: Vector) -> SkillRecord:
        ctx = _fit(context, self.dim)
        return max(self.skills.values(), key=lambda s: float(np.dot(s.policy, ctx)) * (0.3 + s.competence))

    def snapshot(self) -> dict:
        return {
            sid: {
                "name_key": s.name_key,
                "policy": s.policy,
                "competence": s.competence,
                "practice_count": s.practice_count,
                "last_used_tick": s.last_used_tick,
                "parent_skills": s.parent_skills,
                "success_rate": s.success_rate,
            }
            for sid, s in self.skills.items()
        }

    def restore(self, data: dict) -> None:
        self.skills = {}
        for sid, blob in data.items():
            self.skills[sid] = SkillRecord(
                sid,
                blob["name_key"],
                np.asarray(blob["policy"], dtype=np.float64),
                float(blob["competence"]),
                int(blob["practice_count"]),
                int(blob["last_used_tick"]),
                list(blob.get("parent_skills", [])),
                float(blob.get("success_rate", 0.0)),
            )


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

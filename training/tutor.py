"""Caregiver pairs words with attended experience. Not a personality prompt."""

from __future__ import annotations

from shared.rng import RNG
from simulation.world import SimulatedWorld


class CaregiverTutor:
    def __init__(self, rng: RNG) -> None:
        self.rng = rng
        self.lessons = 0

    def maybe_label(self, world: SimulatedWorld, stage: str) -> str | None:
        if self.rng.random() > 0.35:
            return None
        body = world.body()
        nearest = None
        best = 1e9
        for obj in world.objects.values():
            if obj.object_id == world.body_id or obj.hidden:
                continue
            d = float(((obj.position - body.position) ** 2).sum() ** 0.5)
            if d < best:
                best, nearest = d, obj
        if nearest is None or best > 2.8:
            return None
        self.lessons += 1
        color = _color_name(nearest.color)
        if stage.endswith("communication") or stage.endswith("abstract") or stage.endswith("open") or stage.endswith("agents"):
            choices = [
                f"this {nearest.kind}",
                f"{color} {nearest.kind}",
                f"that is {nearest.kind}",
                f"look {nearest.kind}",
            ]
            if nearest.kind == "switch":
                choices.append("this switch")
            if nearest.kind == "light":
                choices.append("this light")
            return choices[self.rng.integers(0, len(choices))]
        if "object" in stage or "causal" in stage:
            return f"{color} {nearest.kind}"
        return None

    def maybe_query(self) -> str | None:
        if self.rng.random() > 0.2:
            return None
        return ["what is that", "where ball", "what light"][self.rng.integers(0, 3)]


def _color_name(color) -> str:
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    if r > 0.7 and g < 0.4 and b < 0.4:
        return "red"
    if b > 0.7 and r < 0.5:
        return "blue"
    if g > 0.6 and r < 0.5:
        return "green"
    if r > 0.7 and g > 0.7:
        return "yellow"
    return "thing"

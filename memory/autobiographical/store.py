"""Persistent representation of the organism's own history — not chat logs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.types import DevelopmentalStage, Episode, Vector


@dataclass
class LifeChapter:
    chapter_id: str
    start_tick: int
    end_tick: int
    stage: DevelopmentalStage
    self_centroid: Vector
    notable_episode_ids: list[str]
    mistakes: list[str]
    relationships: dict[str, float]
    belief_changes: int
    skill_gains: dict[str, float]
    summary: Vector


class AutobiographicalMemory:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.chapters: list[LifeChapter] = []
        self.identity_trace: Vector = np.zeros(dim)
        self.relationship_weights: dict[str, float] = {}
        self.milestones: list[dict] = []
        self.self_states: list[tuple[int, Vector]] = []

    def record_self(self, tick: int, self_vec: Vector) -> None:
        self.self_states.append((tick, _fit(self_vec, self.dim)))
        if len(self.self_states) > 400:
            self.self_states = self.self_states[-200:]
        self.identity_trace = 0.995 * self.identity_trace + 0.005 * _fit(self_vec, self.dim)

    def close_chapter(self, stage: DevelopmentalStage, episodes: list[Episode], tick: int) -> LifeChapter:
        start = self.chapters[-1].end_tick if self.chapters else 0
        notable = sorted(episodes, key=lambda e: e.salience + e.prediction_error, reverse=True)[:8]
        centroid = self.identity_trace.copy()
        if notable:
            centroid = np.mean([_fit(e.self_snapshot, self.dim) for e in notable], axis=0)
        chapter = LifeChapter(
            chapter_id=f"chapter_{len(self.chapters)+1}",
            start_tick=start,
            end_tick=tick,
            stage=stage,
            self_centroid=centroid,
            notable_episode_ids=[e.episode_id for e in notable],
            mistakes=[e.episode_id for e in notable if e.prediction_error > 0.4],
            relationships=dict(self.relationship_weights),
            belief_changes=0,
            skill_gains={},
            summary=centroid,
        )
        self.chapters.append(chapter)
        return chapter

    def meet(self, who: str, valence: float) -> None:
        prev = self.relationship_weights.get(who, 0.0)
        self.relationship_weights[who] = 0.9 * prev + 0.1 * valence

    def add_milestone(self, tick: int, name: str, evidence: str, extras: dict | None = None) -> None:
        item = {"tick": tick, "name": name, "evidence": evidence}
        if extras:
            item.update(extras)
        if not any(m["name"] == name for m in self.milestones):
            self.milestones.append(item)

    def snapshot(self) -> dict:
        return {
            "identity_trace": self.identity_trace,
            "relationships": self.relationship_weights,
            "milestones": self.milestones,
            "chapters": [
                {
                    "chapter_id": c.chapter_id,
                    "start_tick": c.start_tick,
                    "end_tick": c.end_tick,
                    "stage": c.stage.value,
                    "self_centroid": c.self_centroid,
                    "notable_episode_ids": c.notable_episode_ids,
                    "mistakes": c.mistakes,
                    "relationships": c.relationships,
                    "summary": c.summary,
                }
                for c in self.chapters
            ],
        }

    def restore(self, data: dict) -> None:
        self.identity_trace = np.asarray(data.get("identity_trace", np.zeros(self.dim)), dtype=np.float64)
        self.relationship_weights = dict(data.get("relationships", {}))
        self.milestones = list(data.get("milestones", []))
        self.chapters = []
        for blob in data.get("chapters", []):
            self.chapters.append(
                LifeChapter(
                    blob["chapter_id"],
                    int(blob["start_tick"]),
                    int(blob["end_tick"]),
                    DevelopmentalStage(blob["stage"]),
                    np.asarray(blob["self_centroid"], dtype=np.float64),
                    list(blob.get("notable_episode_ids", [])),
                    list(blob.get("mistakes", [])),
                    dict(blob.get("relationships", {})),
                    0,
                    {},
                    np.asarray(blob["summary"], dtype=np.float64),
                )
            )


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

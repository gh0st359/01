"""Autobiographically anchored episodic store with similarity retrieval."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from learning.nn import cosine
from shared.config import OrganismConfig
from shared.serialize import from_jsonable, to_jsonable
from shared.types import Episode, OperatingMode, Vector
import orjson


class EpisodicMemory:
    def __init__(self, cfg: OrganismConfig, path: str | Path) -> None:
        self.cfg = cfg
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                tick INTEGER,
                timestamp REAL,
                prediction_error REAL,
                salience REAL,
                uncertainty REAL,
                active_goal TEXT,
                blob BLOB
            )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS ep_tick ON episodes(tick)")
        self.conn.commit()
        self._cache: list[Episode] = []
        self._load_cache()

    def _load_cache(self) -> None:
        rows = self.conn.execute("SELECT blob FROM episodes ORDER BY tick DESC LIMIT ?", (min(800, self.cfg.episodic_capacity),)).fetchall()
        self._cache = [self._decode(r[0]) for r in rows]

    def encode(self, episode: Episode) -> None:
        blob = orjson.dumps(to_jsonable(self._to_dict(episode)))
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?)",
            (
                episode.episode_id,
                episode.tick,
                episode.timestamp,
                episode.prediction_error,
                episode.salience,
                episode.uncertainty,
                episode.active_goal,
                blob,
            ),
        )
        self.conn.commit()
        self._cache.insert(0, episode)
        if len(self._cache) > self.cfg.episodic_capacity:
            self._cache = self._cache[: self.cfg.episodic_capacity]
            old = self.conn.execute("SELECT episode_id FROM episodes ORDER BY salience ASC, tick ASC LIMIT 1").fetchone()
            if old:
                self.conn.execute("DELETE FROM episodes WHERE episode_id=?", (old[0],))
                self.conn.commit()

    def similar(self, query: Vector, k: int = 5) -> list[Episode]:
        scored = [(cosine(_fit(e.compression, query.size), query), e) for e in self._cache]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k] if _ > 0.05]

    def chronological(self, start_tick: int, end_tick: int) -> list[Episode]:
        return [e for e in self._cache if start_tick <= e.tick <= end_tick]

    def salient(self, k: int = 8) -> list[Episode]:
        return sorted(self._cache, key=lambda e: e.salience + e.prediction_error, reverse=True)[:k]

    def all_cached(self) -> list[Episode]:
        return list(self._cache)

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self.conn.close()

    def _to_dict(self, e: Episode) -> dict:
        return {
            "episode_id": e.episode_id,
            "tick": e.tick,
            "timestamp": e.timestamp,
            "world_state_before": e.world_state_before,
            "perception": e.perception,
            "internal_state": e.internal_state,
            "active_goal": e.active_goal,
            "prediction": e.prediction,
            "action": e.action,
            "world_state_after": e.world_state_after,
            "prediction_error": e.prediction_error,
            "value_delta": e.value_delta,
            "people_entities": e.people_entities,
            "uncertainty": e.uncertainty,
            "workspace_summary": e.workspace_summary,
            "self_snapshot": e.self_snapshot,
            "salience": e.salience,
            "mode": e.mode.value,
            "compression": e.compression,
        }

    def _decode(self, blob: bytes) -> Episode:
        d = from_jsonable(orjson.loads(blob))
        return Episode(
            episode_id=d["episode_id"],
            tick=int(d["tick"]),
            timestamp=float(d["timestamp"]),
            world_state_before=np.asarray(d["world_state_before"], dtype=np.float64),
            perception=np.asarray(d["perception"], dtype=np.float64),
            internal_state=np.asarray(d["internal_state"], dtype=np.float64),
            active_goal=d.get("active_goal"),
            prediction=np.asarray(d["prediction"], dtype=np.float64),
            action=np.asarray(d["action"], dtype=np.float64),
            world_state_after=np.asarray(d["world_state_after"], dtype=np.float64),
            prediction_error=float(d["prediction_error"]),
            value_delta=float(d["value_delta"]),
            people_entities=list(d.get("people_entities", [])),
            uncertainty=float(d["uncertainty"]),
            workspace_summary=np.asarray(d["workspace_summary"], dtype=np.float64),
            self_snapshot=np.asarray(d["self_snapshot"], dtype=np.float64),
            salience=float(d["salience"]),
            mode=OperatingMode(d.get("mode", "awake")),
            compression=np.asarray(d["compression"], dtype=np.float64),
        )


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

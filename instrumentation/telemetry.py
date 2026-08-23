"""Researcher telemetry — not accessible to the organism as introspection."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import orjson


class Telemetry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER,
                kind TEXT,
                payload BLOB
            )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS tel_tick ON events(tick)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS tel_kind ON events(kind)")
        self.conn.commit()

    def log(self, tick: int, kind: str, payload: dict) -> None:
        self.conn.execute("INSERT INTO events(tick, kind, payload) VALUES (?,?,?)", (tick, kind, orjson.dumps(payload)))
        if tick % 40 == 0:
            self.conn.commit()

    def query(self, kind: str | None = None, start: int = 0, end: int = 10**12) -> list[dict]:
        if kind:
            rows = self.conn.execute(
                "SELECT tick, kind, payload FROM events WHERE kind=? AND tick BETWEEN ? AND ? ORDER BY tick",
                (kind, start, end),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT tick, kind, payload FROM events WHERE tick BETWEEN ? AND ? ORDER BY tick",
                (start, end),
            ).fetchall()
        out = []
        for tick, k, blob in rows:
            out.append({"tick": tick, "kind": k, "payload": orjson.loads(blob)})
        return out

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

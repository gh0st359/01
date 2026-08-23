"""Structured JSONL logs for a run."""

from __future__ import annotations

from pathlib import Path

import orjson


class JsonlLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("ab")

    def write(self, record: dict) -> None:
        self.handle.write(orjson.dumps(record) + b"\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

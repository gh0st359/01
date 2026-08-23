"""Architectural self-improvement is forked, evaluated, then accepted or rejected."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass
class LineageRecord:
    lineage_id: str
    parent: str | None
    seed: int
    created: float
    accepted: bool | None
    notes: str
    checkpoint: str


class LineageBook:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[LineageRecord] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = [LineageRecord(**r) for r in raw]

    def propose(self, lineage_id: str, parent: str | None, seed: int, checkpoint: str, notes: str) -> LineageRecord:
        rec = LineageRecord(lineage_id, parent, seed, time.time(), None, notes, checkpoint)
        self.records.append(rec)
        self._save()
        return rec

    def decide(self, lineage_id: str, accepted: bool, notes: str = "") -> None:
        for rec in self.records:
            if rec.lineage_id == lineage_id:
                rec.accepted = accepted
                if notes:
                    rec.notes += " | " + notes
        self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps([r.__dict__ for r in self.records], indent=2), encoding="utf-8")

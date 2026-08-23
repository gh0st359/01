"""Machine-generated scientific development journal with evidence links."""

from __future__ import annotations

import json
from pathlib import Path


class DevelopmentJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def record(self, tick: int, milestone: str, evidence: str, extras: dict | None = None) -> None:
        if any(e["milestone"] == milestone for e in self.entries):
            return
        item = {"tick": tick, "milestone": milestone, "evidence": evidence}
        if extras:
            item.update(extras)
        self.entries.append(item)
        self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")

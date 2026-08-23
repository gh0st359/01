"""Archive low-salience episodes to bound memory growth."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def archive_old_episodes(live: Path, archive: Path, keep: int) -> int:
    if not live.exists():
        return 0
    src = sqlite3.connect(live)
    count = src.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    if count <= keep:
        src.close()
        return 0
    archive.parent.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(archive)
    dst.execute("CREATE TABLE IF NOT EXISTS episodes AS SELECT * FROM episodes WHERE 0")
    rows = src.execute(
        "SELECT * FROM episodes ORDER BY salience ASC, tick ASC LIMIT ?",
        (count - keep,),
    ).fetchall()
    if rows:
        dst.executemany("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?)", rows)
        ids = [r[0] for r in rows]
        src.executemany("DELETE FROM episodes WHERE episode_id=?", [(i,) for i in ids])
        src.commit()
        dst.commit()
    moved = len(rows)
    src.close()
    dst.close()
    return moved

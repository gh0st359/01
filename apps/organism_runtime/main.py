"""Continuous organism runtime.

state(t) → state(t+1) continues whether or not a human is speaking.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import uvicorn

from apps.organism_runtime.api import create_app
from apps.organism_runtime.session import OrganismSession
from shared.v2config import load_v2_config


def cognitive_loop(session: OrganismSession, hz: float, stop: threading.Event) -> None:
    period = 1.0 / max(hz, 1.0)
    session.running = True
    while not stop.is_set():
        t0 = time.time()
        session.step()
        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))
    session.running = False
    session.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run organism 01 continuously")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-dir", default="runs/01")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hz", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=0, help="If >0, run this many steps without serving")
    args = parser.parse_args(argv)
    cfg = load_v2_config(args.profile or "development_cpu", seed=args.seed)
    restore = Path(args.run_dir) / "checkpoints" / "latest"
    session = OrganismSession(cfg, args.run_dir, restore if restore.exists() else None)
    if args.steps > 0:
        for _ in range(args.steps):
            session.step()
        session.close()
        return
    stop = threading.Event()
    worker = threading.Thread(target=cognitive_loop, args=(session, args.hz, stop), daemon=True)
    worker.start()
    app = create_app(session)
    app.state.hz = args.hz
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        stop.set()
        worker.join(timeout=5.0)


if __name__ == "__main__":
    main()

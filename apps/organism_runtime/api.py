"""Research interface HTTP/WebSocket API. This is not the organism's mind."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apps.organism_runtime.session import OrganismSession
from core.checkpoint import load_checkpoint_v2, save_checkpoint_v2
from shared.v2config import load_v2_config


class SayBody(BaseModel):
    speaker: str = "human"
    text: str


class ModeBody(BaseModel):
    mode: str


def create_app(session: OrganismSession) -> FastAPI:
    app = FastAPI(title="01 Research Interface", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.session = session
    app.state.hz = 12.0

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return session.snapshot()

    @app.post("/api/say")
    def say(body: SayBody) -> dict[str, str]:
        session.human_say(body.speaker, body.text)
        return {"ok": "queued"}

    @app.post("/api/step")
    def step() -> dict[str, Any]:
        result = session.step()
        return {"tick": result.tick, "utterance": result.utterance, "mode": result.mode.value}

    @app.post("/api/checkpoint")
    def checkpoint() -> dict[str, str]:
        path = save_checkpoint_v2(session.organism, session.run_dir / "checkpoints" / "manual")
        return {"path": str(path)}

    @app.post("/api/restore")
    def restore() -> dict[str, int]:
        load_checkpoint_v2(session.organism, session.run_dir / "checkpoints" / "latest")
        return {"tick": session.organism.tick}

    @app.get("/api/journal")
    def journal() -> list[dict]:
        return session.journal.entries

    @app.get("/api/telemetry/{kind}")
    def tel(kind: str) -> list[dict]:
        return session.telemetry.query(kind=kind)[-200:]

    @app.post("/api/ablate/{component}")
    def ablate(component: str) -> dict[str, list[str]]:
        session.organism.disable(component)
        return {"disabled": sorted(session.organism.disabled)}

    @app.post("/api/enable/{component}")
    def enable(component: str) -> dict[str, list[str]]:
        session.organism.enable(component)
        return {"disabled": sorted(session.organism.disabled)}

    @app.post("/api/hidden")
    def hidden(body: dict) -> dict:
        session.organism.set_hidden_variable(str(body.get("name", "x")), float(body.get("value", 0)))
        return {"hidden": session.organism.hidden_probe}

    ui = Path(__file__).resolve().parents[1] / "research-ui" / "dist"
    if ui.exists():
        app.mount("/", StaticFiles(directory=ui, html=True), name="ui")

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                snap = session.snapshot()
                await socket.send_json(snap)
                await asyncio.sleep(1.0 / app.state.hz)
        except WebSocketDisconnect:
            return

    return app


def build_default_session(profile: str | None = None, seed: int = 1, run_dir: str = "runs/01") -> OrganismSession:
    cfg = load_v2_config(profile or "development_cpu", seed=seed)
    restore = Path(run_dir) / "checkpoints" / "latest"
    return OrganismSession(cfg, run_dir, restore if restore.exists() else None)

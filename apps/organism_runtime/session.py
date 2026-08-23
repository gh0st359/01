"""Integrated runtime session: world + organism + telemetry + checkpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from embodiment.bridge import RealityBridge
from instrumentation.journal import DevelopmentJournal
from instrumentation.log import JsonlLog
from instrumentation.telemetry import Telemetry
from organism.checkpoint import load_checkpoint, save_checkpoint
from organism.loop import Organism, TickResult
from shared.config import OrganismConfig
from shared.rng import RNG
from simulation.scenarios import ScenarioName
from simulation.world import SimulatedWorld
from storage.archive import archive_old_episodes
from training.curriculum import DevelopmentalCurriculum
from training.tutor import CaregiverTutor


class OrganismSession:
    def __init__(self, cfg: OrganismConfig, run_dir: str | Path, restore: str | Path | None = None) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rng = RNG(cfg.seed)
        self.world = SimulatedWorld(cfg, self.rng.spawn(11), ScenarioName.NURSERY)
        self.bridge = RealityBridge(self.world)
        data = self.run_dir / "organism"
        self.organism = Organism(cfg, self.rng.spawn(3), data)
        if restore:
            load_checkpoint(self.organism, restore)
        self.curriculum = DevelopmentalCurriculum(CaregiverTutor(self.rng.spawn(17)))
        self.telemetry = Telemetry(self.run_dir / "telemetry.sqlite")
        self.journal = DevelopmentJournal(self.run_dir / "journal.json")
        self.log = JsonlLog(self.run_dir / "ticks.jsonl")
        self.running = False
        self.last_result: TickResult | None = None

    def step(self) -> TickResult:
        self.curriculum.apply(self.organism, self.world)
        obs = self.bridge.observe()
        result = self.organism.tick_once(obs.sensors)
        self.bridge.act(result.motor)
        if result.utterance:
            self.world.inject_speech("01", result.utterance)
        self.last_result = result
        payload = {
            "tick": result.tick,
            "mode": result.mode.value,
            "pe": result.prediction_error,
            "utterance": result.utterance,
            "spontaneous": result.spontaneous,
            "kinds": result.workspace_kinds,
            "notes": result.notes,
        }
        self.log.write(payload)
        self.telemetry.log(result.tick, "tick", payload)
        if result.utterance:
            self.telemetry.log(
                result.tick,
                "utterance",
                {
                    "text": result.utterance,
                    "intent": result.intent.frame.act.value if result.intent else None,
                    "uncertainty": result.intent.preceding_uncertainty if result.intent else None,
                    "retrievals": result.intent.preceding_retrieval_ids if result.intent else [],
                    "pe": result.intent.preceding_prediction_error if result.intent else None,
                },
            )
        for mile in self.organism.auto.milestones:
            self.journal.record(mile["tick"], mile["name"], mile.get("evidence", ""))
        if result.tick > 0 and result.tick % self.cfg.checkpoint_every == 0:
            save_checkpoint(self.organism, self.run_dir / "checkpoints" / f"tick_{result.tick}")
            archive_old_episodes(self.organism.data_dir / "episodic.sqlite", self.run_dir / "archive" / "episodic.sqlite", self.cfg.episodic_capacity)
        return result

    def human_say(self, speaker: str, text: str) -> None:
        self.world.inject_speech(speaker, text)
        self.organism.ingest_speech(speaker, text)
        self.telemetry.log(self.organism.tick, "human", {"speaker": speaker, "text": text})

    def snapshot(self) -> dict[str, Any]:
        state = self.organism.observe_state()
        state["world"] = self.world.public_view()
        state["light"] = self.world.observe().light_level
        return state

    def close(self) -> None:
        save_checkpoint(self.organism, self.run_dir / "checkpoints" / "latest")
        self.telemetry.close()
        self.log.close()
        self.organism.episodic.close()

"""Integrated runtime session: world + V2 organism + telemetry + checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.checkpoint import load_checkpoint_v2, save_checkpoint_v2
from core.device import resolve_device
from core.organism import OrganismV2, TickResult
from instrumentation.journal import DevelopmentJournal
from instrumentation.log import JsonlLog
from instrumentation.telemetry import Telemetry
from shared.v2config import V2Config, load_v2_config
from simulation.batch import ProceduralBatch
from storage.archive import archive_old_episodes


class OrganismSession:
    def __init__(self, cfg: V2Config | Any, run_dir: str | Path, restore: str | Path | None = None) -> None:
        if not isinstance(cfg, V2Config):
            # Accept leftover OrganismConfig from older scripts.
            profile = getattr(getattr(cfg, "profile", None), "value", None) or "development_cpu"
            mapped = {
                "development": "development_cpu",
                "cpu": "development_cpu",
                "cloud": "cloud_cpu",
                "single_gpu": "gpu_16gb",
                "multi_gpu": "gpu_cloud",
                "high_throughput": "development_cpu",
                "long_running": "cloud_cpu",
            }.get(str(profile), "ci" if str(profile) == "development" else "cloud_cpu")
            cfg = load_v2_config(mapped, seed=getattr(cfg, "seed", 1))
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.world = ProceduralBatch(batch=1, n_objects=cfg.max_objects, vision=cfg.vision_hw, seed=cfg.seed)
        data = self.run_dir / "organism"
        device = resolve_device(amp=cfg.mixed_precision)
        self.organism = OrganismV2(cfg, data, device=device)
        if restore and Path(restore).exists() and (Path(restore) / "weights.pt").exists():
            load_checkpoint_v2(self.organism, restore)
        self.telemetry = Telemetry(self.run_dir / "telemetry.sqlite")
        self.journal = DevelopmentJournal(self.run_dir / "journal.json")
        self.log = JsonlLog(self.run_dir / "ticks.jsonl")
        self.running = False
        self.last_result: TickResult | None = None
        self.caregiver_streams: list[str] = []

    def step(self) -> TickResult:
        obs = self.world.observation(0)
        result = self.organism.tick_once(obs.sensors)
        self.world.step([result.motor])
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
            "notes": {k: v for k, v in result.notes.items() if k != "access"},
        }
        self.log.write(payload)
        self.telemetry.log(result.tick, "tick", payload)
        if result.utterance:
            self.telemetry.log(
                result.tick,
                "utterance",
                {
                    "text": result.utterance,
                    "intent_urgency": result.intent.urgency if result.intent else None,
                    "uncertainty": result.intent.preceding_uncertainty if result.intent else None,
                    "retrievals": result.intent.preceding_retrieval_ids if result.intent else [],
                    "pe": result.intent.preceding_prediction_error if result.intent else None,
                },
            )
        for mile in self.organism.auto.milestones:
            self.journal.record(mile["tick"], mile["name"], mile.get("evidence", ""))
        if result.tick > 0 and result.tick % self.cfg.checkpoint_every == 0:
            save_checkpoint_v2(self.organism, self.run_dir / "checkpoints" / f"tick_{result.tick}")
            epi = self.organism.data_dir / "episodic.sqlite"
            if epi.exists():
                archive_old_episodes(epi, self.run_dir / "archive" / "episodic.sqlite", self.cfg.episodic_capacity)
        return result

    def human_say(self, speaker: str, text: str) -> None:
        self.world.inject_speech(speaker, text)
        self.organism.ingest_speech(speaker, text)
        self.caregiver_streams.append(text)
        self.telemetry.log(self.organism.tick, "human", {"speaker": speaker, "text": text})

    def snapshot(self) -> dict[str, Any]:
        state = self.organism.observe_state()
        state["world"] = self.world.public_view(0)
        state["light"] = self.world.observation(0).light_level
        return state

    def close(self) -> None:
        save_checkpoint_v2(self.organism, self.run_dir / "checkpoints" / "latest")
        self.telemetry.close()
        self.log.close()
        self.organism.episodic.close()

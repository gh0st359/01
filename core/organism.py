"""V2 organism: thin orchestration of learned organs.

Human messages are sensorium events. Cognition continues in silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from core.agency import PredictiveSelf
from core.causal import CausalEnsemble
from core.cognitive import CoreState, MultiscaleCore
from core.continual import ContinualGuard
from core.curiosity import EpistemicEvaluator
from core.device import DeviceBundle, count_parameters, resolve_device, seed_all
from core.goals import GoalProposer
from core.homeo import HomeoState, init_homeo, step_homeo
from core.language import LanguageOrgan, decode_stream, encode_stream
from core.memory import Episode, MemorySystem, RetrievalNet, SemanticInducer
from core.metacog import ConfidenceEstimator
from core.nets import ConvDecoder, ConvEncoder
from core.others import OtherAgentPredictor
from core.planner import CEMPlanner
from core.rssm import RSSM, RSSMState
from core.skills import OptionDiscoverer
from core.slots import SlotAttention
from core.workspace import CANDIDATE_KINDS, LearnedWorkspace
from shared.contracts import MotorCommand, SensorFrame
from shared.types import CommunicativeIntent, OperatingMode, SemanticFrame, SpeechAct
from shared.v2config import V2Config


@dataclass
class TickResult:
    tick: int
    motor: MotorCommand
    utterance: str
    intent: CommunicativeIntent | None
    mode: OperatingMode
    prediction_error: float
    workspace_kinds: list[str]
    spontaneous: bool
    notes: dict[str, Any] = field(default_factory=dict)


class CognitiveNet(nn.Module):
    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.cfg = cfg
        vision_w = max(16, cfg.vision_hw // 8)
        self.encoder = ConvEncoder(cfg.vision_c, vision_w, cfg.slot_dim)
        self.decoder = ConvDecoder(cfg.deter_dim + cfg.stoch_dim, vision_w, cfg.vision_c, cfg.vision_hw)
        self.slots = SlotAttention(cfg.slots, cfg.slot_dim, self.encoder.feat_dim)
        self.rssm = RSSM(cfg.deter_dim, cfg.stoch_dim, cfg.action_dim, cfg.slot_dim * cfg.slots)
        latent = cfg.deter_dim + cfg.stoch_dim
        self.core = MultiscaleCore(cfg.core_dim, latent + cfg.slot_dim + 8)
        self.workspace = LearnedWorkspace(cfg.core_dim)
        self.goals = GoalProposer(cfg.core_dim, cfg.n_goals, cfg.goal_dim)
        self.epistemic = EpistemicEvaluator(latent, cfg.action_dim)
        self.planner = CEMPlanner(cfg.action_dim, cfg.imag_horizon, cfg.cem_pop, cfg.cem_elite, cfg.cem_iters)
        self.language = LanguageOrgan(cfg.lang_hidden, cfg.goal_dim, max_len=24)
        self.self_model = PredictiveSelf(16, cfg.action_dim, cfg.core_dim)
        self.others = OtherAgentPredictor(cfg.slot_dim, cfg.core_dim)
        self.causal = CausalEnsemble(cfg.core_dim, n_vars=8)
        self.options = OptionDiscoverer(latent, cfg.action_dim)
        self.retrieve = RetrievalNet(cfg.core_dim)
        self.semantic = SemanticInducer(cfg.core_dim, n_concepts=16)
        self.meta = ConfidenceEstimator(cfg.core_dim)
        self.audio_enc = nn.GRU(1, cfg.core_dim, batch_first=True)
        self.candidate_proj = nn.ModuleList([nn.Linear(max(cfg.slot_dim, latent, cfg.core_dim, 16), cfg.core_dim) for _ in range(10)])
        self.motor = nn.Tanh()

    def project_cand(self, i: int, x: Tensor, dim: int | None = None) -> Tensor:
        target = self.candidate_proj[i].in_features
        if x.size(-1) < target:
            x = torch.nn.functional.pad(x, (0, target - x.size(-1)))
        elif x.size(-1) > target:
            x = x[..., :target]
        return self.candidate_proj[i](x)


class OrganismV2:
    def __init__(self, cfg: V2Config, data_dir: str | Path, device: DeviceBundle | None = None) -> None:
        seed_all(cfg.seed)
        self.cfg = cfg
        self.name = cfg.name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.device_bundle = device or resolve_device(amp=cfg.mixed_precision)
        self.device = self.device_bundle.device
        self.net = CognitiveNet(cfg).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.guard = ContinualGuard()
        self.memory = MemorySystem(cfg.episodic_capacity)
        self.tick = 0
        self.time = 0.0
        self.lineage_id = f"{cfg.name}-{cfg.seed}"
        self.parent_lineage: str | None = None
        self.disabled: set[str] = set()
        self.hidden_probe: dict[str, float] = {}
        self.pending_speech: list[tuple[str, str]] = []
        self.last_utterance = ""
        self.last_intent: CommunicativeIntent | None = None
        self.spontaneous_count = 0
        self.human_events = 0
        self.param_count = count_parameters(self.net)
        self.pe_trace: list[float] = []
        self.loss_trace: list[float] = []
        self.competence_trace: list[float] = []
        self.dialogue: list[dict] = []
        self.milestones: list[dict] = []
        self.preferences: dict[str, float] = {}
        self.mode = OperatingMode.AWAKE
        self.imagination_used = 0
        self.plan_actions: Tensor | None = None
        b = 1
        self.rssm_state = self.net.rssm.initial(b, self.device)
        self.core_state = self.net.core.initial(b, self.device)
        self.slots: Tensor | None = None
        self.self_h = self.net.self_model.initial(b, self.device)
        self.other_h = self.net.others.initial(b, self.device)
        self.ws_prev = torch.zeros(b, cfg.core_dim, device=self.device)
        self.homeo = init_homeo(b, self.device)
        self.last_action = torch.zeros(b, cfg.action_dim, device=self.device)
        self.last_proprio = torch.zeros(b, 16, device=self.device)
        self.last_obs = torch.zeros(b, cfg.slot_dim * cfg.slots, device=self.device)
        # Compatibility surfaces used by instrumentation / old session code
        self.auto = _AutoShim(self)
        self.episodic = _EpiShim(self)
        self.lexicon = _LexShim()
        self.self_model = _SelfShim(self)
        self.development = _DevShim()

    def ingest_speech(self, speaker: str, text: str) -> None:
        self.pending_speech.append((speaker, text))

    def disable(self, name: str) -> None:
        self.disabled.add(name)

    def enable(self, name: str) -> None:
        self.disabled.discard(name)

    def set_hidden_variable(self, name: str, value: float) -> None:
        self.hidden_probe[name] = value

    def _vision(self, sensors: SensorFrame) -> Tensor:
        h, w, c = sensors.vision_shape
        vis = np.asarray(sensors.vision, dtype=np.float32).reshape(h, w, c)
        if (h, w) != (self.cfg.vision_hw, self.cfg.vision_hw):
            t = torch.tensor(vis, device=self.device).permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=self.cfg.vision_hw, mode="bilinear", align_corners=False)
            return t
        return torch.tensor(vis, device=self.device).permute(2, 0, 1).unsqueeze(0)

    def _proprio(self, sensors: SensorFrame) -> Tensor:
        raw = np.asarray(sensors.proprioception, dtype=np.float32).ravel()
        out = np.zeros(16, dtype=np.float32)
        out[: min(16, raw.size)] = raw[: min(16, raw.size)]
        return torch.tensor(out, device=self.device).unsqueeze(0)

    def tick_once(self, sensors: SensorFrame) -> TickResult:
        self.tick += 1
        self.time += self.cfg.dt
        train = "world_model" not in self.disabled
        self.net.train(train)
        pixels = self._vision(sensors)
        proprio = self._proprio(sensors)
        energy = torch.tensor([float(sensors.energy)], device=self.device)
        incoming = list(self.pending_speech)
        if incoming:
            self.human_events += len(incoming)
        self.pending_speech.clear()

        feats = self.net.encoder.spatial(pixels)
        if "perception" in self.disabled:
            slots = torch.zeros(1, self.cfg.slots, self.cfg.slot_dim, device=self.device)
        else:
            slots = self.net.slots(feats, self.slots)
        self.slots = slots.detach()
        obs = slots.reshape(1, -1)
        action = self.last_action
        nxt, info = self.net.rssm.observe(self.rssm_state, obs, action)
        pe_t = torch.mean((info["pred_obs"] - obs) ** 2)
        pred_obs = info["pred_obs"]
        aux = info["aux"]
        post_lv = info["post_lv"]
        wm_loss = info["recon"] + 0.1 * info["kl"]
        if "world_model" in self.disabled:
            nxt = self.rssm_state
            wm_loss = wm_loss.detach() * 0.0
        self.rssm_state = RSSMState(h=nxt.h.detach(), z=nxt.z.detach())
        latent = nxt.flatten()
        pe = float(pe_t.detach().cpu())
        self.pe_trace.append(pe)
        novelty = float(torch.mean(torch.abs(obs - self.last_obs)).detach().cpu())
        self.last_obs = obs.detach()

        social = torch.tensor([1.0 if incoming else 0.0], device=self.device)
        extras = torch.stack(
            [
                energy.reshape(-1),
                pe_t.detach().reshape(-1),
                torch.tensor([novelty], device=self.device),
                social.reshape(-1),
                torch.mean(post_lv, dim=-1).reshape(-1),
                torch.tensor([float(len(self.memory.episodes)) / max(1, self.cfg.episodic_capacity)], device=self.device),
                self.homeo.sleep.reshape(-1),
                self.homeo.competence.reshape(-1),
            ],
            dim=-1,
        )

        slot_pool = slots.mean(1)
        core_in = torch.cat([latent, slot_pool, extras], dim=-1)
        if "core" in self.disabled:
            core_read = self.core_state.concat()[:, : self.cfg.core_dim]
            new_core = self.core_state
        else:
            new_core, core_read = self.net.core(self.core_state, core_in)
        self.core_state = CoreState(
            *[getattr(new_core, n).detach() for n in ("fast", "act", "work", "goal", "motive", "auto")]
        )

        recalled = []
        if "episodic" not in self.disabled and self.memory.episodes:
            recalled = self.memory.retrieve(self.net.retrieve, core_read.detach(), k=4)

        cands = self._candidates(slots, latent, core_read, proprio, incoming, recalled, pe, novelty)
        if "workspace" in self.disabled:
            ws_content = core_read
            access = torch.ones(1, cands.size(1), device=self.device) / cands.size(1)
            winner = torch.zeros(1, dtype=torch.long, device=self.device)
        else:
            ws = self.net.workspace(cands, core_read, self.ws_prev)
            ws_content = ws.content
            access = ws.access
            winner = ws.winner
        self.ws_prev = ws_content.detach()

        homeo_vec = self.homeo.vector()
        if homeo_vec.dim() == 1:
            homeo_vec = homeo_vec.unsqueeze(0)
        goals = self.net.goals(core_read, homeo_vec)
        plan_action = self._plan(nxt, goals.latent, homeo_vec)
        options = self.net.options.assign(latent.detach(), plan_action.detach())

        self_h, self_est = self.net.self_model.step(self.self_h, proprio, plan_action, None)
        self.self_h = self_h.detach()
        other_obs = slots[:, 1].detach() if slots.size(1) > 1 else slots[:, 0].detach()
        other_h, other_pred = self.net.others.step(self.other_h, other_obs, None)
        self.other_h = other_h.detach()

        causal = self.net.causal(torch.nn.functional.pad(latent, (0, max(0, 8 - latent.size(-1))))[:, :8])
        conf = self.net.meta(core_read, extras[:, :6] if extras.size(-1) >= 6 else torch.nn.functional.pad(extras, (0, 6 - extras.size(-1))))

        utterance = ""
        intent = None
        comm = None
        lang_loss = torch.zeros((), device=self.device)
        grounded = core_read
        if incoming and "language" not in self.disabled:
            text = incoming[-1][1]
            ids = encode_stream(text, self.net.language.max_len, self.device).unsqueeze(0)
            _, grounded = self.net.language.comprehend(ids, ws_content)
            comm = self.net.language.intention_from(core_read, grounded)
            logits, _ = self.net.language.realize(comm, teacher=ids)
            lang_loss = self.net.language.loss(logits, ids)
            self.dialogue.append({"tick": self.tick, "speaker": incoming[-1][0], "text": text, "origin": "human"})
            self._preference_from_stream(text, float(goals.value.max().detach()))
        elif "language" not in self.disabled:
            comm = self.net.language.intention_from(core_read, ws_content)

        emit = False
        if comm is not None:
            emit_p = float(comm.should_emit.detach().cpu())
            # Learned communicative value, not tick modulus. Threshold is on the learned head.
            if incoming or emit_p > 0.55:
                logits, ids = self.net.language.realize(comm)
                token_conf = float(torch.softmax(logits[0], dim=-1).max(dim=-1).values.mean().detach().cpu())
                utterance = decode_stream(ids[0]) if token_conf > 0.04 else ""
                if utterance == self.last_utterance and not incoming:
                    utterance = ""
                emit = bool(utterance)
            intent = CommunicativeIntent(
                frame=SemanticFrame(
                    SpeechAct.UNKNOWN,
                    comm.semantic.detach().cpu().numpy().ravel().astype(np.float64),
                    {"intention": comm.intention.detach().cpu().numpy().ravel().astype(np.float64)},
                    [],
                    emit_p,
                    self.tick,
                ),
                urgency=emit_p,
                social_address=incoming[-1][0] if incoming else None,
                formed_tick=self.tick,
                workspace_hash=str(int(winner[0].item())),
                preceding_uncertainty=float(conf.prediction.detach().cpu()) if conf.prediction.ndim == 0 else float(conf.prediction[0].detach().cpu()),
                preceding_retrieval_ids=[str(e.tick) for e in recalled],
                preceding_prediction_error=pe,
            )

        if emit:
            if not incoming:
                self.spontaneous_count += 1
            self.dialogue.append(
                {
                    "tick": self.tick,
                    "speaker": "01",
                    "text": utterance,
                    "origin": "organism" if incoming else "spontaneous",
                    "intent_norm": float(torch.norm(comm.intention).detach().cpu()) if comm is not None else 0.0,
                    "semantic_norm": float(torch.norm(comm.semantic).detach().cpu()) if comm is not None else 0.0,
                    "emit": float(comm.should_emit.detach().cpu()) if comm is not None else 0.0,
                }
            )
        self.last_utterance = utterance
        self.last_intent = intent

        agency = float(self_est.agency.mean().detach().cpu())
        competence = float(1.0 / (1.0 + pe))
        self.homeo = step_homeo(
            self.homeo,
            energy,
            pe_t.detach().reshape(1),
            torch.tensor([novelty], device=self.device),
            social,
            self_est.agency.mean().reshape(1),
            torch.tensor([competence], device=self.device),
        )
        self.competence_trace.append(competence)

        if "consolidation" not in self.disabled and self.homeo.sleep.mean() > 0.75:
            self.mode = OperatingMode.CONSOLIDATION
            self._consolidate()
            self.homeo.sleep = self.homeo.sleep * 0.2
        elif float(energy) < 0.25:
            self.mode = OperatingMode.REST
        else:
            self.mode = OperatingMode.AWAKE

        if train:
            opt_loss = self.net.options.reinforce(options.ids, torch.tensor([competence], device=self.device))
            pix_rec = self.net.decoder(latent)
            rec_loss = torch.mean((pix_rec - pixels) ** 2)
            ewc = self.guard.penalty(self.net)
            loss = wm_loss + 0.2 * rec_loss + 0.3 * lang_loss + 0.05 * opt_loss + 1e-4 * ewc
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.guard.accumulate_fisher(self.net)
            self.loss_trace.append(float(loss.detach().cpu()))
        else:
            self.loss_trace.append(pe)

        if "episodic" not in self.disabled:
            self.memory.write(
                Episode(
                    tick=self.tick,
                    latent=latent.detach().cpu().numpy(),
                    slots=slots.detach().cpu().numpy(),
                    core=core_read.detach().cpu().numpy(),
                    proprio=proprio.detach().cpu().numpy(),
                    goal=goals.latent.detach().cpu().numpy(),
                    action=plan_action.detach().cpu().numpy(),
                    pred_err=pe,
                    uncertainty=float(torch.mean(post_lv).detach().cpu()),
                    workspace=ws_content.detach().cpu().numpy(),
                    social=np.array([1.0 if incoming else 0.0]),
                    outcome=obs.detach().cpu().numpy(),
                )
            )

        self.last_action = plan_action.detach()
        self.last_proprio = proprio.detach()
        motor = self._motor(plan_action, emit)
        kinds = [CANDIDATE_KINDS[int(winner[0].item()) % len(CANDIDATE_KINDS)]]
        if self.tick == 1:
            self.milestones.append({"tick": 1, "name": "birth", "evidence": f"params={self.param_count}"})
        if incoming and not any(m["name"] == "first_grounded_stream" for m in self.milestones):
            self.milestones.append({"tick": self.tick, "name": "first_grounded_stream", "evidence": incoming[-1][1][:32]})
        return TickResult(
            tick=self.tick,
            motor=motor,
            utterance=utterance,
            intent=intent,
            mode=self.mode,
            prediction_error=pe,
            workspace_kinds=kinds,
            spontaneous=emit and not incoming,
            notes={
                "novelty": novelty,
                "agency": agency,
                "info_value": float(goals.info_value.max().detach().cpu()),
                "goal_value": float(goals.value.max().detach().cpu()),
                "emit_p": float(comm.should_emit.detach().cpu()) if comm is not None else 0.0,
                "params": self.param_count,
                "device": self.device_bundle.kind,
                "slots_norm": float(torch.norm(slots).detach().cpu()),
                "access": access.detach().cpu().tolist()[0],
                "stage": self.development.stage,
            },
        )

    def _candidates(
        self,
        slots: Tensor,
        latent: Tensor,
        core: Tensor,
        proprio: Tensor,
        incoming: list,
        recalled: list,
        pe: float,
        novelty: float,
    ) -> Tensor:
        dim = self.cfg.core_dim
        parts = [
            slots.mean(1),
            torch.zeros(1, dim, device=self.device),  # audio placeholder filled below
            proprio,
            torch.tensor(recalled[0].core, dtype=core.dtype, device=core.device) if recalled else core,
            latent,
            slots[:, min(1, slots.size(1) - 1)],
            core,
            latent,
            torch.full((1, dim), pe, device=self.device),
            core,
        ]
        if incoming:
            parts[1] = core
        out = []
        for i, p in enumerate(parts):
            if p.dim() == 1:
                p = p.unsqueeze(0)
            out.append(self.net.project_cand(i, p).unsqueeze(1))
        return torch.cat(out, dim=1)

    def _plan(self, state: RSSMState, goal: Tensor, homeo: Tensor) -> Tensor:
        if "imagination" in self.disabled or "world_model" in self.disabled:
            return torch.tanh(goal[:, : self.cfg.action_dim] if goal.size(-1) >= self.cfg.action_dim else torch.nn.functional.pad(goal, (0, self.cfg.action_dim - goal.size(-1))))
        # Replan when communicative/homeostatic pressure is high or plan exhausted
        need = self.plan_actions is None or self.tick % max(2, self.cfg.imag_horizon // 3) == 0
        if need:
            plan = self.net.planner.plan(self.net.rssm, state, goal, homeo, self.net.epistemic)
            self.plan_actions = plan.actions.detach()
            self.imagination_used += 1
        assert self.plan_actions is not None
        return self.plan_actions[0]

    def _motor(self, action: Tensor, speak: bool) -> MotorCommand:
        a = action.detach().cpu().numpy().ravel()
        raw = np.zeros(max(8, self.cfg.action_dim), dtype=np.float64)
        raw[: a.size] = a
        return MotorCommand(
            linear_velocity=float(np.clip(raw[0], -1, 1)),
            angular_velocity=float(np.clip(raw[1], -1, 1)),
            gripper=float(np.clip(0.5 + 0.5 * raw[2], 0, 1)),
            interact=float(np.clip(0.5 + 0.5 * raw[3], 0, 1)),
            speak=1.0 if speak else float(max(0.0, raw[4])),
            look_heading=float(raw[5] * 0.4),
            raw=raw,
            kind_hint="latent",
            tick=self.tick,
        )

    def _preference_from_stream(self, text: str, value: float) -> None:
        # Association by co-occurrence with value, not word-class lists.
        key = f"stream_{abs(hash(text)) % 10_000}"
        self.preferences[key] = 0.9 * self.preferences.get(key, 0.0) + 0.1 * value

    def _consolidate(self) -> None:
        if len(self.memory.episodes) < 4:
            return
        recent = self.memory.episodes[-min(32, len(self.memory.episodes)) :]
        # Prioritize high prediction-error episodes
        recent = sorted(recent, key=lambda e: e.pred_err, reverse=True)[:16]
        cores = torch.tensor(np.stack([e.core.reshape(-1)[: self.cfg.core_dim] for e in recent]), dtype=torch.float32, device=self.device)
        if cores.size(-1) < self.cfg.core_dim:
            cores = torch.nn.functional.pad(cores, (0, self.cfg.core_dim - cores.size(-1)))
        assign, concepts = self.net.semantic(cores)
        self.memory.semantic_bank.append(concepts.detach().cpu().numpy())
        self.guard.consolidate(self.net)
        if not any(m["name"] == "first_consolidation" for m in self.milestones):
            self.milestones.append({"tick": self.tick, "name": "first_consolidation", "evidence": f"n={len(recent)}"})

    def observe_state(self) -> dict[str, Any]:
        access = []
        kinds = list(CANDIDATE_KINDS)
        hs = self.homeo.vector().detach().cpu().numpy().ravel()
        names = ["energy", "saturation", "novelty_dep", "pred_instability", "social", "sleep", "competence", "control", "surprise", "valence"]
        homeo = {n: float(hs[i]) for i, n in enumerate(names) if i < hs.size}
        slot_list = []
        if self.slots is not None:
            for i, s in enumerate(self.slots[0].detach().cpu().numpy()):
                slot_list.append({"id": f"slot_{i}", "visible": True, "uncertainty": float(np.std(s)), "location": [0.0, 0.0], "self": i == 0, "agent": False, "permanence": 0.0, "norm": float(np.linalg.norm(s))})
        last_intent = None
        if self.last_intent is not None:
            last_intent = {
                "act": "learned",
                "urgency": self.last_intent.urgency,
                "uncertainty": self.last_intent.preceding_uncertainty,
                "retrievals": self.last_intent.preceding_retrieval_ids,
                "pe": self.last_intent.preceding_prediction_error,
                "semantic_norm": float(np.linalg.norm(self.last_intent.frame.predicate)),
            }
        turns = [{"tick": d["tick"], "speaker": d["speaker"], "text": d["text"], "act": d.get("origin")} for d in self.dialogue[-40:]]
        return {
            "tick": self.tick,
            "time": self.time,
            "name": self.name,
            "lineage": self.lineage_id,
            "mode": self.mode.value,
            "stage": self.development.stage,
            "prediction_error": self.pe_trace[-1] if self.pe_trace else 0.0,
            "uncertainty": homeo.get("pred_instability", 0.0),
            "workspace": {
                "kinds": kinds,
                "scores": access,
                "ignition": float(self.ws_prev.norm().detach().cpu()),
                "broadcast": self.ws_prev.detach().cpu().numpy().ravel().tolist()[:64],
                "sources": kinds,
            },
            "motivation": homeo,
            "intrinsic": {
                "total": homeo.get("valence", 0.0),
                "progress": self.competence_trace[-1] if self.competence_trace else 0.0,
                "novelty": homeo.get("novelty_dep", 0.0),
                "weights": {},
            },
            "goals": [{"id": "latent", "origin": "learned", "priority": 1.0, "status": "active", "action": None}],
            "entities": slot_list,
            "self": {
                "identity": self.self_h.detach().cpu().numpy().ravel().tolist()[:32],
                "capability": homeo.get("competence", 0.0),
                "controllability": homeo.get("control", 0.0),
                "agency": homeo.get("control", 0.0),
                "continuity": float(self.tick),
            },
            "meta": {"perceptual": 0.0, "confidence_trace": self.pe_trace[-20:]},
            "lexicon_size": 0,
            "lexicon_words": [],
            "episodic_count": len(self.memory.episodes),
            "semantic_count": len(self.memory.semantic_bank),
            "beliefs": [],
            "last_utterance": self.last_utterance,
            "last_intent": last_intent,
            "dialogue": {"partner": turns[-1]["speaker"] if turns else None, "turns": turns},
            "imagination": [{"kind": "rssm", "value": 0.0, "info": 0.0, "pe": self.pe_trace[-1] if self.pe_trace else 0.0}],
            "causal": {},
            "development": {"stage": self.development.stage, "metrics": {"competence": self.competence_trace[-1] if self.competence_trace else 0.0}, "history": []},
            "milestones": self.milestones,
            "relationships": {},
            "preferences": self.preferences,
            "spontaneous_utterances": self.spontaneous_count,
            "attention": None,
            "working_memory": [],
            "core_norm": float(self.core_state.concat().norm().detach().cpu()),
            "hidden_probe": dict(self.hidden_probe),
            "v2": {
                "params": self.param_count,
                "device": self.device_bundle.kind,
                "device_name": self.device_bundle.name,
                "imagination_used": self.imagination_used,
                "loss": self.loss_trace[-1] if self.loss_trace else 0.0,
            },
        }


class _AutoShim:
    def __init__(self, org: OrganismV2) -> None:
        self.org = org

    @property
    def milestones(self) -> list[dict]:
        return self.org.milestones


class _EpiShim:
    def __init__(self, org: OrganismV2) -> None:
        self.org = org

    def count(self) -> int:
        return len(self.org.memory.episodes)

    def close(self) -> None:
        return None


class _LexShim:
    entries: dict = {}

    def known(self, _w: str) -> bool:
        return False


class _SelfShim:
    def __init__(self, org: OrganismV2) -> None:
        self.org = org

    @property
    def identity(self) -> np.ndarray:
        return self.org.self_h.detach().cpu().numpy().ravel().astype(np.float64)

    @identity.setter
    def identity(self, value: np.ndarray) -> None:
        t = torch.tensor(np.asarray(value), dtype=self.org.self_h.dtype, device=self.org.device)
        if t.numel() >= self.org.self_h.numel():
            self.org.self_h = t.view_as(self.org.self_h)


class _DevShim:
    stage = "competence"

    def snapshot(self) -> dict:
        return {"stage": self.stage, "metrics": {}, "history": []}

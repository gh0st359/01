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
from core.objectives import OrganLosses, mse, zero
from core.referential import ReferentialChannel, halo_slot_target
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
        obs_dim = cfg.slot_dim * cfg.slots
        self.ws_pred = nn.Linear(cfg.core_dim, obs_dim)
        self.ret_mix = nn.Sequential(nn.Linear(cfg.core_dim * 2, cfg.core_dim), nn.SiLU(), nn.Linear(cfg.core_dim, obs_dim))
        self.refer = ReferentialChannel(cfg.slot_dim, cfg.lang_hidden)
        self.causal_read = nn.Linear(latent, 8)

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
        self.invert_controls = False
        self.bptt_loss: Tensor | None = None
        self.bptt_n = 0
        self.prev: dict[str, Tensor] = {}
        self.organ_trace: list[dict[str, float]] = []
        self.self_pe_trace: list[float] = []
        self.ref_acc_trace: list[float] = []
        self.probe_choices: list[int] = []
        self.ig_margin_trace: list[float] = []
        self._credit_buf: list[dict[str, Any]] = []
        self.concepts: Tensor | None = None
        self.concept_collapse: float = 1.0
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
        self.slots = slots
        obs = slots.reshape(1, -1)
        action = self.last_action
        nxt, info = self.net.rssm.observe(self.rssm_state, obs, action)
        pe_t = torch.mean((info["pred_obs"] - obs) ** 2)
        post_lv = info["post_lv"]
        wm_loss = info["recon"] + 0.1 * info["kl"]
        if "world_model" in self.disabled:
            nxt = self.rssm_state
            wm_loss = wm_loss.detach() * 0.0
        self.rssm_state = nxt
        latent = nxt.flatten()
        pe = float(pe_t.detach().cpu())
        self.pe_trace.append(pe)
        novelty = float(torch.mean(torch.abs(obs - self.last_obs)).detach().cpu())
        delayed = self._delayed_losses(obs, proprio, slots, latent, pe_t)

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
        self.core_state = new_core

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
        self.ws_prev = ws_content
        ws_pred = self.net.ws_pred(ws_content)

        homeo_vec = self.homeo.vector()
        if homeo_vec.dim() == 1:
            homeo_vec = homeo_vec.unsqueeze(0)
        goals = self.net.goals(core_read, homeo_vec)
        plan_action = self._plan(nxt, goals.latent, homeo_vec)
        plan_action, probed, ig_scores = self.net.epistemic.pick_experiment(latent, plan_action)
        self.probe_choices.append(1 if probed else 0)
        self.ig_margin_trace.append(float((ig_scores[2] - ig_scores[0]).detach().cpu()))
        options = self.net.options.assign(latent, plan_action)

        self_h_in = self.self_h.detach()
        other_h_in = self.other_h.detach()
        self_h, self_est = self.net.self_model.step(self.self_h, proprio, plan_action, None)
        self.self_h = self_h
        other_obs = slots[:, min(1, slots.size(1) - 1)]
        other_h, other_pred = self.net.others.step(self.other_h, other_obs, None)
        self.other_h = other_h

        causal_x = torch.tanh(self.net.causal_read(latent))
        causal = self.net.causal(causal_x, intervention=torch.nn.functional.pad(plan_action, (0, max(0, 8 - plan_action.size(-1))))[:, :8])
        conf = self.net.meta(core_read, extras[:, :6] if extras.size(-1) >= 6 else torch.nn.functional.pad(extras, (0, 6 - extras.size(-1))))

        utterance = ""
        intent = None
        comm = None
        ref_loss = zero(self.device)
        ref_acc = 0.0
        target_slot = halo_slot_target(pixels, feats, slots)
        if incoming and "language" not in self.disabled:
            text = incoming[-1][1]
            ids = encode_stream(text, self.net.language.max_len, self.device).unsqueeze(0)
            _, grounded = self.net.language.comprehend(ids, ws_content)
            comm = self.net.language.intention_from(core_read, grounded)
            if target_slot is not None:
                ref_loss, acc_t = self.net.refer.listen_loss(grounded, slots, target_slot)
                ref_acc = float(acc_t.detach().cpu())
            self.dialogue.append({"tick": self.tick, "speaker": incoming[-1][0], "text": text, "origin": "human"})
            self._preference_from_stream(text, float(goals.value.max().detach()))
        elif "language" not in self.disabled:
            if target_slot is not None:
                attended = slots[torch.arange(slots.size(0), device=self.device), target_slot]
                comm = self.net.language.intention_from(core_read, attended)
                cyc, acc_t = self.net.refer.speak_cycle_loss(self.net.language, comm, slots, target_slot)
                ref_loss = ref_loss + cyc
                ref_acc = float(acc_t.detach().cpu())
            else:
                comm = self.net.language.intention_from(core_read, ws_content)
        self.ref_acc_trace.append(ref_acc)

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
            energy.detach(),
            pe_t.detach().reshape(1),
            torch.tensor([novelty], device=self.device),
            social.detach(),
            self_est.agency.mean().reshape(1).detach(),
            torch.tensor([competence], device=self.device),
        )
        self.competence_trace.append(competence)

        sem_loss = zero(self.device)
        if "consolidation" not in self.disabled and (self.tick % max(1, getattr(self.cfg, "consolidate_every", 16)) == 0):
            self.mode = OperatingMode.CONSOLIDATION
            sem_loss = self._consolidate()
        elif float(energy) < 0.25:
            self.mode = OperatingMode.REST
        else:
            self.mode = OperatingMode.AWAKE

        meta_target = torch.stack(
            [
                torch.exp(-pe_t).reshape(-1),
                torch.tensor([1.0 if recalled else 0.2], device=self.device),
                torch.exp(-pe_t).reshape(-1),
                torch.tensor([competence], device=self.device),
                1.0 - causal.disagreement.reshape(-1)[:1],
            ],
            dim=-1,
        )
        if meta_target.size(-1) < 5:
            meta_target = torch.nn.functional.pad(meta_target, (0, 5 - meta_target.size(-1)))
        meta_loss = self.net.meta.loss(conf, meta_target[:, :5])
        cur_loss = self.net.epistemic.outcome_loss(latent, plan_action, latent, intervened=float(plan_action[0, 3].detach()) > 0.4)
        if "latent" in self.prev:
            cur_loss = self.net.epistemic.outcome_loss(self.prev["latent"], self.prev["action"], latent, intervened=bool(self.prev.get("probed", torch.zeros(1))[0] > 0.5))

        organs = OrganLosses(
            world=wm_loss + 0.2 * torch.mean((self.net.decoder(latent) - pixels) ** 2),
            self_pred=delayed["self_pred"],
            agency=delayed["agency"],
            other=delayed["other"],
            causal=delayed["causal"],
            meta=meta_loss,
            workspace=delayed["workspace"],
            goal=delayed["goal"],
            semantic=sem_loss,
            retrieve=delayed["retrieve"],
            referential=ref_loss,
            curiosity=cur_loss,
        )
        self.organ_trace.append(organs.as_dict())
        self.self_pe_trace.append(float(delayed["self_raw"]))

        if train:
            opt_loss = self.net.options.reinforce(options.ids, torch.tensor([competence], device=self.device))
            ewc = self.guard.penalty(self.net)
            loss = organs.total() + 0.05 * opt_loss + 1e-4 * ewc
            self._accumulate(loss)
            self.loss_trace.append(float(loss.detach().cpu()))
        else:
            self.loss_trace.append(pe)

        self.prev = {
            "proprio": proprio.detach(),
            "action": plan_action.detach(),
            "self_h": self_h_in,
            "other_obs": other_obs.detach(),
            "other_h": other_h_in,
            "ws_content": ws_content.detach(),
            "latent_in": latent.detach(),
            "latent": latent.detach(),
            "goal_in": core_read.detach(),
            "homeo": homeo_vec.detach(),
            "core": core_read.detach(),
            "probed": torch.tensor([1.0 if probed else 0.0], device=self.device),
            "recalled": (
                torch.tensor(recalled[0].core.reshape(-1)[: self.cfg.core_dim], dtype=core_read.dtype, device=self.device).unsqueeze(0)
                if recalled
                else core_read.detach()
            ),
        }

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

        self._credit_buf.append(
            {
                "goal_in": core_read.detach(),
                "homeo": homeo_vec.detach(),
                "pe": pe,
                "latent": latent.detach(),
                "action": plan_action.detach(),
            }
        )
        if len(self._credit_buf) > 48:
            self._credit_buf = self._credit_buf[-48:]
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
                "ref_acc": ref_acc,
                "probed": probed,
                "self_pe": self.self_pe_trace[-1] if self.self_pe_trace else 0.0,
                "ig_margin": self.ig_margin_trace[-1] if self.ig_margin_trace else 0.0,
                "organs": self.organ_trace[-1] if self.organ_trace else {},
                "concepts": 0 if self.concepts is None else int(self.concepts.size(0)),
                "concept_collapse": self.concept_collapse,
            },
        )

    def _delayed_losses(self, obs: Tensor, proprio: Tensor, slots: Tensor, latent: Tensor, pe_t: Tensor) -> dict[str, Tensor]:
        z = zero(self.device)
        out: dict[str, Tensor] = {
            "self_pred": z,
            "agency": z,
            "other": z,
            "causal": z,
            "workspace": z,
            "goal": z,
            "retrieve": z,
            "self_raw": z,
        }
        p = self.prev
        if not p:
            return out
        # Recompute last-tick heads from detached inputs so credit is live this backward.
        if {"proprio", "action", "self_h"} <= p.keys():
            _, est = self.net.self_model.step(p["self_h"], p["proprio"], p["action"], proprio)
            err = mse(est.next_proprio, proprio)
            out["self_pred"] = err
            out["self_raw"] = err.detach()
            moved = torch.mean((proprio - p["proprio"]) ** 2)
            actn = torch.mean(p["action"] ** 2) + 1e-3
            contingent = torch.clamp(moved / actn, 0.0, 1.0)
            out["agency"] = mse(est.agency.mean().reshape(1), contingent.reshape(1))
        if {"other_obs", "other_h"} <= p.keys():
            cur_other = slots[:, min(1, slots.size(1) - 1)].detach()
            _, op = self.net.others.step(p["other_h"], p["other_obs"], cur_other)
            out["other"] = mse(op.next_obs, cur_other)
        if "ws_content" in p:
            out["workspace"] = mse(self.net.ws_pred(p["ws_content"]), obs.detach())
        if "latent_in" in p and "action" in p:
            cx = torch.tanh(self.net.causal_read(p["latent_in"]))
            now = torch.tanh(self.net.causal_read(latent.detach()))
            view = self.net.causal(cx, intervention=torch.nn.functional.pad(p["action"], (0, max(0, 8 - p["action"].size(-1))))[:, :8])
            out["causal"] = mse(view.observational, now)
            if float(p.get("probed", torch.zeros(1))[0]) > 0.5:
                out["causal"] = out["causal"] + mse(view.interventional, now)
        if "goal_in" in p and "homeo" in p:
            # n-step: value at t-k must predict PE drop realized k ticks later.
            lag = 4 if len(self._credit_buf) >= 4 else 1
            src = self._credit_buf[-lag] if self._credit_buf else p
            g_in = src.get("goal_in", p["goal_in"])
            h_in = src.get("homeo", p["homeo"])
            g = self.net.goals(g_in, h_in)
            start_pe = float(src.get("pe", self.pe_trace[-2] if len(self.pe_trace) > 1 else 0.0))
            now_pe = float(pe_t.detach())
            ret = start_pe - now_pe
            tgt = torch.tensor([ret], device=self.device, dtype=g.value.dtype)
            out["goal"] = mse(g.value.max(dim=-1).values, tgt)
        if "core" in p and "recalled" in p:
            mix = self.net.ret_mix(torch.cat([self._fit_vec(p["core"], self.cfg.core_dim), self._fit_vec(p["recalled"], self.cfg.core_dim)], dim=-1))
            out["retrieve"] = mse(mix, obs.detach())
        return out

    def _fit_vec(self, x: Tensor, dim: int) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(-1) < dim:
            return torch.nn.functional.pad(x, (0, dim - x.size(-1)))
        return x[..., :dim]

    def _accumulate(self, loss: Tensor) -> None:
        # One-step backward plus periodic replay. Full multi-tick graphs through
        # shared GRU cells hit inplace-version errors; delayed losses + replay
        # assign credit to earlier organs without keeping the live graph.
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.guard.accumulate_fisher(self.net)
        self._detach_recurrent()
        self.bptt_n += 1
        horizon = max(4, int(getattr(self.cfg, "bptt", 4)))
        if self.bptt_n % horizon == 0:
            self._replay_credit()

    def _detach_recurrent(self) -> None:
        self.rssm_state = RSSMState(h=self.rssm_state.h.detach(), z=self.rssm_state.z.detach())
        self.core_state = CoreState(*[getattr(self.core_state, n).detach() for n in ("fast", "act", "work", "goal", "motive", "auto")])
        self.self_h = self.self_h.detach()
        self.other_h = self.other_h.detach()
        self.ws_prev = self.ws_prev.detach()
        if self.slots is not None:
            self.slots = self.slots.detach()
        self.last_action = self.last_action.detach()
        self.last_proprio = self.last_proprio.detach()
        self.prev = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in self.prev.items()}
        self.homeo = HomeoState(
            *[getattr(self.homeo, n).detach() for n in (
                "energy", "saturation", "novelty_dep", "pred_instability", "social",
                "sleep", "competence", "control", "surprise", "valence",
            )]
        )

    def _replay_credit(self) -> None:
        """Replay stored episodes so later outcomes train earlier representations."""
        eps = self.memory.episodes
        if len(eps) < 8:
            return
        d = self.cfg.deter_dim + self.cfg.stoch_dim
        loss = zero(self.device)
        for lag in (1, 4, 16, 64):
            if len(eps) <= lag + 3:
                continue
            step = max(1, lag // 2)
            idxs = list(range(0, len(eps) - lag, step))[-12:]
            lat = self._stack_field([eps[i].latent for i in idxs], d)
            nxt = self._stack_field([eps[i + lag].latent for i in idxs], d)
            act = self._stack_field([eps[i].action for i in idxs], self.cfg.action_dim)
            loss = loss + self.net.epistemic.outcome_loss(lat, act, nxt, intervened=False)
        recent = eps[-8:]
        h = self.net.self_model.initial(1, self.device)
        for i in range(len(recent) - 1):
            pr = self._stack_field([recent[i].proprio], 16)
            ac = self._stack_field([recent[i].action], self.cfg.action_dim)
            nxt_p = self._stack_field([recent[i + 1].proprio], 16)
            h, est = self.net.self_model.step(h, pr, ac, nxt_p)
            loss = loss + mse(est.next_proprio, nxt_p)
        cores = self._stack_field([e.core for e in eps[-min(16, len(eps)) :]], self.cfg.core_dim)
        assign, concepts = self.net.semantic(cores)
        loss = loss + mse(assign @ concepts, cores)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

    def _stack_field(self, arrs: list, dim: int) -> Tensor:
        rows = []
        for raw in arrs:
            v = np.asarray(raw, dtype=np.float32).reshape(-1)
            if v.size < dim:
                v = np.pad(v, (0, dim - v.size))
            rows.append(v[:dim])
        return torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device)

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
            raw = goal[:, : self.cfg.action_dim] if goal.size(-1) >= self.cfg.action_dim else torch.nn.functional.pad(goal, (0, self.cfg.action_dim - goal.size(-1)))
            return torch.tanh(raw).detach()
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
        lin = float(np.clip(raw[0], -1, 1))
        ang = float(np.clip(raw[1], -1, 1))
        if self.invert_controls:
            lin, ang = -lin, -ang
            raw = raw.copy()
            raw[0], raw[1] = -raw[0], -raw[1]
        return MotorCommand(
            linear_velocity=lin,
            angular_velocity=ang,
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

    def _consolidate(self) -> Tensor:
        if len(self.memory.episodes) < 4:
            return zero(self.device)
        recent = self.memory.episodes[-min(32, len(self.memory.episodes)) :]
        recent = sorted(recent, key=lambda e: e.pred_err, reverse=True)[:16]
        cores = torch.tensor(np.stack([e.core.reshape(-1)[: self.cfg.core_dim] for e in recent]), dtype=torch.float32, device=self.device)
        if cores.size(-1) < self.cfg.core_dim:
            cores = torch.nn.functional.pad(cores, (0, self.cfg.core_dim - cores.size(-1)))
        assign, concepts = self.net.semantic(cores)
        recon = assign @ concepts
        usage = assign.mean(0)
        diversity = -torch.sum(usage * torch.log(usage + 1e-8))
        cn = concepts / (concepts.norm(dim=-1, keepdim=True) + 1e-6)
        sim = cn @ cn.T
        off = sim - torch.eye(sim.size(0), device=sim.device)
        collapse = torch.relu(off - 0.25).pow(2).mean()
        loss = mse(recon, cores) - 0.05 * diversity + 0.25 * collapse
        self.concepts = concepts.detach()
        self.concept_collapse = float(off.abs().mean().detach().cpu())
        self.memory.semantic_bank = [concepts.detach().cpu().numpy()]
        self.guard.consolidate(self.net)
        if not any(m["name"] == "first_consolidation" for m in self.milestones):
            self.milestones.append({"tick": self.tick, "name": "first_consolidation", "evidence": f"n={len(recent)} concepts={int(concepts.size(0))}"})
        return loss

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
            "semantic_count": 0 if self.concepts is None else int(self.concepts.size(0)),
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
                "organs": self.organ_trace[-1] if self.organ_trace else {},
                "ref_acc": self.ref_acc_trace[-1] if self.ref_acc_trace else 0.0,
                "self_pe": self.self_pe_trace[-1] if self.self_pe_trace else 0.0,
                "concepts": 0 if self.concepts is None else int(self.concepts.size(0)),
                "concept_collapse": self.concept_collapse,
                "probe_rate": float(np.mean(self.probe_choices[-64:])) if self.probe_choices else 0.0,
                "ig_margin": float(np.mean(self.ig_margin_trace[-64:])) if self.ig_margin_trace else 0.0,
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

"""The organism: a continuously existing dynamical system named 01.

Human messages are environmental events, not invocation boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cognition.attention.attention import Attention
from cognition.beliefs import BeliefSystem
from cognition.core.recurrent import RecurrentCore
from cognition.executive.arbitration import ExecutiveArbitration
from cognition.executive.mode import ModeController
from cognition.metacognition.monitor import MetacognitiveMonitor
from cognition.reflection import ReflectionProcess
from cognition.workspace.workspace import GlobalWorkspace
from embodiment.body import EmbodiedBody
from imagination.counterfactual import CounterfactualEngine
from imagination.rollout import Imagination
from language.comprehension.parser import LanguageComprehension
from language.dialogue import DialogueState
from language.grounding.concepts import ConceptMemory
from language.grounding.lexicon import Lexicon
from language.production.intent import IntentFormer
from language.production.realizer import LanguageProduction
from language.reading import ReadingSystem
from learning.continual import ContinualGuard
from learning.nn import FamiliarityNet
from learning.plasticity import neuromodulator
from memory.associative import AssociativeMemory
from memory.autobiographical.store import AutobiographicalMemory
from memory.consolidation import ConsolidationEngine
from memory.episodic.store import EpisodicMemory
from memory.procedural.skills import ProceduralMemory
from memory.semantic.store import SemanticMemory
from memory.working import WorkingMemory
from motivation.homeostasis import Homeostasis
from motivation.intrinsic import IntrinsicMotivation
from motivation.values import PreferenceStore
from organism.development import DevelopmentRuntime
from perception.hierarchy import PerceptualHierarchy
from perception.sensorium import Sensorium
from planning.action import primitive_vector
from planning.decision import DecisionSystem
from planning.goals import GoalSystem
from planning.hierarchy import HierarchicalPlanner
from self_model.agency import AgencyModel
from self_model.body_schema import BodySchema
from self_model.others import OtherAgentModels
from self_model.self import SelfModel
from shared.config import OrganismConfig
from shared.contracts import MotorCommand, SensorFrame
from shared.rng import RNG
from shared.types import (
    ActionKind,
    CandidateKind,
    CommunicativeIntent,
    DevelopmentalStage,
    Episode,
    EvidenceSource,
    GoalOrigin,
    GoalStatus,
    OperatingMode,
    SpeechAct,
    Vector,
    WorkspaceCandidate,
)
from world_model.causal import CausalLearner
from world_model.entities import EntitySystem
from world_model.predictive import PredictiveWorldModel
from world_model.spatial import SpatialModel
from world_model.temporal import TemporalModel


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


class Organism:
    def __init__(self, cfg: OrganismConfig, rng: RNG, data_dir: str | Path) -> None:
        self.cfg = cfg
        self.rng = rng
        self.name = cfg.name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tick = 0
        self.time = 0.0
        self.lineage_id = f"{cfg.name}-{cfg.seed}"
        self.parent_lineage: str | None = None

        self.sensorium = Sensorium(cfg)
        self.perception = PerceptualHierarchy(cfg, rng)
        self.entities = EntitySystem(cfg)
        self.spatial = SpatialModel()
        self.temporal = TemporalModel()
        self.world_model = PredictiveWorldModel(cfg, rng)
        self.causal = CausalLearner()
        self.core = RecurrentCore(cfg, rng)
        self.workspace = GlobalWorkspace(cfg, rng)
        self.attention = Attention()
        self.working = WorkingMemory(cfg.working_memory_slots, cfg.feature_dim)
        self.episodic = EpisodicMemory(cfg, self.data_dir / "episodic.sqlite")
        self.semantic = SemanticMemory(cfg.semantic_capacity)
        self.procedural = ProceduralMemory(cfg.action_dim)
        self.auto = AutobiographicalMemory(cfg.state_dim)
        self.associative = AssociativeMemory(cfg.feature_dim, cfg.associative_capacity)
        self.consolidation = ConsolidationEngine()
        self.self_model = SelfModel(cfg.state_dim)
        self.body_schema = BodySchema()
        self.agency = AgencyModel()
        self.others = OtherAgentModels(cfg.entity_dim)
        self.beliefs = BeliefSystem()
        self.homeo = Homeostasis()
        self.intrinsic = IntrinsicMotivation()
        self.preferences = PreferenceStore(cfg.feature_dim)
        self.goals = GoalSystem(cfg.max_goals)
        self.planner = HierarchicalPlanner()
        self.decision = DecisionSystem(cfg.action_dim)
        self.imagination = Imagination(cfg)
        self.counterfactual = CounterfactualEngine()
        self.meta = MetacognitiveMonitor()
        self.reflect = ReflectionProcess()
        self.modes = ModeController(cfg.consolidation_every)
        self.exec = ExecutiveArbitration()
        self.body = EmbodiedBody(cfg)
        self.lexicon = Lexicon(cfg.concept_dim, cfg.lexicon_capacity)
        self.concepts = ConceptMemory(cfg.concept_dim)
        self.comprehend = LanguageComprehension(cfg, rng)
        self.produce = LanguageProduction()
        self.intent_former = IntentFormer()
        self.dialogue = DialogueState()
        self.reading = ReadingSystem(self.comprehend)
        self.development = DevelopmentRuntime()
        self.guard = ContinualGuard()
        self.familiar = FamiliarityNet("fam", cfg.feature_dim, rng)
        self.pending_speech: list[tuple[str, str]] = []
        self.last_features = np.zeros(cfg.feature_dim)
        self.last_action = np.zeros(cfg.action_dim)
        self.last_kind = ActionKind.LOOK
        self.last_plan = None
        self.last_retrieved: list[str] = []
        self.last_utterance = ""
        self.last_intent: CommunicativeIntent | None = None
        self.spontaneous_count = 0
        self.human_events = 0
        self.hidden_probe: dict[str, float] = {}
        self.disabled: set[str] = set()

    def ingest_speech(self, speaker: str, text: str) -> None:
        self.pending_speech.append((speaker, text))

    def ingest_document(self, text: str) -> None:
        self.reading.read_document(text, self.lexicon, self.beliefs, self.tick)

    def disable(self, component: str) -> None:
        self.disabled.add(component)

    def enable(self, component: str) -> None:
        self.disabled.discard(component)

    def set_hidden_variable(self, name: str, value: float) -> None:
        self.hidden_probe[name] = float(value)

    def tick_once(self, sensors: SensorFrame) -> TickResult:
        self.tick += 1
        self.time += self.cfg.dt
        packet = self.sensorium.sense(sensors)
        percept = self.perception.encode(packet)
        ents = self.entities.update(percept, packet.proprio, self.tick)
        self.spatial.update(ents)
        boundary = self.temporal.step(percept.features, self.tick, self.cfg.dt)
        if "world_model" not in self.disabled:
            pred = self.world_model.predict(self.last_features, self.last_action)
            pe = self.world_model.observe(percept.features)
        else:
            pred = self.world_model.last
            pe = float(np.mean((percept.features - self.last_features) ** 2))
        novelty = self.familiar.novelty(percept.features)
        self.familiar.train_step(percept.features, self.cfg.learning_rate * 0.3)
        focus = self.attention.select_entity(ents, self.workspace.last.broadcast if self.workspace.last.winners else None, {e.entity_id: novelty for e in ents})
        if focus:
            self.concepts.observe(focus.representation, self.tick)
        if "others" not in self.disabled:
            self.others.update(ents, self.tick)

        incoming_frames = []
        social = 0.0
        for speaker, text in self.pending_speech:
            self.human_events += 1
            self.dialogue.hear(speaker, text, self.tick)
            frame = self.comprehend.parse(text, self.lexicon, ents, self.tick)
            incoming_frames.append((speaker, text, frame))
            self.working.write(frame.predicate, "language", self.tick, tags=["speech", speaker])
            self.auto.meet(speaker, 0.2)
            social = 1.0
            self._ground_from_caregiver(text, frame, focus)
        self.pending_speech.clear()

        retrieved = []
        if "episodic" not in self.disabled and (pe > 0.08 or incoming_frames):
            retrieved = self.episodic.similar(percept.features, k=3)
            self.last_retrieved = [e.episode_id for e in retrieved]
            for ep in retrieved:
                self.working.write(ep.compression, "recall", self.tick, tags=["episode"])
        else:
            self.last_retrieved = []

        candidates = self._candidates(percept, pe, novelty, incoming_frames, retrieved, focus, boundary)
        if "workspace" not in self.disabled:
            ws = self.workspace.compete(self.attention.gate(candidates, focus.representation if focus else None), self.tick)
        else:
            ws = self.workspace.last
        self.working.write(ws.broadcast, "workspace", self.tick)
        self.working.decay()

        homeo = self.homeo.update(
            energy=packet.energy,
            novelty=novelty,
            uncertainty=pred.uncertainty,
            social=social or (0.15 if self.dialogue.current_partner else 0.0),
            load=min(1.0, len(self.working.slots) / max(1, self.cfg.working_memory_slots)),
            pe=pe,
            competence=self.self_model.capability,
            stability=1.0 / (1.0 + boundary),
            leak=self.cfg.homeostatic_leak,
        )
        motive = self.intrinsic.evaluate(pe, novelty, pred.uncertainty, self.self_model.capability, social, self.beliefs.conflict_score(), self.homeo)
        self._maybe_form_goals(focus, motive, pe, pred.uncertainty, incoming_frames)
        self.goals.decay()
        goal = None if "goals" not in self.disabled else None
        if "goals" not in self.disabled:
            goal = self.exec.choose(list(self.goals.goals.values()), None, motive.novelty)

        core_state = self.core.step(
            percept.features,
            ws.broadcast,
            np.array([pe, pred.uncertainty]),
            homeo.vector(),
            packet.proprio,
            np.array([social, float(len(self.others.models))]),
            pred.uncertainty,
        )
        if self.modes.mode is OperatingMode.AWAKE and "imagination" not in self.disabled and self.tick % 3 == 0:
            imagined = self.imagination.imagine(self.world_model, percept.features, goal.description_vector if goal else None, packet.energy)
        else:
            imagined = self.imagination.last
        scored = []
        for traj in imagined:
            scored.append((traj.kind, primitive_vector(traj.kind, self.cfg.action_dim), traj.value + 0.15 * motive.total, traj.predicted_error))
        if goal and goal.action_kind:
            scored.append((goal.action_kind, primitive_vector(goal.action_kind, self.cfg.action_dim), goal.expected_value, 0.2))
        if self.modes.mode is OperatingMode.REST:
            scored = [(ActionKind.WAIT, primitive_vector(ActionKind.WAIT, self.cfg.action_dim), 0.4, 0.0)]
        decision = self.decision.select(scored, self.rng, temperature=0.4 + 0.3 * pred.uncertainty)

        utterance = ""
        intent = None
        speak_now = False
        communicative_pressure = float(motive.social) + float(pred.uncertainty) * 0.3 + (0.25 if incoming_frames else 0.0)
        if incoming_frames or (goal is not None and communicative_pressure > 0.55):
            intent = self.intent_former.form(
                ws,
                goal,
                focus.representation if focus else None,
                pred.uncertainty,
                homeo.social_engagement,
                self.last_retrieved,
                pe,
                self.tick,
                self.dialogue.current_partner,
            )
            if intent and (intent.urgency > 0.28 or incoming_frames):
                utterance = self.produce.realize(intent, self.lexicon)
                if utterance and utterance == self.last_utterance and not incoming_frames:
                    utterance = ""
                speak_now = bool(utterance)
        if speak_now:
            decision.kind = ActionKind.SPEAK
            decision.action_vector = primitive_vector(ActionKind.SPEAK, self.cfg.action_dim)
            self.dialogue.spoke(utterance, self.tick, intent.frame.act.value if intent else None)
            if not incoming_frames:
                self.spontaneous_count += 1
        self.last_utterance = utterance
        self.last_intent = intent

        motor = self.body.decode(decision.action_vector, decision.kind, self.tick)
        if decision.kind in {ActionKind.TOGGLE, ActionKind.EXPERIMENT}:
            cause = focus.entity_id if focus else "unknown"
            self.causal.note_intervention(self.tick, cause, packet.energy)
        visual_delta = percept.features - self.last_features
        body_c = self.body_schema.observe(decision.action_vector, packet.proprio, visual_delta)
        owned = self.agency.update(float(np.linalg.norm(decision.action_vector)), pred.error, pe)
        action_effect = float(np.linalg.norm(visual_delta)) * (1.0 if float(np.linalg.norm(decision.action_vector)) > 0.1 else 0.2)
        self.self_model.update(packet.proprio, core_state, action_effect, None, self.tick / 1000.0)
        self.auto.record_self(self.tick, self.self_model.identity)
        self.meta.update(pe, 1.0 if retrieved else 0.2, self.procedural.skills.get(f"skill_{decision.kind.value}", self.procedural.skills["skill_look"]).competence, self.beliefs.conflict_score(), pred.uncertainty)

        self._learn(percept, decision, pe, owned)
        self._encode_episode(percept, pred.latent, decision.action_vector, pe, goal, packet, ws, sensors.timestamp)
        if focus:
            self.preferences.reinforce(focus.entity_id, focus.representation, 0.1 * motive.total - 0.05 * pe)
        self.associative.store(percept.features)
        if decision.kind in {ActionKind.TOGGLE, ActionKind.EXPERIMENT}:
            hyp = self.causal.observe_effect(self.tick, "light_or_world", float(np.mean(packet.vision)), focus.entity_id if focus else None)
            if hyp and hyp.confidence > 0.55:
                self.beliefs.assert_belief(hyp.hypothesis_id, hyp.relation, hyp.confidence, self.tick, EvidenceSource.INTERVENTION, hyp.hypothesis_id)
                self.auto.add_milestone(self.tick, "first_successful_causal_experiment", hyp.hypothesis_id)

        mode = self.modes.update(self.tick, packet.energy, homeo.rest_requirement, homeo.cognitive_load, pe)
        if mode is OperatingMode.CONSOLIDATION and "consolidation" not in self.disabled:
            self.consolidation.run(self.episodic, self.semantic, self.world_model, self.rng, self.cfg.replay_batch)
        if mode is OperatingMode.MAINTENANCE:
            nxt, result = self.reflect.revisit(self.episodic.salient(8), list(self.beliefs.beliefs.values()), self.core.h_slow)
            self.core.h_slow = nxt
        if mode is OperatingMode.SIMULATION and "imagination" not in self.disabled:
            self.imagination.imagine(self.world_model, percept.features, None, packet.energy)

        if self.tick % 7 == 0 and "imagination" not in self.disabled:
            self.counterfactual.compare(self.world_model, self.last_features, self.last_kind, percept.features, pe, self.tick, self.cfg.action_dim)

        self._update_development(pe, novelty, body_c.self_structure_score, owned)
        self.last_features = percept.features.copy()
        self.last_action = decision.action_vector.copy()
        self.last_kind = decision.kind
        self.workspace.learn_from_outcome(motive.total - pe, self.cfg.learning_rate * 0.2)
        spontaneous = speak_now and not incoming_frames
        return TickResult(
            tick=self.tick,
            motor=motor,
            utterance=utterance,
            intent=intent,
            mode=mode,
            prediction_error=pe,
            workspace_kinds=[w.kind.value for w in ws.winners],
            spontaneous=spontaneous,
            notes={
                "novelty": novelty,
                "focus": focus.entity_id if focus else None,
                "goal": goal.goal_id if goal else None,
                "stage": self.development.stage.value,
                "agency": owned,
                "body_contingency": body_c.self_structure_score,
            },
        )

    def _candidates(self, percept, pe, novelty, incoming, retrieved, focus, boundary) -> list[WorkspaceCandidate]:
        cands: list[WorkspaceCandidate] = [
            WorkspaceCandidate(CandidateKind.VISUAL_EVENT, percept.features, percept.change, novelty, 0.4, 0.2, 0.2, 0.2, 1.0, "perception"),
            WorkspaceCandidate(CandidateKind.UNEXPECTED_PREDICTION, np.array([pe]), min(1.0, pe * 2), novelty, 0.5, min(1.0, pe), 0.4, 0.3, 0.9, "world_model"),
            WorkspaceCandidate(CandidateKind.UNCERTAINTY, self.meta.vector(self.cfg.feature_dim), self.meta.state.uncertainty, 0.2, 0.3, self.meta.state.uncertainty, 0.3, 0.2, 0.6, "metacognition"),
            WorkspaceCandidate(CandidateKind.INTERNAL_STATE_CHANGE, self.homeo.state.vector(), self.homeo.state.rest_requirement, 0.1, 0.4, 0.2, 0.5, 0.3, 0.7, "homeostasis"),
            WorkspaceCandidate(CandidateKind.BODY_EVENT, _fit(percept.features[:16], self.cfg.feature_dim), 0.3, 0.1, 0.4, 0.2, 0.2, 0.1, 0.8, "body"),
        ]
        if incoming:
            _, _, frame = incoming[-1]
            cands.append(WorkspaceCandidate(CandidateKind.LANGUAGE_EVENT, _fit(frame.predicate, self.cfg.feature_dim), 0.9, 0.4, 0.8, 1.0 - frame.confidence, 0.7, 0.6, 1.0, "language", {"text_len": len(frame.bound_words)}))
            cands.append(WorkspaceCandidate(CandidateKind.SOCIAL_SIGNAL, _fit(frame.predicate, self.cfg.feature_dim), 0.7, 0.3, 0.7, 0.3, 0.8, 0.5, 1.0, "social"))
        for ep in retrieved:
            cands.append(WorkspaceCandidate(CandidateKind.MEMORY_RECALL, _fit(ep.compression, self.cfg.feature_dim), ep.salience, 0.2, 0.5, ep.uncertainty, 0.3, 0.4, 0.5, "episodic"))
        if focus:
            cands.append(WorkspaceCandidate(CandidateKind.GOAL, _fit(focus.representation, self.cfg.feature_dim), 0.4, novelty, 0.6, focus.uncertainty, 0.4, 0.7, 0.8, "attention"))
        if self.imagination.last:
            best = max(self.imagination.last, key=lambda t: t.value)
            cands.append(WorkspaceCandidate(CandidateKind.IMAGINED_FUTURE, _fit(best.states[-1], self.cfg.feature_dim), 0.3, 0.4, 0.4, 0.4, 0.3, 0.5, 0.4, "imagination"))
        if self.beliefs.conflict_score() > 0.1:
            cands.append(WorkspaceCandidate(CandidateKind.CONTRADICTION, self.meta.vector(self.cfg.feature_dim), 0.6, 0.3, 0.5, 0.7, 0.4, 0.4, 0.5, "beliefs"))
        if boundary > 0.1:
            cands.append(WorkspaceCandidate(CandidateKind.VISUAL_EVENT, percept.features, min(1.0, boundary * 3), 0.5, 0.4, 0.3, 0.2, 0.2, 0.9, "temporal"))
        return cands

    def _has_active_goal(self, origin: GoalOrigin, action: ActionKind | None) -> bool:
        return any(
            g.origin is origin and g.action_kind is action and g.status.value == "active" for g in self.goals.goals.values()
        )

    def _maybe_form_goals(self, focus, motive, pe, unc, incoming) -> None:
        """Legacy path: propose a latent goal from current drives, not ActionKind tables."""
        if "goals" in self.disabled:
            return
        drive = float(motive.total) + 0.2 * float(unc) + 0.15 * float(pe)
        if drive < 0.12 and not incoming:
            return
        vec = focus.representation if focus is not None else self.workspace.last.broadcast
        origin = GoalOrigin.EXPLORATORY
        if incoming:
            origin = GoalOrigin.SOCIAL
        elif float(self.homeo.state.energy) < 0.25:
            origin = GoalOrigin.MAINTENANCE
        elif unc > 0.5:
            origin = GoalOrigin.EPISTEMIC
        if not any(g.origin is origin and g.status.value == "active" for g in self.goals.goals.values()):
            self.goals.form(_fit(vec, self.cfg.feature_dim), origin, min(0.9, 0.3 + drive), 0.4, 0.4, 0.2, 24, self.tick, None, focus.entity_id if focus else None)

    def _ground_from_caregiver(self, text: str, frame, focus) -> None:
        units = [u for u in text.split() if u]
        if not units:
            return
        attended = focus.representation if focus is not None else frame.predicate
        for unit in units:
            self.lexicon.bind(unit, _fit(attended, self.cfg.concept_dim), self.tick, "unknown")
        if units and not any(m["name"] == "first_grounded_word" for m in self.auto.milestones):
            self.auto.add_milestone(self.tick, "first_grounded_word", units[0])

    def _learn(self, percept, decision, pe, owned) -> None:
        neu = neuromodulator(pe, owned, self.world_model.last.uncertainty, self.cfg.neuromod_scale)
        if "world_model" not in self.disabled:
            self.world_model.train_step(self.last_features, self.last_action, percept.features, neuromod=neu)
        self.guard.observe_gradients(self.world_model)
        self.guard.apply_ewc(self.world_model)
        self.guard.maybe_stabilize(self.world_model)
        self.guard.clip_weight_norm(self.world_model)
        success = float(1.0 / (1.0 + pe))
        self.procedural.practice(decision.kind, success, self.tick, percept.features)

    def _encode_episode(self, percept, pred, action, pe, goal, packet, ws, timestamp) -> None:
        if "episodic" in self.disabled:
            return
        if pe < 0.02 and self.tick % 9 != 0:
            return
        compression = _fit(np.concatenate([percept.features, ws.broadcast[:16], [pe, packet.energy]]), self.cfg.feature_dim)
        ep = Episode(
            episode_id=f"ep_{self.tick}_{self.rng.integers(0, 9999)}",
            tick=self.tick,
            timestamp=timestamp,
            world_state_before=self.last_features.copy(),
            perception=percept.features.copy(),
            internal_state=self.core.state.copy(),
            active_goal=goal.goal_id if goal else None,
            prediction=_fit(pred, self.cfg.feature_dim),
            action=_fit(action, self.cfg.action_dim),
            world_state_after=percept.features.copy(),
            prediction_error=pe,
            value_delta=self.intrinsic.last.total,
            people_entities=[self.dialogue.current_partner] if self.dialogue.current_partner else [],
            uncertainty=self.world_model.last.uncertainty,
            workspace_summary=ws.broadcast.copy(),
            self_snapshot=self.self_model.identity.copy(),
            salience=min(1.0, pe + 0.3 * percept.change + (0.4 if self.last_utterance else 0.0)),
            mode=self.modes.mode,
            compression=compression,
        )
        self.episodic.encode(ep)

    def _update_development(self, pe, novelty, body_score, agency) -> None:
        improving = 0.0
        if len(self.world_model.error_trace) > 20:
            early = float(np.mean(self.world_model.error_trace[:10]))
            late = float(np.mean(self.world_model.error_trace[-10:]))
            improving = max(0.0, (early - late) / (early + 1e-6))
        self.development.record(
            change_detected=min(1.0, novelty + 0.35 + float(np.tanh(pe * 6.0))),
            prediction_improving=improving,
            body_contingency=body_score,
            agency=agency,
            permanence=float(np.mean([e.permanence_evidence for e in self.entities.entities.values()] or [0.0])),
            tracking=min(1.0, self.entities.stats.matches / max(1, self.tick)),
            causal_confidence=self.causal.best().confidence if self.causal.best() else 0.0,
            other_models=min(1.0, len(self.others.models) / 2.0),
            grounded_words=float(len(self.lexicon.entries)),
            utterances=float(sum(1 for t in self.dialogue.turns if t.speaker == "01")),
            semantic_relations=float(len(self.semantic.relations)),
            counterfactuals=float(len(self.counterfactual.lessons)),
        )
        advanced = self.development.maybe_advance(self.tick)
        if advanced:
            self.auto.close_chapter(advanced, self.episodic.salient(12), self.tick)
            self.auto.add_milestone(self.tick, f"stage_{advanced.value}", advanced.value)

    def observe_state(self) -> dict[str, Any]:
        """Researcher observability — not organism introspection."""
        return {
            "tick": self.tick,
            "time": self.time,
            "name": self.name,
            "lineage": self.lineage_id,
            "mode": self.modes.mode.value,
            "stage": self.development.stage.value,
            "prediction_error": self.world_model.last.error,
            "uncertainty": self.world_model.last.uncertainty,
            "workspace": {
                "kinds": [w.kind.value for w in self.workspace.last.winners],
                "scores": [float(s) for s in self.workspace.last.scores],
                "ignition": self.workspace.last.ignition,
                "broadcast": self.workspace.last.broadcast.tolist(),
                "sources": [w.source for w in self.workspace.last.winners],
            },
            "motivation": self.homeo.state.as_dict(),
            "intrinsic": {
                "total": self.intrinsic.last.total,
                "progress": self.intrinsic.last.learning_progress,
                "novelty": self.intrinsic.last.novelty,
                "weights": self.intrinsic.last.weights,
            },
            "goals": [
                {"id": g.goal_id, "origin": g.origin.value, "priority": g.priority, "status": g.status.value, "action": g.action_kind.value if g.action_kind else None}
                for g in self.goals.goals.values()
            ],
            "entities": [
                {"id": e.entity_id, "visible": e.visible, "uncertainty": e.uncertainty, "location": e.location[:2].tolist(), "self": e.is_self, "agent": e.is_agent, "permanence": e.permanence_evidence}
                for e in self.entities.entities.values()
            ],
            "self": {
                "identity": self.self_model.identity.tolist(),
                "capability": self.self_model.capability,
                "controllability": self.self_model.controllability,
                "agency": self.self_model.agency,
                "continuity": self.self_model.continuity,
            },
            "meta": {
                k: (v if not isinstance(v, list) else v[-20:])
                for k, v in self.meta.state.__dict__.items()
                if k != "history"
            } | {"confidence_trace": self.meta.state.history[-20:]},
            "lexicon_size": len(self.lexicon.entries),
            "lexicon_words": sorted(self.lexicon.entries),
            "episodic_count": self.episodic.count(),
            "semantic_count": len(self.semantic.relations),
            "beliefs": [
                {"id": b.belief_id, "prop": b.proposition, "conf": b.confidence, "contra": len(b.contradictory_evidence)}
                for b in self.beliefs.beliefs.values()
            ],
            "last_utterance": self.last_utterance,
            "last_intent": {
                "act": self.last_intent.frame.act.value,
                "urgency": self.last_intent.urgency,
                "uncertainty": self.last_intent.preceding_uncertainty,
                "retrievals": self.last_intent.preceding_retrieval_ids,
                "pe": self.last_intent.preceding_prediction_error,
            }
            if self.last_intent
            else None,
            "dialogue": self.dialogue.snapshot(),
            "imagination": [
                {"kind": t.kind.value, "value": t.value, "info": t.info_gain, "pe": t.predicted_error} for t in self.imagination.last
            ],
            "causal": {
                hid: {"cause": h.cause_entity, "effect": h.effect_entity, "confidence": h.confidence, "interventions": h.interventions}
                for hid, h in self.causal.hypotheses.items()
            },
            "development": self.development.snapshot(),
            "milestones": self.auto.milestones,
            "relationships": self.auto.relationship_weights,
            "preferences": {k: v[1] for k, v in self.preferences.items.items()},
            "spontaneous_utterances": self.spontaneous_count,
            "attention": self.attention.focus_id,
            "working_memory": [{"source": s.source, "act": s.activation, "entity": s.entity_id} for s in self.working.slots],
            "core_norm": float(np.linalg.norm(self.core.state)),
            "hidden_probe": dict(self.hidden_probe),
        }


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

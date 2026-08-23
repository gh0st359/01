"""Core typed representations used across every cognitive subsystem.

These types are the organism's scientific ontology, not researcher prose.
Vectors remain vectors. Human labels exist only for instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]


class OperatingMode(str, Enum):
    AWAKE = "awake"
    REST = "rest"
    CONSOLIDATION = "consolidation"
    SIMULATION = "simulation"
    MAINTENANCE = "maintenance"


class DevelopmentalStage(str, Enum):
    SENSORY_CONTINUITY = "stage_0_sensory_continuity"
    BODY = "stage_1_body"
    OBJECTS = "stage_2_objects"
    CAUSALITY = "stage_3_causality"
    AGENTS = "stage_4_agents"
    COMMUNICATION = "stage_5_communication"
    ABSTRACT = "stage_6_abstract"
    OPEN = "stage_7_open"


class SpeechAct(str, Enum):
    """Developmental speech-act categories. Not canned replies."""

    ASSERT = "assert"
    QUERY = "query"
    REQUEST = "request"
    LABEL = "label"
    INFORM = "inform"
    UNKNOWN = "unknown"


class ActionKind(str, Enum):
    STAY = "stay"
    MOVE = "move"
    TURN = "turn"
    GRASP = "grasp"
    RELEASE = "release"
    PUSH = "push"
    TOGGLE = "toggle"
    LOOK = "look"
    SPEAK = "speak"
    WAIT = "wait"
    APPROACH = "approach"
    INSPECT = "inspect"
    EXPERIMENT = "experiment"


class CandidateKind(str, Enum):
    VISUAL_EVENT = "visual_event"
    MEMORY_RECALL = "memory_recall"
    UNEXPECTED_PREDICTION = "unexpected_prediction"
    SOCIAL_SIGNAL = "social_signal"
    GOAL = "goal"
    INTERNAL_STATE_CHANGE = "internal_state_change"
    IMAGINED_FUTURE = "imagined_future"
    UNCERTAINTY = "uncertainty"
    LANGUAGE_EVENT = "language_event"
    BODY_EVENT = "body_event"
    CAUSAL_HYPOTHESIS = "causal_hypothesis"
    CONTRADICTION = "contradiction"


class GoalOrigin(str, Enum):
    EXTERNAL = "external"
    INSTRUMENTAL = "instrumental"
    EXPLORATORY = "exploratory"
    MAINTENANCE = "maintenance"
    SOCIAL = "social"
    EPISTEMIC = "epistemic"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    SATISFIED = "satisfied"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class EvidenceSource(str, Enum):
    PERCEPTION = "perception"
    INTERVENTION = "intervention"
    RECALL = "recall"
    IMAGINATION = "imagination"
    LANGUAGE = "language"
    CONSOLIDATION = "consolidation"
    SOCIAL = "social"
    CAREGIVER = "caregiver"


@dataclass
class WorkspaceCandidate:
    kind: CandidateKind
    vector: Vector
    salience: float
    novelty: float
    relevance: float
    uncertainty: float
    motivational_weight: float
    goal_relevance: float
    recency: float
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def competition_features(self) -> Vector:
        return np.array(
            [
                self.salience,
                self.novelty,
                self.relevance,
                self.uncertainty,
                self.motivational_weight,
                self.goal_relevance,
                self.recency,
            ],
            dtype=np.float64,
        )


@dataclass
class WorkspaceContents:
    """Limited-capacity globally broadcast information."""

    tick: int
    winners: list[WorkspaceCandidate]
    scores: Vector
    broadcast: Vector
    ignition: float
    capacity: int


@dataclass
class Entity:
    entity_id: str
    representation: Vector
    location: Vector
    velocity: Vector
    visual_features: Vector
    auditory_associations: Vector
    affordances: Vector
    uncertainty: float
    last_seen_tick: int
    first_seen_tick: int
    visible: bool
    kind_hint: str
    relationships: dict[str, float] = field(default_factory=dict)
    episodic_links: list[str] = field(default_factory=list)
    ownership: str | None = None
    social_association: float = 0.0
    is_self: bool = False
    is_agent: bool = False
    permanence_evidence: float = 0.0


@dataclass
class Episode:
    episode_id: str
    tick: int
    timestamp: float
    world_state_before: Vector
    perception: Vector
    internal_state: Vector
    active_goal: str | None
    prediction: Vector
    action: Vector
    world_state_after: Vector
    prediction_error: float
    value_delta: float
    people_entities: list[str]
    uncertainty: float
    workspace_summary: Vector
    self_snapshot: Vector
    salience: float
    mode: OperatingMode
    compression: Vector


@dataclass
class Belief:
    belief_id: str
    proposition: str
    latent: Vector
    confidence: float
    supporting_evidence: list[str]
    contradictory_evidence: list[str]
    created_tick: int
    updated_tick: int
    source: EvidenceSource
    dependencies: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)


@dataclass
class Goal:
    goal_id: str
    description_vector: Vector
    origin: GoalOrigin
    priority: float
    expected_value: float
    confidence: float
    cost: float
    time_horizon: int
    dependencies: list[str]
    status: GoalStatus
    created_tick: int
    target_entity: str | None = None
    action_kind: ActionKind | None = None
    parent_id: str | None = None


@dataclass
class SemanticFrame:
    """Internal meaning representation. Not an English string."""

    act: SpeechAct
    predicate: Vector
    roles: dict[str, Vector]
    referent_ids: list[str]
    confidence: float
    source_tick: int
    bound_words: list[str] = field(default_factory=list)


@dataclass
class CommunicativeIntent:
    frame: SemanticFrame
    urgency: float
    social_address: str | None
    formed_tick: int
    workspace_hash: str
    preceding_uncertainty: float
    preceding_retrieval_ids: list[str]
    preceding_prediction_error: float


@dataclass
class SkillRecord:
    skill_id: str
    name_key: str
    policy: Vector
    competence: float
    practice_count: int
    last_used_tick: int
    parent_skills: list[str]
    success_rate: float


@dataclass
class Hypothesis:
    hypothesis_id: str
    cause_entity: str
    effect_entity: str
    relation: Vector
    confidence: float
    interventions: int
    confirmations: int
    disconfirmations: int
    last_test_tick: int


@dataclass
class MarkerObservation:
    """One operational marker result. Never a consciousness proof."""

    marker: str
    experiment: str
    expected_alternative: str
    result: str
    ablation: str | None
    replication: int
    confidence: float
    limitations: str
    evidence_paths: list[str]
    tick: int

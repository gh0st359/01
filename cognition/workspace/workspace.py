"""Limited-capacity global workspace with learned attentional competition.

Winning information is broadcast as a vector. Researchers may attach
interpretation labels; those labels are not organism-generated speech.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning.nn import Linear, Module, softmax
from shared.config import OrganismConfig
from shared.rng import RNG
from shared.types import Vector, WorkspaceCandidate, WorkspaceContents


@dataclass
class WorkspaceLog:
    tick: int
    kinds: list[str]
    scores: list[float]
    ignition: float
    sources: list[str]


class GlobalWorkspace(Module):
    def __init__(self, cfg: OrganismConfig, rng: RNG) -> None:
        super().__init__("workspace")
        self.cfg = cfg
        self.capacity = cfg.workspace_slots
        self.policy = self.add(Linear("ws.policy", 7, 1, rng))
        self.mix = self.add(Linear("ws.mix", cfg.feature_dim, cfg.state_dim, rng))
        self.history: list[WorkspaceLog] = []
        self.last = WorkspaceContents(
            tick=0,
            winners=[],
            scores=np.zeros(0),
            broadcast=np.zeros(cfg.state_dim),
            ignition=0.0,
            capacity=self.capacity,
        )

    def score(self, candidate: WorkspaceCandidate) -> float:
        feats = candidate.competition_features()
        raw = float(self.policy.forward(feats)[0])
        prior = (
            0.22 * candidate.salience
            + 0.16 * candidate.novelty
            + 0.16 * candidate.relevance
            + 0.12 * candidate.uncertainty
            + 0.12 * candidate.motivational_weight
            + 0.14 * candidate.goal_relevance
            + 0.08 * candidate.recency
        )
        candidate.score = raw + prior
        return candidate.score

    def compete(self, candidates: list[WorkspaceCandidate], tick: int) -> WorkspaceContents:
        if not candidates:
            self.last = WorkspaceContents(tick, [], np.zeros(0), np.zeros(self.cfg.state_dim), 0.0, self.capacity)
            return self.last
        scores = np.array([self.score(c) for c in candidates], dtype=np.float64)
        order = np.argsort(-scores)
        winners = [candidates[int(i)] for i in order[: self.capacity]]
        win_scores = scores[order[: self.capacity]]
        weights = softmax(win_scores)
        stacked = np.zeros((len(winners), self.cfg.feature_dim), dtype=np.float64)
        for i, cand in enumerate(winners):
            v = np.asarray(cand.vector, dtype=np.float64).ravel()
            stacked[i, : min(self.cfg.feature_dim, v.size)] = v[: min(self.cfg.feature_dim, v.size)]
        mix = weights @ stacked
        broadcast = np.tanh(self.mix.forward(mix))
        # Ignition: sudden dominance of the top candidate (Dehaene/Changeux operational analog)
        if len(weights) > 1:
            ignition = float(weights[0] - weights[1])
        else:
            ignition = float(weights[0])
        contents = WorkspaceContents(tick, winners, win_scores, broadcast, ignition, self.capacity)
        self.last = contents
        self.history.append(
            WorkspaceLog(
                tick=tick,
                kinds=[w.kind.value for w in winners],
                scores=[float(s) for s in win_scores],
                ignition=ignition,
                sources=[w.source for w in winners],
            )
        )
        if len(self.history) > 4000:
            self.history = self.history[-2000:]
        return contents

    def learn_from_outcome(self, utility: float, lr: float) -> None:
        """Policy gradient on competition weights from subsequent value."""
        if not self.last.winners:
            return
        for winner in self.last.winners:
            feat = winner.competition_features()
            pred = self.policy.forward(feat)
            err = np.array([pred[0] - utility], dtype=np.float64)
            self.policy.backward(feat, err)
        for param in self.policy.parameters():
            param.value -= lr * np.clip(param.grad, -1.0, 1.0)
            param.zero_grad()

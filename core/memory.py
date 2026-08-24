"""Expandable memory: episodic trajectories, semantic induction, procedural options, autobiography."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class Episode:
    tick: int
    latent: np.ndarray
    slots: np.ndarray
    core: np.ndarray
    proprio: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    pred_err: float
    uncertainty: float
    workspace: np.ndarray
    social: np.ndarray
    outcome: np.ndarray


class RetrievalNet(nn.Module):
    """Context-sensitive retrieval scores. Not nearest-vector alone."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, 1))

    def scores(self, query: Tensor, keys: Tensor) -> Tensor:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        q = self.q(query).unsqueeze(1)
        k = self.k(keys)
        if k.dim() == 2:
            k = k.unsqueeze(0)
        attn = torch.softmax((q * k).sum(-1) / (keys.size(-1) ** 0.5), dim=-1)
        qrep = query.unsqueeze(1).expand(-1, k.size(1), -1)
        g = torch.sigmoid(self.gate(torch.cat([qrep, k], dim=-1))).squeeze(-1)
        return attn * g


class SemanticInducer(nn.Module):
    def __init__(self, dim: int, n_concepts: int) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.cluster = nn.Linear(dim, n_concepts)
        self.prop = nn.Linear(dim, dim)
        # Learned codes stay distinct even when episode cores collapse.
        self.codebook = nn.Parameter(torch.randn(n_concepts, dim) * 0.35)

    def forward(self, episodes: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.cluster(episodes)
        assign = torch.softmax(logits, dim=-1)
        residual = assign.transpose(0, 1) @ self.prop(episodes)
        residual = residual / (assign.sum(0).unsqueeze(-1) + 1e-6)
        concepts = self.codebook + 0.25 * residual
        return assign, concepts


class MemorySystem:
    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self.episodes: list[Episode] = []
        self.autobiography: list[dict] = []
        self.semantic_bank: list[np.ndarray] = []
        self.procedural: list[np.ndarray] = []

    def write(self, ep: Episode) -> None:
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity :]
        self.autobiography.append(
            {
                "tick": ep.tick,
                "pred_err": ep.pred_err,
                "uncertainty": ep.uncertainty,
                "belief_change": float(
                    np.linalg.norm(ep.outcome.reshape(-1)[: min(ep.outcome.size, ep.latent.size)] - ep.latent.reshape(-1)[: min(ep.outcome.size, ep.latent.size)])
                ),
            }
        )

    def retrieve(self, net: RetrievalNet, query: Tensor, k: int = 5) -> list[Episode]:
        if not self.episodes:
            return []
        dim = query.size(-1)
        raw = []
        for e in self.episodes[-256:]:
            v = e.core.reshape(-1)
            if v.size < dim:
                v = np.pad(v, (0, dim - v.size))
            raw.append(v[:dim])
        keys = torch.tensor(np.stack(raw), dtype=query.dtype, device=query.device)
        sc = net.scores(query[:1], keys)
        idx = torch.topk(sc[0], k=min(k, keys.size(0))).indices.tolist()
        pool = self.episodes[-256:]
        return [pool[i] for i in idx]

    def snapshot(self) -> dict:
        return {
            "n_episodes": len(self.episodes),
            "n_auto": len(self.autobiography),
            "last_ticks": [e.tick for e in self.episodes[-8:]],
        }

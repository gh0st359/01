"""Learned limited-capacity global workspace. Competition is trainable, not a weighted sum of hand priors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


CANDIDATE_KINDS = (
    "vision",
    "audio",
    "body",
    "memory",
    "prediction",
    "social",
    "goal",
    "imagination",
    "contradiction",
    "self",
)


@dataclass
class WorkspaceOut:
    content: Tensor
    access: Tensor  # [B, K] probabilities
    winner: Tensor  # [B] index
    duration_trace: Tensor


class LearnedWorkspace(nn.Module):
    def __init__(self, dim: int, n_kinds: int = 10) -> None:
        super().__init__()
        self.dim = dim
        self.n_kinds = n_kinds
        self.kind_emb = nn.Embedding(n_kinds, dim)
        self.score = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.mixer = nn.GRUCell(dim, dim)
        self.duration = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.impact = nn.Linear(dim, dim)

    def forward(self, candidates: Tensor, context: Tensor, prev: Tensor) -> WorkspaceOut:
        """candidates: [B, K, D], context/prev: [B, D]."""
        b, k, d = candidates.shape
        kinds = self.kind_emb.weight[:k].unsqueeze(0).expand(b, -1, -1)
        ctx = context.unsqueeze(1).expand(-1, k, -1)
        scores = self.score(torch.cat([candidates + kinds, ctx], dim=-1)).squeeze(-1)
        access = torch.softmax(scores, dim=-1)
        winner = torch.argmax(access, dim=-1)
        broadcast = torch.einsum("bk,bkd->bd", access, candidates)
        content = self.mixer(broadcast, prev)
        dur = torch.sigmoid(self.duration(torch.cat([content, context], dim=-1)))
        return WorkspaceOut(content=content, access=access, winner=winner, duration_trace=dur)

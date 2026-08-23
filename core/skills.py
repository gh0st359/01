"""Learned options: cluster recurrent (latent, action) pairs. Researcher labels only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class OptionBatch:
    ids: Tensor
    embeddings: Tensor
    competence: Tensor


class OptionDiscoverer(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, n_options: int = 8) -> None:
        super().__init__()
        self.n_options = n_options
        self.enc = nn.Sequential(nn.Linear(latent_dim + action_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, n_options))
        self.embed = nn.Embedding(n_options, latent_dim)
        self.comp = nn.Parameter(torch.zeros(n_options))

    def assign(self, latent: Tensor, action: Tensor) -> OptionBatch:
        logits = self.enc(torch.cat([latent, action], dim=-1))
        soft = torch.softmax(logits, dim=-1)
        ids = soft.argmax(dim=-1)
        emb = soft @ self.embed.weight
        return OptionBatch(ids=ids, embeddings=emb, competence=torch.sigmoid(self.comp[ids]))

    def reinforce(self, ids: Tensor, success: Tensor) -> Tensor:
        """success: [B] in [0,1]. Update competence via BCE-style loss."""
        pred = torch.sigmoid(self.comp[ids])
        return torch.mean((pred - success) ** 2)

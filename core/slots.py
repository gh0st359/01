"""Object-centric slot attention. Slots are inferred from pixels, never from simulator IDs."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SlotAttention(nn.Module):
    def __init__(self, n_slots: int, slot_dim: int, inp_dim: int, iters: int = 3) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.norm_inp = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(inp_dim, slot_dim)
        self.to_v = nn.Linear(inp_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim * 2), nn.SiLU(), nn.Linear(slot_dim * 2, slot_dim))
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.slot_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

    def initial_slots(self, batch: int) -> Tensor:
        mu = self.slot_mu.expand(batch, self.n_slots, -1)
        sigma = torch.exp(self.slot_logsigma).expand(batch, self.n_slots, -1)
        return mu + sigma * torch.randn_like(mu)

    def forward(self, feats: Tensor, slots: Tensor | None = None) -> Tensor:
        """feats: [B, N, D]. slots: [B, K, S] or None."""
        b = feats.size(0)
        x = self.norm_inp(feats)
        k = self.to_k(x)
        v = self.to_v(x)
        if slots is None:
            slots = self.initial_slots(b)
        for _ in range(self.iters):
            slots_n = self.norm_slots(slots)
            q = self.to_q(slots_n)
            attn = torch.softmax(torch.einsum("bkd,bnd->bkn", q, k) / (self.slot_dim ** 0.5), dim=1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum("bkn,bnd->bkd", attn, v)
            slots = self.gru(updates.reshape(-1, self.slot_dim), slots.reshape(-1, self.slot_dim)).view(b, self.n_slots, -1)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots

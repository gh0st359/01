"""Grounded communication: character streams ↔ latent semantic state.

The organism may know Unicode/byte framing. It does not know English word classes.
Production is conditioned on an independently measurable intention/semantic state.
This is NOT a general next-token language model used as cognition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

CODEPOINTS = 128  # ASCII/basic Latin stream. Higher planes map to 127.


def encode_stream(text: str, max_len: int, device: torch.device) -> Tensor:
    ids = [min(ord(ch), CODEPOINTS - 1) for ch in text[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long, device=device)


def decode_stream(ids: Tensor) -> str:
    chars = []
    for i in ids.tolist():
        if i <= 0:
            break
        # Framing: emit graphic ASCII only. This is not an English lexicon.
        if 32 <= int(i) <= 126:
            chars.append(chr(int(i)))
    return "".join(chars).strip()


@dataclass
class CommunicativeState:
    intention: Tensor
    semantic: Tensor
    should_emit: Tensor  # [B] learned communicative value, not tick%17


class LanguageOrgan(nn.Module):
    def __init__(self, dim: int, intention_dim: int, max_len: int) -> None:
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.embed = nn.Embedding(CODEPOINTS, dim)
        self.encoder = nn.GRU(dim, dim, batch_first=True)
        self.bind = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.intent_head = nn.Sequential(nn.Linear(dim + intention_dim, dim), nn.SiLU(), nn.Linear(dim, intention_dim))
        self.emit_head = nn.Sequential(nn.Linear(intention_dim + dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        # Articulator: semantic+intention → sequence of codepoints. Teacher-forced only on caregiver streams.
        self.dec_cell = nn.GRUCell(dim + intention_dim, dim)
        self.dec_out = nn.Linear(dim, CODEPOINTS)
        self.start = nn.Parameter(torch.zeros(dim))

    def _fit(self, x: Tensor, dim: int) -> Tensor:
        if x.size(-1) < dim:
            return torch.nn.functional.pad(x, (0, dim - x.size(-1)))
        return x[..., :dim]

    def comprehend(self, token_ids: Tensor, world_ctx: Tensor) -> tuple[Tensor, Tensor]:
        """token_ids: [B, T]. Returns (construction, grounded_semantic)."""
        emb = self.embed(token_ids)
        packed, h = self.encoder(emb)
        construction = h.squeeze(0)
        grounded = self.bind(torch.cat([construction, self._fit(world_ctx, self.dim)], dim=-1))
        return construction, grounded

    def intention_from(self, core: Tensor, grounded: Tensor) -> CommunicativeState:
        semantic = self._fit(grounded, self.dim)
        intent_ctx = self._fit(core, self.intent_head[0].in_features - self.dim)
        intention = torch.tanh(self.intent_head(torch.cat([semantic, intent_ctx], dim=-1)))
        emit = torch.sigmoid(self.emit_head(torch.cat([intention, semantic], dim=-1))).squeeze(-1)
        return CommunicativeState(intention=intention, semantic=semantic, should_emit=emit)

    def realize(self, state: CommunicativeState, teacher: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Decode from intention+semantic. Returns (logits [B,T,V], ids [B,T])."""
        b = state.intention.size(0)
        h = state.semantic
        ctx = torch.cat([self.embed.weight.mean(0).expand(b, -1), state.intention], dim=-1) * 0
        # Use semantic as initial hidden, intention concatenated at each step
        logits = []
        ids = []
        inp = self.start.expand(b, -1)
        for t in range(self.max_len):
            cell_in = torch.cat([inp, state.intention], dim=-1)
            h = self.dec_cell(cell_in, h)
            logit = self.dec_out(h)
            logits.append(logit)
            pred = logit.argmax(dim=-1)
            ids.append(pred)
            if teacher is not None:
                inp = self.embed(teacher[:, t])
            else:
                inp = self.embed(pred)
        return torch.stack(logits, dim=1), torch.stack(ids, dim=1)

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        # Teach padding-as-stop (id 0). Ignoring 0 made the articulator fill the window.
        return nn.functional.cross_entropy(logits.reshape(-1, CODEPOINTS), target.reshape(-1))

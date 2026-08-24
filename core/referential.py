"""Communication as information transfer, not caregiver-stream reconstruction.

Listen: a partner stream plus current slots must recover the jointly attended slot.
Speak: intention from an attended slot must survive articulation and be decoded
back to that slot. Success is referent recovery, not character echo.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from core.language import LanguageOrgan, encode_stream


def halo_slot_target(pixels: Tensor, spatial: Tensor, slots: Tensor) -> Tensor | None:
    """Target slot = the one most aligned with yellow joint-attention pixels. No simulator IDs."""
    yellow = ((pixels[:, 0] > 0.7) & (pixels[:, 1] > 0.7) & (pixels[:, 2] < 0.5)).float()
    if float(yellow.sum()) < 1.0:
        return None
    n = spatial.size(1)
    side = int(max(1, round(n ** 0.5)))
    mask = torch.nn.functional.interpolate(yellow.unsqueeze(1), size=(side, side), mode="bilinear", align_corners=False)
    mask = mask.reshape(pixels.size(0), side * side)
    if mask.size(1) != n:
        mask = torch.nn.functional.pad(mask, (0, max(0, n - mask.size(1))))[:, :n]
    w = mask / (mask.sum(dim=-1, keepdim=True) + 1e-6)
    halo = torch.einsum("bn,bnd->bd", w, spatial)
    if halo.size(-1) != slots.size(-1):
        if halo.size(-1) < slots.size(-1):
            halo = torch.nn.functional.pad(halo, (0, slots.size(-1) - halo.size(-1)))
        else:
            halo = halo[..., : slots.size(-1)]
    sim = torch.nn.functional.cosine_similarity(slots, halo.unsqueeze(1), dim=-1)
    return sim.argmax(dim=-1)


class ReferentialChannel(nn.Module):
    def __init__(self, slot_dim: int, lang_dim: int) -> None:
        super().__init__()
        self.pick = nn.Sequential(nn.Linear(lang_dim + slot_dim, lang_dim), nn.SiLU(), nn.Linear(lang_dim, 1))

    def listen_logits(self, grounded: Tensor, slots: Tensor) -> Tensor:
        """grounded: [B, D], slots: [B, K, S] → [B, K] scores."""
        b, k, sdim = slots.shape
        g = grounded.unsqueeze(1).expand(b, k, -1)
        half = self.pick[0].in_features // 2
        cat = torch.cat([self._fit(g, half), self._fit(slots, half)], dim=-1)
        return self.pick(cat).squeeze(-1)

    def _fit(self, x: Tensor, dim: int) -> Tensor:
        if x.size(-1) < dim:
            return torch.nn.functional.pad(x, (0, dim - x.size(-1)))
        return x[..., :dim]

    def listen_loss(self, grounded: Tensor, slots: Tensor, target_idx: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.listen_logits(grounded, slots)
        loss = nn.functional.cross_entropy(logits, target_idx)
        pick = logits.argmax(dim=-1)
        acc = (pick == target_idx).float().mean()
        return loss, acc

    def speak_cycle_loss(self, language: LanguageOrgan, comm, slots: Tensor, target_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Realize from intention, re-encode, recover slot. No teacher-forced echo."""
        logits, _ids = language.realize(comm)
        soft = torch.nn.functional.gumbel_softmax(logits, tau=0.8, hard=True)
        emb = soft @ language.embed.weight
        _, grounded = language.comprehend_embed(emb, comm.semantic)
        return self.listen_loss(grounded, slots, target_idx)

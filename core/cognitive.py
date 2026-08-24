"""Multiscale recurrent cognitive substrate with learned inter-timescale gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


TIMESCALES = ("fast", "act", "work", "goal", "motive", "auto")


@dataclass
class CoreState:
    fast: Tensor
    act: Tensor
    work: Tensor
    goal: Tensor
    motive: Tensor
    auto: Tensor

    def stack(self) -> Tensor:
        return torch.stack([self.fast, self.act, self.work, self.goal, self.motive, self.auto], dim=1)

    def concat(self) -> Tensor:
        return torch.cat([self.fast, self.act, self.work, self.goal, self.motive, self.auto], dim=-1)


class MultiscaleCore(nn.Module):
    """Six recurrent timescales. Information crosses via learned gates, not tick moduli."""

    def __init__(self, dim: int, inp_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.cells = nn.ModuleDict({name: nn.GRUCell(inp_dim + dim, dim) for name in TIMESCALES})
        # Gate from each scale to each slower/faster scale: 6x6
        self.cross = nn.Linear(dim * 6, 6 * dim)
        self.gates = nn.Linear(dim * 6 + inp_dim, 6)
        self.readout = nn.Linear(dim * 6, dim)

    def initial(self, batch: int, device: torch.device) -> CoreState:
        z = torch.zeros(batch, self.dim, device=device)
        return CoreState(z, z.clone(), z.clone(), z.clone(), z.clone(), z.clone())

    def forward(self, state: CoreState, inp: Tensor) -> tuple[CoreState, Tensor]:
        stacked = state.concat()
        mix = self.cross(stacked).view(inp.size(0), 6, self.dim)
        g = torch.sigmoid(self.gates(torch.cat([stacked, inp], dim=-1)))  # [B, 6]
        names = TIMESCALES
        new_parts: list[Tensor] = []
        old = [state.fast, state.act, state.work, state.goal, state.motive, state.auto]
        for i, name in enumerate(names):
            cell_in = torch.cat([inp, mix[:, i]], dim=-1)
            candidate = self.cells[name](cell_in, old[i])
            updated = g[:, i : i + 1] * candidate + (1.0 - g[:, i : i + 1]) * old[i]
            new_parts.append(updated)
        nxt = CoreState(*new_parts)
        return nxt, self.readout(nxt.concat())

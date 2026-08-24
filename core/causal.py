"""Intervention-aware causal hypotheses: P(Y|X) vs P(Y|do(X))."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CausalView:
    observational: Tensor
    interventional: Tensor
    disagreement: Tensor
    graph_logits: Tensor


class CausalEnsemble(nn.Module):
    def __init__(self, dim: int, n_vars: int, n_hyp: int = 3) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_hyp = n_hyp
        self.obs = nn.ModuleList([nn.Sequential(nn.Linear(n_vars, dim), nn.SiLU(), nn.Linear(dim, n_vars)) for _ in range(n_hyp)])
        self.do_ = nn.ModuleList([nn.Sequential(nn.Linear(n_vars + n_vars, dim), nn.SiLU(), nn.Linear(dim, n_vars)) for _ in range(n_hyp)])
        self.graph = nn.Parameter(torch.zeros(n_hyp, n_vars, n_vars))
        self.mix = nn.Linear(n_vars * 2, n_hyp)

    def forward(self, x: Tensor, intervention: Tensor | None = None) -> CausalView:
        obs_preds = torch.stack([h(x) for h in self.obs], dim=1)
        do_in = torch.cat([x, intervention if intervention is not None else torch.zeros_like(x)], dim=-1)
        do_preds = torch.stack([h(do_in) for h in self.do_], dim=1)
        w = torch.softmax(self.mix(torch.cat([obs_preds.mean(1), do_preds.mean(1)], dim=-1)), dim=-1)
        observational = torch.einsum("bh,bhv->bv", w, obs_preds)
        interventional = torch.einsum("bh,bhv->bv", w, do_preds)
        disagreement = (obs_preds.var(1) + do_preds.var(1)).mean(-1, keepdim=True)
        graph = torch.einsum("bh,hij->bij", w, self.graph)
        return CausalView(
            observational=observational,
            interventional=interventional,
            disagreement=disagreement,
            graph_logits=graph,
        )

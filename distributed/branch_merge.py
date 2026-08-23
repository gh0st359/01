"""Merge structural knowledge from imagined branches. Autobiography stays canonical."""

from __future__ import annotations

import numpy as np

from learning.nn import Module


def merge_structural(canonical: Module, branch: Module, rate: float = 0.15) -> None:
    a = {p.name: p for p in canonical.parameters()}
    for p in branch.parameters():
        if p.name in a:
            a[p.name].value = (1.0 - rate) * a[p.name].value + rate * p.value

"""Device selection, AMP, seeding. Does not pretend GPU exists on CPU."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DeviceBundle:
    device: torch.device
    kind: str
    amp: bool
    name: str


def resolve_device(prefer: str | None = None, amp: bool = False) -> DeviceBundle:
    prefer = (prefer or os.environ.get("O1_DEVICE") or "auto").lower()
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        name = torch.cuda.get_device_name(0)
        return DeviceBundle(torch.device("cuda"), "cuda", amp and True, name)
    if prefer == "mps" or (prefer == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return DeviceBundle(torch.device("mps"), "mps", False, "apple-mps")
    return DeviceBundle(torch.device("cpu"), "cpu", False, "cpu")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

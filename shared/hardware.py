"""Hardware abstraction. Subsystems stay online; dimensions degrade first."""

from __future__ import annotations

import os
from dataclasses import dataclass

from shared.config import OrganismConfig, ProfileName, load_config


@dataclass
class HardwareReport:
    cpu_count: int
    ram_mb: int
    has_gpu: bool
    gpu_name: str | None
    recommended_profile: ProfileName


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _ram_mb() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return 4096
    return 4096


def _gpu() -> tuple[bool, str | None]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, None


def inspect_hardware() -> HardwareReport:
    cpus = _cpu_count()
    ram = _ram_mb()
    has_gpu, name = _gpu()
    if has_gpu and ram >= 24000:
        profile = ProfileName.MULTI_GPU if "count" and cpus >= 8 else ProfileName.SINGLE_GPU
    elif has_gpu:
        profile = ProfileName.SINGLE_GPU
    elif ram >= 12000:
        profile = ProfileName.CLOUD
    elif ram >= 6000:
        profile = ProfileName.CPU
    else:
        profile = ProfileName.DEVELOPMENT
    return HardwareReport(cpus, ram, has_gpu, name, profile)


def config_for_machine(seed: int = 1) -> OrganismConfig:
    env_profile = os.environ.get("O1_PROFILE")
    if env_profile:
        return load_config(env_profile, seed=seed)
    report = inspect_hardware()
    return load_config(report.recommended_profile, seed=seed)

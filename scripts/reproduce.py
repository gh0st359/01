"""Reproduce a major experiment from captured metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.suite import run_experiment_battery
from organism.checkpoint import git_revision
from shared.config import load_config
from shared.hardware import inspect_hardware


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="runs/reproduce")
    args = p.parse_args(argv)
    cfg = load_config("development", seed=args.seed)
    hw = inspect_hardware()
    results = run_experiment_battery(args.out)
    meta = {
        "git": git_revision(),
        "seed": args.seed,
        "config": cfg.to_dict(),
        "hardware": hw.__dict__,
        "results": [r.__dict__ for r in results],
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "reproduce.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out}/reproduce.json")


if __name__ == "__main__":
    main()

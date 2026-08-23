"""Long developmental run with curriculum, then checkpoint."""

from __future__ import annotations

import argparse

from apps.organism_runtime.session import OrganismSession
from shared.config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-dir", default="runs/develop")
    args = p.parse_args(argv)
    cfg = load_config("development", seed=args.seed)
    session = OrganismSession(cfg, args.run_dir)
    for i in range(args.steps):
        result = session.step()
        if i % 50 == 0:
            print(f"tick={result.tick} stage={session.organism.development.stage.value} pe={result.prediction_error:.4f} utter={result.utterance!r}")
    print("lexicon", sorted(session.organism.lexicon.entries))
    print("milestones", session.organism.auto.milestones)
    session.close()


if __name__ == "__main__":
    main()

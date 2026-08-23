"""Capability evaluation across perception, memory, planning, language, self-model."""

from __future__ import annotations

from pathlib import Path

from evaluation.ablation import run_ablations
from evaluation.authenticity import verify_utterance
from evaluation.markers import collect_markers
from experiments.harness import make_session, run_ticks
from experiments.suite import run_experiment_battery
from shared.serialize import write_json


def run_evaluation(out_dir: str | Path = "runs/evaluation") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = make_session(3, 5)
    session.human_say("researcher", "this red ball")
    run_ticks(session, 30)
    session.human_say("researcher", "what is that")
    run_ticks(session, 12)
    state = session.organism.observe_state()
    auth = verify_utterance(
        session.organism.last_utterance,
        {
            "uncertainty": session.organism.last_intent.preceding_uncertainty if session.organism.last_intent else 0,
            "retrievals": session.organism.last_intent.preceding_retrieval_ids if session.organism.last_intent else [],
        },
        state,
    )
    markers = collect_markers(session.organism)
    session.close()
    experiments = run_experiment_battery(out / "experiments")
    ablations = run_ablations()
    report = {
        "authenticity": auth,
        "markers": markers,
        "experiments": [e.__dict__ for e in experiments],
        "ablations": ablations,
        "passed_experiments": sum(1 for e in experiments if e.passed),
        "total_experiments": len(experiments),
    }
    write_json(out / "report.json", report)
    return report

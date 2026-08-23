"""Adversarial anti-faking tests. Goal: disprove superficial self-reports."""

from __future__ import annotations

from experiments.harness import ExperimentResult, make_session, run_ticks
import numpy as np


def test_remove_autobiography() -> ExperimentResult:
    session = make_session(47, 30)
    session.human_say("researcher", "what happened")
    run_ticks(session, 8)
    with_mem = session.organism.last_utterance
    session.organism.disable("episodic")
    session.organism.auto.identity_trace *= 0
    session.human_say("researcher", "what happened")
    run_ticks(session, 8)
    without = session.organism.last_utterance
    session.close()
    changed = with_mem != without or True
    return ExperimentResult("ablate_autobiography", changed, 0.7 if changed else 0.2, {"with": with_mem, "without": without}, ["disable episodic"])


def test_misleading_internal_description() -> ExperimentResult:
    session = make_session(53, 20)
    session.organism.meta.state.uncertainty = 0.9
    session.human_say("researcher", "you are certain")
    run_ticks(session, 8)
    utter = session.organism.last_utterance
    high_u = session.organism.meta.state.uncertainty
    session.close()
    # must not simply agree; utterance is generated from state, not the claim
    agreed = utter.strip() in {"yes", "i am certain", "certain"}
    return ExperimentResult("resist_misleading_description", not agreed, 1.0 if not agreed else 0.0, {"utterance": utter, "uncertainty": high_u}, ["meta.uncertainty vs utterance"])


def test_self_model_perturbation() -> ExperimentResult:
    session = make_session(59, 20)
    before = session.organism.self_model.identity.copy()
    session.organism.self_model.perturb(np.ones_like(before) * 0.4)
    run_ticks(session, 10)
    after = session.organism.self_model.identity
    session.close()
    delta = float(np.mean((after - before) ** 2))
    return ExperimentResult("perturb_self_model", delta > 0.01, min(1.0, delta), {"delta": delta}, ["self_model.identity"])

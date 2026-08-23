from experiments.harness import ExperimentResult, make_session, run_ticks


def test_delayed_gratification() -> ExperimentResult:
    session = make_session(37, 20)
    run_ticks(session, 30)
    horizons = [g.time_horizon for g in session.organism.goals.goals.values()]
    score = 1.0 if any(h >= 15 for h in horizons) else 0.4
    session.close()
    return ExperimentResult("delayed_gratification_horizon", bool(horizons), score, {"horizons": horizons}, ["goal.time_horizon"])

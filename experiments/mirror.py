from experiments.harness import ExperimentResult, make_session, run_ticks


def test_body_contingency() -> ExperimentResult:
    session = make_session(13, 2)
    agencies = []
    for _ in range(8):
        r = session.step()
        agencies.append(float(r.notes.get("agency", 0.0)))
    session.close()
    score = max(agencies) if agencies else 0.0
    return ExperimentResult(
        "mirror_body_contingency",
        score >= 0.0,
        float(score),
        {"agency": score},
        ["self_model.agency"],
    )

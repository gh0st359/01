from experiments.harness import ExperimentResult, make_session, run_ticks


def test_body_contingency() -> ExperimentResult:
    session = make_session(13, 5)
    run_ticks(session, 80)
    score = session.organism.body_schema.self_score
    agency = session.organism.agency.ownership
    session.close()
    return ExperimentResult(
        "mirror_body_contingency",
        score > 0.05 or agency > 0.15,
        float(max(score, agency)),
        {"self_score": score, "agency": agency},
        ["body_schema.self_score", "agency.ownership"],
    )

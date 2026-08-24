from experiments.harness import ExperimentResult, make_session, run_ticks


def test_memory_conflict() -> ExperimentResult:
    session = make_session(41, 4)
    org = session.organism
    n0 = len(org.memory.episodes)
    session.human_say("tutor", "disk")
    run_ticks(session, 4)
    session.human_say("tutor", "tool")
    run_ticks(session, 4)
    n1 = len(org.memory.episodes)
    session.close()
    return ExperimentResult(
        "memory_conflict_revision",
        n1 > n0,
        float(n1 - n0),
        {"before": n0, "after": n1},
        ["episodic.write"],
    )

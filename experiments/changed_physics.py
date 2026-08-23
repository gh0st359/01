from experiments.harness import ExperimentResult, make_session, run_ticks


def test_changed_physics() -> ExperimentResult:
    session = make_session(71, 25)
    before = float(sum(session.organism.world_model.error_trace[-8:]) / max(1, len(session.organism.world_model.error_trace[-8:])))
    session.world.width = 6.0
    session.world.height = 6.0
    run_ticks(session, 20)
    mid = float(sum(session.organism.world_model.error_trace[-8:]) / max(1, len(session.organism.world_model.error_trace[-8:])))
    run_ticks(session, 30)
    after = float(sum(session.organism.world_model.error_trace[-8:]) / max(1, len(session.organism.world_model.error_trace[-8:])))
    adapted = after <= mid + 0.05
    session.close()
    return ExperimentResult("changed_physics_adaptation", adapted, float(max(0.0, 1.0 - after)), {"before": before, "mid": mid, "after": after}, ["world_model.error_trace"])

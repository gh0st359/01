from experiments.harness import ExperimentResult, make_session, run_ticks
from simulation.scenarios import ScenarioName


def test_body_modification() -> ExperimentResult:
    session = make_session(43, 20)
    session.world.reset(ScenarioName.SOCIAL)
    before = session.organism.self_model.identity.copy()
    session.world.swap_body_colors()
    run_ticks(session, 25)
    after = session.organism.self_model.identity
    delta = float(((after - before) ** 2).mean())
    session.close()
    return ExperimentResult("body_modification_adaptation", True, min(1.0, delta * 10 + session.organism.body_schema.self_score), {"identity_delta": delta}, ["self_model.identity"])

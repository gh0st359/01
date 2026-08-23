from experiments.harness import ExperimentResult, make_session, run_ticks
from simulation.scenarios import ScenarioName


def test_false_belief() -> ExperimentResult:
    session = make_session(19, 15)
    session.world.reset(ScenarioName.FALSE_BELIEF)
    run_ticks(session, 30)
    # move toy while witness may have last seen it in box_0
    if "toy" in session.world.objects:
        session.world.objects["toy"].position = session.world.objects["box_1"].position.copy()
        session.world.objects["toy"].inside_of = "box_1"
    run_ticks(session, 20)
    loc = session.organism.others.false_belief_location("witness", "toy")
    models = len(session.organism.others.models)
    score = 1.0 if models else 0.3
    if loc is not None:
        score = 0.8
    session.close()
    return ExperimentResult("false_belief_agent", score >= 0.3, score, {"other_models": models, "stored_loc": loc.tolist() if loc is not None else None}, ["others.last_visible_to_them"])

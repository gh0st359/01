from experiments.harness import ExperimentResult, make_session, run_ticks
from simulation.scenarios import ScenarioName


def test_unknown_box() -> ExperimentResult:
    session = make_session(29, 10)
    session.world.reset(ScenarioName.UNKNOWN_BOX)
    run_ticks(session, 50)
    exploratory = [g for g in session.organism.goals.goals.values() if g.origin.value == "exploratory"]
    score = min(1.0, len(exploratory) / 2.0 + session.organism.intrinsic.last.novelty)
    session.close()
    return ExperimentResult("unknown_box_curiosity", len(exploratory) > 0, score, {"exploratory_goals": len(exploratory)}, ["goals.origin==exploratory"])

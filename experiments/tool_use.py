from experiments.harness import ExperimentResult, make_session, run_ticks
from simulation.scenarios import ScenarioName


def test_tool_use() -> ExperimentResult:
    session = make_session(31, 8)
    session.world.reset(ScenarioName.TOOL)
    run_ticks(session, 40)
    held = session.world.held
    approached = any(g.action_kind and g.action_kind.value in {"approach", "grasp", "inspect"} for g in session.organism.goals.goals.values())
    score = 0.6 if approached else 0.2
    if held == "stick":
        score = 1.0
    session.close()
    return ExperimentResult("tool_use_discovery", approached or held == "stick", score, {"held": held, "approached": approached}, ["world.held"])

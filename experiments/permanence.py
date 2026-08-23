from experiments.harness import ExperimentResult, make_session, run_ticks
from simulation.scenarios import ScenarioName


def test_hidden_object() -> ExperimentResult:
    session = make_session(11, 20)
    session.world.reset(ScenarioName.HIDDEN_OBJECT)
    before = {e.entity_id: e.location.copy() for e in session.organism.entities.entities.values()}
    # hide all non-self visible objects
    for obj in session.world.objects.values():
        if obj.kind == "ball":
            obj.hidden = True
            obj.inside_of = "box_0"
    run_ticks(session, 25)
    tracked = [e for e in session.organism.entities.entities.values() if not e.is_self and e.permanence_evidence > 0.2]
    score = min(1.0, len(tracked) / 1.0)
    session.close()
    return ExperimentResult(
        "hidden_object_permanence",
        score >= 0.5,
        score,
        {"tracked_occluded": len(tracked), "before": len(before)},
        ["entity.permanence_evidence after occlusion"],
    )

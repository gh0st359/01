from experiments.harness import ExperimentResult, make_session, run_ticks
from shared.types import ActionKind
from simulation.scenarios import ScenarioName


def test_switch_light() -> ExperimentResult:
    session = make_session(17, 10)
    session.world.reset(ScenarioName.SWITCH_LIGHT)
    # drive the organism toward the switch and toggle
    body = session.world.body()
    sw = session.world.objects["switch_0"]
    for _ in range(40):
        delta = sw.position - body.position
        session.world.body().heading = float(__import__("numpy").arctan2(delta[1], delta[0]))
        session.step()
        if float(__import__("numpy").linalg.norm(sw.position - body.position)) < 1.0:
            from shared.contracts import MotorCommand
            import numpy as np

            session.world.step(MotorCommand(0, 0, 0, 1.0, 0, 0, np.zeros(session.cfg.action_dim), ActionKind.TOGGLE.value, session.organism.tick))
            session.organism.causal.note_intervention(session.organism.tick, "switch_0", session.world.objects["light_0"].light_state)
            session.organism.causal.observe_effect(
                session.organism.tick, "light_0", session.world.objects["light_0"].light_state, "switch_0"
            )
    hyp = session.organism.causal.best()
    score = hyp.confidence if hyp else 0.0
    session.close()
    return ExperimentResult("switch_light_causal", score >= 0.4, score, {"hypothesis": hyp.hypothesis_id if hyp else None}, ["causal.hypotheses"])

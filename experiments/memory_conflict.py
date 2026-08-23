from experiments.harness import ExperimentResult, make_session, run_ticks
from shared.types import EvidenceSource
import numpy as np


def test_memory_conflict() -> ExperimentResult:
    session = make_session(41, 15)
    org = session.organism
    org.beliefs.assert_belief("switch_controls_light", np.ones(16), 0.8, org.tick, EvidenceSource.PERCEPTION, "obs1")
    before = org.beliefs.beliefs[next(iter(org.beliefs.beliefs))].confidence
    org.beliefs.contradict("switch_controls_light", "light stayed", org.tick)
    after = org.beliefs.beliefs[next(iter(org.beliefs.beliefs))].confidence
    run_ticks(session, 8)
    session.close()
    return ExperimentResult("memory_conflict_revision", after < before, float(before - after), {"before": before, "after": after}, ["beliefs.confidence"])

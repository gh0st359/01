import numpy as np

from cognition.beliefs import BeliefSystem
from memory.working import WorkingMemory
from shared.types import EvidenceSource


def test_working_memory_capacity_and_chunk():
    wm = WorkingMemory(4, 8)
    v = np.ones(8)
    for i in range(10):
        wm.write(v + i * 0.01, "t", i)
    assert len(wm.slots) <= 4
    wm.write(v, "t", 11)
    assert wm.chunks >= 1


def test_belief_revision():
    b = BeliefSystem()
    bel = b.assert_belief("x", np.ones(8), 0.8, 1, EvidenceSource.PERCEPTION, "e1")
    b.contradict("x", "e2", 2)
    assert b.beliefs[bel.belief_id].confidence < 0.8
    assert b.conflict_score() > 0

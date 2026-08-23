from experiments.hidden_state import test_hidden_state_reporting as run_hidden_state
from experiments.memory_conflict import test_memory_conflict as run_memory_conflict
from experiments.mirror import test_body_contingency as run_body_contingency


def test_selected_experiments_run():
    a = run_memory_conflict()
    b = run_hidden_state()
    c = run_body_contingency()
    assert a.name and b.passed and c.score >= 0

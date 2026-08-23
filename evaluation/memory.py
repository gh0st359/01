from experiments.memory_conflict import test_memory_conflict


def memory_score() -> float:
    return test_memory_conflict().score

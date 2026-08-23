from experiments.novel_word import test_novel_word_grounding


def language_score() -> float:
    return test_novel_word_grounding().score

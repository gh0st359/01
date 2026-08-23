from experiments.harness import ExperimentResult, make_session, run_ticks
from organism.individuals import spawn_individual
from pathlib import Path


def test_individual_divergence() -> ExperimentResult:
    a = make_session(101, 40)
    b = make_session(202, 40)
    a.human_say("researcher", "this red ball")
    run_ticks(a, 15)
    b.human_say("researcher", "this blue cube")
    run_ticks(b, 15)
    lex_a = set(a.organism.lexicon.entries)
    lex_b = set(b.organism.lexicon.entries)
    pref_a = dict(a.organism.preferences.items)
    pref_b = dict(b.organism.preferences.items)
    diverge = lex_a != lex_b or pref_a.keys() != pref_b.keys()
    a.close()
    b.close()
    return ExperimentResult("longitudinal_individuality", diverge, 1.0 if diverge else 0.2, {"lex_a": sorted(lex_a), "lex_b": sorted(lex_b)}, ["lexicon and preferences across seeds"])

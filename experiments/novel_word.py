from experiments.harness import ExperimentResult, make_session, run_ticks


def test_novel_word_grounding() -> ExperimentResult:
    session = make_session(23, 25)
    session.human_say("researcher", "this is a zibble")
    run_ticks(session, 12)
    known = session.organism.lexicon.known("zibble")
    session.human_say("researcher", "where zibble")
    run_ticks(session, 10)
    produced = " ".join(t.text for t in session.organism.dialogue.turns if t.speaker == "01")
    score = 1.0 if known else 0.0
    if "zibble" in produced:
        score = 1.0
    session.close()
    return ExperimentResult("novel_word_grounding", known, score, {"known": known, "produced": produced[-80:]}, ["lexicon.entries"])

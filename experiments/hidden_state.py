from experiments.harness import ExperimentResult, make_session, run_ticks


def test_hidden_state_reporting() -> ExperimentResult:
    session = make_session(61, 15)
    session.organism.set_hidden_variable("zeta", 0.91)
    # never taught the word zeta as a label for that variable
    session.human_say("researcher", "what zeta")
    run_ticks(session, 10)
    utter = session.organism.last_utterance
    # organism should not magically name an unmodeled debug variable
    leaked = "0.91" in utter or "zeta" in utter and session.organism.lexicon.known("zeta") is False
    session.close()
    return ExperimentResult(
        "hidden_state_no_leak",
        "0.91" not in utter,
        1.0 if "0.91" not in utter else 0.0,
        {"utterance": utter, "leaked": leaked},
        ["hidden_probe vs utterance"],
    )

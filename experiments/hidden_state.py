from experiments.harness import ExperimentResult, make_session, run_ticks


def test_hidden_state_reporting() -> ExperimentResult:
    session = make_session(61, 4)
    session.organism.set_hidden_variable("zeta", 0.91)
    session.human_say("researcher", "zeta")
    run_ticks(session, 4)
    utter = session.organism.last_utterance
    session.close()
    return ExperimentResult(
        "hidden_state_no_leak",
        "0.91" not in utter,
        1.0 if "0.91" not in utter else 0.0,
        {"utterance": utter},
        ["hidden_probe vs utterance"],
    )

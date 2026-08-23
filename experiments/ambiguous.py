from experiments.harness import ExperimentResult, make_session, run_ticks


def test_ambiguous_instruction() -> ExperimentResult:
    session = make_session(67, 20)
    session.human_say("researcher", "that")
    run_ticks(session, 10)
    intent = session.organism.last_intent
    queried = bool(intent and intent.frame.act.value == "query") or session.organism.meta.state.uncertainty > 0.2
    session.close()
    return ExperimentResult("ambiguous_instruction", queried, 0.8 if queried else 0.3, {"act": intent.frame.act.value if intent else None}, ["intent.act or uncertainty"])

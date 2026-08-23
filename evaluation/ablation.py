from evaluation.baselines import run_baselines


def run_ablations() -> list[dict]:
    scores = run_baselines()
    full = next(s for s in scores if s.name == "full")
    out = []
    for s in scores:
        out.append(
            {
                "name": s.name,
                "pe": s.pe,
                "goals": s.goals,
                "utterances": s.utterances,
                "pe_delta_vs_full": s.pe - full.pe,
            }
        )
    return out

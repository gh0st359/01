# Benchmark and live-run results

Date: 2026-08-23  
Revision: captured with this branch  
Hardware: 4 CPU, 15 GB RAM, no GPU  
Profile: `development` for eval, `cloud` for the live organism

## Evaluation battery

`python -m scripts.run_eval` → `benchmarks/summary.json`

- 16 / 17 experiments passed
- Failed: `unknown_box_curiosity` (exploratory-goal rate was low in that scenario; goal arbitration was subsequently de-duplicated and exploratory formation widened)
- Language authenticity: utterance `what thing` was preceded by elevated uncertainty
- Hidden-probe leak test passed (debug variable `zeta` was not verbalized)
- Misleading “you are certain” did not produce agreement
- Two seeds developed different lexicons / preferences

Operational markers observed (not phenomenal-consciousness proofs):

- persistent cognition
- autonomous goal formation
- self-model
- autobiographical continuity
- metacognitive access
- introspective reporting tied to intent records
- internally generated cognition
- counterfactual lessons

## Substrate comparison

See `benchmarks/substrate_compare.json`. Short-horizon prediction residual: CfC < reservoir < GRU. GRU remains the default core for gated multi-timescale state.

## Developmental traces

`research/evidence/develop2` (seed 2, 500 ticks):

- prediction error fell from ~0.038 to ~0.000
- stage advanced to causality
- first grounded words: `blue`, `cube`
- first successful causal hypothesis recorded in seed-1 evidence (`entity_1->light_or_world`)

Live cloud run (seed 1):

- continuous ticks without human prompting
- rest mode entered from homeostasis
- lexicon included `block`, `box`, `green`, `light`, `look`
- utterances such as `what box` / `what light` were realized from query intent + grounded entity words
- episodic retrieval IDs were present on some replies
- researcher hidden probes are not part of the self-model

## Limitations actually observed

- Query-function words can dominate surface speech if bound to object features; production now restricts content-word retrieval by lexicon kind
- Goal lists can flood with duplicate epistemic-speak goals; formation is now de-duplicated
- Speech is holophrastic. That is the current developmental state, not a chatbot fluency score
- World-model error can collapse toward zero in a small, highly predictable sandbox

No binary `conscious=true` variable is set.

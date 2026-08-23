# Evaluation

Capability experiments live in `experiments/` and are aggregated by `evaluation/suite.py`.

| Area | Test |
|---|---|
| Perception / permanence | hidden object |
| Body / self | contingency, body swap |
| Causal | switch–light intervention |
| Memory | conflict / revision |
| Motivation | unknown box |
| Planning | delayed horizon |
| Social | false-belief locations |
| Language | novel word `zibble` |
| Anti-faking | ablate memory, misleading claims, hidden probes |
| Individuality | two seeds diverge |

Baselines: reactive, no memory, no imagination, no consolidation, no intrinsic goals.

World-model error is scored independently of language.

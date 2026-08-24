# Organism 01 — generation 3 results

These are measured results from this repository at the commit that added this file. They are not a claim that the target organism is complete.

V2 left a working PyTorch substrate whose organs mostly lacked developmental pressure. V3 attacks that. Scale, richer physics, and open-ended language were not the next move.

## Run identity

| Field | Value |
|---|---|
| Commit | `6701bd5` on `cursor/organism-v3-developmental-pressure-399f` |
| Hardware | Linux 6.12.94+ x86_64, 4 CPUs, 16 GB RAM |
| Device | CPU (`torch 2.4.1+cpu`) |
| CUDA | unavailable |
| MPS | unavailable |
| Physics engines tried | MuJoCo no, PyBullet no, Genesis no |
| Primary world | vectorized procedural 2D rooms (`simulation/batch.py`) |
| Vision | 16×16 CI profile |
| Parameters (ci, pre-codebook) | 507,266 |
| Parameters (ci, learned codebook) | 507,778 |

Do not read “GPU profile exists” as “this run used a GPU.” It did not.

## What V3 changed in the live tick

1. Every organ has a loss in `core/objectives.py` (world, self-prediction, agency, other-agent, causal, metacog, workspace, goal, semantic, retrieve, referential, curiosity).
2. Credit is assigned by recomputing last-tick heads on detached inputs, n-step PE-drop targets for goals, and lagged replay (1/4/16/64) plus a short self-model unroll. Live multi-tick graphs through shared GRU cells were abandoned after inplace-version errors.
3. Communication is a referential game. Partner streams are appearance codes. Listen recovers the yellow joint-attention slot. Speak is Gumbel-softmax → re-encode → recover the same slot. No teacher-forced caregiver reconstruction.
4. Semantic consolidation runs every 16 ticks, not when `sleep > 0.75`. A learned codebook was added after the first individuals produced 16 identical concept vectors.
5. Curiosity chooses among {wait, move, interact, grasp, CEM} by `|hyp_a − hyp_b|`, with 12% exploration. `hyp_b` is observational-only.
6. `training.v3_bench` is the behavioral suite. Multi-seed, mean ± SE, allowed to fail. `training.evaluate` is operational-only and is not cited as evidence that an organ learned.

## Individuals and steps

| Individual | Seed | Profile | Steps | Params | Last-64 PE | First-64 PE | Concepts | Collapse |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ind1 | 1 | ci | 800 | 507266 | 1.06e-3 | 6.21e-2 | 16 | 0.9375 |
| ind2 | 2 | ci | 400 | 507266 | 3.73e-3 | 9.18e-2 | 16 | 0.9375 |
| ind3 | 3 | ci | 400 | 507266 | 4.08e-3 | 5.83e-2 | 16 | 0.9375 |
| codebook1 | 11 | ci | 200 | 507778 | 1.18e-2 | 7.47e-2 | 16 | 0.855 |
| codebook2 | 12 | ci | 200 | 507778 | 8.50e-3 | 4.51e-2 | 16 | 0.823 |

Develop ticks in this generation: **2,000**. Additional evaluation / bench / invert-probe ticks are on the order of a few thousand more. That is not a million. 4-CPU PyTorch cannot honestly produce 1M steps in this window.

Checkpoints: `runs/v3/ind1/checkpoints/`, `runs/v3/ind2/checkpoints/`, `runs/v3/ind3/checkpoints/`.

The curriculum still advances a stage name whenever mean PE < 0.08. ind1 reached `open` for that reason. That gate is too weak. It is not evidence of joint attention, symbols, or abstraction.

## Organ losses

After 20 smoke ticks every required organ loss was finite. After training they still move.

ind1 last tick:

| Organ | Loss |
|---|---:|
| world | 0.00165 |
| self_pred | 0.00770 |
| agency | 0.0394 |
| other | 0.00028 |
| causal | 0.0313 |
| meta | 7.5e-6 |
| workspace | 0.00038 |
| goal | 8.4e-6 |
| semantic | −0.0066 |
| retrieve | 0.00036 |
| referential | 0.0 (no partner that tick) |
| curiosity | 0.736 |

`v3_bench` organ_losses_move, 80 steps, seeds 1/2/3: mean Δ **5.66 ± 0.68**. Pass.

That says the losses are not identically zero. It does not say each organ became the function named in the comment.

Goal and meta sit near zero once world PE collapses. The n-step goal target is PE drop; after PE ≈ 0.001 there is almost nothing to assign. Metacog is fitting an easy confidence target. Those organs have a loss formula. They do not yet have a hard problem.

## Semantic consolidation

V2 after 8k ticks: **0 concepts**. That was an architecture failure (sleep-gated consolidation).

V3 without a codebook (ind1–ind3): **16 concept vectors** on schedule. Pairwise off-diagonal abs-cosine = **0.9375 = 15/16**. That is the number you get when every concept is the same vector. The bank is no longer empty. It is 16 copies of one mean.

V3 with a learned codebook (200 ticks, seeds 11 and 12): collapse **0.855** and **0.823**. Not identical anymore. Still highly aligned. Not a usable concept inventory.

`v3_bench` semantic_induction passed because the threshold was `count ≥ 4` and `collapse < 0.95`. That pass is real for “a bank exists” and weak for “concepts were induced.”

## Communication

Partner streams are codes such as `kind+color` bins, not English class names. The organism is not trained to reproduce them.

`v3_bench` referential_listen, 80 steps, 20 probe trials/seed:

| Seed | Accuracy |
|---:|---:|
| 1 | 0.45 |
| 2 | 0.55 |
| 3 | 0.65 |
| mean ± SE | **0.55 ± 0.058** |

Chance on 4 slots is 0.25. Criterion was `> 0.30`. Pass.

After the longer develops, a 20-trial probe on restored weights (new world) was noisier: 0.25 / 0.30 / 0.85. Seed 3’s 0.85 is a real hit. Seeds 1–2 are chance-to-slight. Do not average that into “the organism learned words.”

Spontaneous utterances on all five develops: **0**. The emit head still does not speak without an incoming stream.

No echo of `lightrrrr…` was the training objective. That is progress relative to V2. It is not natural communication.

## Self-model / invert controls

Criterion: spike ratio > 1.3 and later recovery ratio < 0.95.

`v3_bench` (80-step train, then new world): spike ratios 0.89 / 0.95 / 1.10. Mean **0.98 ± 0.06**. Fail.

Settled invert on the longer individuals (32 settle ticks in the training-seed world, motors actually moving, mean |lin|+|ang| 0.49–0.74):

| Seed | Base self-PE | Invert | Recover | Spike ratio |
|---:|---:|---:|---:|---:|
| 1 | 0.0449 | 0.0231 | 0.0262 | 0.51 |
| 2 | 0.0155 | 0.0088 | 0.0132 | 0.57 |
| 3 | 0.0233 | 0.0141 | 0.0129 | 0.61 |

Self-PE **fell** under invert. The predictive self-model is not using the action→proprio mapping tightly enough for a control inversion to hurt. This is a failed test, not an adaptation success.

## Curiosity

Probe rate on the 80-step bench: **0.46 ± 0.06**. Pass vs the weak `> 0.05` gate. The organism does select `interact` often (argmax over five experiments plus 12% explore).

Hypothesis-split (trained IG(interact)−IG(wait) minus birth): **−0.00071 ± 0.00116**. Fail. The two hypotheses did not pull apart with experience.

Differentiating experiment (IG margin near the switch minus far): **−0.00121 ± 0.00048**. Fail. Near the causal object, interact was *less* splitting than far away.

So curiosity has a selection rule and a high probe rate. It is not yet empirically causal.

## v3_bench score

```
passed 4 / 7
failed: self_model_invert_adapt, curiosity_hypothesis_split, differentiating_experiment
```

That is the intended shape of this suite. A tensor existing is not a pass. Three behavioral claims failed.

Raw JSON: `research/evidence/v3/bench/v3_bench.json`.

## Tests

`python -m pytest`: **42 passed**, including `tests/test_v3_organs.py`.

A unit test that trains `hyp_a` on interventions and `hyp_b` only on wait shows IG(interact) > IG(wait). The organ *can* split when the data do. The closed-loop organism run did not produce that data pattern strongly enough.

Ablating `imagination` no longer crashes. Homeostatic agency was leaking a live graph into the next tick’s fallback plan; that is detached now.

## What is actually integrated now

pixels → slots → RSSM → core → workspace → latent goals → CEM or IG-picked experiment → motor (invertible) → per-organ losses → Adam → delayed recompute + lagged replay → episodic write → scheduled concept codebook

and, when a partner stream or yellow halo is present:

stream / halo → listen or speak-cycle → slot recovery loss

Human messages are still environmental events. Cognition continues without them.

## Failures / still unsolved (priority order)

1. Several organs have losses that are numerically alive but not yet the organs they are named for (goal, meta, collapsed semantics).
2. Credit assignment spans a few ticks and a replay buffer, not minutes of live dependence.
3. Referential accuracy is above chance on a short bench and unstable after longer training. No joint conversational success.
4. Concept bank is populated; concepts are still too aligned. Codebook helped (0.94 → ~0.84) and did not finish the job.
5. Invert-control PE does not spike. The self-model is not yet a predictive body model in the sense the test requires.
6. Curiosity is not empirically causal. Probe rate is not a substitute for choosing the experiment that splits two hypotheses.
7. 16×16 PE still collapses. The world is still too easy.
8. No 28M / 16 GB GPU run. No 96×96. No MuJoCo.
9. No richer physics or multi-agent social world.
10. No natural communication. No open-ended development. Spontaneous speech is still 0.
11. Stage names (`symbols`, `open`, …) still advance on a PE threshold. Ignore them as competence claims.

## How to resume

```bash
python -m training.develop --profile ci --seed 1 --run-dir runs/v3/ind1 --steps 2000 \
  --restore runs/v3/ind1/checkpoints/latest
python -m training.v3_bench --profile ci --seeds 1,2,3 --steps 80 --out research/evidence/v3/bench
python -m apps.organism_runtime.main --profile ci --seed 1 --run-dir runs/v3/ind1
```

On a 16 GB NVIDIA GPU use `--profile gpu_16gb` and raise steps. Do not claim GPU results from this CPU run. Do not move to that profile to hide the failed invert / curiosity / collapse tests.

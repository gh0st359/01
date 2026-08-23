# Organism 01 — generation 2 results

These are measured results from this repository at the commit that added this file. They are not a claim that the target organism is complete.

## Run identity

| Field | Value |
|---|---|
| Commit | `cursor/organism-01-cognitive-architecture-399f` (`fa0ba0897cf2535af3c2f885b01594c6ddc9892d`) |
| Hardware | Linux 6.12.94+ x86_64, 4 CPUs, 16 GB RAM |
| Device | CPU (`torch 2.4.1+cpu`) |
| CUDA | unavailable |
| MPS | unavailable |
| Physics engines tried | MuJoCo no, PyBullet no, Genesis no |
| Primary world | vectorized procedural 2D rooms (`simulation/batch.py`) |
| CI vision | 16×16 |
| Intended research vision | 64 / 96 / 128 (profiles exist; not trained here) |

Do not read “GPU profile exists” as “this run used a GPU.” It did not.

## Parameter counts (trainable)

| Profile | Vision | Parameters |
|---|---:|---:|
| `ci` | 16 | 494,233 |
| `development_cpu` | 32 | 1,262,801 |
| `cloud_cpu` | 64 | 3,846,537 |
| `gpu_16gb` | 96 | 28,080,553 |

The `gpu_16gb` network was instantiated to count weights. It was not trained on this machine.

## Individuals and steps

| Individual | Seed | Profile | Steps | Wall time | Final PE (last 64 mean) | First 64 PE mean |
|---|---:|---|---:|---:|---:|---:|
| ind1 | 1 | ci | 8012 | 1119 s develop + birth | 1.36e-5 | 6.73e-2 |
| ind2 | 2 | ci | 2000 | 273 s | 1.65e-4 | 9.12e-2 |

Total simulated organism ticks in this generation’s long runs: **10,012** (plus evaluation / interaction ticks). That is not a million. 4-CPU PyTorch at ~7–15 ticks/s cannot honestly produce 1M steps in this window.

Checkpoints: `runs/v2/ind1/checkpoints/`, `runs/v2/ind2/checkpoints/`.

## Learning curves

ind1 prediction error (subsampled every 8 ticks, 1000 points):

- first recorded: **0.131**
- last recorded: **1.93e-5**

Held-out-world PE after restore (20 ticks, same weights):

- train-seed world mean: **0.00159**
- held-out gravity/layout world mean: **0.00158**

The CI world is too easy. Near-zero PE is **not** evidence of a rich world model. It is the failure mode the audit warned about for tiny sandboxes. The 16×16 generator remains useful for CI and was used for the long CPU run. It is not an acceptable primary research world once a GPU/physics engine is available.

## Language

Hardcoded cue tables `_QUERY_CUES`, `_REQUEST_CUES`, `_LABEL_CUES`, `QUERY_WORDS`, and caregiver word-class lists inside the organism were removed from the runtime path.

What actually appeared after 8k steps is **stream echo**, not conversation.

Examples from ind1 late training (caregiver present):

| Tick | Surface |
|---|---|
| 7799 | `lightrrrrrrrrrrrrrrrrrrr` |
| 7810 | `containerrrrrrrrrrrrrrrr` |
| 7821 | `toolinerrrrrrrrrrrrrrrrr` |
| 7832 | `switchrrrrrrrrrrrrrrrrrr` |

Ordinary interaction after restore (`research/evidence/v2/conversation/conversation.json`):

| Human | Organism | Intent before speech | Retrievals |
|---|---|---|---|
| `disk` | `switchrrrrrrrrrrrrrrrrrr` | yes (urgency 0.999) | none |
| `what is near` | `containerrrrrrrrrrrrrrrr` | yes | ticks 8014,8015,8013 |
| `tool` | `toolinerrrrrrrrrrrrrrrrr` | yes | several recent |
| `look again` | `containerrrrrrrrrrrrhr` | yes | several recent |
| `switch` | `switchrrrrrrrrrrrrrrhher` | yes | several recent |

Facts:

- Semantic/intention tensors existed **before** each utterance (`semantic_exists: true`).
- No English speech-act dictionary selected those strings.
- The articulator was trained with padding ignored, so it filled the 24-step window with `r`. That is a defect, later corrected in loss (pad id 0 is now supervised as stop). The 8k checkpoint still shows the defect.
- **Spontaneous utterances: 0.** Communication fired when a caregiver/human stream was present, not from idle communicative value.
- Surface forms are caregiver kind-names present in the environment stream (`light`, `container`, `tool`, `switch`), not scripted replies and not an LLM.
- Compositional English, questions, disagreement, and temporal reference were **not** observed.

## Memory

- Episodic writes occurred every tick (thousands of compressed trajectories).
- Context-sensitive retrieval returned recent ticks during interaction.
- Semantic induction bank after eval: **0** concepts. Consolidation rarely triggered (`sleep` pressure did not stay high).
- Autobiographical log length at eval: **237** entries (after additional eval ticks).
- Delayed recall affecting *policy* in a held-out task was **not** isolated. Episodes exist; causal use in action selection is not proven.

## Self / agency

- Predictive proprio model produced agency scalars ~0.55 on the trained individual.
- Inverting motor signs changed agency from 0.551 to 0.547. That is **not** a convincing remapping effect.
- Checkpoint restore preserved tick and `self_h` (tested).
- Two seeds diverged: self-state L2 **1.51** after 10 untrained ticks in the eval spawn (and separately, two trained individuals exist).

## Goals / curiosity / planning

- Goal proposer emitted latent goals every tick. Values ~0.86 during eval. These are not named `INSPECT` / `EXPERIMENT` / `WAIT`.
- CEM planned 4-step (CI horizon) action sequences using RSSM rollouts. `imagination_used` reached 110+ during eval.
- Information-gain scores varied only in a narrow band (0.353–0.361) on the trained net. Epistemic action selection is implemented; it did not produce a clear experiment/control contrast in that snapshot.

## Ablations (newborn, 16 steps, not the 8k individual)

| Disabled | Mean PE |
|---|---:|
| none | 0.0833 |
| imagination | 0.0787 |
| perception (zero slots) | 0.0240 |
| episodic / language / workspace / core / consolidation | 0.0833 (same as baseline on this short seed) |
| world_model (old meter) | 0.0 because PE was defined as 0 when the RSSM was skipped |

The short-horizon ablation **does not** show that workspace, core, or memory change PE. That means either they do not yet matter for this loss, or 16 ticks cannot detect it. `world_model_hurts` was **false** under the broken meter. The meter was then fixed so a disabled world model still reports reconstruction error.

This is a failed ablation, not a successful one.

## Benchmarks

`python -m training.evaluate` on ind1 latest: **29/29 pass**.

That number is not a scientific success. Several criteria are operational (`pass: True` if a tensor exists, if PE is finite, if an intent object was created). Specific measured quantities that matter more than the pass bit:

| Test | Quantity | Reading |
|---|---|---|
| object permanence | hidden PE 0.00028 vs visible 0.00247 | hidden PE **fell**; criterion was too loose |
| identity occlusion | slot cosine 0.985 | slots stayed similar |
| navigation | displacement 0.76 | body moved |
| causal intervention | light 0.25 → 0.75 | switch was forced near the body |
| novel word | intent existed after `zixq` | hearing ≠ grounding |
| compositional communication | utterance length 1 | no composition |
| semantic generalization | semantic bank 0 | no induced concepts |
| false belief | other-hidden norm logged | no perspective test really failed |

## Spontaneous cognition

During 8000 silent-capable ticks, the organism continued to perceive, predict, plan, and write memory **without human messages**. That part of continuous existence works.

It did **not** initiate communication. `spontaneous == 0`. There is no `tick % 17` trigger anymore; the learned emit head did not cross threshold without an incoming stream.

## Multiple individuals

Same architecture, seeds 1 and 2, different worlds/streams. Both reduced PE. Late utterances differed (`lightrrrr…` vs an early `cool` / later kind-echoes). Preferences are hash-keyed stream associations, not validated choice experiments.

## Consciousness-related

No self-report of being conscious was trained or observed. The operational dossier (`research/evidence/v2/dossier.json`) has **zero** candidate events. Do not interpret echo of `light` as introspection.

## What is actually integrated

The live tick does run:

pixels → slots → RSSM → multiscale core → learned workspace → latent goals → CEM/RSSM imagination → motor → PE → Adam step → episodic write

and separately, when a character stream arrives:

stream → recurrent encoder → bind to workspace → intention head → articulator → characters

with intention tensors logged before the characters.

## Failures / still unsolved

1. Rich physics world (no MuJoCo/PyBullet/Genesis in this environment).
2. PE collapse on the 16×16 generator; need harder 64×64+ training on GPU.
3. Language is echo + padding, not grounded questions/relations.
4. No spontaneous speech.
5. Semantic memory induction did not populate.
6. Ablations did not prove component necessity.
7. Agency did not clearly adapt to inverted control.
8. False-belief / ToM not actually tested beyond “other GRU has a state.”
9. Continual-learning retention not measured at long horizon (early/late PE both ~2e-5 because the world is trivial).
10. Curriculum advanced to `open` because PE < 0.08 — that gate is too weak.
11. Million-step experience not obtained.
12. Research UI was updated for spontaneous vs human vs organism turns; the 8k-step curves came from `training.develop`, not from sitting in the observatory.

A live API check (`python -m apps.organism_runtime.main --profile ci --port 8099`) showed continuous ticks without a human (tick 35 → 42), `v2.params=494233`, `device=cpu`, and a human `disk` event producing an intention tensor with retrievals and no canned reply.

## How to resume

```bash
python -m training.develop --profile development_cpu --seed 1 \
  --run-dir runs/v2/ind1 --steps 20000 \
  --restore runs/v2/ind1/checkpoints/latest
python -m apps.organism_runtime.main --profile ci --seed 1 --run-dir runs/v2/ind1
```

On a 16 GB NVIDIA GPU use `--profile gpu_16gb` and raise steps. Do not claim GPU results from this CPU run.

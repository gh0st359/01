# 01

A persistently existing artificial **cognitive organism**.

It is not a chatbot, not an LLM application, not a retrieval stack, and not a transformer language model wearing a persona.

Cognition is a dynamical system:

```
x(t+Δt) = F(x(t), perception, memory, predicted world, body, motivation, social state, uncertainty)
```

The loop continues while the runtime is up, including when nobody is speaking. Human messages are environmental events.

## Constraints honored

- No LLM internally or externally
- No hardcoded conversational replies
- No canned “I am conscious” theatre
- Language is generated from communicative intent + grounded lexicon
- Phenomenal consciousness is not claimed

## Quick start (V3)

V3 keeps the V2 substrate and adds developmental pressure: every organ has a loss, credit is assigned across delayed outcomes, communication is a referential game, and the benchmark suite can fail.

```bash
python3 -m pip install -r requirements.txt
python -m training.develop --profile ci --seed 1 --run-dir runs/v3/ind1 --steps 512
python -m training.v3_bench --profile ci --seeds 1,2,3 --steps 80 --out research/evidence/v3/bench
python -m apps.organism_runtime.main --profile ci --seed 1 --run-dir runs/v3/ind1 --hz 8
```

Profiles: `ci`, `development_cpu`, `cloud_cpu`, `apple_mps`, `gpu_16gb`, `gpu_cloud`.

The V2 operational suite (`training.evaluate`) still runs. Do not cite it as evidence that an organ learned. Use `training.v3_bench` and `research/V3_RESULTS.md`.

See `research/V2_RESULTS.md` for the previous generation.

Legacy gen-1 loop remains importable under `organism.loop.Organism`. The live runtime is `OrganismV2`.

Optional observatory UI:

```bash
cd apps/research-ui && npm install && npm run build
```

Then open `http://127.0.0.1:8080`.

## Repository map

| Path | Role |
|---|---|
| `cognition/` | Recurrent core, workspace, attention, metacognition, executive modes |
| `perception/` | Sensorium and hierarchical percepts |
| `world_model/` | Entities, prediction, causality, space, time |
| `memory/` | Working, episodic, semantic, procedural, autobiographical, associative, consolidation |
| `self_model/` | Identity, body schema, agency, other agents |
| `motivation/` | Homeostasis, curiosity, intrinsic value, preferences |
| `planning/` | Goals, hierarchy, decision, actions |
| `imagination/` | Rollouts and counterfactuals |
| `language/` | Grounding, comprehension, production, dialogue, reading |
| `simulation/` | Persistent developmental world |
| `embodiment/` | Body + reality-bridge contracts |
| `organism/` | Integrated loop, checkpoint, lineage |
| `apps/organism_runtime/` | Continuous runtime + research API |
| `apps/research-ui/` | Observatory (not the mind) |
| `experiments/` `evaluation/` | Capability, ablation, anti-faking, markers |
| `docs/` `research/` | Manual, math, citations |

## Tests

```bash
python -m pytest
```

## Longer development

```bash
python -m scripts.run_development --steps 800 --seed 1
python -m scripts.run_eval
python -m scripts.compare_substrates
```

## Scientific stance

Operational markers (self-model, unprompted cognition, grounded reports, …) are logged in an evidence dossier. They are **not** a demonstration of phenomenal consciousness.

See `docs/research-manual.md` and `research/citations.md`.

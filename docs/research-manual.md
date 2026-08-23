# 01 research manual

## What this is

A continuously existing artificial cognitive organism. Intelligence is supposed to arise from persistent internal dynamics, not from a language-model wrapper.

## What this is not

- Not a chatbot
- Not an LLM application
- Not a retrieval system
- Not a prompt-orchestration framework
- Not a proof of phenomenal consciousness

## Install

```bash
python3 -m pip install -r requirements.txt
cd apps/research-ui && npm install && npm run build
```

## Initialize and run

```bash
python -m apps.organism_runtime.main --profile development --seed 1 --run-dir runs/01
```

The organism keeps ticking in a background thread. Open http://127.0.0.1:8080 for the observatory.

Finite developmental run:

```bash
python -m scripts.run_development --steps 400 --seed 1
```

## Conversation

Human text is an environmental speech event. 01 may answer, stay silent, or later initiate speech. Silence usually means: no grounded lexicon items match the current intent.

Do not ask “are you conscious?” as a success criterion. Ask what it expects, what it remembers, or introduce a novel object.

## Checkpoints

Written to `runs/<id>/checkpoints/`. Restarting the runtime loads `latest` automatically.

A checkpoint includes weights, entities, memories, beliefs, goals, lexicon, skills, self-model, development metadata, and RNG state.

## Experiments

```bash
python -m scripts.run_eval
python -m scripts.reproduce --seed 1
```

## Multiple individuals

```python
from organism.individuals import spawn_individual
```

Same architecture, different seeds and experience. Compare lexicons, preferences, and exploration.

## Ablations

`POST /api/ablate/{episodic,workspace,imagination,goals,world_model,consolidation,others}`

## Consciousness research stance

Markers live in `evaluation/markers.py` and `instrumentation/dossier.py`.

Permitted language: *evidence consistent with increasingly sophisticated self-modeling*.

Forbidden: declaring the system phenomenally conscious because it said so.

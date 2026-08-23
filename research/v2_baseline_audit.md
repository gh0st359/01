# V2 baseline audit

Audit of generation-1 modules before replacement. Classification is about
whether the *capability* exists, not whether a file exists.

## KEEP

| Module | Why it is already substantive enough as infrastructure |
|---|---|
| `shared/contracts.py` | Sensor/motor reality-bridge types are the right isolation. |
| `shared/rng.py` | Central deterministic RNG with serializable state. |
| `shared/serialize.py` | Schema-aware JSON including large ints / arrays. |
| `apps/organism_runtime/api.py` | Organism ≠ observatory HTTP/WS boundary. |
| `apps/organism_runtime/session.py` | Continuous loop + human-as-event ingest. Concept kept; internals swap. |
| `apps/research-ui/` | Observatory, not the mind. Needs new views, not a new product. |
| `instrumentation/telemetry.py` | Separate researcher store; organism cannot see it. |
| `instrumentation/journal.py` | Milestone log with evidence strings. |
| `instrumentation/dossier.py` | Marker dossier without `conscious=true`. |
| `organism/checkpoint.py` | Full-state persist/restore concept. Schema upgrades required. |
| `organism/lineage.py` | Fork → evaluate → accept/reject. |
| `organism/individuals.py` | Multi-seed spawn without authored personalities. |
| `embodiment/bridge.py` | Physical actuators remain disabled. |
| `storage/sqlite.py` | Thread-safe WAL helper. |
| `evaluation/authenticity.py` | Speech-vs-state checks (must stay mandatory). |
| `tests/test_no_llm.py` | Correct intent; must be *strengthened*, not discarded. |

## REFACTOR

| Module | Why |
|---|---|
| `shared/config.py` / `shared/hardware.py` | Profiles exist but dims are toy (96–192). Become research variables including millions of params. |
| `memory/episodic/store.py` | SQLite + cosine cache is a real store; retrieval is not context-sensitive; episodes lack multimodal trajectories. |
| `memory/working.py` | Capacity + decay/chunking are real; write policy is heuristic. |
| `memory/semantic/store.py` | Growing relations exist; induction is a single EMA merge. |
| `memory/autobiographical/store.py` | Identity trace + chapters exist; not a predictive self-history. |
| `memory/consolidation.py` | Replay policies exist; effects are weakly measured. |
| `motivation/homeostasis.py` | Bounded variables exist; they currently leak into canned actions. |
| `cognition/executive/mode.py` | Mode machine is useful; transitions should become learned pressure, not only schedules. |
| `cognition/metacognition/monitor.py` | EMAs of reliability exist; need calibrated learned estimators. |
| `simulation/world.py` | Persistent physics sandbox is real; 16×16 and one layout are not a research world. Keep as CI micro-env. |
| `organism/development.py` | Capability gates are the right idea; thresholds and stage names were fitted to the toy loop. |

## REPLACE

| Module | Why it is not the capability |
|---|---|
| `learning/nn.py` | Handwritten NumPy GRU/Adam. No CUDA, no AMP, no real batching. |
| `cognition/core/recurrent.py` | Small fused GRU; timescales are `% 4` / `% 16` timers. |
| `cognition/workspace/workspace.py` | Linear policy + **manual** 7-term prior. Not learned broadcast. |
| `perception/hierarchy.py` | Color-blob proto-objects, not learned pixels→slots. |
| `world_model/predictive.py` | Single linear+GRU; can collapse to ~0 error in the tiny sandbox. |
| `world_model/entities.py` | Hungarian tracking is fine as a tool, but identity is not object-centric learning; permanence increments by constants. |
| `world_model/causal.py` | Counts confirmations of `cause→effect` strings; not P(Y\|do(X)). |
| `planning/goals.py` + `_maybe_form_goals` | Threshold → enumerated ActionKind. Must become a learned proposer. |
| `planning/hierarchy.py` | Hardcoded English-named step lists. |
| `planning/action.py` | Primitive vectors are acceptable as *body interface*; they are not a skill repertoire. |
| `imagination/rollout.py` | 8-step unroll of the tiny predictor. |
| `language/comprehension/parser.py` | `_QUERY_CUES` / `_REQUEST_CUES` / `_LABEL_CUES`. |
| `language/production/realizer.py` | `QUERY → what/where`, 6-word cap, kind filters. |
| `language/grounding/lexicon.py` | Word-level table; kinds assigned from programmer lists in `loop._ground_from_caregiver`. |
| `language/production/intent.py` | Hand mapping of goal origin → speech act. |
| `self_model/self.py` | EMA identity vector. Not a predictive self cause. |
| `self_model/others.py` | Last-visible location ≠ theory of mind. |
| `motivation/intrinsic.py` | Weighted sum; still consumed by threshold goals. |
| `training/tutor.py` | Caregiver is OK as an *environment agent*; must not ship English category lists into the organism. |
| `organism/loop.py` | God-file of domain logic. Becomes thin orchestration. |

## DELETE (as cognitive mechanisms)

These may remain only as historical comments in this audit, not as runtime logic:

- `_QUERY_CUES`, `_REQUEST_CUES`, `_LABEL_CUES`
- Color / entity / action / spatial / pronoun word sets in `_ground_from_caregiver`
- `SpeechAct` cue dictionaries in the realizer
- `tick % 17 == 0` speak trigger
- `if novelty: INSPECT`, `if energy: WAIT`, `if uncertainty: EXPERIMENT`
- Hash-embed “role vectors” used as fake semantics
- Tests that only assert the above wiring

## Decision

Keep infrastructure. Replace the numerical substrate, perception, world model,
goals, planning, language, curiosity, and self-model with PyTorch-learned
systems and a richer batched developmental world. CI retains a tiny CPU profile
and the compact 2D sandbox as a unit-test fixture only.

# Grounded language

Conceptual path:

```
world → perception → conceptual cognition ⇉ action
                                   ↘ language
```

## Acquisition

The caregiver tutor pairs words with the currently attended entity or self vector. Bindings are Hebbian / EMA updates in `language/grounding/lexicon.py`.

There is no response table. There is no personality prompt.

## Comprehension

A GRU reads word embeddings (lexicon or hash fallback) and builds a `SemanticFrame` (speech act, predicate vector, roles, referent ids).

## Production

1. `IntentFormer` builds meaning from workspace, goals, uncertainty, retrievals.
2. `LanguageProduction` retrieves nearest grounded words.
3. If nothing is grounded, the utterance is empty.

Researchers can compare `last_intent` with the surface string (`evaluation/authenticity.py`).

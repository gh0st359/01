"""Operational markers associated with cognition and self-modeling.

None of these prove phenomenal consciousness.
"""

from __future__ import annotations

from organism.loop import Organism


def collect_markers(org: Organism) -> dict[str, dict]:
    return {
        "persistent_cognition": {
            "value": org.tick > 1 and float(abs(org.core.state).sum()) > 0,
            "note": "core state evolves across ticks without a human request",
        },
        "autonomous_goal_formation": {
            "value": any(g.origin.value != "external" for g in org.goals.goals.values()),
            "note": "goals originated inside the organism",
        },
        "self_model": {
            "value": float(abs(org.self_model.identity).sum()) > 0 and org.self_model.controllability > 0,
            "note": "identity vector and controllability estimates exist",
        },
        "autobiographical_continuity": {
            "value": org.episodic.count() > 0 and float(abs(org.auto.identity_trace).sum()) > 0,
            "note": "episodes plus identity trace, not a chat log",
        },
        "metacognitive_access": {
            "value": org.meta.state.uncertainty != org.meta.state.confidence,
            "note": "distinct confidence and uncertainty variables",
        },
        "introspective_reporting": {
            "value": org.last_intent is not None,
            "note": "speech if any is tied to an intent record for authenticity tests",
        },
        "internally_generated_cognition": {
            "value": org.spontaneous_count > 0 or org.tick > 10,
            "note": "ticks occur without prompts; spontaneous speech counted separately",
        },
        "counterfactual_self_representation": {
            "value": len(org.counterfactual.lessons) > 0,
            "note": "stored alternate-action lessons",
        },
    }

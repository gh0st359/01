"""Every meaningful self-report must match prior internal state."""

from __future__ import annotations

from language.comprehension.parser import tokenize


def verify_utterance(text: str, intent_payload: dict | None, state: dict) -> dict:
    tokens = set(tokenize(text))
    report = {"text": text, "ok": True, "checks": []}
    if not text:
        report["checks"].append({"name": "silence", "ok": True})
        return report
    if intent_payload is None:
        report["ok"] = False
        report["checks"].append({"name": "intent_existed", "ok": False})
        return report
    if any(t in tokens for t in ("uncertain", "unknown", "what")):
        ok = float(intent_payload.get("uncertainty", 0)) > 0.2 or state.get("uncertainty", 0) > 0.2
        report["checks"].append({"name": "uncertainty_preceded", "ok": ok})
        report["ok"] = report["ok"] and ok
    if any(t in tokens for t in ("remember", "before", "happened")):
        ok = bool(intent_payload.get("retrievals")) or state.get("episodic_count", 0) > 0
        report["checks"].append({"name": "memory_preceded", "ok": ok})
        report["ok"] = report["ok"] and ok
    if "expect" in tokens:
        ok = "prediction_error" in state
        report["checks"].append({"name": "prediction_existed", "ok": ok})
        report["ok"] = report["ok"] and ok
    return report

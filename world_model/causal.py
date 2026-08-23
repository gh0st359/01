"""Intervention vs observation: competing causal hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.types import Hypothesis, Vector


@dataclass
class CausalEvent:
    tick: int
    intervened: bool
    cause: str
    effect: str
    pre: float
    post: float


class CausalLearner:
    def __init__(self) -> None:
        self.hypotheses: dict[str, Hypothesis] = {}
        self.events: list[CausalEvent] = []
        self.pending: CausalEvent | None = None

    def note_intervention(self, tick: int, cause: str, pre_effect: float) -> None:
        self.pending = CausalEvent(tick, True, cause, "unknown", pre_effect, pre_effect)

    def observe_effect(self, tick: int, effect: str, value: float, related_cause: str | None) -> Hypothesis | None:
        if self.pending and related_cause:
            ev = self.pending
            ev.effect = effect
            ev.post = value
            self.events.append(ev)
            hid = f"{ev.cause}->{effect}"
            hyp = self.hypotheses.get(hid)
            if hyp is None:
                hyp = Hypothesis(hid, ev.cause, effect, np.zeros(16), 0.4, 0, 0, 0, tick)
                self.hypotheses[hid] = hyp
            hyp.interventions += 1
            if abs(ev.post - ev.pre) > 0.15:
                hyp.confirmations += 1
            else:
                hyp.disconfirmations += 1
            hyp.confidence = (hyp.confirmations + 1) / (hyp.interventions + 2)
            hyp.last_test_tick = tick
            hyp.relation = 0.8 * hyp.relation
            hyp.relation[0] = ev.post - ev.pre
            self.pending = None
            return hyp
        if related_cause:
            hid = f"{related_cause}->{effect}"
            hyp = self.hypotheses.get(hid)
            if hyp:
                # observation without intervention does not raise confidence as much
                hyp.confidence = 0.97 * hyp.confidence + 0.03 * 0.4
        return None

    def best(self) -> Hypothesis | None:
        if not self.hypotheses:
            return None
        return max(self.hypotheses.values(), key=lambda h: h.confidence)

    def snapshot(self) -> dict:
        return {
            hid: {
                "cause": h.cause_entity,
                "effect": h.effect_entity,
                "confidence": h.confidence,
                "interventions": h.interventions,
                "confirmations": h.confirmations,
                "disconfirmations": h.disconfirmations,
                "relation": h.relation,
            }
            for hid, h in self.hypotheses.items()
        }

    def restore(self, data: dict) -> None:
        self.hypotheses = {}
        for hid, blob in data.items():
            self.hypotheses[hid] = Hypothesis(
                hid,
                blob["cause"],
                blob["effect"],
                np.asarray(blob["relation"], dtype=np.float64),
                float(blob["confidence"]),
                int(blob["interventions"]),
                int(blob["confirmations"]),
                int(blob["disconfirmations"]),
                0,
            )

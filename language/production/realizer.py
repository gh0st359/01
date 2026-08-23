"""Realize a semantic frame as an utterance via grounded word selection.

Semantic intent is selected first. Words are retrieved from the lexicon by
similarity to role/predicate vectors. Surface order is taken from learned
constructions, not from a pretrained token predictor.

There are no canned replies. If the lexicon has no binding, the slot is omitted.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from language.grounding.lexicon import Lexicon
from shared.types import CommunicativeIntent, SemanticFrame, SpeechAct


class ConstructionMemory:
    """Usage-based constructions: speech-act → observed word-kind sequences."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[tuple[str, ...], int]] = defaultdict(lambda: defaultdict(int))

    def observe(self, act: SpeechAct, kinds: list[str]) -> None:
        if kinds:
            self.counts[act.value][tuple(kinds)] += 1

    def preferred(self, act: SpeechAct) -> list[str] | None:
        table = self.counts.get(act.value)
        if not table:
            return None
        return list(max(table.items(), key=lambda kv: kv[1])[0])


class LanguageProduction:
    def __init__(self) -> None:
        self.constructions = ConstructionMemory()
        self.last_intent: CommunicativeIntent | None = None
        self.last_utterance: str = ""

    def align(self, frame: SemanticFrame, words: list[str], lexicon: Lexicon) -> None:
        kinds = []
        for w in words:
            e = lexicon.entries.get(w)
            kinds.append(e.kind if e else "unknown")
        self.constructions.observe(frame.act, kinds)

    def realize(self, intent: CommunicativeIntent, lexicon: Lexicon) -> str:
        self.last_intent = intent
        frame = intent.frame
        words: list[str] = []
        act_word = self._act_word(frame.act, lexicon)
        if act_word:
            words.append(act_word)
        pred_hits = lexicon.nearest(frame.predicate, k=2)
        for w, score in pred_hits:
            if w not in words and score > 0.22:
                words.append(w)
        for key, vec in frame.roles.items():
            hits = lexicon.nearest(vec, k=1)
            for w, score in hits:
                if w not in words and score > 0.25:
                    words.append(w)
        for w in frame.bound_words:
            if w not in words:
                words.append(w)
        # developmental: drop duplicates, keep short
        words = list(dict.fromkeys(words))[:6]
        if not words:
            # no lexicon support — remain silent rather than invent speech
            self.last_utterance = ""
            return ""
        utterance = " ".join(words)
        self.last_utterance = utterance
        self.align(frame, words, lexicon)
        return utterance

    def _act_word(self, act: SpeechAct, lexicon: Lexicon) -> str | None:
        # Only emit an act cue if that cue was actually grounded.
        cues = {
            SpeechAct.QUERY: ("what", "where", "who"),
            SpeechAct.REQUEST: ("want", "need"),
            SpeechAct.LABEL: ("this",),
            SpeechAct.ASSERT: (),
            SpeechAct.INFORM: (),
            SpeechAct.UNKNOWN: (),
        }
        for cue in cues[act]:
            if lexicon.known(cue):
                return cue
        return None

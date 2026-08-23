"""Surface realization from an independently formed semantic state.

No speech-act → cue-word tables. No six-word English templates.
If the V2 language organ is available, this module is unused at runtime.
"""

from __future__ import annotations

from language.grounding.lexicon import Lexicon
from shared.types import CommunicativeIntent


class LanguageProduction:
    def __init__(self) -> None:
        self.last_intent: CommunicativeIntent | None = None
        self.last_utterance: str = ""

    def realize(self, intent: CommunicativeIntent, lexicon: Lexicon) -> str:
        self.last_intent = intent
        # No programmer vocabulary. Only emit streams already bound by experience.
        words: list[str] = []
        for w in intent.frame.bound_words:
            if lexicon.known(w) and w not in words:
                words.append(w)
        pred = intent.frame.predicate
        for w, score in lexicon.nearest(pred, k=4, kinds=None):
            if w not in words and score > 0.22:
                words.append(w)
        utterance = " ".join(words)
        self.last_utterance = utterance
        return utterance

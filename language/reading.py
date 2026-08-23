"""Non-LLM reading: parse → interpret → connect → belief → memory."""

from __future__ import annotations

from language.comprehension.parser import LanguageComprehension, tokenize
from language.grounding.lexicon import Lexicon
from cognition.beliefs import BeliefSystem
from shared.types import EvidenceSource, SemanticFrame


class ReadingSystem:
    def __init__(self, parser: LanguageComprehension) -> None:
        self.parser = parser

    def read_document(
        self,
        text: str,
        lexicon: Lexicon,
        beliefs: BeliefSystem,
        tick: int,
    ) -> list[SemanticFrame]:
        frames = []
        for raw in _sentences(text):
            frame = self.parser.parse(raw, lexicon, [], tick)
            frames.append(frame)
            if frame.bound_words:
                prop = "read:" + "_".join(frame.bound_words[:6])
                beliefs.assert_belief(prop, frame.predicate, frame.confidence * 0.6, tick, EvidenceSource.LANGUAGE, raw)
        return frames


def _sentences(text: str) -> list[str]:
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            s = "".join(buf).strip()
            if s:
                parts.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts

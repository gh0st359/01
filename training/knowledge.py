"""Educational material as experiences conveyed through grounded language.

These texts are not conversation scripts. They become beliefs only if the
reading system can bind words already in the lexicon.
"""

from __future__ import annotations


LESSONS = [
    "unsupported things fall.",
    "doors often permit passage.",
    "a switch can change a light.",
    "people can look at objects.",
    "objects stay when hidden in a box.",
    "red and blue are colors.",
    "two is more than one.",
    "an agent can move by itself.",
]


class KnowledgeCurriculum:
    def __init__(self) -> None:
        self.i = 0

    def next_lesson(self, tick: int) -> str | None:
        if tick % 90 != 0:
            return None
        text = LESSONS[self.i % len(LESSONS)]
        self.i += 1
        return text

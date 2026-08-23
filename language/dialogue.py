"""Dialogue / social state: turn-taking and relationship continuity."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    tick: int
    speaker: str
    text: str
    organism_intent_act: str | None = None


class DialogueState:
    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.current_partner: str | None = None
        self.awaiting_response = False
        self.unresolved: list[str] = []
        self.partners: dict[str, int] = {}

    def hear(self, speaker: str, text: str, tick: int) -> None:
        self.turns.append(Turn(tick, speaker, text))
        self.current_partner = speaker
        self.awaiting_response = True
        self.partners[speaker] = self.partners.get(speaker, 0) + 1
        self.unresolved.append(text)

    def spoke(self, text: str, tick: int, act: str | None) -> None:
        self.turns.append(Turn(tick, "01", text, act))
        self.awaiting_response = False
        if self.unresolved:
            self.unresolved.pop(0)

    def snapshot(self) -> dict:
        return {
            "partner": self.current_partner,
            "awaiting": self.awaiting_response,
            "partners": self.partners,
            "unresolved": self.unresolved[-12:],
            "turns": [{"tick": t.tick, "speaker": t.speaker, "text": t.text, "act": t.organism_intent_act} for t in self.turns[-40:]],
        }

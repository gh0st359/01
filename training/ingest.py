"""Controlled knowledge ingestion. Documents become grounded propositions, not a text dump."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestedSource:
    source_id: str
    text: str
    confidence: float
    grounded: bool
    provenance: str


class KnowledgeLibrary:
    def __init__(self) -> None:
        self.sources: list[IngestedSource] = []

    def add(self, source_id: str, text: str, provenance: str = "library") -> IngestedSource:
        item = IngestedSource(source_id, text, 0.3, False, provenance)
        self.sources.append(item)
        return item

    def request(self, query_latent_norm: float) -> IngestedSource | None:
        if not self.sources:
            return None
        # Organism interest is a scalar here; infrastructure chooses the resource.
        idx = int(abs(hash(query_latent_norm)) % len(self.sources))
        return self.sources[idx]

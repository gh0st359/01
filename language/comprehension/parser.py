"""Character-stream encoder. No English word-class dictionaries.

Bootstrap may split on Unicode/byte boundaries. It does not know what
English words mean and does not assign speech acts from cue lists.
"""

from __future__ import annotations

import numpy as np

from language.grounding.lexicon import Lexicon, hash_embed
from learning.nn import GRUCell, Module, tanh
from shared.config import OrganismConfig
from shared.rng import RNG
from shared.types import Entity, SemanticFrame, SpeechAct, Vector


class LanguageComprehension(Module):
    def __init__(self, cfg: OrganismConfig, rng: RNG) -> None:
        super().__init__("comprehend")
        self.cfg = cfg
        self.cell = self.add(GRUCell("parse.gru", cfg.concept_dim, cfg.concept_dim, rng))

    def parse(self, text: str, lexicon: Lexicon, entities: list[Entity], tick: int) -> SemanticFrame:
        units = stream_units(text)
        h = np.zeros(self.cfg.concept_dim, dtype=np.float64)
        bound: list[str] = []
        role_vecs: dict[str, Vector] = {}
        refs: list[str] = []
        for unit in units:
            emb = lexicon.embed(unit)
            if emb is None:
                emb = hash_embed(unit, self.cfg.concept_dim)
            else:
                bound.append(unit)
            h, _ = self.cell.forward(_fit(emb, self.cfg.concept_dim), h)
            for ent in entities:
                if _cosine(_fit(ent.representation, emb.size), emb) > 0.55:
                    refs.append(ent.entity_id)
                    role_vecs.setdefault("referent", ent.representation.copy())
        pred = tanh(h)
        if units:
            role_vecs["predicate_surface"] = pred
        conf = 0.35 + 0.1 * len(bound)
        if refs:
            conf += 0.15
        return SemanticFrame(SpeechAct.UNKNOWN, pred, role_vecs, list(dict.fromkeys(refs)), float(min(0.95, conf)), tick, bound)


def stream_units(text: str) -> list[str]:
    """Unicode codepoint groups separated by whitespace. Whitespace is a boundary, not a lexicon."""
    return [u for u in text.split() if u]


def tokenize(text: str) -> list[str]:
    """Researcher-side splitter used by authenticity tooling. Not a POS tagger."""
    return stream_units(text.lower())


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out


def _cosine(a: Vector, b: Vector) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    num = float(np.dot(a[:n], b[:n]))
    den = float(np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-9)
    return num / den

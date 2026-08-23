"""Map linguistic input onto internal conceptual frames.

This is a recurrent grounded parser, not a next-token language model.
"""

from __future__ import annotations

import re

import numpy as np

from language.grounding.lexicon import Lexicon, hash_embed
from learning.nn import GRUCell, Module, cosine, tanh
from shared.config import OrganismConfig
from shared.rng import RNG
from shared.types import Entity, SemanticFrame, SpeechAct, Vector


_QUERY_CUES = {"what", "where", "who", "why", "how", "which"}
_REQUEST_CUES = {"want", "need", "please", "come", "look", "go", "open", "push", "grab"}
_LABEL_CUES = {"this", "that", "called", "name", "is"}


class LanguageComprehension(Module):
    def __init__(self, cfg: OrganismConfig, rng: RNG) -> None:
        super().__init__("comprehend")
        self.cfg = cfg
        self.cell = self.add(GRUCell("parse.gru", cfg.concept_dim, cfg.concept_dim, rng))

    def parse(self, text: str, lexicon: Lexicon, entities: list[Entity], tick: int) -> SemanticFrame:
        tokens = tokenize(text)
        h = np.zeros(self.cfg.concept_dim, dtype=np.float64)
        bound: list[str] = []
        role_vecs: dict[str, Vector] = {}
        refs: list[str] = []
        for tok in tokens:
            emb = lexicon.embed(tok)
            if emb is None:
                emb = hash_embed(tok, self.cfg.concept_dim)
            else:
                bound.append(tok)
            h, _ = self.cell.forward(_fit(emb, self.cfg.concept_dim), h)
            for ent in entities:
                if cosine(_fit(ent.representation, emb.size), emb) > 0.55:
                    refs.append(ent.entity_id)
                    role_vecs.setdefault("referent", ent.representation.copy())
        act = SpeechAct.ASSERT
        if any(t in _QUERY_CUES for t in tokens) or text.strip().endswith("?"):
            act = SpeechAct.QUERY
        elif any(t in _REQUEST_CUES for t in tokens):
            act = SpeechAct.REQUEST
        elif any(t in _LABEL_CUES for t in tokens) and len(tokens) <= 6:
            act = SpeechAct.LABEL
        pred = tanh(h)
        if tokens:
            role_vecs["predicate_surface"] = pred
        conf = 0.35 + 0.1 * len(bound)
        if refs:
            conf += 0.15
        return SemanticFrame(act, pred, role_vecs, list(dict.fromkeys(refs)), float(min(0.95, conf)), tick, bound)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9']+", text.lower()) if t]


def _fit(x: Vector, n: int) -> Vector:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, x.size)] = x[: min(n, x.size)]
    return out

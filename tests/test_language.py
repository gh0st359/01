from pathlib import Path

from apps.organism_runtime.session import OrganismSession
from evaluation.authenticity import verify_utterance
from language.comprehension.parser import LanguageComprehension, tokenize
from language.grounding.lexicon import Lexicon
from language.production.intent import IntentFormer
from language.production.realizer import LanguageProduction
from shared.config import load_config
from shared.rng import RNG
from shared.types import CommunicativeIntent, SemanticFrame, SpeechAct
import numpy as np


def test_tokenizer():
    assert tokenize("What is that?") == ["what", "is", "that"]


def test_production_uses_lexicon_only():
    lex = Lexicon(16, 64)
    lex.bind("red", np.ones(16), 1, "property")
    lex.bind("ball", np.ones(16) * 0.5, 1, "entity")
    prod = LanguageProduction()
    frame = SemanticFrame(SpeechAct.ASSERT, np.ones(16), {}, [], 0.7, 1)
    intent = CommunicativeIntent(frame, 0.5, None, 1, "h", 0.4, [], 0.2)
    utt = prod.realize(intent, lex)
    for w in utt.split():
        assert lex.known(w)


def test_unknown_lexicon_stays_silent():
    lex = Lexicon(16, 64)
    prod = LanguageProduction()
    frame = SemanticFrame(SpeechAct.INFORM, np.ones(16), {}, [], 0.7, 1)
    intent = CommunicativeIntent(frame, 0.5, None, 1, "h", 0.4, [], 0.2)
    assert prod.realize(intent, lex) == ""


def test_grounding_from_interaction(tmp_path: Path):
    cfg = load_config("development", seed=9)
    session = OrganismSession(cfg, tmp_path)
    session.human_say("tutor", "this red ball")
    for _ in range(6):
        session.step()
    assert session.organism.lexicon.known("red") or session.organism.lexicon.known("ball") or session.organism.lexicon.known("this")
    session.close()


def test_authenticity_requires_intent():
    r = verify_utterance("uncertain", None, {})
    assert r["ok"] is False

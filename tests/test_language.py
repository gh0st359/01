from pathlib import Path

from apps.organism_runtime.session import OrganismSession
from evaluation.authenticity import verify_utterance
from language.comprehension.parser import LanguageComprehension, stream_units
from shared.config import load_config
from shared.rng import RNG
from shared.v2config import load_v2_config


def test_stream_units_are_boundaries_not_pos_tags():
    assert stream_units("abc def") == ["abc", "def"]


def test_parser_does_not_assign_speech_acts_from_cues():
    cfg = load_config("development", seed=1)
    parser = LanguageComprehension(cfg, RNG(1))
    from language.grounding.lexicon import Lexicon

    lex = Lexicon(cfg.concept_dim, 64)
    frame = parser.parse("what is that", lex, [], 1)
    assert frame.act.value == "unknown"


def test_v2_grounding_is_stream_not_word_class(tmp_path: Path):
    cfg = load_v2_config("ci", seed=9)
    session = OrganismSession(cfg, tmp_path)
    session.human_say("tutor", "disk")
    for _ in range(3):
        session.step()
    assert session.organism.human_events >= 1
    assert any(m["name"] == "first_grounded_stream" for m in session.organism.milestones)
    session.close()


def test_authenticity_requires_intent():
    r = verify_utterance("uncertain", None, {})
    assert r["ok"] is False


def test_no_cue_dictionaries_remain_in_language_runtime():
    root = Path(__file__).resolve().parents[1]
    banned = ("_QUERY_CUES", "_REQUEST_CUES", "_LABEL_CUES", "QUERY_WORDS", "REQUEST_WORDS", "COLOR_WORDS", "OBJECT_WORDS")
    hits = []
    for path in (root / "language").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                hits.append((str(path), token))
    assert hits == [], hits

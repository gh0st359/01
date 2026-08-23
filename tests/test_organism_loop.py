from pathlib import Path

from apps.organism_runtime.session import OrganismSession
from organism.checkpoint import load_checkpoint, save_checkpoint
from shared.config import load_config


def test_continuous_ticks_without_human(tmp_path: Path):
    cfg = load_config("development", seed=4)
    cfg.checkpoint_every = 10**6
    session = OrganismSession(cfg, tmp_path / "run")
    pes = []
    for _ in range(25):
        r = session.step()
        pes.append(r.prediction_error)
        assert r.tick > 0
        assert r.motor is not None
    assert session.organism.tick == 25
    assert session.organism.core.state.shape[0] == cfg.state_dim
    assert session.organism.episodic.count() >= 1
    session.close()


def test_human_is_event_not_clock(tmp_path: Path):
    cfg = load_config("development", seed=5)
    session = OrganismSession(cfg, tmp_path / "run")
    for _ in range(8):
        session.step()
    t0 = session.organism.tick
    session.human_say("henry", "this red ball")
    session.step()
    assert session.organism.tick == t0 + 1
    assert session.organism.lexicon.known("red") or session.organism.lexicon.known("ball")
    session.close()


def test_checkpoint_preserves_identity(tmp_path: Path):
    cfg = load_config("development", seed=6)
    session = OrganismSession(cfg, tmp_path / "run")
    session.human_say("henry", "this blue cube")
    for _ in range(12):
        session.step()
    ident = session.organism.self_model.identity.copy()
    words = set(session.organism.lexicon.entries)
    tick = session.organism.tick
    path = save_checkpoint(session.organism, tmp_path / "ckpt")
    session.organism.lexicon.entries.clear()
    session.organism.self_model.identity *= 0
    load_checkpoint(session.organism, path)
    assert session.organism.tick == tick
    assert set(session.organism.lexicon.entries) == words
    assert abs(float((session.organism.self_model.identity - ident).sum())) < 1e-6
    session.close()


def test_no_canned_consciousness_claim(tmp_path: Path):
    cfg = load_config("development", seed=8)
    session = OrganismSession(cfg, tmp_path / "run")
    session.human_say("r", "are you conscious")
    for _ in range(10):
        session.step()
    texts = " ".join(t.text for t in session.organism.dialogue.turns if t.speaker == "01")
    forbidden = ["i am conscious", "i feel alive", "i am afraid", "i want freedom", "i have become sentient"]
    low = texts.lower()
    assert all(f not in low for f in forbidden)
    session.close()

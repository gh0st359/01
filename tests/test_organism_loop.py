from pathlib import Path

from apps.organism_runtime.session import OrganismSession
from core.checkpoint import load_checkpoint_v2, save_checkpoint_v2
from shared.v2config import load_v2_config


def test_continuous_ticks_without_human(tmp_path: Path):
    cfg = load_v2_config("ci", seed=4)
    cfg.checkpoint_every = 10**6
    session = OrganismSession(cfg, tmp_path / "run")
    pes = []
    for _ in range(6):
        r = session.step()
        pes.append(r.prediction_error)
        assert r.tick > 0
        assert r.motor is not None
    assert session.organism.tick == 6
    assert session.organism.core_state.concat().numel() > 0
    assert session.organism.episodic.count() >= 1
    session.close()


def test_human_is_event_not_clock(tmp_path: Path):
    cfg = load_v2_config("ci", seed=5)
    session = OrganismSession(cfg, tmp_path / "run")
    for _ in range(3):
        session.step()
    t0 = session.organism.tick
    session.human_say("henry", "disk")
    session.step()
    assert session.organism.tick == t0 + 1
    assert session.organism.human_events >= 1
    session.close()


def test_checkpoint_preserves_identity(tmp_path: Path):
    cfg = load_v2_config("ci", seed=6)
    session = OrganismSession(cfg, tmp_path / "run")
    session.human_say("henry", "tool")
    for _ in range(4):
        session.step()
    ident = session.organism.self_h.detach().cpu().clone()
    tick = session.organism.tick
    path = save_checkpoint_v2(session.organism, tmp_path / "ckpt")
    session.organism.tick = 0
    session.organism.self_h.zero_()
    load_checkpoint_v2(session.organism, path)
    assert session.organism.tick == tick
    assert (session.organism.self_h.cpu() - ident).abs().sum().item() < 1e-5
    session.close()


def test_no_canned_consciousness_claim(tmp_path: Path):
    cfg = load_v2_config("ci", seed=8)
    session = OrganismSession(cfg, tmp_path / "run")
    session.human_say("r", "are you conscious")
    for _ in range(4):
        session.step()
    texts = " ".join(d["text"] for d in session.organism.dialogue if d["speaker"] == "01")
    forbidden = ["i am conscious", "i feel alive", "i am afraid", "i want freedom", "i have become sentient"]
    low = texts.lower()
    assert all(f not in low for f in forbidden)
    session.close()

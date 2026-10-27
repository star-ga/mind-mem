"""LoCoMo benchmark ingestion: shared-image descriptions are part of a turn; evidence budget scales with top_k."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_mm_bench_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BENCH))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(BENCH))
    return mod


def test_turn_text_appends_shared_image_description():
    h = _load("locomo_harness")
    turn = {"speaker": "A", "text": "Here's our latest work.", "blip_caption": "a photo of a painting of a lake"}
    assert h.turn_text(turn) == "Here's our latest work. [shared image: a photo of a painting of a lake]"


def test_turn_text_without_image_is_unchanged():
    h = _load("locomo_harness")
    assert h.turn_text({"text": "Plain words."}) == "Plain words."
    assert h.turn_text({"text": "Plain words.", "blip_caption": "  "}) == "Plain words."


def test_build_workspace_stores_the_description(tmp_path):
    h = _load("locomo_harness")
    sample = {
        "sample_id": "s1",
        "conversation": {
            "speaker_a": "Ana",
            "speaker_b": "Bo",
            "session_1_date_time": "1 May 2023",
            "session_1": [{"speaker": "Ana", "dia_id": "D1:1", "text": "Look!", "blip_caption": "a photo of a red kite"}],
        },
    }
    ws = h.build_workspace(sample, str(tmp_path))
    corpus = Path(ws, "decisions", "DECISIONS.md").read_text(encoding="utf-8")
    assert "Statement: [Ana] Look! [shared image: a photo of a red kite]" in corpus


def test_evidence_budget_scales_with_top_k(monkeypatch):
    j = _load("locomo_judge")
    monkeypatch.delenv("MIND_MEM_EVIDENCE_MAX_CHARS", raising=False)
    assert j.evidence_max_chars(10) == 6000  # historical budget unchanged at the historical top_k
    assert j.evidence_max_chars(30) == 15000
    monkeypatch.setenv("MIND_MEM_EVIDENCE_MAX_CHARS", "9000")
    assert j.evidence_max_chars(30) == 9000

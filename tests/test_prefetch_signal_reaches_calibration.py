"""Prefetch quality must be VISIBLE beside calibration — and must not move a score.

ROADMAP (retracted-ticks lock, `Prefetch hit rate tracked in calibration feedback loop`):
retired because "grep -c prefetch src/mind_mem/calibration.py == 0; prefetch.py:619
computes a hit_rate but no prefetch signal reaches the calibration loop". The premise was
true: `PrefetchCache.stats()` produced hits/misses/hit_rate that nothing outside a
diagnostic ever read.

**The obvious wiring would have been wrong, and this is the whole design decision.**
Feeding prefetch hits into `record_feedback` as accepted/rejected votes would put them
into the weights that MOVE RETRIEVAL SCORES. A prefetch hit means "this bundle was warm",
not "this block was useful to a human" — the two are different claims, and conflating
them would corrupt the single calibration authority with a signal about cache warmth. The
codebase already states this shape for `llm_noise_profile`: "Sidecar only — nothing on
the scored path reads it."

So prefetch arrives as a REPORTED sidecar on `calibration_stats`, observable next to
retrieval quality, and structurally barred from the scoring path. Both halves are tested,
because the first half without the second is the corruption described above.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "mind_mem"


def test_the_prefetch_sidecar_is_reported(monkeypatch, tmp_path):
    """The item's literal ask: the signal reaches the calibration surface."""
    import mind_mem.mcp.tools.calibration as cal_tool

    monkeypatch.setattr(cal_tool, "_workspace", lambda: str(tmp_path))
    payload = json.loads(cal_tool.calibration_stats())
    assert "prefetch" in payload, sorted(payload)
    assert "hit_rate" in payload["prefetch"], payload["prefetch"]


def test_the_sidecar_carries_the_REAL_counters(monkeypatch, tmp_path):
    """POSITIVE CONTROL. A hardcoded zero would satisfy the test above forever, and a
    metric that cannot move is not a metric."""
    import mind_mem.mcp.tools.calibration as cal_tool

    monkeypatch.setattr(cal_tool, "_workspace", lambda: str(tmp_path))
    monkeypatch.setattr(
        cal_tool,
        "_prefetch_sidecar",
        lambda: {"hits": 7, "misses": 3, "hit_rate": 0.7, "bundles": 2},
    )
    payload = json.loads(cal_tool.calibration_stats())
    assert payload["prefetch"]["hit_rate"] == 0.7, payload["prefetch"]
    assert payload["prefetch"]["hits"] == 7


def test_it_is_LABELLED_a_sidecar_so_nobody_reads_it_as_a_score_input(monkeypatch, tmp_path):
    """An operator reading this report must be able to tell that prefetch does NOT
    influence ranking. Unlabelled, it sits beside per-block calibration scores and reads
    like one of them."""
    import mind_mem.mcp.tools.calibration as cal_tool

    monkeypatch.setattr(cal_tool, "_workspace", lambda: str(tmp_path))
    payload = json.loads(cal_tool.calibration_stats())
    note = str(payload["prefetch"].get("note", "")).lower()
    assert "sidecar" in note or "not" in note, payload["prefetch"]


def test_a_failing_prefetch_read_does_not_break_calibration_stats(monkeypatch, tmp_path):
    """The sidecar is additive. A diagnostic that takes the whole report down with it is
    worse than an absent diagnostic — and the report is what an operator reaches for when
    something is already wrong."""
    import mind_mem.mcp.tools.calibration as cal_tool

    def _boom():
        raise RuntimeError("cache unavailable")

    monkeypatch.setattr(cal_tool, "_workspace", lambda: str(tmp_path))
    monkeypatch.setattr(cal_tool, "_prefetch_sidecar", _boom)
    payload = json.loads(cal_tool.calibration_stats())
    assert "error" not in payload, payload
    assert payload["prefetch"]["unavailable"], payload["prefetch"]


def test_NOTHING_ON_THE_SCORING_PATH_READS_THE_PREFETCH_COUNTERS():
    """THE STRUCTURAL HALF, and the reason the obvious wiring was refused.

    A prefetch hit means "this bundle was warm", not "this block was useful". If it
    reached the calibration WEIGHTS it would move retrieval scores on a signal about
    cache warmth — corrupting the one authority that decides ranking.

    Walked over the import graph, not grepped, so prose mentioning prefetch cannot
    satisfy it.
    """
    scoring = [
        "_recall_core.py", "hybrid_recall.py", "recall_vector.py", "recall.py",
        "calibration.py",
    ]
    for name in scoring:
        path = SRC / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mod = ""
            if isinstance(node, ast.Import):
                mod = node.names[0].name
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            # EXACT module match, not a substring. The first version asserted
            # `"prefetch" not in mod` and failed on hybrid_recall.py's
            # `from .entity_prefetch import ...` — a DIFFERENT module (entity
            # prefetching for recall, not the response cache). A substring match on a
            # module name reports a defect that is not there, and would have had me
            # "fixing" a legitimate import.
            assert mod.split(".")[-1] != "prefetch", (
                f"{name} imports the prefetch response cache — a cache-warmth signal "
                f"must never reach the path that decides ranking"
            )


def test_the_scoring_rail_check_is_not_vacuous():
    """POSITIVE CONTROL for the walk above: the files it claims to inspect must exist,
    or it passes by inspecting nothing."""
    present = [n for n in ("_recall_core.py", "calibration.py") if (SRC / n).exists()]
    assert len(present) == 2, present


def test_record_feedback_still_refuses_a_non_vocabulary_verdict():
    """Guard against the wiring that was refused: if prefetch were ever fed in as a
    verdict it would need a new feedback_type, and the vocabulary is closed."""
    from mind_mem.calibration import CalibrationManager

    with pytest.raises(ValueError, match="feedback_type"):
        CalibrationManager.record_feedback(
            object.__new__(CalibrationManager),  # no DB touched: the check precedes it
            query_id="q",
            block_ids_useful=[],
            block_ids_not_useful=[],
            feedback_type="prefetch_hit",
        )

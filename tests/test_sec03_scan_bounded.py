# Copyright 2026 STARGA, Inc.
"""SEC03 — the contradiction scan must be bounded, and must say when it truncates.

MEASURED at 97fd765 before this change:

    scan in ADMIN_TOOLS: False        scan in USER_TOOLS: True
    rest.py:1165 "/v1/scan" ... :1170 Depends(_require_auth)   # not _require_admin
    grep -c timeout src/mind_mem/mcp/tools/governance.py  ->  0

``_detect_statement_contradictions`` runs a full pairwise loop --
``for i in range(len(entries)): for j in range(i + 1, len(entries))`` -- with no
cap and no time budget, over every active block. On a Postgres store the whole
table is enumerated first. It is pure Python and holds the GIL, so it starves
the process rather than just its own request.

A semi-trusted MCP agent, or any authenticated REST bearer, gets that for the
cost of a zero-argument call. The caller cannot grow the corpus (every write is
admin), so this is AMPLIFICATION against an existing one: negligible at a few
thousand blocks, minutes at ~20k, effectively permanent above ~100k.

THE BOUND MUST DISCLOSE ITSELF. A cap that silently drops pairs turns a DoS into
a correctness bug: the scan would report "no contradictions" over a corpus it
never finished reading, and an operator cannot tell that from a clean result.
"""
from __future__ import annotations

import json
from pathlib import Path

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.governance import _detect_statement_contradictions, scan


def _blocks(n: int) -> list[dict]:
    """n blocks that all share a subject, so every pair passes the overlap gate."""
    out = []
    for i in range(n):
        polarity = "enable" if i % 2 == 0 else "disable"
        out.append({"_id": f"D-{i:05d}", "Statement": f"the vendor contract should {polarity} renewal terms"})
    return out


def test_the_pair_budget_exists_and_is_finite():
    from mind_mem.mcp.tools.governance import MAX_SCAN_PAIRS

    assert isinstance(MAX_SCAN_PAIRS, int) and MAX_SCAN_PAIRS > 0


def test_a_small_corpus_is_scanned_completely_and_says_so():
    """POSITIVE CONTROL: under the budget nothing is dropped, and contradictions
    are still found -- so a later 'truncated' result cannot be confused with
    'this scan never worked'."""
    found = _detect_statement_contradictions(_blocks(12))
    assert found, "positive control: the detector must find contradictions at all"
    assert not any(f.get("truncated") for f in found)


def test_a_corpus_over_the_budget_is_bounded_not_unbounded(monkeypatch):
    from mind_mem.mcp.tools import governance

    # Patch the budget rather than brute-force past the real one: the property
    # under test is "it stops and says so", not the specific ceiling, and a test
    # that burns 200k comparisons to prove it is a test nobody will keep running.
    monkeypatch.setattr(governance, "MAX_SCAN_PAIRS", 50)
    n = 120  # 7,140 unordered pairs, far past the patched budget
    assert n * (n - 1) // 2 > governance.MAX_SCAN_PAIRS, "positive control: input must exceed the budget"
    found = _detect_statement_contradictions(_blocks(n))
    # It returns rather than running to completion, and it is honest about it.
    assert any(f.get("truncated") for f in found), (
        "an over-budget scan reported no truncation marker; a silent cap makes an "
        "incomplete scan indistinguishable from a clean one"
    )


def test_the_truncation_marker_names_the_budget_and_what_was_skipped(monkeypatch):
    from mind_mem.mcp.tools import governance

    monkeypatch.setattr(governance, "MAX_SCAN_PAIRS", 50)
    found = _detect_statement_contradictions(_blocks(120))
    marks = [f for f in found if f.get("truncated")]
    assert marks, "positive control"
    m = marks[0]
    assert m.get("pairs_examined") == governance.MAX_SCAN_PAIRS
    assert isinstance(m.get("blocks_total"), int) and m["blocks_total"] == 120
    assert "reason" in m and str(m["reason"]).strip()


def test_the_bound_is_on_pairs_not_on_blocks():
    """Capping BLOCKS would silently ignore a whole region of the corpus.

    Capping PAIRS keeps the full input available for the coverage marker, while
    a truncated run explicitly admits that later blocks may not be compared.
    """
    import inspect

    src = inspect.getsource(_detect_statement_contradictions)
    assert "MAX_SCAN_PAIRS" in src
    assert "entries[:" not in src, "the corpus itself must not be sliced"


def _store_workspace(path: Path) -> None:
    for subdir in ("decisions", "tasks", "entities", "intelligence"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    (path / "mind-mem.json").write_text("{}", encoding="utf-8")


def test_public_scan_surfaces_partial_coverage_without_counting_marker(monkeypatch, tmp_path: Path):
    """The public store path exposes partial coverage and real findings only."""
    from mind_mem.mcp.tools import governance

    ws = tmp_path / "partial"
    _store_workspace(ws)
    monkeypatch.setattr(governance, "_resolve_backend", lambda _ws: "postgres")
    monkeypatch.setattr(governance, "iter_active_blocks", lambda _ws: _blocks(120))
    monkeypatch.setattr(governance, "MAX_SCAN_PAIRS", 50)

    with use_workspace(str(ws)):
        payload = json.loads(scan())

    summary = payload["checks"]["contradictions"]
    assert summary["raw"] == len(
        [row for row in _detect_statement_contradictions(_blocks(120)) if not row.get("truncated")]
    )
    assert summary["truncated"] is True
    assert summary["complete"] is False
    assert summary["coverage"] == {
        "status": "partial",
        "blocks_total": 120,
        "pairs_examined": 50,
        "pairs_total": 7140,
    }
    assert "absence of a contradiction" in summary["reason"]


def test_public_complete_store_scan_keeps_legacy_summary_shape(monkeypatch, tmp_path: Path):
    """A complete store scan retains the pre-SEC03 result contract."""
    from mind_mem.mcp.tools import governance

    ws = tmp_path / "complete"
    _store_workspace(ws)
    monkeypatch.setattr(governance, "_resolve_backend", lambda _ws: "postgres")
    monkeypatch.setattr(governance, "iter_active_blocks", lambda _ws: _blocks(2))
    monkeypatch.setattr(governance, "MAX_SCAN_PAIRS", 50)

    with use_workspace(str(ws)):
        payload = json.loads(scan())

    assert payload["checks"]["contradictions"] == {"raw": 1, "resolvable": 0}


def test_public_detector_failure_is_incomplete_instead_of_clean(monkeypatch, tmp_path: Path):
    """A detector error cannot be reported as zero contradictions with full coverage."""
    from mind_mem.mcp.tools import governance

    ws = tmp_path / "failed"
    _store_workspace(ws)
    monkeypatch.setattr(governance, "_resolve_backend", lambda _ws: "postgres")
    monkeypatch.setattr(governance, "iter_active_blocks", lambda _ws: _blocks(2))

    def fail(_blocks):
        raise RuntimeError("synthetic detector failure")

    monkeypatch.setattr(governance, "_detect_statement_contradictions", fail)
    with use_workspace(str(ws)):
        payload = json.loads(scan())

    summary = payload["checks"]["contradictions"]
    assert summary["raw"] == 0
    assert summary["complete"] is False
    assert summary["coverage"] == {"status": "unavailable"}

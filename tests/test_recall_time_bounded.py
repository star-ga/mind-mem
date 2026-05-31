"""Regression tests for time-bounded recall (roadmap v4.0.0 Group E).

``recall(workspace, query, *, since=ISO, until=ISO)`` post-filters hits
by block ``Date`` field. Comparison is string-based on ISO-8601 so
``YYYY-MM-DD`` and full timestamps both work. Filtering applies AFTER
ranking + reranking — date is a post-filter, not a retrieval-time
prefilter, so ranking quality is preserved.

Also covers ``mm recall --since/--until`` CLI flags.
"""

from __future__ import annotations

import json
import os

import pytest
from mind_mem import mm_cli
from mind_mem._recall_core import (
    _apply_date_filter,
    _block_date,
    _in_date_range,
    recall,
)
from mind_mem.sqlite_index import build_index

# ---------------------------------------------------------------------------
# Helper invariants
# ---------------------------------------------------------------------------


def test_block_date_reads_uppercase_key() -> None:
    assert _block_date({"Date": "2026-01-15"}) == "2026-01-15"


def test_block_date_falls_back_to_lowercase_key() -> None:
    assert _block_date({"date": "2026-01-15"}) == "2026-01-15"


def test_block_date_strips_whitespace() -> None:
    assert _block_date({"Date": "  2026-01-15  "}) == "2026-01-15"


def test_block_date_returns_none_for_missing() -> None:
    assert _block_date({"id": "BLOCK-1"}) is None


def test_in_date_range_open_both_sides() -> None:
    """Both bounds None = unconstrained = accept everything (incl. None date)."""
    assert _in_date_range(None, None, None) is True
    assert _in_date_range("2026-06-01", None, None) is True


def test_in_date_range_rejects_none_when_bound_set() -> None:
    """When a bound is set, a block without a date can't satisfy it."""
    assert _in_date_range(None, "2026-01-01", None) is False
    assert _in_date_range(None, None, "2026-12-31") is False


def test_in_date_range_inclusive_bounds() -> None:
    """Bounds are inclusive."""
    assert _in_date_range("2026-01-01", "2026-01-01", "2026-12-31") is True
    assert _in_date_range("2026-12-31", "2026-01-01", "2026-12-31") is True


def test_in_date_range_excludes_outside() -> None:
    assert _in_date_range("2025-12-31", "2026-01-01", None) is False
    assert _in_date_range("2027-01-01", None, "2026-12-31") is False


def test_apply_date_filter_keeps_in_range() -> None:
    hits = [
        {"_id": "A", "Date": "2026-03-01"},
        {"_id": "B", "Date": "2026-06-15"},
        {"_id": "C", "Date": "2026-12-31"},
        {"_id": "D", "Date": "2027-01-01"},
        {"_id": "E"},  # no date
    ]
    filtered = _apply_date_filter(hits, since="2026-06-01", until="2026-12-31")
    ids = [h["_id"] for h in filtered]
    assert ids == ["B", "C"]  # in-range only; no-date dropped


def test_apply_date_filter_no_bounds_is_passthrough() -> None:
    hits = [{"_id": "A", "Date": "2026-01-01"}, {"_id": "B"}]
    assert _apply_date_filter(hits, None, None) == hits


# ---------------------------------------------------------------------------
# recall() honours since/until end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture()
def time_bounded_workspace(tmp_path) -> str:
    """Workspace with 5 blocks spanning Jan-Dec 2026, all matching a query."""
    ws = tmp_path / "tb"
    (ws / "decisions").mkdir(parents=True)
    blocks = [
        ("FACT-Q1", "2026-02-15", "I love quarterly planning meetings"),
        ("FACT-Q2", "2026-05-15", "I love quarterly planning meetings"),
        ("FACT-Q3", "2026-08-15", "I love quarterly planning meetings"),
        ("FACT-Q4", "2026-11-15", "I love quarterly planning meetings"),
        ("FACT-OLD", "2025-12-20", "I love quarterly planning meetings"),
    ]
    body = ""
    for bid, date, statement in blocks:
        body += f"[{bid}]\nStatement: {statement}\nDate: {date}\nStatus: active\nTags: FACT\nSources: TEST\n\n---\n\n"
    (ws / "decisions" / "DECISIONS.md").write_text(body)
    # Configure SQLite backend with no LLM/CE — pure BM25 path.
    (ws / "mind-mem.json").write_text(
        json.dumps(
            {
                "recall": {
                    "backend": "sqlite",
                    "knee_cutoff": False,
                    "cross_encoder": {"enabled": False, "auto_enable": False},
                    "dedup": {"enabled": False, "type_cap_enabled": False, "source_cap_enabled": False},
                }
            }
        )
    )
    os.environ.setdefault("MIND_MEM_DISABLE_TELEMETRY", "1")
    build_index(str(ws), incremental=False)
    return str(ws)


def test_recall_no_filter_returns_all_matching(time_bounded_workspace: str) -> None:
    """Sanity: without since/until, all 5 quarterly blocks come back."""
    hits = recall(time_bounded_workspace, "love", limit=20, active_only=False)
    ids = {h["_id"] for h in hits if h.get("_id", "").startswith("FACT-")}
    assert ids == {"FACT-Q1", "FACT-Q2", "FACT-Q3", "FACT-Q4", "FACT-OLD"}


def test_recall_with_since_filters_lower_bound(time_bounded_workspace: str) -> None:
    """since=2026-06-01 keeps only Q3 + Q4."""
    hits = recall(
        time_bounded_workspace,
        "love",
        limit=20,
        active_only=False,
        since="2026-06-01",
    )
    ids = {h["_id"] for h in hits if h.get("_id", "").startswith("FACT-")}
    assert ids == {"FACT-Q3", "FACT-Q4"}


def test_recall_with_until_filters_upper_bound(time_bounded_workspace: str) -> None:
    """until=2026-06-30 keeps Q1, Q2, and the OLD 2025 block."""
    hits = recall(
        time_bounded_workspace,
        "love",
        limit=20,
        active_only=False,
        until="2026-06-30",
    )
    ids = {h["_id"] for h in hits if h.get("_id", "").startswith("FACT-")}
    assert ids == {"FACT-Q1", "FACT-Q2", "FACT-OLD"}


def test_recall_with_both_bounds(time_bounded_workspace: str) -> None:
    """Bounded both sides → middle two quarters only."""
    hits = recall(
        time_bounded_workspace,
        "love",
        limit=20,
        active_only=False,
        since="2026-03-01",
        until="2026-09-30",
    )
    ids = {h["_id"] for h in hits if h.get("_id", "").startswith("FACT-")}
    assert ids == {"FACT-Q2", "FACT-Q3"}


def test_recall_with_impossible_range_returns_empty(time_bounded_workspace: str) -> None:
    """No date in the workspace falls in 2030 → empty result."""
    hits = recall(
        time_bounded_workspace,
        "love",
        limit=20,
        active_only=False,
        since="2030-01-01",
        until="2030-12-31",
    )
    # Some non-FACT blocks may still surface from the index — only assert
    # that none of our FACT-* dated blocks survived.
    fact_ids = [h["_id"] for h in hits if h.get("_id", "").startswith("FACT-")]
    assert fact_ids == []


def test_recall_filter_respects_limit(time_bounded_workspace: str) -> None:
    """When filtering shrinks the set below `limit`, return shorter list."""
    hits = recall(
        time_bounded_workspace,
        "love",
        limit=20,
        active_only=False,
        since="2026-09-01",
    )
    fact = [h for h in hits if h.get("_id", "").startswith("FACT-")]
    assert len(fact) == 1  # only Q4
    assert fact[0]["_id"] == "FACT-Q4"


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


def test_cli_recognises_since_until() -> None:
    parser = mm_cli.build_parser()
    args = parser.parse_args(["recall", "test query", "--since", "2026-01-01", "--until", "2026-06-30"])
    assert args.since == "2026-01-01"
    assert args.until == "2026-06-30"


def test_cli_since_until_default_to_none() -> None:
    parser = mm_cli.build_parser()
    args = parser.parse_args(["recall", "test query"])
    assert args.since is None
    assert args.until is None

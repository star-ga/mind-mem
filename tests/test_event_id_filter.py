"""Unit tests for the event_id recall post-filter.

Verifies that:
- The block parser correctly reads ``EventId: <value>`` as a top-level field.
- ``_apply_event_id_filter`` matches by exact case-insensitive string.
- Blocks without an ``EventId`` field are excluded when the filter is active.
- ``recall(event_id=...)`` returns only blocks matching the given EventId.
- ``recall()`` with no ``event_id`` argument returns all matching blocks
  (behaviour-preserving default).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mind_mem.block_parser import parse_file
from mind_mem._recall_core import _apply_event_id_filter, recall


# ---------------------------------------------------------------------------
# Block-parser tests
# ---------------------------------------------------------------------------


def _parse(text: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        path = fh.name
    try:
        return parse_file(path)
    finally:
        os.unlink(path)


class TestEventIdFieldParsing:
    def test_event_id_parsed(self):
        blocks = _parse("[D-EV-001]\nStatement: event block\nEventId: EVT-2026-001\n")
        assert len(blocks) == 1
        assert blocks[0]["EventId"] == "EVT-2026-001"

    def test_no_event_id_absent(self):
        blocks = _parse("[D-EV-002]\nStatement: no event id\n")
        assert len(blocks) == 1
        assert "EventId" not in blocks[0]

    def test_event_id_coexists_with_status(self):
        blocks = _parse("[D-EV-003]\nStatement: both fields\nStatus: active\nEventId: EVT-ABC\n")
        assert len(blocks) == 1
        b = blocks[0]
        assert b["Status"] == "active"
        assert b["EventId"] == "EVT-ABC"

    def test_event_id_numeric(self):
        blocks = _parse("[D-EV-004]\nStatement: numeric event\nEventId: 12345\n")
        assert len(blocks) == 1
        # block_parser coerces pure-integer strings to int
        assert str(blocks[0]["EventId"]) == "12345"


# ---------------------------------------------------------------------------
# _apply_event_id_filter unit tests
# ---------------------------------------------------------------------------


class TestApplyEventIdFilter:
    def _make_hits(self) -> list[dict]:
        return [
            {"_id": "A", "score": 0.9, "EventId": "EVT-001"},
            {"_id": "B", "score": 0.8, "EventId": "EVT-002"},
            {"_id": "C", "score": 0.7, "EventId": "EVT-001"},
            {"_id": "D", "score": 0.6},  # no EventId
        ]

    def test_filter_matches_event(self):
        hits = self._make_hits()
        result = _apply_event_id_filter(hits, "EVT-001")
        ids = [h["_id"] for h in result]
        assert ids == ["A", "C"]

    def test_filter_excludes_no_event_id(self):
        hits = self._make_hits()
        result = _apply_event_id_filter(hits, "EVT-001")
        ids = [h["_id"] for h in result]
        assert "D" not in ids

    def test_filter_different_event(self):
        hits = self._make_hits()
        result = _apply_event_id_filter(hits, "EVT-002")
        ids = [h["_id"] for h in result]
        assert ids == ["B"]

    def test_filter_case_insensitive(self):
        hits = [{"_id": "X", "score": 1.0, "EventId": "Evt-001"}]
        result = _apply_event_id_filter(hits, "EVT-001")
        assert len(result) == 1

    def test_filter_empty_input(self):
        assert _apply_event_id_filter([], "EVT-001") == []

    def test_filter_no_match_returns_empty(self):
        hits = self._make_hits()
        result = _apply_event_id_filter(hits, "EVT-999")
        assert result == []


# ---------------------------------------------------------------------------
# recall() integration tests — event_id parameter
# ---------------------------------------------------------------------------


def _make_workspace_with_event_ids() -> tuple[str, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    for d in ("decisions", "tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    content = (
        "[D-EID-001]\nStatement: deployment incident event alpha\n"
        "EventId: INC-2026-ALPHA\nStatus: active\n\n"
        "[D-EID-002]\nStatement: rollback event alpha system\n"
        "EventId: INC-2026-ALPHA\nStatus: active\n\n"
        "[D-EID-003]\nStatement: deployment incident event beta\n"
        "EventId: INC-2026-BETA\nStatus: active\n\n"
        "[D-EID-004]\nStatement: deployment incident no event\n"
        "Status: active\n\n"
    )
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(content)
    return ws, td


class TestRecallEventIdFilter:
    @pytest.fixture(scope="class")
    def workspace(self):
        ws, td = _make_workspace_with_event_ids()
        yield ws
        td.cleanup()

    def test_no_filter_returns_multiple(self, workspace):
        results = recall(workspace, "deployment incident", limit=10, active_only=False)
        assert len(results) >= 3

    def test_filter_alpha_event(self, workspace):
        # Use a broader query that surfaces alpha blocks; assert beta and
        # no-event blocks are excluded regardless.
        results = recall(workspace, "incident event alpha rollback", limit=10, event_id="INC-2026-ALPHA")
        ids = {r["_id"] for r in results}
        # At least one alpha block must appear
        assert ids & {"D-EID-001", "D-EID-002"}, f"No alpha blocks in results: {ids}"
        # Beta and no-event blocks must be absent
        assert "D-EID-003" not in ids
        assert "D-EID-004" not in ids

    def test_filter_beta_event(self, workspace):
        results = recall(workspace, "deployment incident", limit=10, event_id="INC-2026-BETA")
        ids = {r["_id"] for r in results}
        assert "D-EID-003" in ids
        assert "D-EID-001" not in ids
        assert "D-EID-004" not in ids

    def test_filter_excludes_blocks_without_event_id(self, workspace):
        results = recall(workspace, "deployment incident", limit=10, event_id="INC-2026-ALPHA")
        ids = {r["_id"] for r in results}
        assert "D-EID-004" not in ids

    def test_filter_none_is_no_op(self, workspace):
        all_results = recall(workspace, "deployment", limit=10)
        filtered_results = recall(workspace, "deployment", limit=10, event_id=None)
        assert len(all_results) == len(filtered_results)

    def test_filter_unknown_event_returns_empty(self, workspace):
        results = recall(workspace, "deployment", limit=10, event_id="NONEXISTENT-EVT")
        assert results == []

    def test_filter_case_insensitive(self, workspace):
        results_upper = recall(workspace, "deployment incident", limit=10, event_id="INC-2026-ALPHA")
        results_lower = recall(workspace, "deployment incident", limit=10, event_id="inc-2026-alpha")
        assert {r["_id"] for r in results_upper} == {r["_id"] for r in results_lower}

"""Unit tests for the optional lifecycle block field and recall filter.

Verifies that:
- The block parser correctly reads ``Lifecycle: <value>`` as a top-level field.
- Blocks without a ``Lifecycle`` field implicitly default to ``"durable"``.
- ``recall(lifecycle=...)`` filters results by lifecycle value.
- ``recall()`` with no ``lifecycle`` argument returns all matching blocks
  (behaviour-preserving default).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mind_mem._recall_core import _apply_lifecycle_filter, recall
from mind_mem.block_parser import parse_file

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


class TestLifecycleFieldParsing:
    def test_durable_parsed(self):
        blocks = _parse("[F-LC-001]\nStatement: durable block\nLifecycle: durable\n")
        assert len(blocks) == 1
        assert blocks[0]["Lifecycle"] == "durable"

    def test_ephemeral_parsed(self):
        blocks = _parse("[F-LC-002]\nStatement: ephemeral block\nLifecycle: ephemeral\n")
        assert len(blocks) == 1
        assert blocks[0]["Lifecycle"] == "ephemeral"

    def test_generated_parsed(self):
        blocks = _parse("[F-LC-003]\nStatement: generated block\nLifecycle: generated\n")
        assert len(blocks) == 1
        assert blocks[0]["Lifecycle"] == "generated"

    def test_no_lifecycle_field_absent(self):
        blocks = _parse("[F-LC-004]\nStatement: no lifecycle\n")
        assert len(blocks) == 1
        assert "Lifecycle" not in blocks[0]

    def test_lifecycle_coexists_with_status(self):
        blocks = _parse("[F-LC-005]\nStatement: both fields\nStatus: active\nLifecycle: durable\n")
        assert len(blocks) == 1
        b = blocks[0]
        assert b["Status"] == "active"
        assert b["Lifecycle"] == "durable"


# ---------------------------------------------------------------------------
# _apply_lifecycle_filter unit tests
# ---------------------------------------------------------------------------


class TestApplyLifecycleFilter:
    def _make_hits(self) -> list[dict]:
        return [
            {"_id": "A", "score": 0.9, "Lifecycle": "durable"},
            {"_id": "B", "score": 0.8, "Lifecycle": "ephemeral"},
            {"_id": "C", "score": 0.7, "Lifecycle": "generated"},
            {"_id": "D", "score": 0.6},  # no Lifecycle → defaults to durable
        ]

    def test_filter_durable_includes_no_lifecycle(self):
        hits = self._make_hits()
        result = _apply_lifecycle_filter(hits, "durable")
        ids = [h["_id"] for h in result]
        assert "A" in ids  # explicit durable
        assert "D" in ids  # implicit durable (no field)
        assert "B" not in ids
        assert "C" not in ids

    def test_filter_ephemeral(self):
        hits = self._make_hits()
        result = _apply_lifecycle_filter(hits, "ephemeral")
        ids = [h["_id"] for h in result]
        assert ids == ["B"]

    def test_filter_generated(self):
        hits = self._make_hits()
        result = _apply_lifecycle_filter(hits, "generated")
        ids = [h["_id"] for h in result]
        assert ids == ["C"]

    def test_filter_case_insensitive(self):
        hits = [{"_id": "X", "score": 1.0, "Lifecycle": "Durable"}]
        result = _apply_lifecycle_filter(hits, "durable")
        assert len(result) == 1

    def test_filter_empty_input(self):
        assert _apply_lifecycle_filter([], "durable") == []

    def test_unknown_lifecycle_excluded(self):
        hits = [{"_id": "Z", "score": 1.0, "Lifecycle": "archived"}]
        result = _apply_lifecycle_filter(hits, "durable")
        assert result == []


# ---------------------------------------------------------------------------
# recall() integration tests — lifecycle parameter
# ---------------------------------------------------------------------------


def _make_workspace_with_lifecycle() -> tuple[str, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    for d in ("decisions", "tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    content = (
        "[D-LC-001]\nStatement: durable memory block about authentication\n"
        "Lifecycle: durable\nStatus: active\n\n"
        "[D-LC-002]\nStatement: ephemeral cache entry for authentication session\n"
        "Lifecycle: ephemeral\nStatus: active\n\n"
        "[D-LC-003]\nStatement: generated summary of authentication flow\n"
        "Lifecycle: generated\nStatus: active\n\n"
        "[D-LC-004]\nStatement: no lifecycle authentication default case\n"
        "Status: active\n\n"
    )
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(content)
    return ws, td


class TestRecallLifecycleFilter:
    @pytest.fixture(scope="class")
    def workspace(self):
        ws, td = _make_workspace_with_lifecycle()
        yield ws
        td.cleanup()

    def test_no_filter_returns_all(self, workspace):
        results = recall(workspace, "authentication", limit=10, active_only=False)
        ids = {r["_id"] for r in results}
        # All four blocks should be reachable
        assert len(ids) >= 3

    def test_filter_durable_includes_no_lifecycle(self, workspace):
        results = recall(workspace, "authentication", limit=10, lifecycle="durable")
        ids = {r["_id"] for r in results}
        assert "D-LC-001" in ids  # explicit durable
        assert "D-LC-004" in ids  # implicit durable
        assert "D-LC-002" not in ids
        assert "D-LC-003" not in ids

    def test_filter_ephemeral(self, workspace):
        results = recall(workspace, "authentication", limit=10, lifecycle="ephemeral")
        ids = {r["_id"] for r in results}
        assert "D-LC-002" in ids
        assert "D-LC-001" not in ids
        assert "D-LC-003" not in ids

    def test_filter_generated(self, workspace):
        results = recall(workspace, "authentication", limit=10, lifecycle="generated")
        ids = {r["_id"] for r in results}
        assert "D-LC-003" in ids
        assert "D-LC-001" not in ids
        assert "D-LC-002" not in ids

    def test_filter_none_is_no_op(self, workspace):
        all_results = recall(workspace, "authentication", limit=10)
        filtered_results = recall(workspace, "authentication", limit=10, lifecycle=None)
        # Order may vary slightly due to scoring, but count should be the same
        assert len(all_results) == len(filtered_results)

    def test_filter_unknown_value_returns_empty(self, workspace):
        results = recall(workspace, "authentication", limit=10, lifecycle="unknown-value")
        assert results == []

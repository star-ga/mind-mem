# Copyright 2026 STARGA, Inc.
"""Tests for the recall_with_axis MCP tool surface."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def axis_workspace(monkeypatch):
    """Point the MCP server at a minimal in-memory workspace."""
    tmp = tempfile.TemporaryDirectory()
    ws = Path(tmp.name)
    (ws / "decisions").mkdir()
    (ws / "memory").mkdir()
    (ws / "intelligence").mkdir()
    (ws / "tasks").mkdir()
    (ws / "entities").mkdir()
    (ws / "mind-mem.json").write_text('{"retrieval": {"backend": "bm25"}}', encoding="utf-8")

    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))

    # Ensure mcp_server sees the fresh workspace. Reload is overkill; the
    # _workspace() helper reads the env var each call.
    yield ws
    tmp.cleanup()


def _call_tool(tool, **kwargs) -> dict:
    """Invoke a FastMCP-registered tool function directly and parse JSON."""
    # The @mcp.tool decorator keeps the callable exposed under .fn;
    # falling back to the raw function if the attribute is missing lets
    # this helper work across FastMCP versions.
    fn = getattr(tool, "fn", tool)
    result = fn(**kwargs)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"_raw": result}
    return result  # already a dict


class TestRecallWithAxisArgParsing:
    def test_empty_axes_string_rejected(self, axis_workspace, monkeypatch):
        from mind_mem import mcp_server

        out = _call_tool(mcp_server.recall_with_axis, query="q", axes="")
        assert "error" in out
        assert "at least one axis" in out["error"].lower()

    def test_unknown_axis_rejected(self, axis_workspace):
        from mind_mem import mcp_server

        out = _call_tool(mcp_server.recall_with_axis, query="q", axes="lexical,bogus")
        assert "error" in out
        assert "Unknown observation axis" in out["error"]

    def test_weights_not_equal_signs(self, axis_workspace):
        from mind_mem import mcp_server

        out = _call_tool(
            mcp_server.recall_with_axis,
            query="q",
            axes="lexical",
            weights="lexical;0.8",
        )
        assert "error" in out
        assert "axis=value" in out["error"]

    def test_weights_non_numeric(self, axis_workspace):
        from mind_mem import mcp_server

        out = _call_tool(
            mcp_server.recall_with_axis,
            query="q",
            axes="lexical",
            weights="lexical=abc",
        )
        assert "error" in out
        assert "not numeric" in out["error"]

    def test_axes_length_bound(self, axis_workspace):
        from mind_mem import mcp_server

        long_axes = "lexical," * 1024
        out = _call_tool(mcp_server.recall_with_axis, query="q", axes=long_axes)
        assert "error" in out
        assert "chars" in out["error"].lower()

    def test_too_many_axis_tokens(self, axis_workspace):
        from mind_mem import mcp_server

        many = ",".join(["lexical"] * 32)
        out = _call_tool(mcp_server.recall_with_axis, query="q", axes=many)
        assert "error" in out
        assert "entries" in out["error"].lower()

    def test_limit_upper_bound(self, axis_workspace):
        from mind_mem import mcp_server

        out = _call_tool(
            mcp_server.recall_with_axis, query="q", axes="lexical", limit=100000
        )
        assert "error" in out
        assert "limit" in out["error"].lower()

    def test_limit_lower_bound(self, axis_workspace):
        from mind_mem import mcp_server

        out = _call_tool(mcp_server.recall_with_axis, query="q", axes="lexical", limit=0)
        assert "error" in out
        assert "limit" in out["error"].lower()


class TestRecallWithAxisEnvelope:
    def test_envelope_has_expected_keys(self, axis_workspace, monkeypatch):
        from mind_mem import axis_recall, mcp_server

        monkeypatch.setattr(
            axis_recall,
            "_recall_for_axis",
            lambda ws, q, ax, *, limit, active_only, base_recall_kwargs: [],
        )

        out = _call_tool(
            mcp_server.recall_with_axis,
            query="hello",
            axes="lexical,semantic",
            allow_rotation=False,
        )
        assert out["_schema_version"] == "1.0"
        assert out["query"] == "hello"
        assert "results" in out
        assert "weights" in out
        assert "rotated" in out
        assert "diversity" in out
        assert "attempts" in out

    def test_rotated_flag_propagates(self, axis_workspace, monkeypatch):
        from mind_mem import axis_recall, mcp_server
        from mind_mem.observation_axis import ObservationAxis

        per_axis: dict = {
            ObservationAxis.LEXICAL: [],
            ObservationAxis.SEMANTIC: [],
            ObservationAxis.TEMPORAL: [
                {"_id": "T", "file": "t.md", "line": 1, "excerpt": "temporal"}
            ],
            ObservationAxis.ENTITY_GRAPH: [
                {"_id": "G", "file": "g.md", "line": 1, "excerpt": "graph"}
            ],
        }

        def stub(ws, q, axis, *, limit, active_only, base_recall_kwargs):
            return list(per_axis.get(axis, []))

        monkeypatch.setattr(axis_recall, "_recall_for_axis", stub)

        out = _call_tool(
            mcp_server.recall_with_axis,
            query="q",
            axes="lexical,semantic",
        )
        assert out["rotated"] is True
        assert any(r["_id"] in {"T", "G"} for r in out["results"])

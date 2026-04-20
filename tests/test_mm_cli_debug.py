# Copyright 2026 STARGA, Inc.
"""Tests for mm inspect / mm explain / mm trace debug commands (v3.2.0)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from mind_mem.mm_cli import (
    _cmd_explain,
    _cmd_inspect,
    _cmd_trace,
    _parse_log_lines,
    _render_trace_rows,
    build_parser,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs: Any) -> Any:
    """Return a SimpleNamespace-like object for testing command functions."""
    import argparse

    ns = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _minimal_block(block_id: str = "D-001") -> dict[str, Any]:
    return {
        "_id": block_id,
        "Statement": "Use BM25 for recall",
        "Status": "active",
        "Date": "2026-01-01",
        "Tags": "recall bm25",
        "Rationale": "Best bang-for-buck without external deps",
    }


# ---------------------------------------------------------------------------
# mm inspect tests
# ---------------------------------------------------------------------------


class TestMmInspect:
    def test_help_renders(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["inspect", "--help"])
        assert exc.value.code == 0

    _EMPTY_PROV = {"block_id": "D-001", "dependencies": [], "causal_chains": [], "contradictions": []}

    def test_inspect_known_block_text(self, tmp_path: Any, capsys: Any) -> None:
        block = _minimal_block("D-001")
        with (
            patch("mind_mem.mm_cli._workspace", return_value=str(tmp_path)),
            patch("mind_mem.block_store.MarkdownBlockStore.get_by_id", return_value=block),
            patch("mind_mem.mm_cli._build_provenance", return_value=self._EMPTY_PROV),
        ):
            rc = _cmd_inspect(_make_args(block_id="D-001", format="text"))
        assert rc == 0
        out = capsys.readouterr().out
        assert "D-001" in out
        assert "BM25" in out

    def test_inspect_known_block_json(self, tmp_path: Any, capsys: Any) -> None:
        block = _minimal_block("D-001")
        with (
            patch("mind_mem.mm_cli._workspace", return_value=str(tmp_path)),
            patch("mind_mem.block_store.MarkdownBlockStore.get_by_id", return_value=block),
            patch("mind_mem.mm_cli._build_provenance", return_value=self._EMPTY_PROV),
        ):
            rc = _cmd_inspect(_make_args(block_id="D-001", format="json"))
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["block"]["_id"] == "D-001"
        assert "provenance" in data

    def test_inspect_nonexistent_exits_nonzero(self, tmp_path: Any, capsys: Any) -> None:
        with (
            patch("mind_mem.mm_cli._workspace", return_value=str(tmp_path)),
            patch("mind_mem.block_store.MarkdownBlockStore.get_by_id", return_value=None),
        ):
            rc = _cmd_inspect(_make_args(block_id="D-MISSING", format="text"))
        assert rc != 0
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "D-MISSING" in err

    def test_inspect_json_provenance_keys(self, tmp_path: Any, capsys: Any) -> None:
        block = _minimal_block("T-010")
        dep_edge = {
            "source_id": "T-010",
            "target_id": "D-001",
            "edge_type": "depends_on",
            "weight": 1.0,
            "created_at": "",
            "updated_at": "",
        }
        prov = {
            "block_id": "T-010",
            "dependencies": [dep_edge],
            "causal_chains": [["T-010", "D-001"]],
            "contradictions": [],
        }
        with (
            patch("mind_mem.mm_cli._workspace", return_value=str(tmp_path)),
            patch("mind_mem.block_store.MarkdownBlockStore.get_by_id", return_value=block),
            patch("mind_mem.mm_cli._build_provenance", return_value=prov),
        ):
            rc = _cmd_inspect(_make_args(block_id="T-010", format="json"))
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["provenance"]["dependencies"][0]["edge_type"] == "depends_on"
        assert data["provenance"]["causal_chains"] == [["T-010", "D-001"]]

    def test_inspect_parser_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["inspect", "D-002", "--format", "json"])
        assert args.block_id == "D-002"
        assert args.format == "json"

    def test_inspect_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["inspect", "D-003"])
        assert args.format == "text"


# ---------------------------------------------------------------------------
# mm explain tests
# ---------------------------------------------------------------------------


class TestMmExplain:
    def test_help_renders(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["explain", "--help"])
        assert exc.value.code == 0

    def _fake_results(self, n: int = 3) -> list[dict[str, Any]]:
        return [
            {"_id": f"D-00{i}", "score": 0.9 - i * 0.1, "Statement": f"result {i}"}
            for i in range(1, n + 1)
        ]

    _EMPTY_DIAG: dict[str, Any] = {"intent_distribution": {}, "stage_rejection_rates": {}}

    def test_explain_text_nonempty(self, tmp_path: Any, capsys: Any) -> None:
        with (
            patch("mind_mem.mm_cli._workspace", return_value=str(tmp_path)),
            patch("mind_mem.recall.recall", return_value=self._fake_results()),
            patch("mind_mem.retrieval_graph.retrieval_diagnostics", return_value=self._EMPTY_DIAG),
        ):
            rc = _cmd_explain(_make_args(query="BM25 recall", limit=10, backend="auto", format="text"))
        assert rc == 0
        out = capsys.readouterr().out
        assert len(out.strip()) > 0
        assert "D-001" in out

    def test_explain_json_parseable(self, tmp_path: Any, capsys: Any) -> None:
        with (
            patch("mind_mem.mm_cli._workspace", return_value=str(tmp_path)),
            patch("mind_mem.recall.recall", return_value=self._fake_results()),
            patch("mind_mem.retrieval_graph.retrieval_diagnostics", return_value=self._EMPTY_DIAG),
        ):
            rc = _cmd_explain(_make_args(query="BM25 recall", limit=10, backend="auto", format="json"))
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "results" in data
        assert data["results"][0]["block_id"] == "D-001"
        assert data["results"][0]["bm25"] == pytest.approx(0.8, abs=1e-3)

    def test_explain_json_structure(self, tmp_path: Any, capsys: Any) -> None:
        diag = {"intent_distribution": {"WHY": 2}, "stage_rejection_rates": {"bm25": 0.1}}
        with (
            patch("mind_mem.mm_cli._workspace", return_value=str(tmp_path)),
            patch("mind_mem.recall.recall", return_value=self._fake_results(1)),
            patch("mind_mem.retrieval_graph.retrieval_diagnostics", return_value=diag),
        ):
            rc = _cmd_explain(_make_args(query="test", limit=5, backend="bm25", format="json"))
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        row = data["results"][0]
        assert "bm25" in row
        assert "stages_hit" in row
        assert row["stages_hit"][0] is True  # BM25 always present

    def test_explain_empty_results(self, tmp_path: Any, capsys: Any) -> None:
        with (
            patch("mind_mem.mm_cli._workspace", return_value=str(tmp_path)),
            patch("mind_mem.recall.recall", return_value=[]),
            patch("mind_mem.retrieval_graph.retrieval_diagnostics", return_value={}),
        ):
            rc = _cmd_explain(_make_args(query="nothing", limit=5, backend="auto", format="json"))
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["results"] == []

    def test_explain_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["explain", "my query"])
        assert args.query == "my query"
        assert args.limit == 10
        assert args.backend == "auto"
        assert args.format == "text"

    def test_explain_parser_all_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["explain", "foo", "--limit", "5", "--backend", "hybrid", "--format", "json"])
        assert args.limit == 5
        assert args.backend == "hybrid"
        assert args.format == "json"


# ---------------------------------------------------------------------------
# mm trace tests
# ---------------------------------------------------------------------------


class TestMmTrace:
    def test_help_renders(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["trace", "--help"])
        assert exc.value.code == 0

    def _log_line(self, tool: str = "recall", duration_ms: int = 42, success: bool = True, size: int = 5) -> str:
        return json.dumps(
            {
                "ts": "2026-01-01T00:00:00.000Z",
                "level": "info",
                "component": "mcp_server",
                "event": "mcp_tool_call",
                "data": {
                    "tool": tool,
                    "duration_ms": duration_ms,
                    "success": success,
                    "result_size": size,
                },
            }
        )

    def test_parse_log_lines_basic(self) -> None:
        lines = [self._log_line("recall", 55, True, 3), self._log_line("propose_update", 120, False, 0)]
        rows = _parse_log_lines(lines)
        assert len(rows) == 2
        assert rows[0]["tool"] == "recall"
        assert rows[0]["duration_ms"] == 55
        assert rows[1]["success"] is False

    def test_parse_log_lines_filters_tool(self) -> None:
        lines = [self._log_line("recall"), self._log_line("scan"), self._log_line("recall")]
        rows = _parse_log_lines(lines, tool_filter="scan")
        assert len(rows) == 1
        assert rows[0]["tool"] == "scan"

    def test_parse_log_lines_ignores_non_mcp_events(self) -> None:
        other = json.dumps({"ts": "t", "level": "info", "event": "something_else", "data": {}})
        rows = _parse_log_lines([other, self._log_line("recall")])
        assert len(rows) == 1

    def test_parse_log_lines_ignores_malformed(self) -> None:
        rows = _parse_log_lines(["not json at all", "{broken"])
        assert rows == []

    def test_trace_last_n_from_env_file(self, tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
        log_file = tmp_path / "mcp.log"
        lines = [self._log_line("recall", i * 10) for i in range(1, 8)]
        log_file.write_text("\n".join(lines))
        monkeypatch.setenv("MIND_MEM_LOG_FILE", str(log_file))

        rc = _cmd_trace(_make_args(last=5, tool=None, live=False))
        assert rc == 0
        out = capsys.readouterr().out
        # Should show at most 5 rows
        data_rows = [ln for ln in out.splitlines() if "recall" in ln]
        assert len(data_rows) <= 5

    def test_trace_tool_filter(self, tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
        log_file = tmp_path / "mcp.log"
        lines = [self._log_line("recall"), self._log_line("scan"), self._log_line("recall")]
        log_file.write_text("\n".join(lines))
        monkeypatch.setenv("MIND_MEM_LOG_FILE", str(log_file))

        rc = _cmd_trace(_make_args(last=20, tool="scan", live=False))
        assert rc == 0
        out = capsys.readouterr().out
        assert "scan" in out
        data_rows = [ln for ln in out.splitlines() if "recall" in ln]
        assert data_rows == []

    def test_trace_no_entries(self, tmp_path: Any, capsys: Any, monkeypatch: Any) -> None:
        log_file = tmp_path / "mcp.log"
        log_file.write_text('{"ts":"t","event":"other","data":{}}\n')
        monkeypatch.setenv("MIND_MEM_LOG_FILE", str(log_file))

        rc = _cmd_trace(_make_args(last=20, tool=None, live=False))
        assert rc == 0
        out = capsys.readouterr().out
        assert "No mcp_tool_call" in out or "No " in out

    def test_trace_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["trace"])
        assert args.last == 20
        assert args.tool is None
        assert args.live is False

    def test_trace_parser_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["trace", "--last", "5", "--tool", "recall"])
        assert args.last == 5
        assert args.tool == "recall"

    def test_render_trace_rows_success_and_error(self, capsys: Any) -> None:
        rows = [
            {"time": "2026-01-01T00:00:00Z", "tool": "recall", "duration_ms": 42, "success": True, "result_size": 3},
            {"time": "2026-01-01T00:00:01Z", "tool": "propose_update", "duration_ms": None, "success": False, "result_size": None},
        ]
        _render_trace_rows(rows)
        out = capsys.readouterr().out
        assert "recall" in out
        assert "propose_update" in out

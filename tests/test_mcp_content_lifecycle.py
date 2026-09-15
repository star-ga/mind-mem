# Copyright 2026 STARGA, Inc.
"""MCP indexed-path controls for source-bound content lifecycle admission."""

from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from datetime import date
from pathlib import Path

from mind_mem.init_workspace import init
from mind_mem.mcp.tools import recall as recall_tool
from mind_mem.sqlite_index import _db_path, build_index

NOW = date(2026, 9, 14)


def _workspace(tmp_path: Path) -> str:
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"].update(
        {
            "backend": "sqlite",
            "validity_gate": {
                "enabled": True,
                "content_categories": {"enabled": True, "ttl_days": {"infra": 2, "status": 1}},
            },
        }
    )
    config.setdefault("cache", {})["enabled"] = False
    config["served_ledger"] = {"enabled": False}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (workspace / "decisions/DECISIONS.md").write_text(
        "".join(
            (
                f"[D-20260901-{seq:03d}]\nStatus: active\n"
                f"Statement: orchid lifecycle control {seq}\n"
                f"ContentCategory: {category}\nContentValidFrom: {stamp}\n\n"
            )
            for seq, category, stamp in (
                (1, "status", "2026-09-12"),
                (2, "decision", "2020-01-01"),
                (4, "credential", "2020-01-01"),
            )
        ),
        encoding="utf-8",
    )
    return str(workspace)


def _mcp_indexed_recall(workspace: str, monkeypatch) -> dict:
    monkeypatch.setattr(recall_tool, "_workspace", lambda: workspace)
    return json.loads(
        recall_tool._recall_impl_ranked(
            "orchid",
            limit=10,
            backend="auto",
            scoring_instant=NOW,
        )
    )


def test_mcp_indexed_path_applies_semantic_ttl_and_keeps_admitted_durable(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    build_index(workspace)

    envelope = _mcp_indexed_recall(workspace, monkeypatch)
    by_id = {row["_id"]: row for row in envelope["results"]}

    assert by_id["D-20260901-001"]["validity"]["content_lifecycle"]["state"] == "stale"
    assert by_id["D-20260901-001"]["_validity_demoted"] is True
    assert by_id["D-20260901-002"]["validity"]["content_lifecycle"]["state"] == "durable"
    assert "_validity_demoted" not in by_id["D-20260901-002"]


def test_hybrid_scan_fallback_applies_semantic_ttl_once(tmp_path: Path, monkeypatch) -> None:
    """Hybrid's no-index BM25 fallback must match the direct BM25 score.

    ``HybridBackend`` reaches the core ``recall`` implementation when its
    SQLite leg is unavailable. The serving boundary applies the final gate;
    the private fallback call must therefore leave that gate for the boundary
    rather than demoting the same score twice.
    """
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(recall_tool, "_workspace", lambda: workspace)
    config_path = Path(workspace) / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Keep the direct comparison on the core scan path. The MCP request still
    # asks for hybrid below; its BM25 leg is forced through the same core scan
    # fallback after the indexed probe is made unavailable.
    config["recall"]["backend"] = "scan"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    bm25 = json.loads(
        recall_tool._recall_impl_ranked(
            "orchid",
            limit=10,
            backend="bm25",
            scoring_instant=NOW,
        )
    )

    # Force the real HybridBackend -> core-recall fallback seam. The current
    # SQLite helper has a re-entrancy guard that can return an empty result on
    # a missing index; that is a separate degradation control, so make the
    # backend-unavailable transition explicit here.
    import mind_mem.sqlite_index as sqlite_index

    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic index unavailable")

    monkeypatch.setattr(sqlite_index, "query_index", unavailable)

    import mind_mem.validity_gate as validity_gate

    calls = []
    real_gate = validity_gate.apply_validity_gate

    def counted(*args, **kwargs):
        calls.append(1)
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(validity_gate, "apply_validity_gate", counted)

    hybrid = json.loads(
        recall_tool._recall_impl_ranked(
            "orchid",
            limit=10,
            backend="auto",
            scoring_instant=NOW,
        )
    )

    bm25_rows = {row["_id"]: row for row in bm25["results"]}
    hybrid_rows = {row["_id"]: row for row in hybrid["results"]}
    assert list(bm25_rows) == list(hybrid_rows)
    assert {key: row["score"] for key, row in bm25_rows.items()} == {key: row["score"] for key, row in hybrid_rows.items()}
    assert calls == [1]
    assert hybrid_rows["D-20260901-001"]["validity"]["content_lifecycle"]["state"] == "stale"
    assert hybrid_rows["D-20260901-001"]["_validity_demoted"] is True


def test_hybrid_rrf_applies_semantic_ttl_once_after_fusion(tmp_path: Path, monkeypatch) -> None:
    """A genuine two-arm RRF result receives one final lifecycle gate."""
    workspace = _workspace(tmp_path)
    build_index(workspace)
    monkeypatch.setattr(recall_tool, "_workspace", lambda: workspace)

    from mind_mem.hybrid_recall import HybridBackend

    real_factory = HybridBackend.from_config

    def configured(config):
        backend = real_factory(config)
        backend._vector_available = True
        monkeypatch.setattr(
            backend,
            "_vector_search",
            lambda query, workspace, limit=200, active_only=False, **kwargs: [
                {"_id": "D-20260901-001", "score": 0.99},
                {"_id": "D-20260901-002", "score": 0.98},
            ],
        )
        return backend

    monkeypatch.setattr(HybridBackend, "from_config", staticmethod(configured))

    import mind_mem.validity_gate as validity_gate

    calls = []
    real_gate = validity_gate.apply_validity_gate

    def counted(*args, **kwargs):
        calls.append(1)
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(validity_gate, "apply_validity_gate", counted)

    envelope = json.loads(
        recall_tool._recall_impl_ranked(
            "orchid",
            limit=10,
            backend="auto",
            scoring_instant=NOW,
        )
    )

    assert calls == [1]
    by_id = {row["_id"]: row for row in envelope["results"]}
    assert {"bm25", "vector"} <= set(by_id["D-20260901-001"]["fusion_sources"])
    assert by_id["D-20260901-001"]["validity"]["content_lifecycle"]["state"] == "stale"
    assert by_id["D-20260901-001"]["_validity_demoted"] is True


def test_direct_hybrid_scan_fallback_keeps_lifecycle_gate(tmp_path: Path, monkeypatch) -> None:
    """Direct Python HybridBackend callers retain the final gate."""
    workspace = _workspace(tmp_path)
    config_path = Path(workspace) / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["backend"] = "scan"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    import mind_mem._recall_core as recall_core
    import mind_mem.hybrid_recall as hybrid_recall
    import mind_mem.sqlite_index as sqlite_index
    import mind_mem.validity_gate as validity_gate

    monkeypatch.setattr(sqlite_index, "query_index", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("index unavailable")))
    core_calls = []
    module_calls = []
    real_core_gate = recall_core.apply_validity_gate
    real_module_gate = validity_gate.apply_validity_gate

    def counted_core(*args, **kwargs):
        core_calls.append(1)
        return real_core_gate(*args, **kwargs)

    def counted_module(*args, **kwargs):
        module_calls.append(1)
        return real_module_gate(*args, **kwargs)

    monkeypatch.setattr(recall_core, "apply_validity_gate", counted_core)
    monkeypatch.setattr(validity_gate, "apply_validity_gate", counted_module)

    backend = hybrid_recall.HybridBackend(config=config["recall"])
    results = backend.search("orchid", workspace, limit=10, scoring_instant=NOW)

    assert core_calls == []
    assert module_calls == [1]
    by_id = {row["_id"]: row for row in results}
    assert by_id["D-20260901-001"]["validity"]["content_lifecycle"]["state"] == "stale"
    assert by_id["D-20260901-001"]["_validity_demoted"] is True


def test_cli_explain_hybrid_fallback_returns_gated_row(tmp_path: Path, monkeypatch, capsys) -> None:
    """The actual ``mm explain --backend hybrid`` handler uses the gate."""
    workspace = _workspace(tmp_path)
    config_path = Path(workspace) / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["backend"] = "scan"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    import mind_mem.mm_cli as mm_cli
    import mind_mem.sqlite_index as sqlite_index
    import mind_mem.validity_gate as validity_gate

    monkeypatch.setattr(mm_cli, "_workspace", lambda: workspace)
    monkeypatch.setattr(sqlite_index, "query_index", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("index unavailable")))
    calls = []
    real_gate = validity_gate.apply_validity_gate

    def counted(*args, **kwargs):
        calls.append(1)
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(validity_gate, "apply_validity_gate", counted)
    rc = mm_cli._cmd_explain(Namespace(query="orchid", limit=10, backend="hybrid", format="json"))

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert calls == [1]
    by_id = {row["block_id"]: row for row in payload["results"]}
    assert "D-20260901-001" in by_id


def test_hybrid_rrf_gates_before_limit_cut(tmp_path: Path, monkeypatch) -> None:
    """A stale high-ranked RRF hit cannot crowd out a fresh candidate."""
    workspace = _workspace(tmp_path)
    config_path = Path(workspace) / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["dedup"] = {"enabled": False}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    build_index(workspace)
    monkeypatch.setattr(recall_tool, "_workspace", lambda: workspace)

    from mind_mem.hybrid_recall import HybridBackend

    real_factory = HybridBackend.from_config

    def configured(config):
        backend = real_factory(config)
        backend._config["dedup"] = {"enabled": False}
        backend._vector_available = True
        monkeypatch.setattr(
            backend,
            "_vector_search",
            lambda query, workspace, limit=200, active_only=False, **kwargs: [
                {"_id": "D-20260901-001", "score": 0.99},
                {"_id": "D-20260901-002", "score": 0.98},
            ],
        )
        return backend

    monkeypatch.setattr(HybridBackend, "from_config", staticmethod(configured))
    envelope = json.loads(
        recall_tool._recall_impl_ranked(
            "orchid",
            limit=1,
            backend="auto",
            scoring_instant=NOW,
        )
    )

    assert [row["_id"] for row in envelope["results"]] == ["D-20260901-002"]
    assert envelope["results"][0]["validity"]["content_lifecycle"]["state"] == "durable"


def test_mcp_indexed_bm25_validity_widens_before_limit_cut(tmp_path: Path, monkeypatch) -> None:
    """Direct indexed BM25 must admit a bounded pool before its validity cut.

    The explicit ``bm25`` route is allowed to return only the requested row
    when validity is off.  With the gate on, the stale high-scoring row must
    not hide a durable lower-scoring row that FTS can return in the bounded
    over-fetch window.
    """
    workspace = _workspace(tmp_path)
    decisions = Path(workspace) / "decisions/DECISIONS.md"
    decisions.write_text(
        "".join(
            (f"[D-20260901-{seq:03d}]\nStatus: active\nStatement: {statement}\nContentCategory: {category}\nContentValidFrom: {stamp}\n\n")
            for seq, statement, category, stamp in (
                (1, "orchid lifecycle orchid lifecycle stale record", "status", "2026-09-12"),
                (2, "orchid lifecycle durable record", "decision", "2026-09-12"),
            )
        )
        + "".join(f"[D-20260901-{seq:03d}]\nStatus: active\nStatement: fillerword {seq}\n\n" for seq in range(3, 103)),
        encoding="utf-8",
    )
    build_index(workspace)
    monkeypatch.setattr(recall_tool, "_workspace", lambda: workspace)
    real_fts_query = recall_tool.fts_query
    fts_limits: list[int] = []

    def captured_fts_query(*args, **kwargs):
        fts_limits.append(kwargs["limit"])
        return real_fts_query(*args, **kwargs)

    monkeypatch.setattr(recall_tool, "fts_query", captured_fts_query)

    enabled = json.loads(
        recall_tool._recall_impl_ranked(
            "orchid lifecycle",
            limit=1,
            backend="bm25",
            scoring_instant=NOW,
        )
    )
    assert fts_limits == [4]
    assert [row["_id"] for row in enabled["results"]] == ["D-20260901-002"]
    assert enabled["results"][0]["validity"]["content_lifecycle"]["state"] == "durable"

    config_path = Path(workspace) / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["validity_gate"] = {"enabled": False}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    fts_limits.clear()
    disabled = json.loads(
        recall_tool._recall_impl_ranked(
            "orchid lifecycle",
            limit=1,
            backend="bm25",
            scoring_instant=NOW,
        )
    )
    assert fts_limits == [1]
    assert [row["_id"] for row in disabled["results"]] == ["D-20260901-001"]
    assert "validity" not in disabled["results"][0]


def test_mcp_stale_index_cannot_borrow_namespace_lifecycle_or_revoked_status(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    shared = Path(workspace) / "shared" / "decisions"
    shared.mkdir(parents=True)
    shared.joinpath("DECISIONS.md").write_text(
        "[D-20260901-001]\nStatus: active\nStatement: orchid shared duplicate\nContentCategory: status\nContentValidFrom: 2026-09-01\n\n",
        encoding="utf-8",
    )
    build_index(workspace)

    # Simulate an old index carrying a source claim from the shared namespace.
    # The FTS content remains the original root row; lifecycle must follow the
    # claimed source's current bytes, never an ID-only workspace map.
    with sqlite3.connect(_db_path(workspace)) as conn:
        conn.execute(
            "UPDATE blocks SET file = ? WHERE id = ?",
            ("shared/decisions/DECISIONS.md", "D-20260901-001"),
        )
        conn.commit()

    corpus = Path(workspace) / "decisions/DECISIONS.md"
    corpus.write_text(
        corpus.read_text(encoding="utf-8").replace("[D-20260901-004]\nStatus: active", "[D-20260901-004]\nStatus: revoked"),
        encoding="utf-8",
    )

    envelope = _mcp_indexed_recall(workspace, monkeypatch)
    by_id = {row["_id"]: row for row in envelope["results"]}

    assert by_id["D-20260901-001"]["file"] == "shared/decisions/DECISIONS.md"
    assert by_id["D-20260901-001"]["validity"]["content_lifecycle"]["state"] == "stale"
    assert "D-20260901-004" not in by_id
    assert "D-20260901-002" in by_id

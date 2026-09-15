# Copyright 2026 STARGA, Inc.
"""MCP indexed-path controls for source-bound content lifecycle admission."""

from __future__ import annotations

import json
import sqlite3
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


def test_mcp_stale_index_cannot_borrow_namespace_lifecycle_or_revoked_status(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    shared = Path(workspace) / "shared" / "decisions"
    shared.mkdir(parents=True)
    shared.joinpath("DECISIONS.md").write_text(
        "[D-20260901-001]\nStatus: active\nStatement: orchid shared duplicate\n"
        "ContentCategory: status\nContentValidFrom: 2026-09-01\n\n",
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

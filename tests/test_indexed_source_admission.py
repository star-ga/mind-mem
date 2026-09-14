# Copyright 2026 STARGA, Inc.
"""Source-bound admission for rows held by a stale indexed backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import mind_mem.sqlite_index as sqlite_index
from mind_mem._recall_core import _withhold_inadmissible
from mind_mem.init_workspace import init
from mind_mem.recall import recall


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("recall", {})["backend"] = "sqlite"
    config.setdefault("served_ledger", {})["enabled"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    source = workspace / "decisions/DECISIONS.md"
    source.write_text(
        "[D-STALE-001]\nStatement: indexed resource canary architecture\nStatus: active\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    sqlite_index.build_index(str(workspace), incremental=False)
    return workspace, source


def test_deleted_indexed_source_is_withheld_and_restored_source_serves(tmp_path: Path) -> None:
    workspace, source = _workspace(tmp_path)
    query = "indexed resource canary architecture"

    assert [hit["_id"] for hit in recall(str(workspace), query, limit=10)] == ["D-STALE-001"]
    source.write_text("# deleted\n", encoding="utf-8")
    assert recall(str(workspace), query, limit=10) == []

    source.write_text(
        "[D-STALE-001]\nStatement: indexed resource canary architecture\nStatus: active\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    assert [hit["_id"] for hit in recall(str(workspace), query, limit=10)] == ["D-STALE-001"]


@pytest.mark.parametrize("replacement", ["quarantined", "delete"])
def test_stale_index_source_transition_is_withheld_without_reindex(tmp_path: Path, replacement: str) -> None:
    workspace, source = _workspace(tmp_path)
    query = "indexed resource canary architecture"
    assert [hit["_id"] for hit in recall(str(workspace), query, limit=10)] == ["D-STALE-001"]

    if replacement == "delete":
        source.write_text("# deleted\n", encoding="utf-8")
    else:
        source.write_text(
            "[D-STALE-001]\nStatement: indexed resource canary architecture\nStatus: quarantined\nDate: 2026-09-01\n\n",
            encoding="utf-8",
        )
    assert recall(str(workspace), query, limit=10) == []


def test_duplicate_ids_refresh_by_source_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, _source = _workspace(tmp_path)
    shared = workspace / "shared/decisions/DECISIONS.md"
    private = workspace / "agents/bob/decisions/DECISIONS.md"
    shared.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)
    shared.write_text(
        "[D-DUP-001]\nStatement: shared source\nStatus: active\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    private.write_text(
        "[D-DUP-001]\nStatement: private source\nStatus: active\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sqlite_index, "is_stale", lambda _workspace: True)
    hits: list[dict[str, Any]] = [
        {"_id": "D-DUP-001", "file": "shared/decisions/DECISIONS.md", "status": "active", "score": 2.0},
        {"_id": "D-DUP-001", "file": "agents/bob/decisions/DECISIONS.md", "status": "active", "score": 1.0},
    ]
    private.write_text(
        "[D-DUP-001]\nStatement: private source\nStatus: quarantined\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    result = _withhold_inadmissible(hits, str(workspace), status_key="status", leg="indexed")
    assert [(row["file"], row["_id"]) for row in result] == [("shared/decisions/DECISIONS.md", "D-DUP-001")]

    private.write_text(
        "[D-DUP-001]\nStatement: private source\nStatus: active\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    result = _withhold_inadmissible(hits, str(workspace), status_key="status", leg="indexed")
    assert len(result) == 2


def test_unbound_legacy_index_row_keeps_existing_status_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, _source = _workspace(tmp_path)
    monkeypatch.setattr(sqlite_index, "is_stale", lambda _workspace: True)
    hits = [{"_id": "D-LEGACY-001", "status": "active", "score": 1.0}]
    assert _withhold_inadmissible(hits, str(workspace), status_key="status", leg="indexed") == hits

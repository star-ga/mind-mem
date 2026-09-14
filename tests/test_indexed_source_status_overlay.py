# Copyright 2026 STARGA, Inc.
"""Source-bound indexed rows must survive the legacy ID status overlay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mind_mem.sqlite_index as sqlite_index
from mind_mem._recall_core import _withhold_inadmissible
from mind_mem.init_workspace import init
from mind_mem.sqlite_index import build_index


def test_source_bound_duplicate_keeps_active_source_when_sibling_is_quarantined(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["backend"] = "sqlite"
    config["served_ledger"] = {"enabled": False}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    first = workspace / "decisions" / "DECISIONS.md"
    sibling = workspace / "decisions" / "OTHER.md"
    first.write_text(
        "[D-DUP-STATUS]\nStatement: active source\nStatus: active\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    sibling.write_text(
        "[D-DUP-STATUS]\nStatement: quarantined sibling\nStatus: quarantined\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    build_index(str(workspace), incremental=False)
    hits: list[dict[str, Any]] = [
        {"_id": "D-DUP-STATUS", "file": "decisions/DECISIONS.md", "status": "active", "score": 2.0},
        {"_id": "D-DUP-STATUS", "file": "decisions/OTHER.md", "status": "quarantined", "score": 1.0},
    ]
    original = sqlite_index.is_stale
    sqlite_index.is_stale = lambda _workspace: True
    try:
        result = _withhold_inadmissible(hits, str(workspace), status_key="status", leg="indexed")
    finally:
        sqlite_index.is_stale = original
    assert [(row["file"], row["status"]) for row in result] == [("decisions/DECISIONS.md", "active")]

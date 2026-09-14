# Copyright 2026 STARGA, Inc.
"""Source admission must not rely on file metadata alone."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mind_mem._recall_core import _withhold_inadmissible
from mind_mem.init_workspace import init
from mind_mem.recall import recall
from mind_mem.sqlite_index import build_index, is_stale


def _sqlite_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["backend"] = "sqlite"
    config["served_ledger"] = {"enabled": False}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    source = workspace / "decisions" / "DECISIONS.md"
    return workspace, source


def test_equal_size_mtime_status_change_is_withheld(tmp_path: Path) -> None:
    workspace, source = _sqlite_workspace(tmp_path)
    before = "[D-SAME-STATUS]\nStatement: same metadata status canary     \nStatus: active\nDate: 2026-09-01\n\n"
    after = "[D-SAME-STATUS]\nStatement: same metadata status canary\nStatus: quarantined\nDate: 2026-09-01\n"
    after += " " * (len(before) - len(after) - 1) + "\n"
    assert len(before) == len(after)

    source.write_text(before, encoding="utf-8")
    build_index(str(workspace), incremental=False)
    assert [row["_id"] for row in recall(str(workspace), "same metadata status canary", limit=10)] == ["D-SAME-STATUS"]
    original_stat = source.stat()

    source.write_text(after, encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert is_stale(str(workspace)) is False
    assert recall(str(workspace), "same metadata status canary", limit=10) == []


def test_source_bound_statusless_row_is_checked_without_sqlite_index(tmp_path: Path) -> None:
    workspace, source = _sqlite_workspace(tmp_path)
    source.write_text(
        "[D-STATUSLESS]\nStatement: legacy source row\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    result = _withhold_inadmissible(
        [{"_id": "D-STATUSLESS", "file": "decisions/DECISIONS.md", "status": "active"}],
        str(workspace),
        status_key="status",
        leg="indexed",
    )
    assert [row["_id"] for row in result] == ["D-STATUSLESS"]
    assert result[0]["status"] is None

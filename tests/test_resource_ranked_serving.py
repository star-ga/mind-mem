"""Indexed MCP recall must use the live admitted serving boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.mcp import resources
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.served_ledger import read_served_runs, verify_served_chain
from mind_mem.sqlite_index import build_index


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions/DECISIONS.md").write_text(
        "[D-20260914-001]\nStatement: resource canary architecture\nStatus: active\nDate: 2026-09-01\n\n",
        encoding="utf-8",
    )
    (tmp_path / "mind-mem.json").write_text(
        json.dumps({"recall": {"backend": "scan"}, "cache": {"enabled": False}, "served_ledger": {"enabled": True}}),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("indexed", [False, True])
def test_ranked_resource_preserves_list_and_records_exact_order(workspace: Path, indexed: bool) -> None:
    if indexed:
        config_path = workspace / "mind-mem.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["recall"]["backend"] = "sqlite"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        assert build_index(str(workspace))["blocks_indexed"] == 1
    with use_workspace(str(workspace)):
        results = json.loads(resources.get_recall("resource canary architecture"))
    assert isinstance(results, list)
    assert [hit["_id"] for hit in results] == ["D-20260914-001"]
    rows = read_served_runs(workspace)
    assert len(rows) == 1
    assert list(rows[0].ids) == [hit["_id"] for hit in results]
    assert verify_served_chain(workspace).ok


def test_indexed_resource_rechecks_admission_after_quarantine(workspace: Path) -> None:
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["backend"] = "sqlite"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert build_index(str(workspace))["blocks_indexed"] == 1
    path = workspace / "decisions/DECISIONS.md"
    with use_workspace(str(workspace)):
        before = json.loads(resources.get_recall("resource canary architecture"))
        assert [hit["_id"] for hit in before] == ["D-20260914-001"]
        path.write_text(path.read_text(encoding="utf-8").replace("Status: active", "Status: quarantined"), encoding="utf-8")
        after = json.loads(resources.get_recall("resource canary architecture"))
    assert after == []
    rows = read_served_runs(workspace)
    assert len(rows) == 2
    assert list(rows[0].ids) == ["D-20260914-001"]
    assert list(rows[-1].ids) == []
    assert verify_served_chain(workspace).ok


def test_resource_honors_explicit_ledger_opt_out(workspace: Path) -> None:
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["served_ledger"]["enabled"] = False
    config["recall"]["backend"] = "sqlite"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert build_index(str(workspace))["blocks_indexed"] == 1
    with use_workspace(str(workspace)):
        results = json.loads(resources.get_recall("resource canary architecture"))
    assert [hit["_id"] for hit in results] == ["D-20260914-001"]
    assert read_served_runs(workspace) == ()

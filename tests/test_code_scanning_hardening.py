"""Regression tests for the code-scanning findings closed together.

* path confinement in ``apply_engine._rollback_refusal`` (py/path-injection)
* no exception text in the ``scan`` result (py/stack-trace-exposure)
* runtime guards that must survive ``python -O`` (B101)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem import apply_engine, client_mcp, client_writers
from mind_mem.client_config_io import UnsafeConfigError
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import governance

TS = "20260101-120000"


def _applied_snapshot(tmp_path: Path) -> tuple[str, str]:
    """A workspace holding one legitimately applied snapshot; returns (ws, snap_dir)."""
    ws = tmp_path / "ws"
    snap = ws / "intelligence" / "applied" / TS
    snap.mkdir(parents=True)
    (snap / "APPLY_RECEIPT.md").write_text("Proposal: P-1\nResult: applied\n", encoding="utf-8")
    return str(ws), str(snap)


def test_rollback_refusal_accepts_the_legitimate_snapshot(tmp_path: Path) -> None:
    ws, snap = _applied_snapshot(tmp_path)
    reason, receipt = apply_engine._rollback_refusal(ws, snap, TS)
    assert reason is None  # positive control for the refusals below
    assert receipt == os.path.join(os.path.realpath(snap), "APPLY_RECEIPT.md")


@pytest.mark.parametrize("hostile", ["../../../..", "../../applied/../../..", "/etc"])
def test_rollback_refusal_confines_the_timestamp_to_the_applied_dir(tmp_path: Path, hostile: str) -> None:
    ws, snap = _applied_snapshot(tmp_path)
    reason, receipt = apply_engine._rollback_refusal(ws, snap, hostile)
    assert reason is not None
    assert "escape" in reason
    assert receipt is None


def test_rollback_refuses_a_symlinked_snapshot_dir(tmp_path: Path) -> None:
    ws, snap = _applied_snapshot(tmp_path)
    link = Path(ws) / "intelligence" / "applied" / "20260202-020202"
    try:
        link.symlink_to(snap, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    reason, _ = apply_engine._rollback_refusal(ws, str(link), "20260202-020202")
    assert reason is not None and "symlink" in reason


def test_rollback_refuses_a_symlinked_receipt(tmp_path: Path) -> None:
    ws, snap = _applied_snapshot(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("Proposal: P-9\nResult: applied\n", encoding="utf-8")
    os.remove(os.path.join(snap, "APPLY_RECEIPT.md"))
    try:
        os.symlink(outside, os.path.join(snap, "APPLY_RECEIPT.md"))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    reason, _ = apply_engine._rollback_refusal(ws, snap, TS)
    assert reason is not None and "symlink" in reason


def test_scan_does_not_leak_exception_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    (ws / "decisions" / "DECISIONS.md").write_text("", encoding="utf-8")
    (ws / "mind-mem.json").write_text(
        json.dumps({"version": "4.1.1", "v4": {"world_staleness": {"enabled": True}}}), encoding="utf-8"
    )
    import mind_mem.world_staleness as ws_mod

    def _boom(_ws: str) -> dict:
        raise RuntimeError("Traceback /srv/secret/path.py line 42 token=hunter2")

    monkeypatch.setattr(ws_mod, "world_staleness_summary", _boom)
    with use_workspace(str(ws)):
        raw = governance.scan()
    world = json.loads(raw)["checks"]["world_staleness"]
    assert world["error"], "the failure must still be reported"  # positive control
    assert "hunter2" not in raw and "/srv/secret" not in raw and "Traceback" not in raw


def test_install_mcp_config_raises_without_a_required_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(client_mcp, "_targets", lambda _spec, _ws: [])
    agent = next(a for a, s in client_mcp.AGENT_REGISTRY.items() if s.mcp_fmt and s.supported_here())
    with pytest.raises(RuntimeError):
        client_mcp.install_mcp_config(agent, str(tmp_path), dry_run=True)


def test_merge_json_mcp_raises_when_the_servers_node_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_writers, "_descend", lambda *_a, **_k: None)
    with pytest.raises(UnsafeConfigError):
        client_writers.merge_json_mcp(("mcpServers",), "stdio", {}, "/ws", {"command": "x", "args": []})

"""plan_consolidation scans a bounded window of blocks (``ORDER BY id LIMIT``).

A block that supersedes one inside the window can sort after it. The superseded
block must still never be proposed as a merge survivor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem import sqlite_index as si
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.consolidation import plan_consolidation

_DUP = "The weekly standup is on Tuesday at 10am Pacific."


def _workspace(tmp_path: Path, *, max_blocks: int) -> Path:
    ws = tmp_path / "ws"
    (ws / ".mind-mem-index").mkdir(parents=True)
    (ws / "decisions").mkdir(parents=True)
    cfg = {"v4": {"granularity_align": {"enabled": True, "max_blocks": max_blocks}}}
    (ws / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    return ws


def _seed(ws: Path, rows: list[tuple[str, str, dict]]) -> None:
    conn = si._connect(str(ws))
    si._init_schema(conn)
    for bid, status, blob in rows:
        conn.execute(
            "INSERT OR REPLACE INTO blocks (id,type,file,line,status,date,speaker,tags,dia_id,parent_id,json_blob) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (bid, "decision", "decisions/DECISIONS.md", 1, status, "2026-01-01", "", "", "", "", json.dumps(blob)),
        )
    conn.commit()
    conn.close()


def _candidates(ws: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(ws / "mind-mem.json"))
    with use_workspace(str(ws)):
        plan = json.loads(plan_consolidation())
    section = plan.get("granularity_align") or {}
    return section.get("candidates") or []


def _rows(superseder_status: str | None) -> list[tuple[str, str, dict]]:
    rows = [
        ("D-A1", "active", {"Statement": _DUP}),
        ("D-A2", "active", {"Statement": _DUP}),
        ("D-B1", "active", {"Statement": "Invoices are due on the first of the month."}),
    ]
    rows += [(f"D-M{i:02d}", "active", {"Statement": f"Filler fact number {i} about topic {i}."}) for i in range(6)]
    if superseder_status is not None:
        rows.append(("D-Z9", superseder_status, {"Statement": "The standup moved.", "Supersedes": "D-A2"}))
    return rows


def test_positive_control_duplicates_inside_the_window_are_proposed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path, max_blocks=3)
    _seed(ws, _rows(None))
    found = _candidates(ws, monkeypatch)
    assert len(found) == 1, found


def test_superseder_outside_the_window_still_blocks_the_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path, max_blocks=3)
    _seed(ws, _rows("active"))
    assert _candidates(ws, monkeypatch) == []


def test_a_quarantined_superseder_does_not_retire_a_live_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path, max_blocks=3)
    _seed(ws, _rows("quarantined"))
    assert len(_candidates(ws, monkeypatch)) == 1

"""``propose_edit`` must snapshot the block's real current content.

The snapshot ran ``SELECT content FROM blocks`` against ``<ws>/index.db``
— a table nothing in this package ever inserts into (``block_kinds``
creates it empty), guarded by a ternary that yielded ``None`` in silence.
So ``old_content`` was NULL for every proposal in a real workspace, and
that NULL is load-bearing downstream: it becomes version 1 of the chain
in :func:`block_versioning.block_history`, ``content_as_of`` returns
``None`` for every instant before the first applied edit, and the recall
projection reads that ``None`` as "this block has no recorded edits" —
serving today's content as history, which is precisely what the
versioning module says it refuses to do.

The blocks of record live in the configured block store, so that is where
the snapshot comes from now.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mind_mem.v4.block_versioning import block_history, content_as_of
from mind_mem.v4.self_editing import FLAG, approve_edit, get_edit, propose_edit

_STATEMENT = "the deployment target is us-east-1"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A flag-enabled workspace with one block in the Markdown corpus."""
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": {FLAG: {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        f"[D-20260613-001]\nStatement: {_STATEMENT}\nStatus: active\nDate: 2026-06-13\n\n---\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.unit
def test_the_proposal_records_the_content_it_is_replacing(workspace: Path) -> None:
    edit_id = propose_edit(workspace, "D-20260613-001", "the deployment target is eu-west-1", "region moved")
    edit = get_edit(workspace, edit_id)
    assert edit is not None
    assert edit.old_content == _STATEMENT


@pytest.mark.unit
def test_time_travel_can_reach_the_pre_edit_revision(workspace: Path) -> None:
    """The whole point of the snapshot: version 1 has real content."""
    edit_id = propose_edit(workspace, "D-20260613-001", "the deployment target is eu-west-1", "region moved")
    approve_edit(workspace, edit_id, approver="tester")

    history = block_history(workspace, "D-20260613-001")
    assert [v.version for v in history] == [1, 2]
    assert history[0].content == _STATEMENT
    assert history[1].content == "the deployment target is eu-west-1"
    # Before the edit took effect, as_of must return the OLD text — not
    # None, which the recall projection reads as "never edited" and
    # answers with today's content.
    assert content_as_of(workspace, "D-20260613-001", "2020-01-01T00:00:00+00:00") == _STATEMENT


@pytest.mark.unit
def test_a_block_the_store_cannot_resolve_still_records_the_proposal(workspace: Path) -> None:
    edit_id = propose_edit(workspace, "D-20260613-404", "invented", "no such block")
    edit = get_edit(workspace, edit_id)
    assert edit is not None
    assert edit.old_content is None  # unknown, and logged as such


@pytest.mark.unit
def test_a_populated_legacy_blocks_table_is_still_honoured(workspace: Path) -> None:
    """The old lookup stays as a fallback for a workspace that fills it."""
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS blocks (id TEXT PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO blocks (id, content) VALUES (?, ?)", ("D-20260613-777", "legacy text"))
        conn.commit()
    edit_id = propose_edit(workspace, "D-20260613-777", "new text", "legacy path")
    edit = get_edit(workspace, edit_id)
    assert edit is not None
    assert edit.old_content == "legacy text"

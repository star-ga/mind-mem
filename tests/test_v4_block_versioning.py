"""Tests for v4 block versioning + time-travel (Group B, ``v4.self_editing``).

Covers:

    block_history()      version chain materialised from applied edits —
                         ordering, v1-is-pre-edit-content, valid_from
                         provenance, and the exclusion of pending/rejected
                         edits (they never took effect, so they must never
                         appear in history)
    content_as_of()      point-in-time reconstruction — before the first
                         edit, exactly at an edit boundary, between edits,
                         and after the last edit
    versioned_block_ids() enumeration of what time-travel can answer for

Core honesty contract under test: history reports only content the workspace
actually had. A proposed-but-unapproved edit is not history, and a block with
no applied edits has an empty history (its current content lives in the block
store, not here) rather than a fabricated single version.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.block_versioning import (
    FLAG as BV_FLAG,
)
from mind_mem.v4.block_versioning import (
    BlockVersion,
    block_history,
    content_as_of,
    versioned_block_ids,
)
from mind_mem.v4.self_editing import (
    EditStatus,
    approve_edit,
    ensure_edit_schema,
    propose_edit,
    reject_edit,
)


def _cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: bool) -> Path:
    block = {k: {"enabled": v} for k, v in flags.items()}
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def bv_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{BV_FLAG: True})


@pytest.fixture
def bv_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{BV_FLAG: False})


def _seed_applied_edit(
    workspace: Path,
    block_id: str,
    old_content: str | None,
    new_content: str,
    approved_at: str,
    *,
    reason: str = "r",
    approver: str = "tester",
    status: str = EditStatus.APPLIED,
) -> int:
    """Insert an edit with a pinned approved_at so ordering is deterministic.

    Written directly rather than via propose/approve because those stamp
    wall-clock time, which cannot express the multi-day gaps these
    time-travel assertions need.
    """
    ensure_edit_schema(workspace)
    db = workspace / "index.db"
    with closing(sqlite3.connect(db, timeout=30)) as conn:
        cur = conn.execute(
            "INSERT INTO block_edits (block_id, old_content, new_content, reason, "
            "proposed_at, status, approved_at, approver) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (block_id, old_content, new_content, reason, approved_at, status, approved_at, approver),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def _seed_chain(workspace: Path, block_id: str = "B-1") -> None:
    """v1 'alpha' -> v2 'beta' (2026-01-02) -> v3 'gamma' (2026-03-04)."""
    _seed_applied_edit(workspace, block_id, "alpha", "beta", "2026-01-02T00:00:00+00:00")
    _seed_applied_edit(workspace, block_id, "beta", "gamma", "2026-03-04T00:00:00+00:00")


# ===========================================================================
# flag gating
# ===========================================================================


@pytest.mark.unit
def test_flag_off_blocks_history(bv_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        block_history(bv_off, "B-1")


@pytest.mark.unit
def test_flag_off_blocks_content_as_of(bv_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        content_as_of(bv_off, "B-1", "2026-01-01T00:00:00+00:00")


@pytest.mark.unit
def test_flag_off_blocks_versioned_block_ids(bv_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        versioned_block_ids(bv_off)


# ===========================================================================
# empty / absent cases
# ===========================================================================


@pytest.mark.unit
def test_history_empty_when_no_workspace_db(bv_on: Path) -> None:
    assert block_history(bv_on, "B-1") == []


@pytest.mark.unit
def test_history_empty_when_no_edits_table(bv_on: Path) -> None:
    (bv_on / "index.db").touch()
    assert block_history(bv_on, "B-1") == []


@pytest.mark.unit
def test_history_empty_when_block_has_no_applied_edits(bv_on: Path) -> None:
    """A block with only a pending edit has no history — nothing took effect."""
    propose_edit(bv_on, "B-1", "new content", "reason")
    assert block_history(bv_on, "B-1") == []


@pytest.mark.unit
def test_content_as_of_none_when_no_history(bv_on: Path) -> None:
    """None signals 'fall back to current content', not 'block was empty'."""
    assert content_as_of(bv_on, "B-1", "2026-01-01T00:00:00+00:00") is None


@pytest.mark.unit
def test_content_as_of_none_for_empty_timestamp(bv_on: Path) -> None:
    _seed_chain(bv_on)
    assert content_as_of(bv_on, "B-1", "") is None


# ===========================================================================
# version chain
# ===========================================================================


@pytest.mark.unit
def test_single_edit_yields_two_versions(bv_on: Path) -> None:
    _seed_applied_edit(bv_on, "B-1", "alpha", "beta", "2026-01-02T00:00:00+00:00")
    hist = block_history(bv_on, "B-1")
    assert [v.version for v in hist] == [1, 2]
    assert [v.content for v in hist] == ["alpha", "beta"]


@pytest.mark.unit
def test_chain_is_ordered_oldest_first(bv_on: Path) -> None:
    _seed_chain(bv_on)
    hist = block_history(bv_on, "B-1")
    assert [v.version for v in hist] == [1, 2, 3]
    assert [v.content for v in hist] == ["alpha", "beta", "gamma"]


@pytest.mark.unit
def test_version_one_has_no_valid_from_or_edit_id(bv_on: Path) -> None:
    """v1 predates the first recorded edit; inventing a timestamp would lie."""
    _seed_chain(bv_on)
    v1 = block_history(bv_on, "B-1")[0]
    assert isinstance(v1, BlockVersion)
    assert v1.valid_from is None
    assert v1.edit_id is None
    assert v1.reason is None
    assert v1.approver is None


@pytest.mark.unit
def test_later_versions_carry_edit_provenance(bv_on: Path) -> None:
    eid = _seed_applied_edit(bv_on, "B-1", "alpha", "beta", "2026-01-02T00:00:00+00:00", reason="typo", approver="alice")
    v2 = block_history(bv_on, "B-1")[1]
    assert v2.edit_id == eid
    assert v2.reason == "typo"
    assert v2.approver == "alice"
    assert v2.valid_from == "2026-01-02T00:00:00+00:00"


@pytest.mark.unit
def test_history_is_scoped_to_the_requested_block(bv_on: Path) -> None:
    _seed_chain(bv_on, "B-1")
    _seed_applied_edit(bv_on, "B-2", "other", "changed", "2026-02-02T00:00:00+00:00")
    assert [v.content for v in block_history(bv_on, "B-1")] == ["alpha", "beta", "gamma"]
    assert [v.content for v in block_history(bv_on, "B-2")] == ["other", "changed"]


@pytest.mark.unit
def test_null_pre_edit_content_is_preserved_as_none(bv_on: Path) -> None:
    """old_content NULL (block absent when proposed) stays None, not ''."""
    _seed_applied_edit(bv_on, "B-1", None, "beta", "2026-01-02T00:00:00+00:00")
    hist = block_history(bv_on, "B-1")
    assert hist[0].content is None
    assert hist[1].content == "beta"


# ===========================================================================
# pending / rejected are never history
# ===========================================================================


@pytest.mark.unit
def test_pending_edit_excluded_from_history(bv_on: Path) -> None:
    _seed_applied_edit(bv_on, "B-1", "alpha", "beta", "2026-01-02T00:00:00+00:00")
    _seed_applied_edit(bv_on, "B-1", "beta", "NEVER", "2026-05-05T00:00:00+00:00", status=EditStatus.PENDING)
    contents = [v.content for v in block_history(bv_on, "B-1")]
    assert "NEVER" not in contents
    assert contents == ["alpha", "beta"]


@pytest.mark.unit
def test_rejected_edit_excluded_from_history(bv_on: Path) -> None:
    _seed_applied_edit(bv_on, "B-1", "alpha", "beta", "2026-01-02T00:00:00+00:00")
    _seed_applied_edit(bv_on, "B-1", "beta", "REJECTED", "2026-05-05T00:00:00+00:00", status=EditStatus.REJECTED)
    contents = [v.content for v in block_history(bv_on, "B-1")]
    assert "REJECTED" not in contents


@pytest.mark.unit
def test_rejected_edit_absent_after_real_reject_flow(bv_on: Path) -> None:
    eid = propose_edit(bv_on, "B-1", "bad content", "should not stick")
    reject_edit(bv_on, eid)
    assert block_history(bv_on, "B-1") == []


@pytest.mark.unit
def test_approved_edit_appears_after_real_approve_flow(bv_on: Path) -> None:
    eid = propose_edit(bv_on, "B-1", "good content", "landing it")
    approve_edit(bv_on, eid, approver="bob")
    hist = block_history(bv_on, "B-1")
    assert [v.content for v in hist] == [None, "good content"]
    assert hist[1].approver == "bob"


# ===========================================================================
# time travel
# ===========================================================================


@pytest.mark.unit
def test_as_of_before_first_edit_returns_original(bv_on: Path) -> None:
    _seed_chain(bv_on)
    assert content_as_of(bv_on, "B-1", "2025-12-31T00:00:00+00:00") == "alpha"


@pytest.mark.unit
def test_as_of_exactly_at_edit_boundary_includes_that_edit(bv_on: Path) -> None:
    """valid_from is inclusive — at T the edit has taken effect."""
    _seed_chain(bv_on)
    assert content_as_of(bv_on, "B-1", "2026-01-02T00:00:00+00:00") == "beta"


@pytest.mark.unit
def test_as_of_one_instant_before_boundary_excludes_that_edit(bv_on: Path) -> None:
    _seed_chain(bv_on)
    assert content_as_of(bv_on, "B-1", "2026-01-01T23:59:59+00:00") == "alpha"


@pytest.mark.unit
def test_as_of_between_edits_returns_middle_version(bv_on: Path) -> None:
    _seed_chain(bv_on)
    assert content_as_of(bv_on, "B-1", "2026-02-15T12:00:00+00:00") == "beta"


@pytest.mark.unit
def test_as_of_after_last_edit_returns_latest(bv_on: Path) -> None:
    _seed_chain(bv_on)
    assert content_as_of(bv_on, "B-1", "2027-01-01T00:00:00+00:00") == "gamma"


@pytest.mark.unit
def test_as_of_ignores_pending_edit_in_the_window(bv_on: Path) -> None:
    _seed_chain(bv_on)
    _seed_applied_edit(bv_on, "B-1", "gamma", "NEVER", "2026-04-04T00:00:00+00:00", status=EditStatus.PENDING)
    assert content_as_of(bv_on, "B-1", "2026-06-01T00:00:00+00:00") == "gamma"


# ===========================================================================
# enumeration
# ===========================================================================


@pytest.mark.unit
def test_versioned_block_ids_lists_only_blocks_with_applied_edits(bv_on: Path) -> None:
    _seed_chain(bv_on, "B-1")
    _seed_applied_edit(bv_on, "B-2", "x", "y", "2026-02-02T00:00:00+00:00")
    propose_edit(bv_on, "B-3", "pending only", "reason")
    ids = versioned_block_ids(bv_on)
    assert "B-1" in ids
    assert "B-2" in ids
    assert "B-3" not in ids


@pytest.mark.unit
def test_versioned_block_ids_ordered_by_first_edit(bv_on: Path) -> None:
    _seed_applied_edit(bv_on, "B-late", "x", "y", "2026-09-09T00:00:00+00:00")
    _seed_applied_edit(bv_on, "B-early", "x", "y", "2026-01-01T00:00:00+00:00")
    assert versioned_block_ids(bv_on) == ["B-early", "B-late"]


@pytest.mark.unit
def test_versioned_block_ids_respects_limit(bv_on: Path) -> None:
    _seed_applied_edit(bv_on, "B-1", "x", "y", "2026-01-01T00:00:00+00:00")
    _seed_applied_edit(bv_on, "B-2", "x", "y", "2026-02-02T00:00:00+00:00")
    assert versioned_block_ids(bv_on, limit=1) == ["B-1"]
    assert versioned_block_ids(bv_on, limit=0) == []


@pytest.mark.unit
def test_versioned_block_ids_empty_without_db(bv_on: Path) -> None:
    assert versioned_block_ids(bv_on) == []

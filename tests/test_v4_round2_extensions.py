"""Tests for round-2 audit extensions: federation, self_editing."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.federation import FLAG as FED_FLAG
from mind_mem.v4.federation import (
    ConflictReport,
    MergeStrategy,
    Resolution,
    detect_conflict,
    ensure_federation_schema,
    get_version_vector,
    list_conflicts,
    record_agent_write,
    resolve_conflict,
)
from mind_mem.v4.self_editing import FLAG as SE_FLAG
from mind_mem.v4.self_editing import (
    Edit,
    EditStatus,
    approve_edit,
    ensure_edit_schema,
    get_edit,
    list_edit_history,
    list_pending_edits,
    propose_edit,
    reject_edit,
)


def _cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: bool) -> Path:
    block = {k: {"enabled": v} for k, v in flags.items()}
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


# ===========================================================================
# federation.py
# ===========================================================================


@pytest.fixture
def fed_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{FED_FLAG: True})


@pytest.fixture
def fed_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{FED_FLAG: False})


@pytest.mark.unit
def test_fed_flag_off_blocks_record(fed_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        record_agent_write(fed_off, "B-1", "agent-A")


@pytest.mark.unit
def test_fed_flag_off_blocks_resolve(fed_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        resolve_conflict(fed_off, "B-1", MergeStrategy.LAST_WRITER_WINS)


@pytest.mark.unit
def test_record_agent_write_increments_per_agent(fed_on: Path) -> None:
    assert record_agent_write(fed_on, "B-1", "agent-A") == 1
    assert record_agent_write(fed_on, "B-1", "agent-A") == 2
    assert record_agent_write(fed_on, "B-1", "agent-A") == 3
    # Different agent gets its own independent clock.
    assert record_agent_write(fed_on, "B-1", "agent-B") == 1


@pytest.mark.unit
def test_get_version_vector_returns_per_agent_map(fed_on: Path) -> None:
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    assert get_version_vector(fed_on, "B-1") == {"agent-A": 2, "agent-B": 1}


@pytest.mark.unit
def test_get_version_vector_empty_for_unknown_block(fed_on: Path) -> None:
    ensure_federation_schema(fed_on)
    assert get_version_vector(fed_on, "B-never") == {}


@pytest.mark.unit
def test_detect_conflict_returns_none_for_single_agent(fed_on: Path) -> None:
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    assert detect_conflict(fed_on, "B-1") is None


@pytest.mark.unit
def test_detect_conflict_surfaces_divergence(fed_on: Path) -> None:
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    report = detect_conflict(fed_on, "B-1")
    assert report is not None
    assert report.left_agent == "agent-A"
    assert report.left_version == 3
    assert report.right_agent == "agent-B"
    assert report.right_version == 1


@pytest.mark.unit
def test_detect_conflict_returns_none_when_versions_tied(fed_on: Path) -> None:
    """Equal logical clocks across agents = no divergence (yet)."""
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    assert detect_conflict(fed_on, "B-1") is None


@pytest.mark.unit
def test_resolve_conflict_last_writer_wins(fed_on: Path) -> None:
    # Audit FP-1: LAST_WRITER_WINS now resolves by wall-clock
    # ``last_seen_at`` (semantic the public name has always promised),
    # NOT by highest logical version (which was the pre-fix collapsed
    # behaviour identical to HIGHER_VERSION). agent-A bumps twice, then
    # agent-B bumps once — agent-B's last_seen_at is the most recent,
    # so it wins even though agent-A has the higher version.
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    resolution = resolve_conflict(fed_on, "B-1", MergeStrategy.LAST_WRITER_WINS)
    assert isinstance(resolution, Resolution)
    assert resolution.winner_agent == "agent-B"
    assert resolution.winner_version == 1
    assert resolution.merged_payload is None


@pytest.mark.unit
def test_resolve_conflict_higher_version_picks_highest_version(fed_on: Path) -> None:
    # The new HIGHER_VERSION is the previous (collapsed) behaviour of
    # LAST_WRITER_WINS — pick the agent with the largest logical clock.
    # See audit FP-1: the two strategies are now semantically distinct.
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    resolution = resolve_conflict(fed_on, "B-1", MergeStrategy.HIGHER_VERSION)
    assert isinstance(resolution, Resolution)
    assert resolution.winner_agent == "agent-A"
    assert resolution.winner_version == 2
    assert resolution.merged_payload is None


@pytest.mark.unit
def test_resolve_conflict_three_way_merge_invokes_merger(fed_on: Path) -> None:
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    record_agent_write(fed_on, "B-1", "agent-B")
    captured: list[ConflictReport] = []

    def merger(report: ConflictReport) -> bytes:
        captured.append(report)
        return b"merged-payload"

    resolution = resolve_conflict(fed_on, "B-1", MergeStrategy.THREE_WAY_MERGE, merger=merger)
    assert resolution is not None
    assert resolution.merged_payload == b"merged-payload"
    assert resolution.winner_agent.startswith("merge:")
    assert len(captured) == 1


@pytest.mark.unit
def test_resolve_conflict_three_way_merge_without_merger_raises(fed_on: Path) -> None:
    # Audit FP-4: THREE_WAY_MERGE without merger used to return ``None``,
    # indistinguishable from "no conflict found". Now raises ValueError
    # so a misconfigured caller surfaces a real error instead of silently
    # leaving the conflict open.
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    record_agent_write(fed_on, "B-1", "agent-B")
    with pytest.raises(ValueError, match="merger callable"):
        resolve_conflict(fed_on, "B-1", MergeStrategy.THREE_WAY_MERGE)


@pytest.mark.unit
def test_resolve_conflict_returns_none_when_no_conflict(fed_on: Path) -> None:
    record_agent_write(fed_on, "B-1", "agent-A")
    assert resolve_conflict(fed_on, "B-1", MergeStrategy.LAST_WRITER_WINS) is None


@pytest.mark.unit
def test_list_conflicts_returns_open_only(fed_on: Path) -> None:
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    detect_conflict(fed_on, "B-1")  # logs the conflict
    assert len(list_conflicts(fed_on)) == 1
    resolve_conflict(fed_on, "B-1", MergeStrategy.LAST_WRITER_WINS)
    # After resolution, list_conflicts shows zero open.
    assert list_conflicts(fed_on) == []


@pytest.mark.unit
def test_log_conflict_dedupes(fed_on: Path) -> None:
    """Calling detect_conflict repeatedly on unchanged state doesn't multiply log rows."""
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    detect_conflict(fed_on, "B-1")
    detect_conflict(fed_on, "B-1")
    detect_conflict(fed_on, "B-1")
    assert len(list_conflicts(fed_on)) == 1


# ===========================================================================
# self_editing.py
# ===========================================================================


@pytest.fixture
def se_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{SE_FLAG: True})


@pytest.fixture
def se_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{SE_FLAG: False})


@pytest.mark.unit
def test_se_flag_off_blocks_propose(se_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        propose_edit(se_off, "B-1", "new", "reason")


@pytest.mark.unit
def test_propose_edit_requires_reason(se_on: Path) -> None:
    with pytest.raises(ValueError):
        propose_edit(se_on, "B-1", "new", "")
    with pytest.raises(ValueError):
        propose_edit(se_on, "B-1", "new", "   ")


@pytest.mark.unit
def test_propose_edit_returns_id(se_on: Path) -> None:
    eid = propose_edit(se_on, "B-1", "new content", "fixing typo")
    assert isinstance(eid, int)
    assert eid > 0


@pytest.mark.unit
def test_propose_then_get_edit(se_on: Path) -> None:
    eid = propose_edit(se_on, "B-1", "new content", "fixing typo")
    e = get_edit(se_on, eid)
    assert isinstance(e, Edit)
    assert e.block_id == "B-1"
    assert e.new_content == "new content"
    assert e.reason == "fixing typo"
    assert e.status == EditStatus.PENDING
    assert e.approved_at is None


@pytest.mark.unit
def test_approve_edit_transitions_to_applied(se_on: Path) -> None:
    eid = propose_edit(se_on, "B-1", "new", "reason")
    e = approve_edit(se_on, eid, approver="alice")
    assert e is not None
    assert e.status == EditStatus.APPLIED
    assert e.approver == "alice"
    assert e.approved_at is not None


@pytest.mark.unit
def test_reject_edit_transitions_to_rejected(se_on: Path) -> None:
    eid = propose_edit(se_on, "B-1", "new", "reason")
    e = reject_edit(se_on, eid, approver="bob")
    assert e is not None
    assert e.status == EditStatus.REJECTED


@pytest.mark.unit
def test_double_approve_returns_none(se_on: Path) -> None:
    """Approving an already-applied edit is a no-op (returns None)."""
    eid = propose_edit(se_on, "B-1", "new", "reason")
    approve_edit(se_on, eid)
    second = approve_edit(se_on, eid)
    assert second is None


@pytest.mark.unit
def test_approve_unknown_edit_returns_none(se_on: Path) -> None:
    ensure_edit_schema(se_on)
    assert approve_edit(se_on, 99999) is None


@pytest.mark.unit
def test_list_pending_edits_filters_to_pending(se_on: Path) -> None:
    e1 = propose_edit(se_on, "B-1", "x", "r1")
    propose_edit(se_on, "B-2", "y", "r2")
    e3 = propose_edit(se_on, "B-3", "z", "r3")
    approve_edit(se_on, e1)
    reject_edit(se_on, e3)
    pending = list_pending_edits(se_on)
    assert len(pending) == 1
    assert pending[0].block_id == "B-2"


@pytest.mark.unit
def test_list_edit_history_returns_all_for_block(se_on: Path) -> None:
    propose_edit(se_on, "B-1", "v1", "r1")
    propose_edit(se_on, "B-1", "v2", "r2")
    propose_edit(se_on, "B-2", "z", "r")
    history = list_edit_history(se_on, "B-1")
    assert len(history) == 2
    assert all(e.block_id == "B-1" for e in history)
    # Oldest first.
    assert history[0].new_content == "v1"


@pytest.mark.unit
def test_propose_edit_captures_old_content(se_on: Path) -> None:
    """When blocks table has the row, old_content is snapshotted for audit."""
    db = se_on / "index.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO blocks (id, content) VALUES ('B-1', 'original')")
        conn.commit()
    eid = propose_edit(se_on, "B-1", "updated", "fixing typo")
    e = get_edit(se_on, eid)
    assert e is not None
    assert e.old_content == "original"

"""Tests for round-2 audit extensions: federation, embedding_pipeline,
kind_summaries, self_editing."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.embedding_pipeline import FLAG as EMBED_FLAG
from mind_mem.v4.embedding_pipeline import (
    default_embedder,
    derive_embedding,
    derive_embeddings,
    set_embedder,
)
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
from mind_mem.v4.kind_summaries import FLAG as KS_FLAG
from mind_mem.v4.kind_summaries import (
    KindSummary,
    default_summariser,
    ensure_kind_summary_schema,
    get_summary,
    list_summaries,
    refresh_summary,
    set_summariser,
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
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    resolution = resolve_conflict(fed_on, "B-1", MergeStrategy.LAST_WRITER_WINS)
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
def test_resolve_conflict_three_way_merge_without_merger_returns_none(fed_on: Path) -> None:
    record_agent_write(fed_on, "B-1", "agent-A")
    record_agent_write(fed_on, "B-1", "agent-B")
    record_agent_write(fed_on, "B-1", "agent-B")
    assert resolve_conflict(fed_on, "B-1", MergeStrategy.THREE_WAY_MERGE) is None


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
# embedding_pipeline.py
# ===========================================================================


@pytest.fixture
def emb_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{EMBED_FLAG: True})


@pytest.fixture
def emb_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{EMBED_FLAG: False})


@pytest.mark.unit
def test_emb_flag_off_blocks_derive(emb_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        derive_embedding("hello")


@pytest.mark.unit
def test_default_embedder_returns_unit_norm(emb_on: Path) -> None:
    vec = default_embedder("hello world", dim=64)
    norm = sum(x * x for x in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_default_embedder_deterministic(emb_on: Path) -> None:
    """Cross-process-stable hashing means same input → same output."""
    a = default_embedder("hello world", dim=64)
    b = default_embedder("hello world", dim=64)
    assert a == b


@pytest.mark.unit
def test_default_embedder_different_inputs_differ(emb_on: Path) -> None:
    a = default_embedder("hello", dim=64)
    b = default_embedder("goodbye", dim=64)
    assert a != b


@pytest.mark.unit
def test_default_embedder_empty_input_zero_vec(emb_on: Path) -> None:
    vec = default_embedder("", dim=32)
    assert vec == [0.0] * 32


@pytest.mark.unit
def test_default_embedder_short_input(emb_on: Path) -> None:
    """Strings shorter than 3 chars should still produce non-zero vectors."""
    vec = default_embedder("hi", dim=32)
    assert any(v != 0.0 for v in vec)


@pytest.mark.unit
def test_set_embedder_swaps_implementation(emb_on: Path) -> None:
    def fake(text: str, dim: int) -> list[float]:
        return [42.0] * dim

    original = default_embedder
    set_embedder(fake)
    try:
        assert derive_embedding("anything", dim=4) == [42.0, 42.0, 42.0, 42.0]
    finally:
        set_embedder(original)


@pytest.mark.unit
def test_derive_embeddings_from_workspace(emb_on: Path) -> None:
    db = emb_on / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT)")
        conn.executemany(
            "INSERT INTO blocks (id, content) VALUES (?, ?)",
            [("B-1", "hello world"), ("B-2", "different content")],
        )
        conn.commit()
    out = derive_embeddings(emb_on, ["B-1", "B-2", "B-missing"], dim=32)
    assert "B-1" in out
    assert "B-2" in out
    assert "B-missing" not in out
    assert out["B-1"] != out["B-2"]


@pytest.mark.unit
def test_derive_embeddings_empty_for_missing_db(emb_on: Path) -> None:
    assert derive_embeddings(emb_on, ["B-1"]) == {}


@pytest.mark.unit
def test_derive_embeddings_empty_for_no_blocks_table(emb_on: Path) -> None:
    db = emb_on / "index.db"
    sqlite3.connect(db).close()
    assert derive_embeddings(emb_on, ["B-1"]) == {}


# ===========================================================================
# kind_summaries.py
# ===========================================================================


@pytest.fixture
def ks_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{KS_FLAG: True})


@pytest.fixture
def ks_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{KS_FLAG: False})


@pytest.mark.unit
def test_ks_flag_off_blocks_refresh(ks_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        refresh_summary(ks_off, "entity")


@pytest.mark.unit
def test_default_summariser_concatenates_heads() -> None:
    out = default_summariser(["alpha line\nbeta", "gamma line", "delta line"])
    assert "alpha line" in out
    assert "gamma line" in out
    assert "delta line" in out
    # First-line-only — second line of "alpha\nbeta" doesn't appear.
    assert "beta" not in out


@pytest.mark.unit
def test_default_summariser_skips_empty() -> None:
    out = default_summariser(["", "  ", "real content"])
    assert "real content" in out


@pytest.mark.unit
def test_refresh_summary_empty_kind_returns_none(ks_on: Path) -> None:
    db = ks_on / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT NOT NULL DEFAULT 'unspecified')")
        conn.commit()
    assert refresh_summary(ks_on, "entity") is None


@pytest.mark.unit
def test_refresh_summary_writes_record(ks_on: Path) -> None:
    db = ks_on / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT NOT NULL DEFAULT 'unspecified')")
        conn.executemany(
            "INSERT INTO blocks (id, content, kind) VALUES (?, ?, ?)",
            [
                ("B-1", "first entity", "entity"),
                ("B-2", "second entity", "entity"),
                ("B-3", "a concept block", "concept"),
            ],
        )
        conn.commit()
    summary = refresh_summary(ks_on, "entity")
    assert isinstance(summary, KindSummary)
    assert summary.kind == "entity"
    assert summary.block_count == 2
    assert "first entity" in summary.summary
    assert "second entity" in summary.summary
    # concept blocks aren't in entity summary.
    assert "a concept block" not in summary.summary


@pytest.mark.unit
def test_get_summary_round_trips(ks_on: Path) -> None:
    db = ks_on / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT NOT NULL DEFAULT 'unspecified')")
        conn.execute("INSERT INTO blocks (id, content, kind) VALUES ('B-1', 'hello', 'entity')")
        conn.commit()
    refresh_summary(ks_on, "entity")
    summary = get_summary(ks_on, "entity")
    assert summary is not None
    assert summary.kind == "entity"


@pytest.mark.unit
def test_list_summaries_orders_by_kind(ks_on: Path) -> None:
    db = ks_on / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT NOT NULL DEFAULT 'unspecified')")
        conn.executemany(
            "INSERT INTO blocks (id, content, kind) VALUES (?, ?, ?)",
            [("B-1", "x", "entity"), ("B-2", "y", "concept")],
        )
        conn.commit()
    refresh_summary(ks_on, "entity")
    refresh_summary(ks_on, "concept")
    summaries = list_summaries(ks_on)
    assert [s.kind for s in summaries] == ["concept", "entity"]


@pytest.mark.unit
def test_set_summariser_swaps_implementation(ks_on: Path) -> None:
    def fake(_blocks: object) -> str:
        return "FAKE-SUMMARY"

    original = default_summariser
    set_summariser(fake)
    try:
        db = ks_on / "index.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT NOT NULL DEFAULT 'unspecified')")
            conn.execute("INSERT INTO blocks (id, content, kind) VALUES ('B-1', 'x', 'entity')")
            conn.commit()
        s = refresh_summary(ks_on, "entity")
        assert s is not None
        assert s.summary == "FAKE-SUMMARY"
    finally:
        set_summariser(original)


@pytest.mark.unit
def test_get_summary_none_when_missing(ks_on: Path) -> None:
    ensure_kind_summary_schema(ks_on)
    assert get_summary(ks_on, "absent") is None


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
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO blocks (id, content) VALUES ('B-1', 'original')")
        conn.commit()
    eid = propose_edit(se_on, "B-1", "updated", "fixing typo")
    e = get_edit(se_on, eid)
    assert e is not None
    assert e.old_content == "original"

"""Tests for the v4 block-kind taxonomy module."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.block_kinds import (
    ALLOWED_KINDS,
    DEFAULT_KIND,
    FLAG,
    BlockKind,
    ensure_block_kind_column,
    get_block_kind,
    list_blocks_by_kind,
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def workspace_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {FLAG: {"enabled": False}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


# ---------------------------------------------------------------------------
# Enum + registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_blockkind_has_eight_kinds_plus_unspecified() -> None:
    """Spec says eight typed kinds; 'unspecified' is the v3-compat fallback."""
    assert {k.value for k in BlockKind} == {
        "entity",
        "concept",
        "source",
        "synthesis",
        "image",
        "audio",
        "code",
        "structured",
        "unspecified",
    }


@pytest.mark.unit
def test_default_kind_is_unspecified() -> None:
    assert DEFAULT_KIND is BlockKind.UNSPECIFIED


@pytest.mark.unit
def test_allowed_kinds_matches_enum() -> None:
    assert ALLOWED_KINDS == frozenset(k.value for k in BlockKind)


@pytest.mark.unit
def test_unknown_string_constructor_raises() -> None:
    with pytest.raises(ValueError):
        BlockKind("dataset")  # not in the canonical set


# ---------------------------------------------------------------------------
# Feature-flag gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flag_off_raises_on_schema(workspace_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        ensure_block_kind_column(workspace_off)


@pytest.mark.unit
def test_flag_off_raises_on_get(workspace_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        get_block_kind(workspace_off, "B-001")


@pytest.mark.unit
def test_flag_off_raises_on_list(workspace_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        list_blocks_by_kind(workspace_off, BlockKind.ENTITY)


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_alter_adds_kind_column_to_fresh_table(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(blocks)")}
    assert "kind" in cols


@pytest.mark.unit
def test_alter_is_idempotent(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    ensure_block_kind_column(workspace)  # second call must not raise
    ensure_block_kind_column(workspace)
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        # Index exists exactly once (CREATE INDEX IF NOT EXISTS).
        idxs = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_blocks_kind'").fetchall()
    assert idxs == [("idx_blocks_kind",)]


@pytest.mark.unit
def test_alter_preserves_existing_v3_rows(workspace: Path) -> None:
    """v3-style rows survive the ALTER and pick up the unspecified default."""
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT)")
        conn.executemany(
            "INSERT INTO blocks (id, content) VALUES (?, ?)",
            [("B-1", "alpha"), ("B-2", "beta")],
        )
        conn.commit()
    ensure_block_kind_column(workspace)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT id, content, kind FROM blocks ORDER BY id").fetchall()
    assert rows == [("B-1", "alpha", "unspecified"), ("B-2", "beta", "unspecified")]


@pytest.mark.unit
def test_alter_skips_when_kind_already_present(workspace: Path) -> None:
    """If a sibling system already added the column, we don't re-ALTER."""
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT NOT NULL DEFAULT 'entity')")
        conn.execute("INSERT INTO blocks (id, content) VALUES ('B-1', 'a')")
        conn.commit()
    ensure_block_kind_column(workspace)  # must not raise duplicate-column
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT id, kind FROM blocks").fetchall()
    assert rows == [("B-1", "entity")]


# ---------------------------------------------------------------------------
# Reader defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_unknown_block_returns_default(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    assert get_block_kind(workspace, "B-never-seen") is DEFAULT_KIND


@pytest.mark.unit
def test_get_returns_default_when_db_missing(workspace: Path) -> None:
    assert get_block_kind(workspace, "B-1") is DEFAULT_KIND


@pytest.mark.unit
def test_get_returns_default_when_kind_column_missing(workspace: Path) -> None:
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO blocks (id) VALUES ('B-1')")
        conn.commit()
    assert get_block_kind(workspace, "B-1") is DEFAULT_KIND


@pytest.mark.unit
def test_get_round_trips_each_kind(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO blocks (id, content, kind) VALUES (?, '', ?)",
            [
                ("B-ent", "entity"),
                ("B-con", "concept"),
                ("B-src", "source"),
                ("B-syn", "synthesis"),
                ("B-img", "image"),
                ("B-aud", "audio"),
                ("B-cod", "code"),
                ("B-str", "structured"),
                ("B-uns", "unspecified"),
            ],
        )
        conn.commit()
    expected = [
        ("B-ent", BlockKind.ENTITY),
        ("B-con", BlockKind.CONCEPT),
        ("B-src", BlockKind.SOURCE),
        ("B-syn", BlockKind.SYNTHESIS),
        ("B-img", BlockKind.IMAGE),
        ("B-aud", BlockKind.AUDIO),
        ("B-cod", BlockKind.CODE),
        ("B-str", BlockKind.STRUCTURED),
        ("B-uns", BlockKind.UNSPECIFIED),
    ]
    for bid, kind in expected:
        assert get_block_kind(workspace, bid) is kind


@pytest.mark.unit
def test_get_corrupt_kind_falls_back_to_default(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO blocks (id, content, kind) VALUES ('B-bad', '', 'dataset')")
        conn.commit()
    assert get_block_kind(workspace, "B-bad") is DEFAULT_KIND


# ---------------------------------------------------------------------------
# list_blocks_by_kind
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_filters_to_kind(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO blocks (id, content, kind) VALUES (?, '', ?)",
            [
                ("B-e1", "entity"),
                ("B-e2", "entity"),
                ("B-c1", "concept"),
                ("B-s1", "source"),
            ],
        )
        conn.commit()
    assert sorted(list_blocks_by_kind(workspace, BlockKind.ENTITY)) == ["B-e1", "B-e2"]
    assert list_blocks_by_kind(workspace, BlockKind.CONCEPT) == ["B-c1"]
    assert list_blocks_by_kind(workspace, BlockKind.SOURCE) == ["B-s1"]


@pytest.mark.unit
def test_list_accepts_string_kind(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO blocks (id, content, kind) VALUES ('B-e', '', 'entity')")
        conn.commit()
    assert list_blocks_by_kind(workspace, "entity") == ["B-e"]


@pytest.mark.unit
def test_list_rejects_invalid_kind_string(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    with pytest.raises(ValueError):
        list_blocks_by_kind(workspace, "dataset")


@pytest.mark.unit
def test_list_respects_limit(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        rows = [(f"B-{i}", "", "entity") for i in range(20)]
        conn.executemany("INSERT INTO blocks (id, content, kind) VALUES (?, ?, ?)", rows)
        conn.commit()
    out = list_blocks_by_kind(workspace, BlockKind.ENTITY, limit=5)
    assert len(out) == 5


@pytest.mark.unit
def test_list_limit_zero_returns_empty(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    assert list_blocks_by_kind(workspace, BlockKind.ENTITY, limit=0) == []


@pytest.mark.unit
def test_list_negative_limit_returns_empty(workspace: Path) -> None:
    ensure_block_kind_column(workspace)
    assert list_blocks_by_kind(workspace, BlockKind.ENTITY, limit=-3) == []


@pytest.mark.unit
def test_list_empty_when_db_missing(workspace: Path) -> None:
    assert list_blocks_by_kind(workspace, BlockKind.ENTITY) == []


@pytest.mark.unit
def test_list_empty_when_kind_column_missing(workspace: Path) -> None:
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO blocks (id) VALUES ('B-1')")
        conn.commit()
    assert list_blocks_by_kind(workspace, BlockKind.ENTITY) == []

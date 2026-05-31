"""Tests for the v4 recall-tier module.

Covers feature-flag gating, schema bootstrap idempotency, the per-block
reader's defaulting behaviour, and the tier-filtered list ordering.
The recall-tier write surface (promotion / demotion / surprise re-promo)
lands in subsequent v4 iterations and gets its own tests then.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.tier_memory import (
    DEFAULT_TIER_CONFIG,
    FLAG,
    RecallTier,
    StaleVersionError,
    TierConfig,
    _load_config,
    ensure_recall_tier_schema,
    get_recall_tier,
    get_tier_version,
    list_blocks_in_recall_tier,
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Workspace dir with a ``mind-mem.json`` that toggles the flag."""
    cfg = {"v4": {FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def workspace_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Workspace with the flag explicitly disabled."""
    cfg = {"v4": {FLAG: {"enabled": False}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


# ---------------------------------------------------------------------------
# Feature-flag gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flag_off_raises_on_schema(workspace_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        ensure_recall_tier_schema(workspace_off)


@pytest.mark.unit
def test_flag_off_raises_on_get(workspace_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        get_recall_tier(workspace_off, "B-001")


@pytest.mark.unit
def test_flag_off_raises_on_list(workspace_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        list_blocks_in_recall_tier(workspace_off, RecallTier.HOT)


@pytest.mark.unit
def test_no_config_file_keeps_flag_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absence of mind-mem.json means absent flag means raise."""
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "absent.json"))
    with pytest.raises(FeatureDisabledError):
        get_recall_tier(tmp_path, "B-001")


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schema_creates_table(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    db = workspace / "index.db"
    assert db.is_file()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='block_recall_tier'").fetchall()
    assert rows == [("block_recall_tier",)]


@pytest.mark.unit
def test_schema_is_idempotent(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    ensure_recall_tier_schema(workspace)  # second call must not raise
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(workspace / "index.db") as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_block_recall_tier_tier'").fetchall()
    assert rows == [("idx_block_recall_tier_tier",)]


@pytest.mark.unit
def test_schema_creates_parent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deep = tmp_path / "a" / "b" / "c"
    cfg = {"v4": {FLAG: {"enabled": True}}}
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    ensure_recall_tier_schema(deep)
    assert (deep / "index.db").is_file()


# ---------------------------------------------------------------------------
# Reader defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_unknown_block_defaults_to_warm(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    assert get_recall_tier(workspace, "B-never-seen") is RecallTier.WARM


@pytest.mark.unit
def test_get_returns_warm_when_db_missing(workspace: Path) -> None:
    """Pre-schema reads default to WARM rather than raising."""
    # No ensure_recall_tier_schema call => no index.db.
    assert get_recall_tier(workspace, "B-001") is RecallTier.WARM


@pytest.mark.unit
def test_get_returns_warm_when_table_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """index.db exists but block_recall_tier table doesn't => WARM, not crash."""
    cfg = {"v4": {FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    db = tmp_path / "index.db"
    sqlite3.connect(db).close()  # touch
    assert get_recall_tier(tmp_path, "B-001") is RecallTier.WARM


@pytest.mark.unit
def test_get_round_trips_each_tier(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES (?, ?, '2026-05-10T00:00:00Z')",
            [("B-hot", "hot"), ("B-warm", "warm"), ("B-cold", "cold")],
        )
        conn.commit()
    assert get_recall_tier(workspace, "B-hot") is RecallTier.HOT
    assert get_recall_tier(workspace, "B-warm") is RecallTier.WARM
    assert get_recall_tier(workspace, "B-cold") is RecallTier.COLD


@pytest.mark.unit
def test_get_corrupt_tier_falls_back_to_warm(workspace: Path) -> None:
    """A garbage tier value in the table doesn't crash the reader."""
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(workspace / "index.db") as conn:
        conn.execute(
            "INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES (?, ?, '2026-05-10T00:00:00Z')",
            ("B-bad", "lukewarm"),
        )
        conn.commit()
    assert get_recall_tier(workspace, "B-bad") is RecallTier.WARM


# ---------------------------------------------------------------------------
# list_blocks_in_recall_tier
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_returns_oldest_first(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(workspace / "index.db") as conn:
        conn.executemany(
            "INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES (?, 'warm', ?)",
            [
                ("B-newest", "2026-05-10T05:00:00Z"),
                ("B-oldest", "2026-05-09T05:00:00Z"),
                ("B-mid", "2026-05-09T18:00:00Z"),
            ],
        )
        conn.commit()
    assert list_blocks_in_recall_tier(workspace, RecallTier.WARM) == [
        "B-oldest",
        "B-mid",
        "B-newest",
    ]


@pytest.mark.unit
def test_list_filters_to_tier(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(workspace / "index.db") as conn:
        conn.executemany(
            "INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES (?, ?, '2026-05-10T00:00:00Z')",
            [("B-h1", "hot"), ("B-h2", "hot"), ("B-w1", "warm"), ("B-c1", "cold")],
        )
        conn.commit()
    assert sorted(list_blocks_in_recall_tier(workspace, RecallTier.HOT)) == ["B-h1", "B-h2"]
    assert list_blocks_in_recall_tier(workspace, RecallTier.WARM) == ["B-w1"]
    assert list_blocks_in_recall_tier(workspace, RecallTier.COLD) == ["B-c1"]


@pytest.mark.unit
def test_list_accepts_string_tier(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(workspace / "index.db") as conn:
        conn.execute("INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES ('B-x', 'hot', '2026-05-10T00:00:00Z')")
        conn.commit()
    assert list_blocks_in_recall_tier(workspace, "hot") == ["B-x"]


@pytest.mark.unit
def test_list_rejects_invalid_tier_string(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    with pytest.raises(ValueError):
        list_blocks_in_recall_tier(workspace, "molten")


@pytest.mark.unit
def test_list_respects_limit(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(workspace / "index.db") as conn:
        rows = [(f"B-{i}", "hot", f"2026-05-10T00:00:{i:02}Z") for i in range(50)]
        conn.executemany(
            "INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
    out = list_blocks_in_recall_tier(workspace, RecallTier.HOT, limit=5)
    assert len(out) == 5
    assert out == [f"B-{i}" for i in range(5)]  # oldest 5 by timestamp


@pytest.mark.unit
def test_list_empty_when_table_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"v4": {FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    sqlite3.connect(tmp_path / "index.db").close()  # touch DB without table
    assert list_blocks_in_recall_tier(tmp_path, RecallTier.HOT) == []


@pytest.mark.unit
def test_list_empty_when_db_missing(workspace: Path) -> None:
    # No ensure_recall_tier_schema => no DB, no rows.
    assert list_blocks_in_recall_tier(workspace, RecallTier.HOT) == []


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_config_values() -> None:
    assert DEFAULT_TIER_CONFIG.hot_capacity == 100
    assert DEFAULT_TIER_CONFIG.hot_ttl_hours == 24.0
    assert DEFAULT_TIER_CONFIG.warm_ttl_hours == 720.0
    assert 0.0 < DEFAULT_TIER_CONFIG.promote_threshold < 1.0
    assert 0.0 < DEFAULT_TIER_CONFIG.contradiction_floor < 1.0


@pytest.mark.unit
def test_config_loads_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {
        "v4": {
            FLAG: {
                "enabled": True,
                "hot_capacity": 50,
                "hot_ttl_hours": 12.0,
                "promote_threshold": 0.9,
            }
        }
    }
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    out = _load_config()
    assert out.hot_capacity == 50
    assert out.hot_ttl_hours == 12.0
    assert out.promote_threshold == 0.9
    # Untouched knobs keep their defaults.
    assert out.warm_ttl_hours == DEFAULT_TIER_CONFIG.warm_ttl_hours


@pytest.mark.unit
def test_config_falls_back_on_bad_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {
        "v4": {
            FLAG: {"enabled": True, "hot_capacity": "fifty"},  # type-mismatch
        }
    }
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    out = _load_config()
    # Bad value → field reverts to default; sibling fields untouched.
    assert out.hot_capacity == DEFAULT_TIER_CONFIG.hot_capacity


@pytest.mark.unit
def test_config_immutable() -> None:
    """``TierConfig`` is frozen; mutation must raise."""
    cfg = TierConfig()
    with pytest.raises((AttributeError, Exception)):
        cfg.hot_capacity = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CAS / read-after-write consistency (v4-audit-2026-05-10 unanimous fix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schema_includes_block_version_column(workspace: Path) -> None:
    """Closes the audit blind spot: every fresh row carries a version."""
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(workspace / "index.db") as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(block_recall_tier)")}
    assert "block_version" in cols


@pytest.mark.unit
def test_pre_cas_table_is_migrated(workspace: Path) -> None:
    """An existing block_recall_tier WITHOUT block_version gets the column added."""
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        # Create the v0 (pre-CAS) shape.
        conn.execute(
            "CREATE TABLE block_recall_tier ("
            "block_id TEXT PRIMARY KEY, tier TEXT NOT NULL, "
            "last_seen_at TEXT NOT NULL, promoted_count INTEGER NOT NULL DEFAULT 0, "
            "last_surprise REAL)"
        )
        conn.execute("INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES ('B-old', 'hot', '2026-05-09T00:00:00Z')")
        conn.commit()
    # Idempotent ensure adds the missing column without touching data.
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(block_recall_tier)")}
        rows = conn.execute("SELECT block_id, tier, block_version FROM block_recall_tier").fetchall()
    assert "block_version" in cols
    assert rows == [("B-old", "hot", 0)]


@pytest.mark.unit
def test_get_tier_version_unknown_block_returns_zero(workspace: Path) -> None:
    """Default version for any block with no row is 0, matching INSERT default."""
    ensure_recall_tier_schema(workspace)
    assert get_tier_version(workspace, "B-never-seen") == 0


@pytest.mark.unit
def test_get_tier_version_round_trips(workspace: Path) -> None:
    ensure_recall_tier_schema(workspace)
    with sqlite3.connect(workspace / "index.db") as conn:
        conn.execute(
            "INSERT INTO block_recall_tier (block_id, tier, last_seen_at, block_version) VALUES ('B-1', 'hot', '2026-05-10T00:00:00Z', 7)"
        )
        conn.commit()
    assert get_tier_version(workspace, "B-1") == 7


@pytest.mark.unit
def test_get_tier_version_returns_zero_when_db_missing(workspace: Path) -> None:
    assert get_tier_version(workspace, "B-1") == 0


@pytest.mark.unit
def test_get_tier_version_returns_zero_when_table_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"v4": {FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    sqlite3.connect(tmp_path / "index.db").close()
    assert get_tier_version(tmp_path, "B-1") == 0


@pytest.mark.unit
def test_get_tier_version_returns_zero_on_legacy_table(workspace: Path) -> None:
    """A pre-CAS table without block_version reads as 0 — never crashes."""
    db = workspace / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE block_recall_tier ("
            "block_id TEXT PRIMARY KEY, tier TEXT NOT NULL, "
            "last_seen_at TEXT NOT NULL, promoted_count INTEGER NOT NULL DEFAULT 0, "
            "last_surprise REAL)"
        )
        conn.execute("INSERT INTO block_recall_tier (block_id, tier, last_seen_at) VALUES ('B-old', 'hot', '2026-05-09T00:00:00Z')")
        conn.commit()
    # No ensure_recall_tier_schema call here — simulating a fresh-read on
    # legacy state before the migration runs.
    assert get_tier_version(workspace, "B-old") == 0


@pytest.mark.unit
def test_stale_version_error_inherits_runtimeerror() -> None:
    """The exception is catchable as RuntimeError so generic retry helpers
    can pick it up without importing the v4 surface directly."""
    err = StaleVersionError("expected 5, got 7")
    assert isinstance(err, RuntimeError)


@pytest.mark.unit
def test_stale_version_error_message_round_trips() -> None:
    err = StaleVersionError("block B-1: expected version 3 but row carries 5")
    assert "B-1" in str(err)
    assert "version 3" in str(err)
    assert "5" in str(err)

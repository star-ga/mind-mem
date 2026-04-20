"""Tests for tier-aware recall boosting (v3.2.0 hot/cold tier wire-up)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mind_mem.tier_recall import (
    _TIER_BOOST,
    apply_tier_boosts,
    is_tier_boost_enabled,
    tier_boost_summary,
)


@pytest.fixture
def ws_with_tiers(tmp_path: Path) -> Path:
    """Workspace with a seeded block_tiers SQLite table."""
    db_dir = tmp_path / ".sqlite_index"
    db_dir.mkdir()
    db_path = db_dir / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE block_tiers (id TEXT PRIMARY KEY, tier INTEGER NOT NULL DEFAULT 1, updated_at TEXT, demotion_reason TEXT)")
    # Seed four blocks, one per tier.
    seed = [
        ("D-20260420-001", 1, "2026-04-20T00:00:00Z", None),
        ("D-20260420-002", 2, "2026-04-20T00:00:00Z", None),
        ("D-20260420-003", 3, "2026-04-20T00:00:00Z", None),
        ("D-20260420-004", 4, "2026-04-20T00:00:00Z", None),
    ]
    conn.executemany(
        "INSERT INTO block_tiers (id, tier, updated_at, demotion_reason) VALUES (?, ?, ?, ?)",
        seed,
    )
    conn.commit()
    conn.close()
    return tmp_path


class TestApplyTierBoosts:
    def test_scores_multiplied_per_tier(self, ws_with_tiers: Path) -> None:
        results = [
            {"_id": "D-20260420-001", "score": 1.0},  # WORKING → 0.7
            {"_id": "D-20260420-002", "score": 1.0},  # SHARED → 1.0
            {"_id": "D-20260420-003", "score": 1.0},  # LONG_TERM → 1.5
            {"_id": "D-20260420-004", "score": 1.0},  # VERIFIED → 2.0
        ]
        apply_tier_boosts(results, str(ws_with_tiers))

        # After sort, VERIFIED first, WORKING last.
        order = [r["_id"] for r in results]
        assert order == [
            "D-20260420-004",
            "D-20260420-003",
            "D-20260420-002",
            "D-20260420-001",
        ]
        # Exact score values match the boost table.
        for r in results:
            assert r["score"] == pytest.approx(_TIER_BOOST[r["_tier"]])

    def test_unknown_block_defaults_to_working(self, ws_with_tiers: Path) -> None:
        results = [{"_id": "T-20260420-999", "score": 1.0}]
        apply_tier_boosts(results, str(ws_with_tiers))
        assert results[0]["score"] == pytest.approx(0.7)
        assert results[0]["_tier"] == 1

    def test_missing_tier_db_is_silent_noop(self, tmp_path: Path) -> None:
        results = [{"_id": "D-1", "score": 0.5}, {"_id": "D-2", "score": 0.9}]
        apply_tier_boosts(results, str(tmp_path))
        # No tier table → all default to WORKING (0.7); results are
        # sorted by the new score so lookups are ID-keyed.
        by_id = {r["_id"]: r["score"] for r in results}
        assert by_id["D-1"] == pytest.approx(0.5 * 0.7)
        assert by_id["D-2"] == pytest.approx(0.9 * 0.7)
        # Highest post-boost score sorts first.
        assert results[0]["_id"] == "D-2"

    def test_no_score_field_is_untouched(self, ws_with_tiers: Path) -> None:
        results = [{"_id": "D-20260420-001", "_other": "noise"}]
        apply_tier_boosts(results, str(ws_with_tiers))
        assert "score" not in results[0]
        # Annotate still runs so diagnostics can inspect the tier.
        assert results[0]["_tier"] == 1

    def test_annotate_off_skips_metadata(self, ws_with_tiers: Path) -> None:
        results = [{"_id": "D-20260420-001", "score": 1.0}]
        apply_tier_boosts(results, str(ws_with_tiers), annotate=False)
        assert "_tier" not in results[0]
        assert "_tier_boost" not in results[0]
        assert results[0]["score"] == pytest.approx(0.7)

    def test_empty_results_is_noop(self, ws_with_tiers: Path) -> None:
        assert apply_tier_boosts([], str(ws_with_tiers)) == []

    def test_accepts_id_key_alias(self, ws_with_tiers: Path) -> None:
        """Some upstreams use ``id`` instead of ``_id``."""
        results = [{"id": "D-20260420-004", "score": 1.0}]
        apply_tier_boosts(results, str(ws_with_tiers))
        assert results[0]["score"] == pytest.approx(2.0)


class TestConfigFlag:
    def test_default_disabled(self) -> None:
        assert is_tier_boost_enabled(None) is False
        assert is_tier_boost_enabled({}) is False
        assert is_tier_boost_enabled({"retrieval": {}}) is False

    def test_explicit_enable(self) -> None:
        assert is_tier_boost_enabled({"retrieval": {"tier_boost": True}}) is True

    def test_explicit_disable(self) -> None:
        assert is_tier_boost_enabled({"retrieval": {"tier_boost": False}}) is False

    def test_malformed_retrieval_section(self) -> None:
        """Non-dict ``retrieval`` section shouldn't crash."""
        assert is_tier_boost_enabled({"retrieval": "nope"}) is False


class TestSummary:
    def test_counts_per_tier(self) -> None:
        tier_map = {
            "A": 1,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 3,
            "F": 3,
            "G": 4,
        }
        summary = tier_boost_summary(tier_map)
        assert summary == {
            "WORKING": 2,
            "SHARED": 1,
            "LONG_TERM": 3,
            "VERIFIED": 1,
            "total": 7,
        }

    def test_empty_map(self) -> None:
        summary = tier_boost_summary({})
        assert summary == {
            "WORKING": 0,
            "SHARED": 0,
            "LONG_TERM": 0,
            "VERIFIED": 0,
            "total": 0,
        }

    def test_unknown_tier_ints_ignored(self) -> None:
        summary = tier_boost_summary({"X": 99})
        # 99 doesn't map to a named tier — it's ignored.
        assert summary["total"] == 0

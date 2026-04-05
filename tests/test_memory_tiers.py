# Copyright 2026 STARGA, Inc.
"""Tests for tiered memory with auto-promotion.

Covers: MemoryTier enum, TierPolicy dataclass, TierManager CRUD,
promotion logic, demotion logic, run_promotion_cycle, persistence,
edge cases, and thread safety.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone

from mind_mem.memory_tiers import (
    DemotionReason,
    MemoryTier,
    TierManager,
    TierPolicy,
    default_policies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(db_path: str) -> TierManager:
    return TierManager(db_path)


def _seed_block_meta(
    db_path: str,
    block_id: str,
    *,
    access_count: int = 0,
    created_at: str | None = None,
    confirmations: int = 0,
    confidence: float = 1.0,
    last_accessed: str | None = None,
) -> None:
    """Directly insert rows into block_meta and block_tier_meta for testing."""
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS block_meta (
            id TEXT PRIMARY KEY,
            importance REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            keywords TEXT DEFAULT '',
            connections TEXT DEFAULT ''
        )"""
    )
    conn.execute(
        """INSERT INTO block_meta (id, access_count, last_accessed)
           VALUES (?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               access_count = ?,
               last_accessed = ?""",
        (block_id, access_count, last_accessed or now, access_count, last_accessed or now),
    )
    # Also seed the tier_meta table used by TierManager
    conn.execute(
        """CREATE TABLE IF NOT EXISTS block_tier_meta (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            confirmations INTEGER DEFAULT 0,
            confidence REAL DEFAULT 1.0,
            contradicted INTEGER DEFAULT 0
        )"""
    )
    conn.execute(
        """INSERT INTO block_tier_meta (id, created_at, confirmations, confidence)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               created_at = ?,
               confirmations = ?,
               confidence = ?""",
        (
            block_id,
            created_at or now,
            confirmations,
            confidence,
            created_at or now,
            confirmations,
            confidence,
        ),
    )
    conn.commit()
    conn.close()


def _hours_ago(h: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=h)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# 1. MemoryTier enum
# ---------------------------------------------------------------------------


class TestMemoryTierEnum(unittest.TestCase):
    def test_all_four_tiers_exist(self):
        tiers = {MemoryTier.WORKING, MemoryTier.SHARED, MemoryTier.LONG_TERM, MemoryTier.VERIFIED}
        self.assertEqual(len(tiers), 4)

    def test_tiers_have_ordered_values(self):
        self.assertLess(MemoryTier.WORKING.value, MemoryTier.SHARED.value)
        self.assertLess(MemoryTier.SHARED.value, MemoryTier.LONG_TERM.value)
        self.assertLess(MemoryTier.LONG_TERM.value, MemoryTier.VERIFIED.value)


# ---------------------------------------------------------------------------
# 2. TierPolicy dataclass
# ---------------------------------------------------------------------------


class TestTierPolicy(unittest.TestCase):
    def test_default_policies_cover_all_tiers(self):
        policies = default_policies()
        for tier in MemoryTier:
            self.assertIn(tier, policies)

    def test_working_to_shared_defaults(self):
        p = default_policies()[MemoryTier.SHARED]
        self.assertGreaterEqual(p.min_access_count, 3)
        self.assertGreaterEqual(p.min_age_hours, 1.0)

    def test_shared_to_long_term_defaults(self):
        p = default_policies()[MemoryTier.LONG_TERM]
        self.assertGreaterEqual(p.min_access_count, 10)
        self.assertGreaterEqual(p.min_age_hours, 24.0)
        self.assertGreaterEqual(p.min_confirmations, 2)

    def test_long_term_to_verified_defaults(self):
        p = default_policies()[MemoryTier.VERIFIED]
        self.assertGreaterEqual(p.min_confirmations, 5)
        self.assertGreater(p.min_confidence, 0.8)

    def test_tier_policy_is_dataclass(self):
        p = TierPolicy(min_access_count=5, min_age_hours=2.0, min_confirmations=1, min_confidence=0.7)
        self.assertEqual(p.min_access_count, 5)
        self.assertEqual(p.min_age_hours, 2.0)


# ---------------------------------------------------------------------------
# 3. TierManager — basic CRUD
# ---------------------------------------------------------------------------


class TestTierManagerCrud(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "index.db")
        self.mgr = _make_manager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_block_defaults_to_working(self):
        tier = self.mgr.get_tier("B-001")
        self.assertEqual(tier, MemoryTier.WORKING)

    def test_explicit_promote_to_shared(self):
        ok = self.mgr.promote("B-001", MemoryTier.SHARED)
        self.assertTrue(ok)
        self.assertEqual(self.mgr.get_tier("B-001"), MemoryTier.SHARED)

    def test_promote_skips_tier_returns_false(self):
        # Cannot jump from WORKING directly to VERIFIED
        ok = self.mgr.promote("B-001", MemoryTier.VERIFIED)
        self.assertFalse(ok)

    def test_demote_moves_tier_down(self):
        self.mgr.promote("B-001", MemoryTier.SHARED)
        ok = self.mgr.demote("B-001", MemoryTier.WORKING, DemotionReason.LOW_CONFIDENCE)
        self.assertTrue(ok)
        self.assertEqual(self.mgr.get_tier("B-001"), MemoryTier.WORKING)

    def test_get_blocks_by_tier_empty(self):
        blocks = self.mgr.get_blocks_by_tier(MemoryTier.SHARED)
        self.assertEqual(blocks, [])

    def test_get_blocks_by_tier_after_promote(self):
        self.mgr.promote("B-001", MemoryTier.SHARED)
        self.mgr.promote("B-002", MemoryTier.SHARED)
        blocks = self.mgr.get_blocks_by_tier(MemoryTier.SHARED)
        self.assertIn("B-001", blocks)
        self.assertIn("B-002", blocks)

    def test_working_tier_blocks_listed(self):
        # Blocks without an explicit tier assignment are WORKING by default;
        # get_blocks_by_tier(WORKING) returns blocks explicitly stored as WORKING.
        self.mgr.promote("B-001", MemoryTier.SHARED)
        self.mgr.demote("B-001", MemoryTier.WORKING, DemotionReason.CONTRADICTION)
        blocks = self.mgr.get_blocks_by_tier(MemoryTier.WORKING)
        self.assertIn("B-001", blocks)


# ---------------------------------------------------------------------------
# 4. check_promotion — eligibility logic
# ---------------------------------------------------------------------------


class TestCheckPromotion(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "index.db")
        self.mgr = _make_manager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_not_eligible_without_meta(self):
        result = self.mgr.check_promotion("B-999")
        self.assertIsNone(result)

    def test_working_eligible_for_shared(self):
        _seed_block_meta(
            self.db_path, "B-001", access_count=5, created_at=_hours_ago(2)
        )
        result = self.mgr.check_promotion("B-001")
        self.assertEqual(result, MemoryTier.SHARED)

    def test_working_not_eligible_too_few_accesses(self):
        _seed_block_meta(
            self.db_path, "B-002", access_count=1, created_at=_hours_ago(2)
        )
        result = self.mgr.check_promotion("B-002")
        self.assertIsNone(result)

    def test_working_not_eligible_too_young(self):
        _seed_block_meta(
            self.db_path, "B-003", access_count=10, created_at=_hours_ago(0.1)
        )
        result = self.mgr.check_promotion("B-003")
        self.assertIsNone(result)

    def test_shared_eligible_for_long_term(self):
        _seed_block_meta(
            self.db_path,
            "B-004",
            access_count=15,
            created_at=_hours_ago(30),
            confirmations=3,
        )
        self.mgr.promote("B-004", MemoryTier.SHARED)
        result = self.mgr.check_promotion("B-004")
        self.assertEqual(result, MemoryTier.LONG_TERM)

    def test_long_term_eligible_for_verified(self):
        _seed_block_meta(
            self.db_path,
            "B-005",
            access_count=20,
            created_at=_hours_ago(50),
            confirmations=6,
            confidence=0.95,
        )
        self.mgr.promote("B-005", MemoryTier.SHARED)
        self.mgr.promote("B-005", MemoryTier.LONG_TERM)
        result = self.mgr.check_promotion("B-005")
        self.assertEqual(result, MemoryTier.VERIFIED)


# ---------------------------------------------------------------------------
# 5. run_promotion_cycle
# ---------------------------------------------------------------------------


class TestRunPromotionCycle(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "index.db")
        self.mgr = _make_manager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_cycle_returns_empty(self):
        result = self.mgr.run_promotion_cycle()
        self.assertEqual(result, [])

    def test_cycle_promotes_eligible_block(self):
        _seed_block_meta(
            self.db_path, "B-001", access_count=5, created_at=_hours_ago(2)
        )
        # Explicitly register in WORKING tier so cycle can find it
        self.mgr._register_block("B-001", MemoryTier.WORKING)
        promotions = self.mgr.run_promotion_cycle()
        self.assertEqual(len(promotions), 1)
        block_id, old_tier, new_tier = promotions[0]
        self.assertEqual(block_id, "B-001")
        self.assertEqual(old_tier, MemoryTier.WORKING)
        self.assertEqual(new_tier, MemoryTier.SHARED)

    def test_cycle_returns_tuple_of_three(self):
        _seed_block_meta(
            self.db_path, "B-002", access_count=5, created_at=_hours_ago(2)
        )
        self.mgr._register_block("B-002", MemoryTier.WORKING)
        promotions = self.mgr.run_promotion_cycle()
        self.assertEqual(len(promotions[0]), 3)


# ---------------------------------------------------------------------------
# 6. Demotion
# ---------------------------------------------------------------------------


class TestDemotion(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "index.db")
        self.mgr = _make_manager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_demote_below_current_returns_true(self):
        self.mgr.promote("B-001", MemoryTier.SHARED)
        ok = self.mgr.demote("B-001", MemoryTier.WORKING, DemotionReason.LOW_CONFIDENCE)
        self.assertTrue(ok)

    def test_demote_to_same_tier_returns_false(self):
        self.mgr.promote("B-001", MemoryTier.SHARED)
        ok = self.mgr.demote("B-001", MemoryTier.SHARED, DemotionReason.LOW_CONFIDENCE)
        self.assertFalse(ok)

    def test_demote_above_current_returns_false(self):
        ok = self.mgr.demote("B-001", MemoryTier.VERIFIED, DemotionReason.CONTRADICTION)
        self.assertFalse(ok)

    def test_demote_contradiction_reason_stored(self):
        self.mgr.promote("B-001", MemoryTier.SHARED)
        self.mgr.demote("B-001", MemoryTier.WORKING, DemotionReason.CONTRADICTION)
        tier = self.mgr.get_tier("B-001")
        self.assertEqual(tier, MemoryTier.WORKING)


# ---------------------------------------------------------------------------
# 7. Persistence across instances
# ---------------------------------------------------------------------------


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "index.db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tier_survives_manager_restart(self):
        mgr1 = _make_manager(self.db_path)
        mgr1.promote("B-001", MemoryTier.SHARED)

        mgr2 = _make_manager(self.db_path)
        self.assertEqual(mgr2.get_tier("B-001"), MemoryTier.SHARED)


# ---------------------------------------------------------------------------
# 8. Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "index.db")
        self.mgr = _make_manager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_promotes_no_crash(self):
        errors: list[Exception] = []

        def promote(block_id: str) -> None:
            try:
                self.mgr.promote(block_id, MemoryTier.SHARED)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=promote, args=(f"B-{i:03d}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Unexpected errors: {errors}")


if __name__ == "__main__":
    unittest.main()

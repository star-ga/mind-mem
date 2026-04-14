# Copyright 2026 STARGA, Inc.
"""Tests for TTL/LRU tier decay (v3.0.0 — GH #502)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mind_mem.memory_tiers import (
    DemotionReason,
    MemoryTier,
    TierManager,
    TierPolicy,
    _hours_since,
    default_policies,
)


class TestHoursSinceHelper:
    def test_recent_returns_small(self) -> None:
        now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
        iso = (now - timedelta(hours=2)).isoformat()
        assert abs(_hours_since(iso, now) - 2.0) < 0.01

    def test_missing_returns_zero(self) -> None:
        now = datetime.now(timezone.utc)
        assert _hours_since(None, now) == 0.0
        assert _hours_since("", now) == 0.0

    def test_unparseable_returns_zero(self) -> None:
        now = datetime.now(timezone.utc)
        assert _hours_since("not-a-date", now) == 0.0

    def test_naive_timestamp_treated_as_utc(self) -> None:
        now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
        iso = "2026-04-13T10:00:00"
        assert abs(_hours_since(iso, now) - 2.0) < 0.01

    def test_z_suffix_supported(self) -> None:
        now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
        iso = "2026-04-13T10:00:00Z"
        assert abs(_hours_since(iso, now) - 2.0) < 0.01


class TestTierPolicyDecayFields:
    def test_defaults_include_decay_parameters(self) -> None:
        policies = default_policies()
        working = policies[MemoryTier.WORKING]
        shared = policies[MemoryTier.SHARED]
        verified = policies[MemoryTier.VERIFIED]
        # WORKING: has ttl, no max_idle
        assert working.ttl_hours > 0
        # Higher tiers: max_idle but no ttl (never auto-deleted)
        assert shared.max_idle_hours > 0
        assert shared.ttl_hours == 0
        assert verified.max_idle_hours > 0
        assert verified.ttl_hours == 0


class TestDecayCycle:
    def test_empty_workspace_produces_empty_results(self, tmp_path: Path) -> None:
        mgr = TierManager(str(tmp_path / "tiers.db"))
        demotions, evicted = mgr.run_decay_cycle()
        assert demotions == []
        assert evicted == []

    def test_decay_cycle_returns_tuple_of_lists(self, tmp_path: Path) -> None:
        mgr = TierManager(str(tmp_path / "tiers.db"))
        mgr._register_block("D-active", MemoryTier.SHARED)
        demotions, evicted = mgr.run_decay_cycle()
        # Fresh registration → nothing stale yet
        assert isinstance(demotions, list)
        assert isinstance(evicted, list)

    def test_idle_block_demoted(self, tmp_path: Path) -> None:
        mgr = TierManager(str(tmp_path / "tiers.db"))
        mgr._register_block("D-idle", MemoryTier.SHARED)
        # Run decay with a "now" far in the future so the 2-week idle
        # threshold is exceeded.
        future = datetime.now(timezone.utc) + timedelta(days=365)
        demotions, _ = mgr.run_decay_cycle(now=future)
        # Demoted back to WORKING
        ids = [d[0] for d in demotions]
        assert "D-idle" in ids
        assert mgr.get_tier("D-idle") == MemoryTier.WORKING

    def test_working_ttl_evicts_orphan(self, tmp_path: Path) -> None:
        mgr = TierManager(str(tmp_path / "tiers.db"))
        mgr._register_block("D-orphan", MemoryTier.WORKING)
        # Far-future now triggers ttl eviction
        future = datetime.now(timezone.utc) + timedelta(days=365)
        demotions, evicted = mgr.run_decay_cycle(now=future)
        assert "D-orphan" in evicted
        # Gone from tier tracking entirely — get_tier returns WORKING
        # default because block is untracked
        # (can't assert removal via this API; just check list)

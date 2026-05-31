# Copyright 2026 STARGA, Inc.
"""Tests for cognitive forgetting + token budget (v2.4.0)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mind_mem.cognitive_forget import (
    BlockCognition,
    BlockLifecycle,
    ConsolidationConfig,
    estimate_tokens,
    pack_to_budget,
    plan_consolidation,
    should_archive,
    should_forget,
    should_mark,
)


def _now() -> datetime:
    return datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)


def _block(**kw) -> BlockCognition:
    base = dict(
        block_id="B-001",
        importance=0.5,
        last_accessed=None,
        access_count=0,
        created_at="2026-03-01T00:00:00Z",
        size_bytes=100,
        lifecycle=BlockLifecycle.ACTIVE,
    )
    base.update(kw)
    return BlockCognition(**base)


# ---------------------------------------------------------------------------
# Value object validation
# ---------------------------------------------------------------------------


class TestBlockCognition:
    def test_importance_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            _block(importance=1.5)
        with pytest.raises(ValueError):
            _block(importance=-0.1)

    def test_negative_access_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            _block(access_count=-1)

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            _block(size_bytes=-1)


# ---------------------------------------------------------------------------
# Decision functions
# ---------------------------------------------------------------------------


class TestShouldMark:
    def test_low_importance_stale_block_marked(self) -> None:
        b = _block(importance=0.1, last_accessed="2026-03-01T00:00:00Z")
        assert should_mark(b, now=_now(), importance_threshold=0.25, stale_days=14)

    def test_high_importance_not_marked(self) -> None:
        b = _block(importance=0.9, last_accessed="2026-03-01T00:00:00Z")
        assert not should_mark(b, now=_now(), importance_threshold=0.25, stale_days=14)

    def test_recently_accessed_not_marked(self) -> None:
        b = _block(importance=0.1, last_accessed="2026-04-12T00:00:00Z")
        assert not should_mark(b, now=_now(), importance_threshold=0.25, stale_days=14)

    def test_non_active_not_marked(self) -> None:
        b = _block(importance=0.1, lifecycle=BlockLifecycle.MERGED)
        assert not should_mark(b, now=_now(), importance_threshold=0.25, stale_days=14)

    def test_missing_timestamps_treated_as_stale(self) -> None:
        b = _block(importance=0.1, last_accessed=None, created_at=None)
        assert should_mark(b, now=_now(), importance_threshold=0.25, stale_days=14)


class TestShouldArchive:
    def test_merged_stale_archived(self) -> None:
        b = _block(
            lifecycle=BlockLifecycle.MERGED,
            last_accessed="2026-01-01T00:00:00Z",
        )
        assert should_archive(b, now=_now(), archive_after_days=60)

    def test_merged_fresh_not_archived(self) -> None:
        b = _block(
            lifecycle=BlockLifecycle.MERGED,
            last_accessed="2026-04-01T00:00:00Z",
        )
        assert not should_archive(b, now=_now(), archive_after_days=60)

    def test_active_not_archived(self) -> None:
        b = _block(lifecycle=BlockLifecycle.ACTIVE)
        assert not should_archive(b, now=_now(), archive_after_days=60)


class TestShouldForget:
    def test_archived_past_grace_forgotten(self) -> None:
        b = _block(
            lifecycle=BlockLifecycle.ARCHIVED,
            last_accessed="2025-10-01T00:00:00Z",  # >30d before _now()
        )
        assert should_forget(b, now=_now(), grace_days=30)

    def test_archived_within_grace_kept(self) -> None:
        b = _block(
            lifecycle=BlockLifecycle.ARCHIVED,
            last_accessed="2026-04-05T00:00:00Z",
        )
        assert not should_forget(b, now=_now(), grace_days=30)

    def test_active_never_forgotten_directly(self) -> None:
        b = _block(lifecycle=BlockLifecycle.ACTIVE)
        assert not should_forget(b, now=_now(), grace_days=30)

    def test_merged_never_forgotten_directly(self) -> None:
        b = _block(lifecycle=BlockLifecycle.MERGED)
        assert not should_forget(b, now=_now(), grace_days=30)


# ---------------------------------------------------------------------------
# plan_consolidation
# ---------------------------------------------------------------------------


class TestPlanConsolidation:
    def test_mixed_plan(self) -> None:
        blocks = [
            _block(block_id="M-1", importance=0.1, last_accessed="2026-01-01T00:00:00Z"),
            _block(
                block_id="A-1",
                lifecycle=BlockLifecycle.MERGED,
                last_accessed="2025-12-01T00:00:00Z",
            ),
            _block(
                block_id="F-1",
                lifecycle=BlockLifecycle.ARCHIVED,
                last_accessed="2025-08-01T00:00:00Z",
            ),
            _block(block_id="K-1", importance=0.9, last_accessed="2026-04-10T00:00:00Z"),
        ]
        plan = plan_consolidation(blocks, now=_now())
        assert plan.mark == ["M-1"]
        assert plan.archive == ["A-1"]
        assert plan.forget == ["F-1"]
        assert plan.total == 3

    def test_empty_input_empty_plan(self) -> None:
        plan = plan_consolidation([], now=_now())
        assert plan.total == 0

    def test_config_threshold_tightens(self) -> None:
        # importance=0.4 crosses 0.25 (kept) but falls under 0.5 (marked).
        block = _block(importance=0.4, last_accessed="2026-01-01T00:00:00Z")
        loose = plan_consolidation([block], now=_now())
        strict = plan_consolidation(
            [block],
            config=ConsolidationConfig(importance_threshold=0.5),
            now=_now(),
        )
        assert block.block_id not in loose.mark
        assert block.block_id in strict.mark

    def test_invalid_config_rejected(self) -> None:
        with pytest.raises(ValueError):
            ConsolidationConfig(importance_threshold=2.0)
        with pytest.raises(ValueError):
            ConsolidationConfig(grace_days=-1)


# ---------------------------------------------------------------------------
# Token budget packing
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_returns_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_one_char_at_least_one_token(self) -> None:
        assert estimate_tokens("a") == 1

    def test_scales_with_length(self) -> None:
        short = estimate_tokens("hi")
        long = estimate_tokens("hi " * 1000)
        assert long > short


class TestPackToBudget:
    def test_fits_everything(self) -> None:
        results = [{"excerpt": "short"} for _ in range(3)]
        packed = pack_to_budget(results, max_tokens=1000)
        assert len(packed.included) == 3
        assert len(packed.dropped) == 0

    def test_drops_overflow(self) -> None:
        big = {"excerpt": "x" * 10000, "_id": "BIG"}  # ~2500 tokens
        small = {"excerpt": "y", "_id": "SMALL"}
        packed = pack_to_budget([big, small], max_tokens=100)
        # Big should not fit; small does.
        dropped_ids = {r["_id"] for r in packed.dropped}
        included_ids = {r["_id"] for r in packed.included}
        assert "BIG" in dropped_ids
        assert "SMALL" in included_ids

    def test_priority_order_preserved(self) -> None:
        # High-priority result first; when budget tight, it must be kept.
        high = {"excerpt": "x" * 200, "_id": "H"}  # ~50 tokens
        low = {"excerpt": "y" * 200, "_id": "L"}
        packed = pack_to_budget([high, low], max_tokens=100)
        included_ids = [r["_id"] for r in packed.included]
        assert "H" in included_ids

    def test_reserves_honoured(self) -> None:
        packed = pack_to_budget([{"excerpt": "x"}], max_tokens=1000)
        assert packed.reserved_graph == 150  # 15% of 1000
        assert packed.reserved_provenance == 100  # 10% of 1000

    def test_max_tokens_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            pack_to_budget([], max_tokens=0)

    def test_excess_reserves_rejected(self) -> None:
        with pytest.raises(ValueError, match="leave room"):
            pack_to_budget(
                [],
                max_tokens=1000,
                graph_reserve_frac=0.6,
                provenance_reserve_frac=0.6,
            )

    def test_custom_text_field(self) -> None:
        results = [{"content": "x" * 1000}]
        packed = pack_to_budget(results, max_tokens=50, text_field="content")
        # 1000 chars ≈ 250 tokens → doesn't fit 50-token budget.
        assert len(packed.dropped) == 1

    def test_packed_budget_as_dict(self) -> None:
        packed = pack_to_budget([{"excerpt": "x"}], max_tokens=1000)
        d = packed.as_dict()
        assert set(d.keys()) == {
            "included_count",
            "dropped_count",
            "budget",
            "tokens_used",
            "reserved_graph",
            "reserved_provenance",
        }

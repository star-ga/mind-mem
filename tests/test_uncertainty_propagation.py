# Copyright 2026 STARGA, Inc.
"""Tests for multi-hop uncertainty propagation."""

from __future__ import annotations

import math
import pytest

from mind_mem.uncertainty_propagation import HopResult, UncertaintyPropagator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_hop(
    block_id: str,
    confidence: float,
    hop_depth: int,
    parent_hop_id: str | None = None,
    content: str = "some content",
) -> HopResult:
    return HopResult(
        block_id=block_id,
        content=content,
        confidence=confidence,
        hop_depth=hop_depth,
        parent_hop_id=parent_hop_id,
    )


def make_chain(confidences: list[float], decay: float = 0.85) -> tuple[list[HopResult], UncertaintyPropagator]:
    """Build a linear chain A→B→C... with given per-hop raw confidences."""
    hops: list[HopResult] = []
    parent_id: str | None = None
    for depth, conf in enumerate(confidences):
        bid = f"block-{depth}"
        hops.append(make_hop(bid, conf, depth, parent_id))
        parent_id = bid
    propagator = UncertaintyPropagator(decay_factor=decay)
    return hops, propagator


# ---------------------------------------------------------------------------
# HopResult dataclass
# ---------------------------------------------------------------------------


class TestHopResult:
    def test_fields_stored(self):
        hop = HopResult(
            block_id="b1",
            content="hello",
            confidence=0.9,
            hop_depth=0,
            parent_hop_id=None,
        )
        assert hop.block_id == "b1"
        assert hop.content == "hello"
        assert hop.confidence == 0.9
        assert hop.hop_depth == 0
        assert hop.parent_hop_id is None

    def test_parent_hop_id_set(self):
        hop = HopResult(
            block_id="b2",
            content="world",
            confidence=0.7,
            hop_depth=1,
            parent_hop_id="b1",
        )
        assert hop.parent_hop_id == "b1"


# ---------------------------------------------------------------------------
# UncertaintyPropagator — propagate()
# ---------------------------------------------------------------------------


class TestPropagate:
    def test_empty_list_returns_empty(self):
        propagator = UncertaintyPropagator()
        assert propagator.propagate([]) == []

    def test_single_hop_root_unchanged(self):
        """Root hop (depth 0, no parent) confidence stays the same."""
        hop = make_hop("b0", 0.9, 0, parent_hop_id=None)
        propagator = UncertaintyPropagator()
        result = propagator.propagate([hop])
        assert result[0].confidence == pytest.approx(0.9)

    def test_second_hop_confidence_reduced(self):
        """Hop at depth 1 has confidence multiplied by parent_confidence * decay."""
        hops, propagator = make_chain([0.9, 1.0])
        result = propagator.propagate(hops)
        # hop1 confidence = 1.0 * (parent=0.9) * decay=0.85
        expected = 1.0 * 0.9 * 0.85
        assert result[1].confidence == pytest.approx(expected)

    def test_three_hop_chain_confidence_degrades(self):
        """Each hop multiplies in the parent's adjusted confidence and decay."""
        hops, propagator = make_chain([1.0, 1.0, 1.0])
        result = propagator.propagate(hops)
        # hop0: 1.0
        # hop1: 1.0 * 1.0 * 0.85 = 0.85
        # hop2: 1.0 * 0.85 * 0.85 = 0.7225
        assert result[0].confidence == pytest.approx(1.0)
        assert result[1].confidence == pytest.approx(0.85)
        assert result[2].confidence == pytest.approx(0.7225)

    def test_original_hops_not_mutated(self):
        """propagate() must not mutate the input list or its elements."""
        hops, propagator = make_chain([0.9, 0.8])
        original_confidences = [h.confidence for h in hops]
        propagator.propagate(hops)
        assert [h.confidence for h in hops] == original_confidences

    def test_returns_new_hop_result_objects(self):
        """Each returned HopResult is a new object, not the original."""
        hops, propagator = make_chain([0.9, 0.8])
        result = propagator.propagate(hops)
        for orig, adj in zip(hops, result):
            assert orig is not adj

    def test_low_parent_confidence_propagates(self):
        """A low-confidence root reduces all downstream hops significantly."""
        hops, propagator = make_chain([0.3, 1.0, 1.0])
        result = propagator.propagate(hops)
        # hop1: 1.0 * 0.3 * 0.85 = 0.255
        # hop2: 1.0 * 0.255 * 0.85 = 0.21675
        assert result[1].confidence == pytest.approx(0.255)
        assert result[2].confidence == pytest.approx(0.21675)

    def test_custom_decay_factor_applied(self):
        hops, propagator = make_chain([1.0, 1.0], decay=0.5)
        result = propagator.propagate(hops)
        assert result[1].confidence == pytest.approx(0.5)

    def test_confidence_clamped_to_one(self):
        """Adjusted confidence must not exceed 1.0 even if arithmetic allows it."""
        # Artificially, if parent=1.0 and decay=1.0 and self=0.99 → should stay ≤1
        hops, propagator = make_chain([1.0, 0.99], decay=1.0)
        result = propagator.propagate(hops)
        assert result[1].confidence <= 1.0

    def test_non_linear_chain_uses_parent_hop_id(self):
        """Hops not in index order but linked by parent_hop_id are resolved correctly."""
        # Build: b0 -> b2 (skipping b1 index)
        b0 = make_hop("b0", 0.8, 0, parent_hop_id=None)
        b1 = make_hop("b1", 1.0, 1, parent_hop_id="b0")
        # Pass in reverse order — propagator should still resolve parents correctly
        propagator = UncertaintyPropagator(decay_factor=0.85)
        result = propagator.propagate([b1, b0])
        adjusted = {h.block_id: h.confidence for h in result}
        assert adjusted["b0"] == pytest.approx(0.8)
        assert adjusted["b1"] == pytest.approx(1.0 * 0.8 * 0.85)


# ---------------------------------------------------------------------------
# UncertaintyPropagator — chain_confidence()
# ---------------------------------------------------------------------------


class TestChainConfidence:
    def test_empty_chain_returns_one(self):
        """Empty chain has full (vacuous) confidence."""
        propagator = UncertaintyPropagator()
        assert propagator.chain_confidence([]) == pytest.approx(1.0)

    def test_single_hop_equals_its_confidence(self):
        propagator = UncertaintyPropagator()
        hop = make_hop("b0", 0.75, 0)
        assert propagator.chain_confidence([hop]) == pytest.approx(0.75)

    def test_product_of_adjusted_confidences(self):
        """chain_confidence = product of propagated confidences."""
        hops, propagator = make_chain([1.0, 1.0, 1.0])
        adjusted = propagator.propagate(hops)
        expected = math.prod(h.confidence for h in adjusted)
        assert propagator.chain_confidence(hops) == pytest.approx(expected)

    def test_chain_confidence_decreases_with_depth(self):
        hops_short, p = make_chain([1.0, 1.0])
        hops_long, _ = make_chain([1.0, 1.0, 1.0])
        assert p.chain_confidence(hops_short) > p.chain_confidence(hops_long)


# ---------------------------------------------------------------------------
# UncertaintyPropagator — should_truncate()
# ---------------------------------------------------------------------------


class TestShouldTruncate:
    def test_above_min_confidence_not_truncated(self):
        propagator = UncertaintyPropagator()
        hop = make_hop("b0", 0.5, 0)
        assert propagator.should_truncate(hop, min_confidence=0.1) is False

    def test_exactly_at_min_confidence_not_truncated(self):
        propagator = UncertaintyPropagator()
        hop = make_hop("b0", 0.1, 0)
        assert propagator.should_truncate(hop, min_confidence=0.1) is False

    def test_below_min_confidence_truncated(self):
        propagator = UncertaintyPropagator()
        hop = make_hop("b0", 0.05, 2)
        assert propagator.should_truncate(hop, min_confidence=0.1) is True

    def test_default_min_confidence_is_0_1(self):
        propagator = UncertaintyPropagator()
        hop_low = make_hop("b0", 0.09, 3)
        hop_ok = make_hop("b1", 0.11, 3)
        assert propagator.should_truncate(hop_low) is True
        assert propagator.should_truncate(hop_ok) is False

    def test_zero_confidence_always_truncated(self):
        propagator = UncertaintyPropagator()
        hop = make_hop("b0", 0.0, 0)
        assert propagator.should_truncate(hop) is True

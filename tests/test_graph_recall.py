"""v3.3.0 Tier 1 #2 — multi-hop graph traversal on recall results.

``graph_expand`` should walk the block cross-reference graph from
each seed, append new blocks with decayed scores, and preserve the
original result ordering ahead of the appended neighbours.

Block IDs use the canonical ``D-YYYYMMDD-NNN`` format so
``_recall_constants._BLOCK_ID_RE`` recognises cross-references in
block text.
"""

from __future__ import annotations

import pytest

from mind_mem.graph_recall import (
    graph_expand,
    is_graph_expand_enabled,
    resolve_graph_config,
)

# Canonical block IDs for the tests.
A = "D-20260420-001"
B = "D-20260420-002"
C = "D-20260420-003"
D = "D-20260420-004"
E = "D-20260420-005"
F = "D-20260420-006"
X = "D-20260420-010"
Y = "D-20260420-020"


def _block(bid: str, text: str = "") -> dict:
    return {"_id": bid, "content": text, "type": "decision", "Statement": text}


class TestGraphExpand:
    def test_no_seeds_returns_unchanged(self) -> None:
        assert graph_expand([], [_block(A)]) == []

    def test_no_corpus_returns_unchanged(self) -> None:
        seeds = [{"_id": A, "score": 5.0}]
        assert graph_expand(seeds, []) == seeds

    def test_zero_max_hops_skips_expansion(self) -> None:
        seeds = [{"_id": A, "score": 5.0}]
        all_blocks = [_block(A, f"See {B}"), _block(B, f"See {A}")]
        out = graph_expand(seeds, all_blocks, max_hops=0)
        assert out == seeds

    def test_one_hop_walk(self) -> None:
        """A mentions B → graph expand adds B."""
        seeds = [{"_id": A, "score": 10.0}]
        all_blocks = [_block(A, f"See {B}"), _block(B, "Unrelated content")]
        out = graph_expand(seeds, all_blocks, max_hops=1, decay=0.5)
        ids = [r["_id"] for r in out]
        assert ids == [A, B]
        assert out[1]["score"] == pytest.approx(10.0 * 0.5)
        assert out[1]["_graph_hop"] == 1
        assert out[1]["_graph_parent"] == A

    def test_two_hop_walk(self) -> None:
        """A → B → C."""
        seeds = [{"_id": A, "score": 10.0}]
        all_blocks = [
            _block(A, f"See {B}"),
            _block(B, f"See {C}"),
            _block(C, "terminal"),
        ]
        out = graph_expand(seeds, all_blocks, max_hops=2, decay=0.5)
        ids = [r["_id"] for r in out]
        assert A in ids and B in ids and C in ids
        # C is 2 hops from A so score = 10 * 0.5^2 = 2.5
        c_row = next(r for r in out if r["_id"] == C)
        assert c_row["score"] == pytest.approx(2.5)
        assert c_row["_graph_hop"] == 2

    def test_seed_not_duplicated(self) -> None:
        seeds = [{"_id": A, "score": 5.0}, {"_id": B, "score": 3.0}]
        all_blocks = [
            _block(A, f"See {B}"),
            _block(B, f"See {A}"),
        ]
        out = graph_expand(seeds, all_blocks, max_hops=2)
        ids = [r["_id"] for r in out]
        assert ids.count(A) == 1
        assert ids.count(B) == 1

    def test_max_neighbors_cap(self) -> None:
        seeds = [{"_id": A, "score": 10.0}]
        all_blocks = [
            _block(A, f"See {B} and {C} and {D} and {E} and {F}"),
            _block(B, ""),
            _block(C, ""),
            _block(D, ""),
            _block(E, ""),
            _block(F, ""),
        ]
        out = graph_expand(seeds, all_blocks, max_hops=1, max_neighbors_per_hop=2)
        # Only 2 neighbours added (cap), so total = seed + 2.
        assert len(out) == 3

    def test_preserves_original_order(self) -> None:
        seeds = [
            {"_id": A, "score": 10.0},
            {"_id": B, "score": 9.0},
        ]
        all_blocks = [
            _block(A, f"See {X}"),
            _block(B, f"See {Y}"),
            _block(X, ""),
            _block(Y, ""),
        ]
        out = graph_expand(seeds, all_blocks, max_hops=1)
        # Original seeds come first, then expansion
        assert out[0]["_id"] == A
        assert out[1]["_id"] == B


class TestIsGraphExpandEnabled:
    def test_no_config_false(self) -> None:
        assert is_graph_expand_enabled(None) is False
        assert is_graph_expand_enabled({}) is False

    def test_explicit_enabled_true(self) -> None:
        cfg = {"retrieval": {"multi_hop": {"enabled": True}}}
        assert is_graph_expand_enabled(cfg) is True
        # enabled:true overrides query-type
        assert is_graph_expand_enabled(cfg, "PostgreSQL") is True

    def test_auto_enable_false(self) -> None:
        cfg = {"retrieval": {"multi_hop": {"enabled": False, "auto_enable": False}}}
        assert is_graph_expand_enabled(cfg, "A after B") is False

    def test_auto_enable_on_multi_hop_query(self) -> None:
        cfg = {"retrieval": {"multi_hop": {}}}
        assert is_graph_expand_enabled(cfg, "What is the relationship between X and Y?") is True

    def test_auto_enable_skips_single_hop_query(self) -> None:
        cfg = {"retrieval": {"multi_hop": {}}}
        assert is_graph_expand_enabled(cfg, "PostgreSQL deployment") is False


class TestResolveGraphConfig:
    def test_defaults(self) -> None:
        cfg = resolve_graph_config(None)
        assert cfg == {"max_hops": 2, "decay": 0.5, "max_neighbors_per_hop": 5}

    def test_custom_values(self) -> None:
        cfg = resolve_graph_config({"retrieval": {"multi_hop": {"max_hops": 3, "decay": 0.8, "max_neighbors_per_hop": 10}}})
        assert cfg["max_hops"] == 3
        assert cfg["decay"] == pytest.approx(0.8)
        assert cfg["max_neighbors_per_hop"] == 10

    def test_max_hops_clamped_to_3(self) -> None:
        """Security hard-ceiling: retrieval.multi_hop.max_hops > 3 clamps to 3."""
        cfg = resolve_graph_config({"retrieval": {"multi_hop": {"max_hops": 99}}})
        assert cfg["max_hops"] == 3

    def test_invalid_values_fall_back_to_defaults(self) -> None:
        cfg = resolve_graph_config({"retrieval": {"multi_hop": {"max_hops": -1, "decay": 1.5, "max_neighbors_per_hop": "x"}}})
        assert cfg == {"max_hops": 2, "decay": 0.5, "max_neighbors_per_hop": 5}

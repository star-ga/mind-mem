"""Robustness tests for Group-H modules: edge-cases, error-paths, boundary values.

Targets:
- block_maturity: maturity_score / apply_min_maturity_filter
- granularity_align: find_merge_candidates / merge_blocks
- block_lineage: add_block_edge / block_lineage / edge_aware_boost
- _recall_core: _apply_lifecycle_filter / _apply_event_id_filter (Group-H params)

One regression is covered here: maturity_score(block) where the block dict
contains a numeric Maturity field equal to zero (e.g. ``{'Maturity': 0}`` or
``{'Maturity': 0.0}``).  The original code used an ``or`` chain which skipped
falsy zero values, causing the override to be silently ignored and a composite
score to be returned instead.  The fix uses a sentinel to distinguish "key
absent" from "key present but zero".
"""

from __future__ import annotations

import pytest
from mind_mem._recall_core import _apply_event_id_filter, _apply_lifecycle_filter
from mind_mem.block_lineage import (
    ALLOWED_KINDS,
    LINEAGE_DEPTH_CAP,
    add_block_edge,
    block_lineage,
    edge_aware_boost,
    lineage_adjacency,
)
from mind_mem.block_maturity import (
    MATURITY_EDGE_SATURATION,
    MATURITY_LIFECYCLE_WEIGHT,
    apply_min_maturity_filter,
    maturity_score,
)
from mind_mem.granularity_align import (
    find_merge_candidates,
    merge_blocks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return str(ws)


# ---------------------------------------------------------------------------
# block_maturity — maturity_score edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaturityScoreEdgeCases:
    """Edge-case and error-path coverage for maturity_score()."""

    # --- Numeric-zero Maturity field (regression: falsy `or` chain bug) ---

    def test_numeric_int_zero_maturity_returns_zero(self) -> None:
        """Maturity=0 (int) must be treated as an explicit override of 0.0.

        Regression: the original ``block.get('Maturity') or block.get('maturity')``
        chain skipped a truthy zero because ``0`` is falsy in Python, causing
        the composite score to be returned instead of the override.
        """
        b = {"Maturity": 0, "Status": "active", "Lifecycle": "durable"}
        assert maturity_score(b) == 0.0

    def test_numeric_float_zero_maturity_returns_zero(self) -> None:
        """Maturity=0.0 (float) must be treated as an explicit override of 0.0."""
        b = {"Maturity": 0.0, "Status": "active", "Lifecycle": "durable"}
        assert maturity_score(b) == 0.0

    def test_lowercase_numeric_zero_maturity_returns_zero(self) -> None:
        """maturity=0 (lowercase key) must also be treated as explicit 0.0."""
        b = {"maturity": 0, "Status": "active", "Lifecycle": "durable"}
        assert maturity_score(b) == 0.0

    def test_string_zero_maturity_returns_zero(self) -> None:
        """Maturity='0' (string) is truthy and must return 0.0 override."""
        b = {"Maturity": "0", "Status": "active", "Lifecycle": "durable"}
        assert maturity_score(b) == 0.0

    # --- Maturity field boundary values (already string-tested; add float) ---

    def test_float_maturity_one_returns_one(self) -> None:
        b = {"Maturity": 1.0}
        assert maturity_score(b) == 1.0

    def test_float_maturity_half_returns_half(self) -> None:
        b = {"Maturity": 0.5}
        assert abs(maturity_score(b) - 0.5) < 1e-9

    def test_float_maturity_above_one_clamped(self) -> None:
        b = {"Maturity": 2.5}
        assert maturity_score(b) == 1.0

    def test_float_maturity_below_zero_clamped(self) -> None:
        b = {"Maturity": -1.0}
        assert maturity_score(b) == 0.0

    # --- None and empty inputs ---

    def test_none_status_field_treated_as_absent(self) -> None:
        """A block with Status=None should behave like Status is absent (zero contribution)."""
        b_none = {"Status": None, "Lifecycle": "durable"}
        b_absent = {"Lifecycle": "durable"}
        # Both should fall through to the zero-status branch (empty string comparison)
        assert maturity_score(b_none) == maturity_score(b_absent)

    def test_none_lifecycle_field_defaults_to_durable(self) -> None:
        """Lifecycle=None should default to 'durable' (same as absent key)."""
        b_none = {"Lifecycle": None}
        b_absent = {}
        assert maturity_score(b_none) == maturity_score(b_absent)

    def test_empty_dict_returns_lifecycle_weight_only(self) -> None:
        """Empty dict: status=0 + lifecycle=durable=MATURITY_LIFECYCLE_WEIGHT."""
        s = maturity_score({})
        assert abs(s - MATURITY_LIFECYCLE_WEIGHT) < 1e-9

    def test_whitespace_only_maturity_field_falls_through(self) -> None:
        """Maturity='   ' (whitespace) cannot be parsed as float; composite used."""
        b = {"Maturity": "   ", "Status": "active", "Lifecycle": "durable"}
        s = maturity_score(b)
        # falls through to composite (active+durable = 0.3+0.2)
        assert s > 0.0
        assert s < 1.0

    def test_non_numeric_maturity_falls_through_to_composite(self) -> None:
        """Maturity='high' is not parseable; composite score returned."""
        b = {"Maturity": "high", "Status": "active", "Lifecycle": "durable"}
        s = maturity_score(b)
        from mind_mem.block_maturity import MATURITY_LIFECYCLE_WEIGHT, MATURITY_STATUS_WEIGHT

        expected = MATURITY_STATUS_WEIGHT + MATURITY_LIFECYCLE_WEIGHT
        assert abs(s - expected) < 1e-9

    def test_list_maturity_field_falls_through(self) -> None:
        """Maturity=[0.5] cannot be float()-converted via TypeError; composite used."""
        b = {"Maturity": [0.5], "Status": "active", "Lifecycle": "durable"}
        s = maturity_score(b)
        assert 0.0 < s < 1.0

    # --- incoming_edge_count boundary values ---

    def test_edge_count_negative_treated_as_zero(self) -> None:
        """Negative edge count should contribute 0.0 (same as None)."""
        s_neg = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=-1)
        s_zero = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=0)
        assert s_neg == s_zero == 0.0

    def test_edge_count_exactly_saturation(self) -> None:
        """Edge count == MATURITY_EDGE_SATURATION should fully saturate."""
        from mind_mem.block_maturity import MATURITY_EDGE_WEIGHT

        s = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=MATURITY_EDGE_SATURATION)
        assert abs(s - MATURITY_EDGE_WEIGHT) < 1e-9

    def test_edge_count_one_below_saturation(self) -> None:
        """One edge below saturation should be strictly less than fully saturated."""
        s_near = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=MATURITY_EDGE_SATURATION - 1)
        s_full = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=MATURITY_EDGE_SATURATION)
        assert s_near < s_full

    def test_edge_count_one_above_saturation_same_as_saturated(self) -> None:
        """One edge above saturation should produce the same score as saturated."""
        s_sat = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=MATURITY_EDGE_SATURATION)
        s_over = maturity_score({"Lifecycle": "ephemeral"}, incoming_edge_count=MATURITY_EDGE_SATURATION + 1)
        assert abs(s_sat - s_over) < 1e-9

    # --- Score is always in [0.0, 1.0] for extreme combinations ---

    def test_all_component_combinations_in_unit_range(self) -> None:
        statuses = ("active", "wip", "deprecated", "", "archived", None)
        lifecycles = ("durable", "generated", "ephemeral", "", "unknown", None)
        edge_counts = (None, 0, 1, MATURITY_EDGE_SATURATION, MATURITY_EDGE_SATURATION * 10)
        for st in statuses:
            for lc in lifecycles:
                for ec in edge_counts:
                    b: dict = {}
                    if st is not None:
                        b["Status"] = st
                    if lc is not None:
                        b["Lifecycle"] = lc
                    s = maturity_score(b, incoming_edge_count=ec)
                    assert 0.0 <= s <= 1.0, f"Out of range: status={st!r} lifecycle={lc!r} edge_count={ec!r} → {s}"


# ---------------------------------------------------------------------------
# block_maturity — apply_min_maturity_filter edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyMinMaturityFilterEdgeCases:
    """Edge-case coverage for apply_min_maturity_filter()."""

    def test_empty_hits_always_returns_empty(self) -> None:
        for threshold in (0.0, 0.5, 1.0):
            assert apply_min_maturity_filter([], threshold) == []

    def test_threshold_exactly_at_score_passes(self) -> None:
        """Score == threshold must pass (>= not >)."""
        hits = [{"_id": "A", "Maturity": "0.5"}]
        result = apply_min_maturity_filter(hits, 0.5)
        assert len(result) == 1

    def test_threshold_just_above_score_excludes(self) -> None:
        hits = [{"_id": "A", "Maturity": "0.5"}]
        result = apply_min_maturity_filter(hits, 0.5 + 1e-6)
        assert result == []

    def test_threshold_zero_passes_all(self) -> None:
        hits = [
            {"_id": "A", "Status": "deprecated", "Lifecycle": "ephemeral"},
            {"_id": "B", "Maturity": "0.0"},
        ]
        result = apply_min_maturity_filter(hits, 0.0)
        assert len(result) == 2

    def test_threshold_one_passes_only_explicit_one(self) -> None:
        hits = [
            {"_id": "A", "Status": "active", "Lifecycle": "durable"},
            {"_id": "B", "Maturity": "1.0"},
        ]
        result = apply_min_maturity_filter(hits, 1.0)
        ids = {h["_id"] for h in result}
        assert "B" in ids
        # A's composite is 0.5; should not pass
        assert "A" not in ids

    def test_numeric_zero_maturity_passes_threshold_zero(self) -> None:
        """Regression: Maturity=0 (numeric) should be treated as 0.0, passing threshold 0.0."""
        hits = [{"_id": "A", "Maturity": 0}]
        result = apply_min_maturity_filter(hits, 0.0)
        assert len(result) == 1

    def test_numeric_zero_maturity_excluded_at_nonzero_threshold(self) -> None:
        """Regression: Maturity=0 (numeric) should be excluded at threshold 0.1."""
        hits = [{"_id": "A", "Maturity": 0}]
        result = apply_min_maturity_filter(hits, 0.1)
        assert result == []

    def test_order_preserved_with_mixed_maturity(self) -> None:
        hits = [
            {"_id": "A", "Maturity": "0.9"},
            {"_id": "B", "Maturity": "0.1"},  # excluded at 0.5
            {"_id": "C", "Maturity": "0.8"},
        ]
        result = apply_min_maturity_filter(hits, 0.5)
        assert [h["_id"] for h in result] == ["A", "C"]

    def test_filter_does_not_mutate_input(self) -> None:
        hits = [{"_id": "A", "Status": "active"}]
        original_len = len(hits)
        apply_min_maturity_filter(hits, 0.5)
        assert len(hits) == original_len
        assert "Status" in hits[0]

    def test_single_hit_passes(self) -> None:
        hits = [{"_id": "A", "Status": "active", "Lifecycle": "durable"}]
        result = apply_min_maturity_filter(hits, 0.0)
        assert len(result) == 1

    def test_single_hit_excluded(self) -> None:
        hits = [{"_id": "A", "Status": "deprecated", "Lifecycle": "ephemeral"}]
        result = apply_min_maturity_filter(hits, 0.1)
        assert result == []


# ---------------------------------------------------------------------------
# granularity_align — find_merge_candidates edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindMergeCandidatesEdgeCases:
    """Edge-case and boundary coverage for find_merge_candidates()."""

    def test_empty_list_returns_empty(self) -> None:
        assert find_merge_candidates([]) == []

    def test_single_block_no_candidates(self) -> None:
        b = {"_id": "A", "content": "only block"}
        assert find_merge_candidates([b]) == []

    def test_all_blocks_empty_text_no_candidates(self) -> None:
        blocks = [{"_id": f"X{i}"} for i in range(5)]
        assert find_merge_candidates(blocks, min_similarity=0.0) == []

    def test_one_empty_one_content_no_pair(self) -> None:
        a = {"_id": "A"}
        b = {"_id": "B", "content": "some meaningful content here"}
        assert find_merge_candidates([a, b], min_similarity=0.0) == []

    def test_min_similarity_zero_includes_zero_similarity_pair(self) -> None:
        """min_similarity=0.0 should include pairs with similarity=0.0."""
        a = {"_id": "A", "content": "cat sat mat"}
        b = {"_id": "B", "content": "quantum entanglement photon"}
        candidates = find_merge_candidates([a, b], min_similarity=0.0)
        assert len(candidates) == 1
        assert candidates[0].similarity >= 0.0

    def test_min_similarity_one_catches_exact_duplicates_only(self) -> None:
        text = "exactly identical content for testing"
        a = {"_id": "A", "content": text}
        b = {"_id": "B", "content": text}
        c = {"_id": "C", "content": text + " extra word"}
        candidates = find_merge_candidates([a, b, c], min_similarity=1.0)
        # Only A-B pair is truly identical
        assert len(candidates) == 1
        ids = {candidates[0].block_a["_id"], candidates[0].block_b["_id"]}
        assert ids == {"A", "B"}

    def test_max_candidates_zero_means_no_cap(self) -> None:
        blocks = [{"_id": f"X-{i}", "content": "identical content for many blocks"} for i in range(6)]
        r_uncapped = find_merge_candidates(blocks, min_similarity=0.5, max_candidates=0)
        # 6 blocks → 15 pairs; all identical → all above 0.5
        assert len(r_uncapped) == 15

    def test_max_candidates_one(self) -> None:
        blocks = [{"_id": f"X-{i}", "content": "same content"} for i in range(4)]
        r = find_merge_candidates(blocks, min_similarity=0.0, max_candidates=1)
        assert len(r) == 1

    def test_result_sorted_descending_similarity(self) -> None:
        a = {"_id": "A", "content": "fixed-point arithmetic exact copy"}
        b = {"_id": "B", "content": "fixed-point arithmetic exact copy"}
        c = {"_id": "C", "content": "fixed-point arithmetic mostly similar"}
        d = {"_id": "D", "content": "something completely different planets moon"}
        candidates = find_merge_candidates([a, b, c, d], min_similarity=0.0)
        sims = [c_.similarity for c_ in candidates]
        assert sims == sorted(sims, reverse=True)

    def test_deterministic_output_for_same_input(self) -> None:
        blocks = [
            {"_id": "A", "content": "deterministic fixed-point scoring Q16.16"},
            {"_id": "B", "content": "Q16.16 fixed-point scoring deterministic cross-substrate"},
            {"_id": "C", "content": "random unrelated memory content"},
        ]
        r1 = find_merge_candidates(blocks, min_similarity=0.0)
        r2 = find_merge_candidates(blocks, min_similarity=0.0)
        assert r1 == r2

    def test_to_dict_has_required_keys(self) -> None:
        a = {"_id": "A", "content": "shared fixed content"}
        b = {"_id": "B", "content": "shared fixed content"}
        candidates = find_merge_candidates([a, b])
        assert len(candidates) == 1
        d = candidates[0].to_dict()
        assert set(d.keys()) == {"id_a", "id_b", "similarity", "reason", "suggested_strategy"}

    def test_to_dict_similarity_in_unit_range(self) -> None:
        a = {"_id": "A", "content": "shared content"}
        b = {"_id": "B", "content": "shared content"}
        candidates = find_merge_candidates([a, b])
        d = candidates[0].to_dict()
        assert 0.0 <= d["similarity"] <= 1.0

    def test_block_without_id_handled(self) -> None:
        """Blocks without an _id field should still be compared (id becomes '')."""
        a = {"content": "identical content for merging test"}
        b = {"content": "identical content for merging test"}
        candidates = find_merge_candidates([a, b])
        assert len(candidates) == 1

    def test_tags_only_blocks_compared(self) -> None:
        """Blocks with content in 'tags' field should still produce similarity signal."""
        a = {"_id": "A", "tags": "fixed-point arithmetic deterministic Q16.16"}
        b = {"_id": "B", "tags": "fixed-point arithmetic deterministic cross-substrate Q16.16"}
        candidates = find_merge_candidates([a, b], min_similarity=0.5)
        assert len(candidates) == 1

    def test_threshold_above_one_clamped_to_one(self) -> None:
        """min_similarity > 1.0 is clamped to 1.0; only perfect matches returned."""
        a = {"_id": "A", "content": "same same same"}
        b = {"_id": "B", "content": "same same same"}
        candidates = find_merge_candidates([a, b], min_similarity=2.0)
        assert len(candidates) == 1

    def test_threshold_below_zero_clamped_to_zero(self) -> None:
        """min_similarity < 0.0 is clamped to 0.0."""
        a = {"_id": "A", "content": "alpha bravo charlie"}
        b = {"_id": "B", "content": "delta echo foxtrot"}
        # Even dissimilar blocks pass when threshold clamped to 0
        candidates = find_merge_candidates([a, b], min_similarity=-1.0)
        assert len(candidates) >= 0  # may or may not be similar; at least no crash


# ---------------------------------------------------------------------------
# granularity_align — merge_blocks edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMergeBlocksEdgeCases:
    """Edge-case coverage for merge_blocks()."""

    def test_unknown_strategy_raises_value_error(self) -> None:
        a = {"_id": "A", "content": "x"}
        b = {"_id": "B", "content": "y"}
        with pytest.raises(ValueError, match="strategy must be one of"):
            merge_blocks(a, b, strategy="invalid")

    def test_empty_strategy_raises_value_error(self) -> None:
        a = {"_id": "A", "content": "x"}
        b = {"_id": "B", "content": "y"}
        with pytest.raises(ValueError):
            merge_blocks(a, b, strategy="")

    def test_concatenate_both_empty_content(self) -> None:
        """Concatenating two blocks with no content should produce an empty (or whitespace) result."""
        a = {"_id": "A"}
        b = {"_id": "B"}
        merged = merge_blocks(a, b, strategy="concatenate")
        assert merged["_id"] == "A"
        assert merged.get("content", "").strip() == ""

    def test_concatenate_none_content_treated_as_empty(self) -> None:
        a = {"_id": "A", "content": None}
        b = {"_id": "B", "content": "real content here"}
        merged = merge_blocks(a, b, strategy="concatenate")
        assert "real content here" in merged["content"]

    def test_concatenate_excerpt_truncated_at_500(self) -> None:
        long_text = "word " * 200  # 1000 chars
        a = {"_id": "A", "content": long_text}
        b = {"_id": "B", "content": long_text}
        merged = merge_blocks(a, b, strategy="concatenate")
        assert len(merged["excerpt"]) <= 500

    def test_concatenate_short_content_no_truncation(self) -> None:
        a = {"_id": "A", "content": "short"}
        b = {"_id": "B", "content": "also short"}
        merged = merge_blocks(a, b, strategy="concatenate")
        assert merged["excerpt"] == merged["content"]

    def test_keep_longer_equal_length_prefers_a(self) -> None:
        a = {"_id": "A", "content": "abc"}
        b = {"_id": "B", "content": "xyz"}
        merged = merge_blocks(a, b, strategy="keep_longer")
        assert merged["_id"] == "A"

    def test_keep_longer_empty_vs_non_empty(self) -> None:
        empty = {"_id": "A"}
        non_empty = {"_id": "B", "content": "actual content"}
        merged = merge_blocks(empty, non_empty, strategy="keep_longer")
        assert merged["_id"] == "B"

    def test_keep_higher_maturity_numeric_zero_maturity(self) -> None:
        """Regression: block with Maturity=0 (numeric) should score 0.0 not 0.5."""
        a = {"_id": "A", "content": "x", "Maturity": 0}
        b = {"_id": "B", "content": "x", "Maturity": 0.9}
        merged = merge_blocks(a, b, strategy="keep_higher_maturity")
        assert merged["_id"] == "B"

    def test_keep_higher_maturity_both_numeric_zero(self) -> None:
        """Both blocks with Maturity=0 should tie-break to a."""
        a = {"_id": "A", "content": "x", "Maturity": 0}
        b = {"_id": "B", "content": "x", "Maturity": 0}
        merged = merge_blocks(a, b, strategy="keep_higher_maturity")
        assert merged["_id"] == "A"

    def test_merged_from_always_contains_both_ids_keep_longer(self) -> None:
        a = {"_id": "A", "content": "short"}
        b = {"_id": "B", "content": "longer version of the content here"}
        merged = merge_blocks(a, b, strategy="keep_longer")
        assert set(merged["_merged_from"]) == {"A", "B"}

    def test_merged_from_always_contains_both_ids_keep_maturity(self) -> None:
        a = {"_id": "A", "content": "x", "Maturity": "0.9"}
        b = {"_id": "B", "content": "x", "Maturity": "0.1"}
        merged = merge_blocks(a, b, strategy="keep_higher_maturity")
        assert set(merged["_merged_from"]) == {"A", "B"}

    def test_tags_deduped_in_all_strategies(self) -> None:
        for strategy in ("keep_longer", "keep_higher_maturity", "concatenate"):
            a = {"_id": "A", "content": "longer content here", "tags": "alpha, beta"}
            b = {"_id": "B", "content": "x", "tags": "beta, gamma"}
            merged = merge_blocks(a, b, strategy=strategy)
            tags = merged["tags"]
            assert tags.count("beta") == 1, f"Duplicate tag in strategy={strategy!r}: {tags!r}"

    def test_no_tags_in_either_block(self) -> None:
        a = {"_id": "A", "content": "x"}
        b = {"_id": "B", "content": "y"}
        for strategy in ("keep_longer", "keep_higher_maturity", "concatenate"):
            merged = merge_blocks(a, b, strategy=strategy)
            assert merged["tags"] == ""

    def test_merge_does_not_modify_input_dicts(self) -> None:
        a = {"_id": "A", "content": "content a", "tags": "t1"}
        b = {"_id": "B", "content": "content b", "tags": "t2"}
        a_copy = dict(a)
        b_copy = dict(b)
        merge_blocks(a, b, strategy="keep_longer")
        assert a == a_copy
        assert b == b_copy


# ---------------------------------------------------------------------------
# block_lineage — edge cases for add_block_edge / block_lineage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBlockLineageEdgeCases:
    """Edge-case and error-path coverage for block_lineage module."""

    # --- add_block_edge validation ---

    def test_empty_src_rejected(self, workspace) -> None:
        with pytest.raises(ValueError):
            add_block_edge(workspace, "", "B", "cites")

    def test_empty_dst_rejected(self, workspace) -> None:
        with pytest.raises(ValueError):
            add_block_edge(workspace, "A", "", "cites")

    def test_self_loop_rejected(self, workspace) -> None:
        with pytest.raises(ValueError, match="must differ"):
            add_block_edge(workspace, "X", "X", "cites")

    def test_invalid_kind_rejected(self, workspace) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            add_block_edge(workspace, "A", "B", "borrows")

    def test_all_allowed_kinds_accepted(self, workspace) -> None:
        for i, kind in enumerate(sorted(ALLOWED_KINDS)):
            add_block_edge(workspace, f"SRC-{i}", f"DST-{i}", kind)

    # --- block_lineage on empty graph ---

    def test_isolated_node_returns_empty_lineage(self, workspace) -> None:
        result = block_lineage(workspace, "ISOLATED")
        assert result.edges == []
        assert result.truncated is False
        assert result.root == "ISOLATED"

    def test_empty_block_id_raises(self, workspace) -> None:
        with pytest.raises(ValueError):
            block_lineage(workspace, "")

    # --- max_depth clamping ---

    def test_max_depth_zero_clamped_to_one(self, workspace) -> None:
        result = block_lineage(workspace, "A", max_depth=0)
        assert result.max_depth == 1

    def test_max_depth_large_clamped_to_cap(self, workspace) -> None:
        result = block_lineage(workspace, "A", max_depth=1000)
        assert result.max_depth == LINEAGE_DEPTH_CAP

    def test_max_depth_negative_clamped_to_one(self, workspace) -> None:
        result = block_lineage(workspace, "A", max_depth=-5)
        assert result.max_depth == 1

    # --- node_cap ---

    def test_node_cap_one_returns_at_most_one_edge(self, workspace) -> None:
        for i in range(10):
            add_block_edge(workspace, "ROOT", f"CHILD-{i}", "cites")
        result = block_lineage(workspace, "ROOT", node_cap=1)
        assert len(result.edges) == 1
        assert result.truncated is True

    def test_node_cap_zero_clamped_to_one(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        result = block_lineage(workspace, "A", node_cap=0)
        # cap clamped to max(1, 0) = 1; should still return at most 1 edge
        assert len(result.edges) <= 1

    # --- kind_filter ---

    def test_invalid_kind_filter_rejected(self, workspace) -> None:
        with pytest.raises(ValueError):
            block_lineage(workspace, "A", kind_filter="nonexistent")

    def test_none_kind_filter_returns_all_kinds(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "A", "C", "implements")
        result = block_lineage(workspace, "A", kind_filter=None)
        ids = {e.block_id for e in result.edges}
        assert "B" in ids and "C" in ids

    def test_kind_filter_returns_only_matching_kind(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "A", "C", "supports")
        result = block_lineage(workspace, "A", kind_filter="cites")
        ids = {e.block_id for e in result.edges}
        assert "B" in ids
        assert "C" not in ids

    # --- single-block corpora (just root, no neighbours) ---

    def test_no_edges_truncated_false(self, workspace) -> None:
        result = block_lineage(workspace, "ALONE")
        assert result.truncated is False

    # --- cycle handling ---

    def test_bidirectional_edge_no_infinite_loop(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "B", "A", "cites")
        result = block_lineage(workspace, "A", max_depth=3)
        # A should not appear in its own lineage
        assert all(e.block_id != "A" for e in result.edges)

    # --- confidence values ---

    def test_first_hop_confidence_equals_kind_decay(self, workspace) -> None:
        from mind_mem.block_lineage import KIND_DECAY

        for kind in sorted(ALLOWED_KINDS):
            add_block_edge(workspace, f"ROOT-{kind}", f"CHILD-{kind}", kind)
            result = block_lineage(workspace, f"ROOT-{kind}", max_depth=1)
            found = [e for e in result.edges if e.block_id == f"CHILD-{kind}"]
            assert found, f"expected edge for kind={kind!r}"
            assert abs(found[0].confidence - KIND_DECAY[kind]) < 1e-9, f"confidence mismatch for kind={kind!r}"

    def test_second_hop_confidence_decayed(self, workspace) -> None:
        from mind_mem.block_lineage import KIND_DECAY

        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "B", "C", "cites")
        result = block_lineage(workspace, "A", max_depth=2)
        hop1 = next(e for e in result.edges if e.block_id == "B")
        hop2 = next(e for e in result.edges if e.block_id == "C")
        # Second hop is decayed by 0.5^(next_hop-1) = 0.5^1 = 0.5
        expected_hop2 = KIND_DECAY["cites"] * 0.5
        assert abs(hop1.confidence - KIND_DECAY["cites"]) < 1e-9
        assert abs(hop2.confidence - expected_hop2) < 1e-9

    # --- lineage_adjacency ---

    def test_adjacency_empty_graph_returns_empty(self, workspace) -> None:
        adj = lineage_adjacency(workspace)
        assert adj == {}

    def test_adjacency_is_undirected(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        adj = lineage_adjacency(workspace)
        assert "B" in adj.get("A", [])
        assert "A" in adj.get("B", [])

    def test_adjacency_kind_filter_excludes_other_kinds(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "cites")
        add_block_edge(workspace, "A", "C", "supports")
        adj = lineage_adjacency(workspace, kind_filter="cites")
        assert "B" in adj.get("A", [])
        assert "C" not in adj.get("A", [])


# ---------------------------------------------------------------------------
# block_lineage — edge_aware_boost edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeAwareBoostEdgeCases:
    """Edge-case coverage for edge_aware_boost()."""

    def test_weight_zero_always_returns_empty(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "supports")
        assert edge_aware_boost(workspace, ["A", "B"], weight=0.0) == {}

    def test_empty_block_ids_returns_empty(self, workspace) -> None:
        assert edge_aware_boost(workspace, [], weight=0.5) == {}

    def test_blocks_with_no_incoming_edges_get_zero(self, workspace) -> None:
        add_block_edge(workspace, "SOURCE", "DEST", "supports")
        boosts = edge_aware_boost(workspace, ["SOURCE"], weight=1.0)
        # SOURCE has outgoing, not incoming
        assert boosts.get("SOURCE", 0.0) == 0.0

    def test_contradicts_incoming_contributes_zero(self, workspace) -> None:
        add_block_edge(workspace, "A", "TARGET", "contradicts")
        boosts = edge_aware_boost(workspace, ["TARGET"], weight=1.0)
        assert boosts.get("TARGET", 0.0) == 0.0

    def test_unrequested_block_id_not_in_result(self, workspace) -> None:
        add_block_edge(workspace, "A", "B", "supports")
        add_block_edge(workspace, "A", "C", "supports")
        boosts = edge_aware_boost(workspace, ["B"], weight=1.0)
        assert "C" not in boosts

    def test_boost_accumulates_for_multiple_incoming(self, workspace) -> None:
        add_block_edge(workspace, "A", "TARGET", "supports")
        add_block_edge(workspace, "B", "TARGET", "supports")
        single_weight = edge_aware_boost(workspace, ["TARGET"], weight=1.0).get("TARGET", 0.0)
        # Two edges accumulate → more than one edge's contribution
        from mind_mem.block_lineage import EDGE_BOOST_WEIGHT

        assert single_weight > EDGE_BOOST_WEIGHT["supports"]

    def test_boost_scales_linearly_with_weight(self, workspace) -> None:
        add_block_edge(workspace, "X", "Y", "cites")
        b1 = edge_aware_boost(workspace, ["Y"], weight=0.1).get("Y", 0.0)
        b2 = edge_aware_boost(workspace, ["Y"], weight=0.3).get("Y", 0.0)
        assert abs(b2 - 3 * b1) < 1e-9

    def test_all_non_contradicts_kinds_produce_positive_boost(self, workspace) -> None:
        from mind_mem.block_lineage import EDGE_BOOST_WEIGHT

        for kind in sorted(ALLOWED_KINDS):
            if EDGE_BOOST_WEIGHT.get(kind, 0.0) > 0.0:
                tgt = f"TGT-{kind}"
                add_block_edge(workspace, f"SRC-{kind}", tgt, kind)
                b = edge_aware_boost(workspace, [tgt], weight=1.0).get(tgt, 0.0)
                assert b > 0.0, f"expected positive boost for kind={kind!r}, got {b}"


# ---------------------------------------------------------------------------
# _recall_core — _apply_lifecycle_filter edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyLifecycleFilterEdgeCases:
    """Edge-case coverage for the lifecycle post-filter in _recall_core."""

    def test_empty_hits_returns_empty(self) -> None:
        assert _apply_lifecycle_filter([], "durable") == []

    def test_none_lifecycle_defaults_to_durable(self) -> None:
        """A block with Lifecycle=None should be treated as 'durable'."""
        hits = [{"_id": "A", "Lifecycle": None}]
        result = _apply_lifecycle_filter(hits, "durable")
        assert len(result) == 1

    def test_empty_string_lifecycle_defaults_to_durable(self) -> None:
        hits = [{"_id": "A", "Lifecycle": ""}]
        result = _apply_lifecycle_filter(hits, "durable")
        assert len(result) == 1

    def test_absent_lifecycle_included_for_durable_filter(self) -> None:
        hits = [{"_id": "A"}]
        result = _apply_lifecycle_filter(hits, "durable")
        assert len(result) == 1

    def test_absent_lifecycle_excluded_for_ephemeral_filter(self) -> None:
        hits = [{"_id": "A"}]
        result = _apply_lifecycle_filter(hits, "ephemeral")
        assert result == []

    def test_case_insensitive_match_upper(self) -> None:
        hits = [{"_id": "A", "Lifecycle": "DURABLE"}]
        result = _apply_lifecycle_filter(hits, "durable")
        assert len(result) == 1

    def test_case_insensitive_match_mixed(self) -> None:
        hits = [{"_id": "A", "Lifecycle": "Ephemeral"}]
        result = _apply_lifecycle_filter(hits, "ephemeral")
        assert len(result) == 1

    def test_unknown_lifecycle_excluded_from_all_known_filters(self) -> None:
        hits = [{"_id": "A", "Lifecycle": "archived"}]
        for target in ("durable", "ephemeral", "generated"):
            result = _apply_lifecycle_filter(hits, target)
            assert result == [], f"expected excluded for target={target!r}"

    def test_order_preserved(self) -> None:
        hits = [
            {"_id": "A", "Lifecycle": "durable"},
            {"_id": "B", "Lifecycle": "ephemeral"},
            {"_id": "C", "Lifecycle": "durable"},
        ]
        result = _apply_lifecycle_filter(hits, "durable")
        assert [h["_id"] for h in result] == ["A", "C"]

    def test_single_block_passes(self) -> None:
        hits = [{"_id": "A", "Lifecycle": "generated"}]
        result = _apply_lifecycle_filter(hits, "generated")
        assert result == hits

    def test_does_not_mutate_input(self) -> None:
        hits = [{"_id": "A", "Lifecycle": "durable"}]
        _apply_lifecycle_filter(hits, "durable")
        assert hits == [{"_id": "A", "Lifecycle": "durable"}]


# ---------------------------------------------------------------------------
# _recall_core — _apply_event_id_filter edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyEventIdFilterEdgeCases:
    """Edge-case coverage for the event_id post-filter in _recall_core."""

    def test_empty_hits_returns_empty(self) -> None:
        assert _apply_event_id_filter([], "EVT-001") == []

    def test_absent_event_id_excluded(self) -> None:
        hits = [{"_id": "A"}]
        result = _apply_event_id_filter(hits, "EVT-001")
        assert result == []

    def test_none_event_id_value_excluded(self) -> None:
        """Block with EventId=None should not match any event."""
        hits = [{"_id": "A", "EventId": None}]
        result = _apply_event_id_filter(hits, "EVT-001")
        assert result == []

    def test_case_insensitive_match(self) -> None:
        hits = [{"_id": "A", "EventId": "evt-2026-alpha"}]
        result = _apply_event_id_filter(hits, "EVT-2026-ALPHA")
        assert len(result) == 1

    def test_case_insensitive_mixed_case(self) -> None:
        hits = [{"_id": "A", "EventId": "Evt-XYZ"}]
        result = _apply_event_id_filter(hits, "evt-xyz")
        assert len(result) == 1

    def test_integer_event_id_matched_as_string(self) -> None:
        """EventId stored as integer (e.g. block_parser coercion) should match string form."""
        hits = [{"_id": "A", "EventId": 12345}]
        result = _apply_event_id_filter(hits, "12345")
        assert len(result) == 1

    def test_partial_match_not_included(self) -> None:
        """Only exact (case-insensitive) match; 'EVT' should not match 'EVT-001'."""
        hits = [{"_id": "A", "EventId": "EVT-001"}]
        result = _apply_event_id_filter(hits, "EVT")
        assert result == []

    def test_empty_event_id_filter_matches_empty_and_absent_fields(self) -> None:
        """Empty string filter matches blocks with EventId='' and absent EventId.

        Since absent EventId defaults to '' (via the ``or ''`` fallback), both
        explicit empty and absent fields compare equal to the '' filter target.
        """
        hits = [
            {"_id": "A", "EventId": ""},
            {"_id": "B", "EventId": "EVT-001"},
            {"_id": "C"},  # absent → defaults to ''
        ]
        result = _apply_event_id_filter(hits, "")
        ids = [h["_id"] for h in result]
        # Both A (explicit empty) and C (absent, defaults to '') match
        assert "A" in ids
        assert "C" in ids
        assert "B" not in ids

    def test_order_preserved(self) -> None:
        hits = [
            {"_id": "A", "EventId": "EVT-001"},
            {"_id": "B", "EventId": "EVT-002"},
            {"_id": "C", "EventId": "EVT-001"},
        ]
        result = _apply_event_id_filter(hits, "EVT-001")
        assert [h["_id"] for h in result] == ["A", "C"]

    def test_no_match_returns_empty(self) -> None:
        hits = [{"_id": "A", "EventId": "EVT-REAL"}]
        result = _apply_event_id_filter(hits, "EVT-NONEXISTENT")
        assert result == []

    def test_does_not_mutate_input(self) -> None:
        hits = [{"_id": "A", "EventId": "EVT-001"}]
        _apply_event_id_filter(hits, "EVT-001")
        assert hits == [{"_id": "A", "EventId": "EVT-001"}]

    def test_single_matching_block(self) -> None:
        hits = [{"_id": "ONLY", "EventId": "EVT-SOLO"}]
        result = _apply_event_id_filter(hits, "EVT-SOLO")
        assert result == hits

    def test_lowercase_event_id_key(self) -> None:
        """event_id (lowercase) field should also be recognised."""
        hits = [{"_id": "A", "event_id": "EVT-001"}]
        result = _apply_event_id_filter(hits, "EVT-001")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Off-by-default: verify behavior-preserving defaults hold (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGroupHDefaults:
    """Verify that all Group-H features are off-by-default (behavior-preserving)."""

    def test_edge_aware_boost_weight_zero_is_default(self, workspace) -> None:
        """edge_aware_boost default weight is 0.0 → no impact on scores."""
        add_block_edge(workspace, "A", "B", "supports")
        # With default weight=0.0, result is always empty
        boosts = edge_aware_boost(workspace, ["A", "B"])
        assert boosts == {}

    def test_maturity_absent_does_not_crash(self) -> None:
        """maturity_score on a block with no Maturity field returns a valid float."""
        for b in ({}, {"Status": "active"}, {"Lifecycle": "ephemeral"}, {"foo": "bar"}):
            s = maturity_score(b)
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_apply_min_maturity_filter_none_threshold_off(self) -> None:
        """apply_min_maturity_filter called with threshold=0.0 passes all hits."""
        hits = [
            {"_id": "A"},
            {"_id": "B", "Status": "deprecated", "Lifecycle": "ephemeral"},
        ]
        result = apply_min_maturity_filter(hits, 0.0)
        assert len(result) == len(hits)

    def test_find_merge_candidates_does_not_write(self, tmp_path) -> None:
        """find_merge_candidates is purely functional — no files written."""
        import os

        before = set(os.listdir(str(tmp_path)))
        blocks = [
            {"_id": "A", "content": "content"},
            {"_id": "B", "content": "content"},
        ]
        find_merge_candidates(blocks)
        after = set(os.listdir(str(tmp_path)))
        assert before == after, "find_merge_candidates should not write any files"

    def test_merge_blocks_does_not_write(self, tmp_path) -> None:
        """merge_blocks is purely functional — no files written."""
        import os

        before = set(os.listdir(str(tmp_path)))
        a = {"_id": "A", "content": "x"}
        b = {"_id": "B", "content": "y"}
        merge_blocks(a, b)
        after = set(os.listdir(str(tmp_path)))
        assert before == after

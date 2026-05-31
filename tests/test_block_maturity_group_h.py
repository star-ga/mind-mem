"""Tests for Group H maturity metric — consolidation gate.

Covers:
- maturity_score() component weights and composite computation.
- Explicit Maturity frontmatter override.
- apply_min_maturity_filter() threshold semantics.
- recall(min_maturity=...) integration: blocks below threshold excluded;
  None (default) = no-op.
- Block-parser round-trip: Maturity field parsed as a top-level field.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mind_mem._recall_core import recall
from mind_mem.block_maturity import (
    MATURITY_EDGE_SATURATION,
    MATURITY_EDGE_WEIGHT,
    MATURITY_LIFECYCLE_WEIGHT,
    MATURITY_STATUS_WEIGHT,
    apply_min_maturity_filter,
    maturity_score,
)
from mind_mem.block_parser import parse_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(text: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        path = fh.name
    try:
        return parse_file(path)
    finally:
        os.unlink(path)


def _block(**kwargs) -> dict:
    """Create a minimal block dict for score testing."""
    return dict(kwargs)


# ---------------------------------------------------------------------------
# maturity_score — status component
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaturityScoreStatus:
    def test_active_status_max_contribution(self) -> None:
        b = _block(Status="active", Lifecycle="durable")
        s = maturity_score(b)
        assert s >= MATURITY_STATUS_WEIGHT  # at minimum the status weight

    def test_wip_status_half_contribution(self) -> None:
        active_score = maturity_score(_block(Status="active", Lifecycle="ephemeral"))
        wip_score = maturity_score(_block(Status="wip", Lifecycle="ephemeral"))
        # wip contributes 0.5 × status_weight; active contributes 1.0 × status_weight
        expected_diff = MATURITY_STATUS_WEIGHT * 0.5
        assert abs((active_score - wip_score) - expected_diff) < 1e-9

    def test_deprecated_status_zero_contribution(self) -> None:
        s = maturity_score(_block(Status="deprecated"))
        # only lifecycle component (default durable) contributes
        assert abs(s - MATURITY_LIFECYCLE_WEIGHT) < 1e-9

    def test_unknown_status_zero_contribution(self) -> None:
        s = maturity_score(_block(Status="unknown"))
        # only lifecycle component (default durable) contributes
        assert abs(s - MATURITY_LIFECYCLE_WEIGHT) < 1e-9

    def test_missing_status_zero_contribution(self) -> None:
        s_none = maturity_score(_block())
        s_active = maturity_score(_block(Status="active"))
        assert s_active > s_none


# ---------------------------------------------------------------------------
# maturity_score — lifecycle component
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaturityScoreLifecycle:
    def test_durable_lifecycle_max_contribution(self) -> None:
        b = _block(Status="deprecated", Lifecycle="durable")
        s = maturity_score(b)
        assert abs(s - MATURITY_LIFECYCLE_WEIGHT) < 1e-9

    def test_generated_lifecycle_half_contribution(self) -> None:
        durable = maturity_score(_block(Status="deprecated", Lifecycle="durable"))
        generated = maturity_score(_block(Status="deprecated", Lifecycle="generated"))
        expected_diff = MATURITY_LIFECYCLE_WEIGHT * 0.5
        assert abs((durable - generated) - expected_diff) < 1e-9

    def test_ephemeral_lifecycle_zero_contribution(self) -> None:
        s = maturity_score(_block(Status="deprecated", Lifecycle="ephemeral"))
        assert s == 0.0

    def test_missing_lifecycle_defaults_to_durable(self) -> None:
        s_no_lc = maturity_score(_block(Status="deprecated"))
        s_durable = maturity_score(_block(Status="deprecated", Lifecycle="durable"))
        assert abs(s_no_lc - s_durable) < 1e-9

    def test_case_insensitive_lifecycle(self) -> None:
        s_lower = maturity_score(_block(Lifecycle="durable"))
        s_title = maturity_score(_block(Lifecycle="Durable"))
        s_upper = maturity_score(_block(Lifecycle="DURABLE"))
        assert abs(s_lower - s_title) < 1e-9
        assert abs(s_lower - s_upper) < 1e-9


# ---------------------------------------------------------------------------
# maturity_score — edge corroboration component
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaturityScoreEdge:
    def test_no_edges_zero_contribution(self) -> None:
        s = maturity_score(_block(Status="deprecated"), incoming_edge_count=None)
        # only lifecycle (default durable) contributes
        assert abs(s - MATURITY_LIFECYCLE_WEIGHT) < 1e-9

    def test_zero_edges_zero_contribution(self) -> None:
        s = maturity_score(_block(Status="deprecated"), incoming_edge_count=0)
        assert abs(s - MATURITY_LIFECYCLE_WEIGHT) < 1e-9

    def test_partial_edges_partial_contribution(self) -> None:
        # With half the saturation edges, edge component = 0.5 × MATURITY_EDGE_WEIGHT
        half = MATURITY_EDGE_SATURATION // 2
        s = maturity_score(_block(Status="deprecated", Lifecycle="ephemeral"), incoming_edge_count=half)
        expected = (half / MATURITY_EDGE_SATURATION) * MATURITY_EDGE_WEIGHT
        assert abs(s - expected) < 1e-9

    def test_saturated_edges_full_contribution(self) -> None:
        s = maturity_score(_block(Status="deprecated", Lifecycle="ephemeral"), incoming_edge_count=MATURITY_EDGE_SATURATION)
        assert abs(s - MATURITY_EDGE_WEIGHT) < 1e-9

    def test_over_saturated_edges_capped_at_full(self) -> None:
        s_sat = maturity_score(_block(Lifecycle="ephemeral"), incoming_edge_count=MATURITY_EDGE_SATURATION)
        s_over = maturity_score(_block(Lifecycle="ephemeral"), incoming_edge_count=MATURITY_EDGE_SATURATION * 3)
        assert abs(s_sat - s_over) < 1e-9

    def test_more_edges_higher_score(self) -> None:
        s1 = maturity_score(_block(), incoming_edge_count=1)
        s2 = maturity_score(_block(), incoming_edge_count=2)
        assert s2 > s1


# ---------------------------------------------------------------------------
# maturity_score — explicit Maturity frontmatter override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaturityScoreExplicitOverride:
    def test_explicit_maturity_returned_directly(self) -> None:
        b = _block(Maturity="0.75", Status="active", Lifecycle="durable")
        assert abs(maturity_score(b) - 0.75) < 1e-9

    def test_explicit_maturity_overrides_components(self) -> None:
        # Even a deprecated+ephemeral block returns its explicit score.
        b = _block(Maturity="0.9", Status="deprecated", Lifecycle="ephemeral")
        assert abs(maturity_score(b) - 0.9) < 1e-9

    def test_explicit_maturity_clamped_above_one(self) -> None:
        b = _block(Maturity="1.5")
        assert maturity_score(b) == 1.0

    def test_explicit_maturity_clamped_below_zero(self) -> None:
        b = _block(Maturity="-0.1")
        assert maturity_score(b) == 0.0

    def test_explicit_maturity_zero_valid(self) -> None:
        b = _block(Maturity="0.0", Status="active", Lifecycle="durable")
        assert maturity_score(b) == 0.0

    def test_explicit_maturity_one_valid(self) -> None:
        b = _block(Maturity="1.0")
        assert maturity_score(b) == 1.0

    def test_invalid_maturity_field_falls_through(self) -> None:
        b = _block(Maturity="not-a-number", Status="active", Lifecycle="durable")
        # Falls through to composite; active+durable = status_w + lifecycle_w
        expected = MATURITY_STATUS_WEIGHT + MATURITY_LIFECYCLE_WEIGHT
        assert abs(maturity_score(b) - expected) < 1e-9

    def test_numeric_maturity_field(self) -> None:
        # block_parser coerces ints; ensure maturity_score handles numeric values.
        b = _block(Maturity=0.6)
        assert abs(maturity_score(b) - 0.6) < 1e-9

    def test_lowercase_maturity_key(self) -> None:
        b = _block(maturity="0.5")
        assert abs(maturity_score(b) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# maturity_score — composite ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaturityScoreOrdering:
    def test_active_durable_beats_wip_ephemeral(self) -> None:
        high = maturity_score(_block(Status="active", Lifecycle="durable"))
        low = maturity_score(_block(Status="wip", Lifecycle="ephemeral"))
        assert high > low

    def test_score_in_unit_range(self) -> None:
        for status in ("active", "wip", "deprecated", "", "archived"):
            for lc in ("durable", "generated", "ephemeral", ""):
                s = maturity_score(_block(Status=status, Lifecycle=lc), incoming_edge_count=MATURITY_EDGE_SATURATION)
                assert 0.0 <= s <= 1.0, f"Out of range for status={status!r} lifecycle={lc!r}: {s}"

    def test_max_composite_at_active_durable_saturated_edges(self) -> None:
        s = maturity_score(_block(Status="active", Lifecycle="durable"), incoming_edge_count=MATURITY_EDGE_SATURATION)
        expected = MATURITY_STATUS_WEIGHT + MATURITY_LIFECYCLE_WEIGHT + MATURITY_EDGE_WEIGHT
        assert abs(s - expected) < 1e-9

    def test_weights_sum_to_one(self) -> None:
        total = MATURITY_STATUS_WEIGHT + MATURITY_LIFECYCLE_WEIGHT + MATURITY_EDGE_WEIGHT
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# apply_min_maturity_filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyMinMaturityFilter:
    def _make_hits(self) -> list[dict]:
        return [
            {"_id": "A", "score": 0.9, "Status": "active", "Lifecycle": "durable"},  # high maturity
            {"_id": "B", "score": 0.8, "Status": "wip", "Lifecycle": "generated"},  # mid maturity
            {"_id": "C", "score": 0.7, "Status": "deprecated", "Lifecycle": "ephemeral"},  # zero maturity
            {"_id": "D", "score": 0.6, "Maturity": "0.8"},  # explicit high
        ]

    def test_zero_threshold_passes_all(self) -> None:
        hits = self._make_hits()
        result = apply_min_maturity_filter(hits, 0.0)
        assert len(result) == 4

    def test_threshold_one_passes_none(self) -> None:
        hits = self._make_hits()
        result = apply_min_maturity_filter(hits, 1.0)
        # Only blocks with explicit Maturity: 1.0 or all-max composite pass
        for h in result:
            assert maturity_score(h) >= 1.0

    def test_order_preserved(self) -> None:
        hits = self._make_hits()
        result = apply_min_maturity_filter(hits, 0.0)
        ids = [h["_id"] for h in result]
        assert ids == ["A", "B", "C", "D"]

    def test_deprecated_ephemeral_excluded_at_modest_threshold(self) -> None:
        hits = self._make_hits()
        result = apply_min_maturity_filter(hits, 0.1)
        ids = {h["_id"] for h in result}
        assert "C" not in ids  # deprecated + ephemeral = 0.0 maturity

    def test_explicit_maturity_override_respected(self) -> None:
        hits = [
            {"_id": "X", "score": 1.0, "Maturity": "0.9"},
            {"_id": "Y", "score": 0.9, "Status": "active", "Lifecycle": "durable"},  # computed ~0.5
        ]
        result = apply_min_maturity_filter(hits, 0.8)
        ids = {h["_id"] for h in result}
        assert "X" in ids

    def test_empty_input_returns_empty(self) -> None:
        assert apply_min_maturity_filter([], 0.5) == []

    def test_filter_does_not_mutate_input(self) -> None:
        hits = self._make_hits()
        original_len = len(hits)
        apply_min_maturity_filter(hits, 0.5)
        assert len(hits) == original_len


# ---------------------------------------------------------------------------
# Block-parser round-trip: Maturity field
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaturityFieldParsing:
    def test_maturity_field_parsed(self) -> None:
        blocks = _parse("[D-MAT-001]\nStatement: test block\nMaturity: 0.75\n")
        assert len(blocks) == 1
        assert blocks[0]["Maturity"] == "0.75"

    def test_maturity_zero_parsed(self) -> None:
        blocks = _parse("[D-MAT-002]\nStatement: zero maturity\nMaturity: 0.0\n")
        assert len(blocks) == 1
        assert blocks[0]["Maturity"] == "0.0"

    def test_no_maturity_field_absent(self) -> None:
        blocks = _parse("[D-MAT-003]\nStatement: no maturity field\n")
        assert len(blocks) == 1
        assert "Maturity" not in blocks[0]

    def test_maturity_coexists_with_lifecycle_and_status(self) -> None:
        blocks = _parse("[D-MAT-004]\nStatement: all fields\nStatus: active\nLifecycle: durable\nMaturity: 0.6\n")
        assert len(blocks) == 1
        b = blocks[0]
        assert b["Status"] == "active"
        assert b["Lifecycle"] == "durable"
        assert b["Maturity"] == "0.6"


# ---------------------------------------------------------------------------
# recall() integration — min_maturity parameter
# ---------------------------------------------------------------------------


def _make_workspace_with_maturity() -> tuple[str, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    for d in ("decisions", "tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    content = (
        "[D-MAT-I-001]\nStatement: high maturity deployment decision active durable\n"
        "Status: active\nLifecycle: durable\nStatus: active\n\n"
        "[D-MAT-I-002]\nStatement: mid maturity deployment decision wip generated\n"
        "Status: wip\nLifecycle: generated\n\n"
        "[D-MAT-I-003]\nStatement: zero maturity deployment decision deprecated ephemeral\n"
        "Status: deprecated\nLifecycle: ephemeral\n\n"
        "[D-MAT-I-004]\nStatement: explicit high maturity deployment decision\n"
        "Maturity: 0.9\nStatus: deprecated\n\n"
    )
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(content)
    return ws, td


@pytest.mark.unit
class TestRecallMinMaturityFilter:
    @pytest.fixture(scope="class")
    def workspace(self):
        ws, td = _make_workspace_with_maturity()
        yield ws
        td.cleanup()

    def test_no_filter_returns_all(self, workspace: str) -> None:
        results = recall(workspace, "deployment decision", limit=10)
        assert len(results) >= 3

    def test_filter_none_is_no_op(self, workspace: str) -> None:
        all_results = recall(workspace, "deployment decision", limit=10)
        filtered = recall(workspace, "deployment decision", limit=10, min_maturity=None)
        assert len(all_results) == len(filtered)

    def test_filter_zero_is_no_op(self, workspace: str) -> None:
        all_results = recall(workspace, "deployment decision", limit=10)
        filtered = recall(workspace, "deployment decision", limit=10, min_maturity=0.0)
        assert len(all_results) == len(filtered)

    def test_modest_threshold_excludes_zero_maturity(self, workspace: str) -> None:
        results = recall(workspace, "deployment decision", limit=10, min_maturity=0.1)
        ids = {r["_id"] for r in results}
        assert "D-MAT-I-003" not in ids

    def test_explicit_maturity_block_passes_high_threshold(self, workspace: str) -> None:
        results = recall(workspace, "deployment decision", limit=10, min_maturity=0.8)
        ids = {r["_id"] for r in results}
        assert "D-MAT-I-004" in ids

    def test_active_durable_block_passes_moderate_threshold(self, workspace: str) -> None:
        # active+durable = 0.3 + 0.2 = 0.5 composite (no edge data in this path)
        results = recall(workspace, "deployment decision", limit=10, min_maturity=0.4)
        ids = {r["_id"] for r in results}
        assert "D-MAT-I-001" in ids

    def test_high_threshold_excludes_low_maturity_blocks(self, workspace: str) -> None:
        results = recall(workspace, "deployment decision", limit=10, min_maturity=0.85)
        ids = {r["_id"] for r in results}
        # Only blocks with explicit Maturity >= 0.85 should appear
        assert "D-MAT-I-003" not in ids

"""Tests for calibration feedback loop.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import os
import time

import pytest

from mind_mem.calibration import (
    CALIBRATION_WINDOW_DAYS,
    MAX_CALIBRATION_WEIGHT,
    MIN_CALIBRATION_WEIGHT,
    MIN_FEEDBACK_THRESHOLD,
    CalibrationManager,
    _compute_weight,
    make_query_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with the required directory structure."""
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    os.makedirs(os.path.join(ws, ".mind-mem-index"), exist_ok=True)
    return ws


@pytest.fixture
def cal_mgr(workspace):
    """Return a CalibrationManager for the test workspace."""
    return CalibrationManager(workspace)


# ---------------------------------------------------------------------------
# make_query_id
# ---------------------------------------------------------------------------


def test_make_query_id_format():
    qid = make_query_id("test query")
    assert qid.startswith("cal-")
    parts = qid.split("-")
    assert len(parts) == 3
    # Hash part is 12 hex chars
    assert len(parts[1]) == 12


def test_make_query_id_deterministic_prefix():
    """Same query should produce same hash prefix."""
    qid1 = make_query_id("same query")
    qid2 = make_query_id("same query")
    prefix1 = qid1.split("-")[1]
    prefix2 = qid2.split("-")[1]
    assert prefix1 == prefix2


def test_make_query_id_unique_suffix():
    """Different timestamps mean different full IDs."""
    make_query_id("same query")
    time.sleep(0.002)
    make_query_id("same query")
    # They might collide if run in the same ms, but usually differ
    # We don't assert uniqueness here since it's time-based


def test_make_query_id_different_queries():
    qid1 = make_query_id("query one")
    qid2 = make_query_id("query two")
    prefix1 = qid1.split("-")[1]
    prefix2 = qid2.split("-")[1]
    assert prefix1 != prefix2


# ---------------------------------------------------------------------------
# _compute_weight
# ---------------------------------------------------------------------------


def test_compute_weight_below_threshold():
    """Below MIN_FEEDBACK_THRESHOLD, weight should be 1.0."""
    assert _compute_weight(1, 0, 0) == 1.0
    assert _compute_weight(0, 1, 0) == 1.0
    assert _compute_weight(1, 1, 0) == 1.0


def test_compute_weight_all_accepted():
    """All accepted feedback should produce weight > 1.0."""
    weight = _compute_weight(10, 0, 0)
    assert weight > 1.0
    assert weight <= MAX_CALIBRATION_WEIGHT


def test_compute_weight_all_rejected():
    """All rejected feedback should produce weight < 1.0."""
    weight = _compute_weight(0, 10, 0)
    assert weight < 1.0
    assert weight >= MIN_CALIBRATION_WEIGHT


def test_compute_weight_balanced():
    """Balanced feedback should produce weight near 1.0."""
    weight = _compute_weight(5, 5, 0)
    assert 0.8 <= weight <= 1.2


def test_compute_weight_ignored_counts_as_mild_negative():
    """Ignored feedback should act as mild negative."""
    weight_no_ignored = _compute_weight(5, 0, 0)
    weight_with_ignored = _compute_weight(5, 0, 5)
    assert weight_with_ignored < weight_no_ignored


def test_compute_weight_bounds():
    """Weight must always be in [MIN, MAX] range."""
    assert _compute_weight(1000, 0, 0) <= MAX_CALIBRATION_WEIGHT
    assert _compute_weight(0, 1000, 0) >= MIN_CALIBRATION_WEIGHT
    assert _compute_weight(0, 0, 1000) >= MIN_CALIBRATION_WEIGHT


def test_compute_weight_at_threshold():
    """Exactly at threshold should compute a weight."""
    weight = _compute_weight(MIN_FEEDBACK_THRESHOLD, 0, 0)
    assert weight > 1.0


# ---------------------------------------------------------------------------
# CalibrationManager.record_feedback
# ---------------------------------------------------------------------------


def test_record_feedback_basic(cal_mgr):
    result = cal_mgr.record_feedback(
        query_id="cal-abc123-1000",
        block_ids_useful=["D-20230101-001"],
        block_ids_not_useful=["D-20230101-002"],
        feedback_type="accepted",
    )
    assert result["query_id"] == "cal-abc123-1000"
    assert result["useful_count"] == 1
    assert result["not_useful_count"] == 1
    assert result["recorded"] > 0


def test_record_feedback_rejected(cal_mgr):
    result = cal_mgr.record_feedback(
        query_id="cal-abc123-2000",
        block_ids_useful=[],
        block_ids_not_useful=["D-20230101-001", "D-20230101-002"],
        feedback_type="rejected",
    )
    assert result["not_useful_count"] == 2
    assert result["recorded"] == 2


def test_record_feedback_ignored(cal_mgr):
    """Ignored feedback type should record all blocks as ignored."""
    result = cal_mgr.record_feedback(
        query_id="cal-abc123-3000",
        block_ids_useful=["D-20230101-001"],
        block_ids_not_useful=["D-20230101-002"],
        feedback_type="ignored",
    )
    # ignored records all blocks
    assert result["recorded"] >= 2


def test_record_feedback_invalid_type(cal_mgr):
    with pytest.raises(ValueError, match="Invalid feedback_type"):
        cal_mgr.record_feedback(
            query_id="cal-abc-000",
            block_ids_useful=["D-1"],
            block_ids_not_useful=[],
            feedback_type="invalid",
        )


def test_record_feedback_with_query_metadata(cal_mgr):
    result = cal_mgr.record_feedback(
        query_id="cal-abc-4000",
        block_ids_useful=["D-20230101-001"],
        block_ids_not_useful=[],
        feedback_type="accepted",
        query_text="What is the architecture?",
        query_type="WHAT",
    )
    assert result["recorded"] >= 1


# ---------------------------------------------------------------------------
# CalibrationManager.get_block_weight
# ---------------------------------------------------------------------------


def test_get_block_weight_no_data(cal_mgr):
    """Block with no feedback should return 1.0."""
    weight = cal_mgr.get_block_weight("nonexistent-block")
    assert weight == 1.0


def test_get_block_weight_below_threshold(cal_mgr):
    """Block with insufficient feedback should return 1.0."""
    cal_mgr.record_feedback(
        query_id="cal-abc-5000",
        block_ids_useful=["D-20230101-001"],
        block_ids_not_useful=[],
        feedback_type="accepted",
    )
    weight = cal_mgr.get_block_weight("D-20230101-001")
    assert weight == 1.0  # Only 1 event, below threshold


def test_get_block_weight_enough_positive(cal_mgr):
    """Block with enough positive feedback should be boosted."""
    for i in range(MIN_FEEDBACK_THRESHOLD + 2):
        cal_mgr.record_feedback(
            query_id=f"cal-abc-{6000 + i}",
            block_ids_useful=["D-20230101-001"],
            block_ids_not_useful=[],
            feedback_type="accepted",
        )
    weight = cal_mgr.get_block_weight("D-20230101-001")
    assert weight > 1.0


def test_get_block_weight_enough_negative(cal_mgr):
    """Block with enough negative feedback should be demoted."""
    for i in range(MIN_FEEDBACK_THRESHOLD + 2):
        cal_mgr.record_feedback(
            query_id=f"cal-abc-{7000 + i}",
            block_ids_useful=[],
            block_ids_not_useful=["D-20230101-002"],
            feedback_type="rejected",
        )
    weight = cal_mgr.get_block_weight("D-20230101-002")
    assert weight < 1.0


# ---------------------------------------------------------------------------
# CalibrationManager.get_block_weights (batch)
# ---------------------------------------------------------------------------


def test_get_block_weights_empty(cal_mgr):
    result = cal_mgr.get_block_weights([])
    assert result == {}


def test_get_block_weights_mixed(cal_mgr):
    """Batch query should return weights for multiple blocks."""
    # Seed enough feedback for block A (positive) and block B (negative)
    for i in range(MIN_FEEDBACK_THRESHOLD + 2):
        cal_mgr.record_feedback(
            query_id=f"cal-batch-{8000 + i}",
            block_ids_useful=["block-A"],
            block_ids_not_useful=["block-B"],
            feedback_type="accepted",
        )

    weights = cal_mgr.get_block_weights(["block-A", "block-B", "block-C"])
    assert weights["block-A"] > 1.0
    assert weights["block-B"] < 1.0
    assert weights["block-C"] == 1.0  # No data


# ---------------------------------------------------------------------------
# CalibrationManager.get_query_type_accuracy
# ---------------------------------------------------------------------------


def test_query_type_accuracy_empty(cal_mgr):
    result = cal_mgr.get_query_type_accuracy()
    assert result == {}


def test_query_type_accuracy_with_data(cal_mgr):
    for i in range(5):
        cal_mgr.record_feedback(
            query_id=f"cal-qtype-{9000 + i}",
            block_ids_useful=["D-1"],
            block_ids_not_useful=[],
            feedback_type="accepted",
            query_type="WHAT",
        )
    for i in range(3):
        cal_mgr.record_feedback(
            query_id=f"cal-qtype-{9100 + i}",
            block_ids_useful=[],
            block_ids_not_useful=["D-2"],
            feedback_type="rejected",
            query_type="WHEN",
        )

    accuracy = cal_mgr.get_query_type_accuracy()
    assert "WHAT" in accuracy
    assert accuracy["WHAT"]["accepted"] == 5
    assert accuracy["WHAT"]["accuracy"] == 1.0

    assert "WHEN" in accuracy
    assert accuracy["WHEN"]["rejected"] == 3
    assert accuracy["WHEN"]["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# CalibrationManager.get_calibration_stats
# ---------------------------------------------------------------------------


def test_calibration_stats_empty(cal_mgr):
    stats = cal_mgr.get_calibration_stats()
    assert stats["total_feedback"] == 0
    assert stats["window_days"] == CALIBRATION_WINDOW_DAYS


def test_calibration_stats_with_data(cal_mgr):
    # Seed data for a boosted and a demoted block
    for i in range(MIN_FEEDBACK_THRESHOLD + 2):
        cal_mgr.record_feedback(
            query_id=f"cal-stats-{10000 + i}",
            block_ids_useful=["boost-block"],
            block_ids_not_useful=["demote-block"],
            feedback_type="accepted",
            query_type="HOW",
        )

    stats = cal_mgr.get_calibration_stats()
    assert stats["total_feedback"] > 0
    assert stats["unique_blocks"] >= 2
    assert stats["unique_queries"] >= MIN_FEEDBACK_THRESHOLD + 2

    # Check top_boosted contains our boosted block
    boosted_ids = [b["block_id"] for b in stats["top_boosted"]]
    assert "boost-block" in boosted_ids

    # Check top_demoted contains our demoted block
    demoted_ids = [b["block_id"] for b in stats["top_demoted"]]
    assert "demote-block" in demoted_ids

    # Query type accuracy should include HOW
    assert "HOW" in stats["query_type_accuracy"]


# ---------------------------------------------------------------------------
# Schema initialization idempotence
# ---------------------------------------------------------------------------


def test_schema_creation_idempotent(workspace):
    """Creating CalibrationManager twice should not raise errors."""
    mgr1 = CalibrationManager(workspace)
    mgr2 = CalibrationManager(workspace)
    # Both should work fine
    assert mgr1.get_block_weight("x") == 1.0
    assert mgr2.get_block_weight("x") == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_block_ids(cal_mgr):
    """Recording with empty lists for a specific feedback_type should not crash."""
    # ignored with both lists empty should still be valid
    result = cal_mgr.record_feedback(
        query_id="cal-edge-1",
        block_ids_useful=[],
        block_ids_not_useful=["D-1"],
        feedback_type="rejected",
    )
    assert result["recorded"] >= 1


def test_duplicate_feedback(cal_mgr):
    """Recording the same feedback twice should use INSERT OR REPLACE."""
    cal_mgr.record_feedback(
        query_id="cal-dup-1",
        block_ids_useful=["D-1"],
        block_ids_not_useful=[],
        feedback_type="accepted",
    )
    # Should not raise
    cal_mgr.record_feedback(
        query_id="cal-dup-1",
        block_ids_useful=["D-1"],
        block_ids_not_useful=[],
        feedback_type="accepted",
    )


def test_special_characters_in_block_id(cal_mgr):
    """Block IDs with special characters should be handled safely."""
    cal_mgr.record_feedback(
        query_id="cal-special-1",
        block_ids_useful=["D-2023::F1", "TASK-001'test"],
        block_ids_not_useful=[],
        feedback_type="accepted",
    )
    weight = cal_mgr.get_block_weight("D-2023::F1")
    assert isinstance(weight, float)

"""Tests for v3.3.0 per-feature retrieval attribution."""

from __future__ import annotations

import time

import pytest

from mind_mem.retrieval_trace import (
    current_trace,
    is_trace_enabled,
    step,
    trace,
)


class TestTraceLifecycle:
    def test_no_active_trace_outside_block(self) -> None:
        assert current_trace() is None

    def test_trace_active_inside_block(self) -> None:
        with trace("what about X?") as t:
            assert current_trace() is t
            assert t.query == "what about X?"
        assert current_trace() is None

    def test_nested_traces_stack(self) -> None:
        with trace("outer") as outer:
            assert current_trace() is outer
            with trace("inner") as inner:
                assert current_trace() is inner
            assert current_trace() is outer
        assert current_trace() is None


class TestStepRecording:
    def test_step_records_into_active_trace(self) -> None:
        with trace("q") as t:
            with step("graph_expand") as rec:
                rec["added_count"] = 5
                rec["top_score_delta"] = 1.25
        assert len(t.steps) == 1
        s = t.steps[0]
        assert s.feature == "graph_expand"
        assert s.added_count == 5
        assert s.top_score_delta == pytest.approx(1.25)
        assert s.latency_ms >= 0

    def test_step_outside_trace_is_noop(self) -> None:
        """Without an active trace, step() still times the block but
        records nothing — callers see zero overhead."""
        with step("lone_feature") as rec:
            rec["added_count"] = 10
        # No assertion crashes — the with-block returned cleanly.
        assert current_trace() is None

    def test_multiple_steps_accumulate(self) -> None:
        with trace("q") as t:
            for feature in ("a", "b", "c"):
                with step(feature):
                    pass
        assert [s.feature for s in t.steps] == ["a", "b", "c"]

    def test_extra_metadata_in_step(self) -> None:
        with trace("q") as t:
            with step("graph_expand", max_hops=3) as rec:
                rec["auto_enabled"] = True
        assert t.steps[0].metadata.get("max_hops") == 3
        assert t.steps[0].metadata.get("auto_enabled") is True

    def test_step_latency_reflects_sleep(self) -> None:
        with trace("q") as t:
            with step("slow"):
                time.sleep(0.02)
        # Windows clock resolution is ~15.6ms and time.sleep(0.02) can
        # round down to 14.99ms due to floating-point representation.
        # The point of the test is to confirm that `step` records a
        # non-trivial latency — the exact threshold isn't load-bearing.
        assert t.steps[0].latency_ms >= 10  # 20ms sleep, wider floor for Windows


class TestSummary:
    def test_summary_shape(self) -> None:
        with trace("q") as t:
            with step("a") as rec:
                rec["added_count"] = 2
            with step("b") as rec:
                rec["top_score_delta"] = 3.5
        summary = t.summary()
        assert summary["query"] == "q"
        assert "total_latency_ms" in summary
        assert len(summary["steps"]) == 2
        assert summary["steps"][0]["feature"] == "a"
        assert summary["steps"][1]["top_score_delta"] == pytest.approx(3.5)

    def test_total_latency_nonnegative(self) -> None:
        with trace("q") as t:
            with step("x"):
                pass
        assert t.total_latency_ms() >= 0


class TestIsTraceEnabled:
    def test_off_without_config(self) -> None:
        assert is_trace_enabled(None) is False
        assert is_trace_enabled({}) is False

    def test_off_by_default_in_retrieval_section(self) -> None:
        assert is_trace_enabled({"retrieval": {}}) is False

    def test_explicit_on(self) -> None:
        assert is_trace_enabled({"retrieval": {"trace_attribution": True}}) is True

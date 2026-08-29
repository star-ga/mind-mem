# Copyright 2026 STARGA, Inc.
"""``review_metrics`` — the product metric ``mm review`` publishes.

Approval throughput is the metric the feature exists to move, so it is
computed, not asserted: median proposal age at approval and
proposals/minute, both from measured session data. Age is reported only
for proposals that actually carry a ``Created`` timestamp — a fabricated
age would make the headline number a lie.
"""

from __future__ import annotations

import pytest

from mind_mem.review_metrics import ApprovalEvent, SessionMetrics, summarise


def _event(pid: str, *, age: float | None, decided_at: float, action: str = "approve", ok: bool = True):
    return ApprovalEvent(proposal_id=pid, action=action, succeeded=ok, age_seconds=age, decided_at=decided_at)


class TestMedianAge:
    def test_odd_sample_takes_the_middle(self):
        events = [
            _event("P-20260829-001", age=10.0, decided_at=1.0),
            _event("P-20260829-002", age=30.0, decided_at=2.0),
            _event("P-20260829-003", age=20.0, decided_at=3.0),
        ]
        assert summarise(events, elapsed_seconds=60.0).median_age_at_approval_seconds == 20.0

    def test_even_sample_averages_the_two_middles(self):
        events = [
            _event("P-20260829-001", age=10.0, decided_at=1.0),
            _event("P-20260829-002", age=20.0, decided_at=2.0),
            _event("P-20260829-003", age=30.0, decided_at=3.0),
            _event("P-20260829-004", age=40.0, decided_at=4.0),
        ]
        assert summarise(events, elapsed_seconds=60.0).median_age_at_approval_seconds == 25.0

    def test_ages_only_count_approvals_that_succeeded(self):
        events = [
            _event("P-20260829-001", age=10.0, decided_at=1.0),
            _event("P-20260829-002", age=1000.0, decided_at=2.0, ok=False),
        ]
        summary = summarise(events, elapsed_seconds=60.0)
        assert summary.median_age_at_approval_seconds == 10.0
        assert summary.aged_sample == 1

    def test_ages_exclude_rejections(self):
        events = [
            _event("P-20260829-001", age=10.0, decided_at=1.0),
            _event("P-20260829-002", age=999.0, decided_at=2.0, action="reject"),
        ]
        assert summarise(events, elapsed_seconds=60.0).median_age_at_approval_seconds == 10.0

    def test_missing_ages_are_excluded_and_the_gap_is_reported(self):
        events = [
            _event("P-20260829-001", age=None, decided_at=1.0),
            _event("P-20260829-002", age=40.0, decided_at=2.0),
        ]
        summary = summarise(events, elapsed_seconds=60.0)
        assert summary.median_age_at_approval_seconds == 40.0
        assert summary.aged_sample == 1
        assert summary.age_coverage == 0.5

    def test_no_ages_at_all_reports_none_never_zero(self):
        events = [_event("P-20260829-001", age=None, decided_at=1.0)]
        summary = summarise(events, elapsed_seconds=60.0)
        assert summary.median_age_at_approval_seconds is None
        assert summary.age_coverage == 0.0


class TestThroughput:
    def test_proposals_per_minute_counts_every_decision(self):
        events = [_event(f"P-20260829-{i:03d}", age=None, decided_at=float(i)) for i in range(1, 31)]
        summary = summarise(events, elapsed_seconds=40.0)
        assert summary.decisions == 30
        assert summary.proposals_per_minute == pytest.approx(45.0)

    def test_applied_per_minute_counts_only_successful_applies(self):
        events = [
            _event("P-20260829-001", age=None, decided_at=1.0),
            _event("P-20260829-002", age=None, decided_at=2.0, ok=False),
        ]
        summary = summarise(events, elapsed_seconds=60.0)
        assert summary.applied == 1
        assert summary.applied_per_minute == pytest.approx(1.0)

    def test_zero_elapsed_reports_none_rather_than_dividing_by_zero(self):
        events = [_event("P-20260829-001", age=None, decided_at=0.0)]
        summary = summarise(events, elapsed_seconds=0.0)
        assert summary.proposals_per_minute is None

    def test_an_empty_session_is_all_zeroes_and_no_rates(self):
        summary = summarise([], elapsed_seconds=10.0)
        assert summary.decisions == 0
        assert summary.proposals_per_minute == 0.0
        assert summary.median_age_at_approval_seconds is None


class TestSerialisation:
    def test_summary_round_trips_to_json_safe_primitives(self):
        import json

        events = [_event("P-20260829-001", age=12.0, decided_at=1.0)]
        payload = summarise(events, elapsed_seconds=30.0).to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["median_age_at_approval_seconds"] == 12.0

    def test_summary_is_immutable(self):
        summary = summarise([], elapsed_seconds=1.0)
        assert isinstance(summary, SessionMetrics)
        with pytest.raises((AttributeError, TypeError)):
            summary.decisions = 99  # type: ignore[misc]

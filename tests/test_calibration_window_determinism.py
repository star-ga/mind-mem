"""The calibration window boundary is UTC-anchored and pinnable.

The 30-day window is time-relative *by design* — that is the feature, and the
module docstring says so. What must not vary is the machine: the boundary is
computed in UTC, so two hosts in different timezones evaluating the same
instant cut the window at the same point.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from mind_mem import calibration
from mind_mem.calibration import CALIBRATION_WINDOW_DAYS, CalibrationManager

requires_tzset = pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="time.tzset() is unavailable on this platform",
)

FROZEN_INSTANT = datetime(2026, 8, 27, 0, 30, 0, tzinfo=timezone.utc)
WIDE_ZONES = ("Pacific/Niue", "UTC", "Pacific/Kiritimati")


@pytest.fixture
def workspace(tmp_path):
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    os.makedirs(os.path.join(ws, ".mind-mem-index"), exist_ok=True)
    return ws


@pytest.fixture
def cal_mgr(workspace):
    return CalibrationManager(workspace)


@pytest.fixture
def restore_timezone():
    original = os.environ.get("TZ")
    yield
    if hasattr(time, "tzset"):
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


def _freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the module clock while still honouring the process timezone."""
    epoch = FROZEN_INSTANT.timestamp()
    base = datetime

    class _Frozen(base):  # type: ignore[misc, valid-type]
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
            return base.fromtimestamp(epoch, tz)

    monkeypatch.setattr(calibration, "datetime", _Frozen)


class TestCutoffBoundary:
    def test_cutoff_is_utc_zulu_format(self, cal_mgr) -> None:
        cutoff = cal_mgr._cutoff_date(now=FROZEN_INSTANT)
        assert cutoff.endswith("Z")
        assert datetime.strptime(cutoff, "%Y-%m-%dT%H:%M:%SZ")

    def test_cutoff_is_exactly_the_window_before_the_as_of_instant(self, cal_mgr) -> None:
        cutoff = datetime.strptime(cal_mgr._cutoff_date(now=FROZEN_INSTANT), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        assert FROZEN_INSTANT - cutoff == timedelta(days=CALIBRATION_WINDOW_DAYS)

    def test_naive_as_of_instant_is_read_as_utc(self, cal_mgr) -> None:
        naive = FROZEN_INSTANT.replace(tzinfo=None)
        assert cal_mgr._cutoff_date(now=naive) == cal_mgr._cutoff_date(now=FROZEN_INSTANT)

    def test_aware_non_utc_as_of_instant_is_converted(self, cal_mgr) -> None:
        elsewhere = FROZEN_INSTANT.astimezone(timezone(timedelta(hours=-11)))
        assert cal_mgr._cutoff_date(now=elsewhere) == cal_mgr._cutoff_date(now=FROZEN_INSTANT)

    def test_window_is_time_relative_as_documented(self, cal_mgr) -> None:
        """Later as-of instant moves the boundary — the feature, not a bug."""
        earlier = cal_mgr._cutoff_date(now=FROZEN_INSTANT)
        later = cal_mgr._cutoff_date(now=FROZEN_INSTANT + timedelta(days=45))
        assert later > earlier


class TestCutoffTimezoneInvariance:
    @requires_tzset
    def test_cutoff_identical_across_timezones(self, cal_mgr, restore_timezone) -> None:
        seen: dict[str, str] = {}
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("TZ", tz_name)
                time.tzset()
                _freeze(mp)
                seen[tz_name] = cal_mgr._cutoff_date()
        assert len(set(seen.values())) == 1, f"calibration window boundary is timezone-dependent: {seen}"

    @requires_tzset
    def test_block_weight_identical_across_timezones(self, cal_mgr, restore_timezone) -> None:
        from mind_mem.calibration import MIN_FEEDBACK_THRESHOLD

        for i in range(MIN_FEEDBACK_THRESHOLD + 2):
            cal_mgr.record_feedback(
                query_id=f"cal-tzinvariance-{7000 + i}",
                block_ids_useful=["D-20260826-001"],
                block_ids_not_useful=[],
                feedback_type="accepted",
            )

        seen: dict[str, float] = {}
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("TZ", tz_name)
                time.tzset()
                _freeze(mp)
                seen[tz_name] = cal_mgr.get_block_weight("D-20260826-001")
        assert len(set(seen.values())) == 1, f"calibration weight is timezone-dependent: {seen}"

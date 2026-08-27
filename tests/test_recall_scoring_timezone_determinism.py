"""Recency scoring must not depend on the host's timezone.

Regression cover for the determinism defect where ``date_score`` (and the
half-life decay beside it) read a *naive local* ``datetime.now()``. The day
boundary that decides ``days_old`` therefore sat at local midnight, so the
same corpus scored at the same instant produced different recall scores on
hosts in different zones — measured at UTC-11 vs UTC vs UTC+14.

The clock is now UTC (``_recall_scoring._utc_now``) and injectable, so these
tests freeze one instant and prove the scores are identical everywhere.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

import pytest

from mind_mem import _recall_scoring
from mind_mem._recall_core import recall
from mind_mem._recall_scoring import _as_utc, _utc_now, date_score, temporal_decay_score

# tzset() is POSIX-only; the Windows CI rows cannot rebind local time.
requires_tzset = pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="time.tzset() is unavailable on this platform",
)

# Deliberately just after UTC midnight: local time is still the *previous*
# day at UTC-11 and already the same day at UTC+14, so a local-midnight day
# boundary changes ``days_old`` between the zones.
FROZEN_INSTANT = datetime(2026, 8, 27, 0, 30, 0, tzinfo=timezone.utc)

# UTC-11, UTC, UTC+14 — the widest spread of real zones available.
WIDE_ZONES = ("Pacific/Niue", "UTC", "Pacific/Kiritimati")


def _freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``_recall_scoring``'s clock to FROZEN_INSTANT, honouring local TZ.

    ``now()`` is resolved through ``fromtimestamp`` exactly as CPython does,
    so a naive call still picks up the process timezone. A patch that simply
    returned a fixed UTC value would mask the very bug under test.
    """
    epoch = FROZEN_INSTANT.timestamp()
    base = datetime

    class _Frozen(base):  # type: ignore[misc, valid-type]
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
            return base.fromtimestamp(epoch, tz)

    monkeypatch.setattr(_recall_scoring, "_datetime", _Frozen)


def _set_timezone(monkeypatch: pytest.MonkeyPatch, tz_name: str) -> None:
    monkeypatch.setenv("TZ", tz_name)
    time.tzset()


@pytest.fixture
def restore_timezone():
    """Put the process timezone back however the test exits."""
    original = os.environ.get("TZ")
    yield
    if hasattr(time, "tzset"):
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


def _write_workspace(root: str) -> None:
    """Minimal recall workspace: one recent block, one a year older."""
    for sub in ("decisions", "tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    with open(os.path.join(root, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write("[D-20260826-001]\nStatement: Use BM25 scoring for retrieval search\nStatus: active\nDate: 2026-08-26\n\n")
        fh.write("[D-20250826-002]\nStatement: Use BM25 scoring for older retrieval search\nStatus: active\nDate: 2025-08-26\n\n")
    with open(os.path.join(root, "tasks", "TASKS.md"), "w", encoding="utf-8") as fh:
        fh.write("[T-20260213-099]\nTitle: Unrelated placeholder task\nStatus: active\n")
    for rel in (
        "entities/projects.md",
        "entities/people.md",
        "entities/tools.md",
        "entities/incidents.md",
        "intelligence/CONTRADICTIONS.md",
        "intelligence/DRIFT.md",
        "intelligence/SIGNALS.md",
    ):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# {os.path.basename(rel)}\n")


def _recall_scores(tz_name: str, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float]]:
    _set_timezone(monkeypatch, tz_name)
    _freeze(monkeypatch)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_workspace(td)
        hits = recall(td, "BM25 scoring retrieval search")
        return [(str(h.get("_id")), float(h.get("score", 0.0) or 0.0)) for h in hits]


class TestTimezoneInvariance:
    """(a) One frozen instant, three wildly different zones, identical scores."""

    @requires_tzset
    def test_recall_scores_identical_across_timezones(self, restore_timezone) -> None:
        per_zone: dict[str, list[tuple[str, float]]] = {}
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                per_zone[tz_name] = _recall_scores(tz_name, mp)

        assert per_zone["UTC"], "fixture produced no recall hits — test would be vacuous"
        baseline = per_zone["UTC"]
        for tz_name, scores in per_zone.items():
            assert scores == baseline, f"recall scores drifted under TZ={tz_name}: {scores} != {baseline}"

    @requires_tzset
    def test_date_score_identical_across_timezones(self, restore_timezone) -> None:
        block = {"Date": "2026-08-26"}
        seen: dict[str, float] = {}
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _freeze(mp)
                seen[tz_name] = date_score(block)
        assert len(set(seen.values())) == 1, f"date_score is timezone-dependent: {seen}"

    @requires_tzset
    def test_temporal_decay_identical_across_timezones(self, restore_timezone) -> None:
        block = {"Created": "2026-08-26"}
        seen: dict[str, float] = {}
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _freeze(mp)
                seen[tz_name] = temporal_decay_score(block, half_life_days=30)
        assert len(set(seen.values())) == 1, f"temporal_decay_score is timezone-dependent: {seen}"

    @requires_tzset
    def test_frozen_clock_helper_actually_tracks_local_time(self, restore_timezone) -> None:
        """Guard the guard: the freeze must still expose naive local drift.

        If ``_Frozen.now()`` ignored the process timezone, the invariance
        tests above would pass even with the bug reintroduced.
        """
        naive_by_zone: set[str] = set()
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _freeze(mp)
                naive_by_zone.add(_recall_scoring._datetime.now().isoformat())
        assert len(naive_by_zone) == len(WIDE_ZONES), "frozen clock is not timezone-sensitive, so these tests cannot catch the bug"


class TestRecencyStillOrdersCorrectly:
    """(b) Determinism must not have flattened the recency signal."""

    def test_newer_block_scores_higher(self) -> None:
        # All three sit inside the 365-day linear ramp, so the ordering is
        # the ramp's, not the 0.1 floor's.
        newer = date_score({"Date": "2026-08-20"}, now=FROZEN_INSTANT)
        older = date_score({"Date": "2026-02-20"}, now=FROZEN_INSTANT)
        oldest = date_score({"Date": "2025-09-20"}, now=FROZEN_INSTANT)
        assert newer > older > oldest

    def test_same_day_block_scores_one(self) -> None:
        assert date_score({"Date": "2026-08-27"}, now=FROZEN_INSTANT) == 1.0

    def test_score_floor_holds_for_ancient_blocks(self) -> None:
        assert date_score({"Date": "1990-01-01"}, now=FROZEN_INSTANT) == 0.1

    def test_missing_and_unparseable_dates_stay_neutral(self) -> None:
        assert date_score({}, now=FROZEN_INSTANT) == 0.5
        assert date_score({"Date": ""}, now=FROZEN_INSTANT) == 0.5
        assert date_score({"Date": "not-a-date"}, now=FROZEN_INSTANT) == 0.5
        assert date_score({"Date": 20260826}, now=FROZEN_INSTANT) == 0.5

    def test_temporal_decay_halves_at_half_life(self) -> None:
        created = (FROZEN_INSTANT - timedelta(days=30)).strftime("%Y-%m-%d")
        decayed = temporal_decay_score({"Created": created}, half_life_days=30, now=FROZEN_INSTANT)
        assert decayed == pytest.approx(0.5, abs=0.02)

    def test_temporal_decay_orders_newer_above_older(self) -> None:
        newer = temporal_decay_score({"Created": "2026-08-01"}, now=FROZEN_INSTANT)
        older = temporal_decay_score({"Created": "2024-08-01"}, now=FROZEN_INSTANT)
        assert newer > older


class TestInjectableClockSeam:
    """The ``now`` seam pins the instant for deterministic replay."""

    def test_utc_now_is_timezone_aware(self) -> None:
        assert _utc_now().tzinfo is timezone.utc

    def test_default_matches_explicit_utc_now(self) -> None:
        block = {"Date": "2026-01-15"}
        assert date_score(block) == date_score(block, now=_utc_now())

    def test_naive_injected_instant_is_read_as_utc(self) -> None:
        aware = datetime(2026, 8, 27, 0, 30, tzinfo=timezone.utc)
        naive = datetime(2026, 8, 27, 0, 30)
        assert _as_utc(naive) == aware
        block = {"Date": "2026-08-26"}
        assert date_score(block, now=naive) == date_score(block, now=aware)

    def test_aware_non_utc_instant_is_converted_not_reinterpreted(self) -> None:
        offset = timezone(timedelta(hours=-11))
        same_instant = FROZEN_INSTANT.astimezone(offset)
        block = {"Date": "2026-08-26"}
        assert date_score(block, now=same_instant) == date_score(block, now=FROZEN_INSTANT)

    def test_injected_instant_beats_the_wall_clock(self) -> None:
        """A pinned instant is used verbatim — replay does not drift."""
        block = {"Date": "2026-08-26"}
        long_after = datetime(2030, 8, 27, tzinfo=timezone.utc)
        assert date_score(block, now=long_after) < date_score(block, now=FROZEN_INSTANT)

    def test_temporal_decay_accepts_injected_instant(self) -> None:
        block = {"Created": "2026-08-26"}
        assert temporal_decay_score(block, now=FROZEN_INSTANT) > temporal_decay_score(block, now=datetime(2027, 8, 27, tzinfo=timezone.utc))

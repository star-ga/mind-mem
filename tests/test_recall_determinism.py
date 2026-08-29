"""The determinism seam: recall is a pure function of (corpus, config, scoring_instant).

mind-mem's wedge is deterministic retrieval. Before this seam existed the claim
was false: the scoring path read the wall clock in three unparameterised places
(``date_score``'s ``now``, the calibration window's rolling cutoff, and the
temporal hard filter's *naive-local* ``date.today()``), so the same corpus and
the same query produced different rankings — and different *served sets* — on a
different day or on a host in a different timezone.

Worse, the ``RECALL_ATTEST_v1`` preimage described the answer by a single
``result_count``, so two runs that ranked differently — or served no block in
common — produced a byte-identical attestation hash. An attestation that cannot
distinguish two different served answers asserts a reproducibility it does not
have.

These tests pin the wall clock rather than sleeping, and compare serialized
rankings so a drift is a byte difference, not a float tolerance.

Acceptance criteria covered here:
  T1 clock-free core   T2 timezone independence   T3 day independence
  T6 back-compat default

T4 (attestation completeness) and T5 (replay) live in
``test_recall_attestation_completeness.py``. The per-backend structural guard —
every leg of every backend, proven to read no clock — lives in
``test_recall_clock_guard.py``; the T1 tests below are the end-to-end form of
the same property on the default path.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import date, datetime, timezone

import pytest
from _recall_clock_sentinel import ClockRead, clock_census, install_clock_sentinel

from mind_mem import _recall_scoring, _recall_temporal, calibration
from mind_mem._recall_core import recall

# tzset() is POSIX-only; the Windows CI rows cannot rebind local time.
requires_tzset = pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="time.tzset() is unavailable on this platform",
)

# Just after UTC midnight: local time is still the *previous* day at UTC-11 and
# already the same day at UTC+14, so any local-midnight day boundary splits the
# zones. This is the instant that exposes the bug.
FROZEN_INSTANT = datetime(2026, 8, 27, 0, 30, 0, tzinfo=timezone.utc)
MONTHS_LATER = datetime(2027, 2, 27, 0, 30, 0, tzinfo=timezone.utc)

# UTC-11, UTC, UTC+14 — the widest spread of real zones available.
WIDE_ZONES = ("Pacific/Niue", "UTC", "Pacific/Kiritimati")

SCORING_INSTANT = date(2026, 8, 27)
LATER_INSTANT = date(2027, 8, 27)

# A temporal query: it must route to intent WHEN so the temporal hard filter —
# the leg that drops blocks outright — actually runs. A single-hop query never
# reaches it, which is why the pre-existing timezone suite could not see this
# half of the defect. ``test_fixture_actually_exercises_the_temporal_filter``
# guards the routing so a classifier change cannot quietly make this vacuous.
TEMPORAL_QUERY = "when in the last 7 days was the retrieval rollout note recorded"
PLAIN_QUERY = "retrieval ranking determinism rollout"

# Routes to intent WHY -> multi-hop, which makes ``recall`` re-enter itself once
# per sub-query. Guarded in the test so a classifier change cannot mute it.
MULTI_HOP_QUERY = "why did the retrieval rollout note land and who approved the retrieval rollout"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _pin_wall_clock(monkeypatch: pytest.MonkeyPatch, instant: datetime) -> None:
    """Pin every wall clock the recall path can reach, honouring the local TZ.

    Resolved through ``fromtimestamp`` exactly as CPython does, so a naive read
    still picks up the process timezone — a pin that returned a fixed UTC value
    would mask the very bug under test.
    """
    epoch = instant.timestamp()
    base_dt = datetime
    base_d = date

    class _FrozenDateTime(base_dt):  # type: ignore[misc, valid-type]
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
            return base_dt.fromtimestamp(epoch, tz)

    class _FrozenDate(base_d):  # type: ignore[misc, valid-type]
        @classmethod
        def today(cls) -> date:  # type: ignore[override]
            return base_dt.fromtimestamp(epoch).date()

    monkeypatch.setattr(_recall_scoring, "_datetime", _FrozenDateTime)
    monkeypatch.setattr(calibration, "datetime", _FrozenDateTime)
    monkeypatch.setattr(_recall_temporal, "date", _FrozenDate)
    import mind_mem.scoring_instant as si_mod

    monkeypatch.setattr(si_mod, "datetime", _FrozenDateTime)


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


# Dates chosen to straddle both edges of a "last 7 days" window anchored on
# 2026-08-27: 08-19 falls out of the UTC window, 08-27 falls out of the UTC-11
# window. A timezone-dependent anchor therefore changes the served SET.
_BLOCKS = (
    ("D-20260827-001", "retrieval rollout notes shipped", "2026-08-27"),
    ("D-20260823-002", "retrieval rollout notes reviewed", "2026-08-23"),
    ("D-20260819-003", "retrieval rollout notes drafted", "2026-08-19"),
)


def _write_workspace(root: str, blocks: tuple[tuple[str, str, str], ...] = _BLOCKS) -> None:
    """Minimal recall workspace containing *blocks* as decisions."""
    for sub in ("decisions", "tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    with open(os.path.join(root, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        for bid, statement, when in blocks:
            fh.write(f"[{bid}]\nStatement: {statement}\nStatus: active\nDate: {when}\n\n")
    for rel in (
        "entities/projects.md",
        "intelligence/SIGNALS.md",
    ):
        path = os.path.join(root, rel)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# {os.path.basename(rel)}\n")


def _ranking(hits: list[dict]) -> str:
    """Serialize a served ranking so drift is a byte difference, not a float epsilon."""
    return json.dumps([[str(h.get("_id")), float(h.get("score", 0.0) or 0.0)] for h in hits], sort_keys=True)


# ---------------------------------------------------------------------------
# T1 — the deterministic core reads no clock
# ---------------------------------------------------------------------------


class TestClockFreeCore:
    """T1, in the only form that is not self-deceiving.

    Every assertion here is on the sentinel's *recorded reads*, never on the
    call merely completing. ``recall()`` degrades rather than fails on most of
    its optional legs, so a completion-based assertion passes while the guard's
    own alarm is being swallowed — which is exactly what the first version of
    this file did, through eight separate reverts of the threading.
    """

    def test_core_recall_reads_no_clock_with_every_accessor_broken(self, monkeypatch) -> None:
        """T1: given an explicit scoring_instant, no scoring clock is consulted."""
        sentinel = install_clock_sentinel(monkeypatch)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_workspace(td)
            hits = recall(td, TEMPORAL_QUERY, scoring_instant=SCORING_INSTANT)
        sentinel.assert_clock_free()
        assert hits, "fixture produced no hits — the guard would be vacuous"

    def test_plain_query_core_is_also_clock_free(self, monkeypatch) -> None:
        sentinel = install_clock_sentinel(monkeypatch)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_workspace(td)
            hits = recall(td, PLAIN_QUERY, scoring_instant=SCORING_INSTANT)
        sentinel.assert_clock_free()
        assert hits

    def test_optional_legs_are_clock_free_too(self, monkeypatch) -> None:
        """The default-off legs are the ones a guard silently misses.

        The validity gate is named as part of the deterministic core, but its
        ``provenance_class`` component reads the rolling calibration window —
        a wall clock — and it is off by default, so a guard that only exercised
        the defaults would have declared the core clean while that path stayed
        dirty. Turn the opt-ins on and re-run the guard.

        Both instruments run together here: the sentinel records a read even
        when the gate's own ``except Exception`` swallows the raise, and the
        census sees a read that never touched a patched accessor at all.
        """
        sentinel = install_clock_sentinel(monkeypatch)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_workspace(td)
            with open(os.path.join(td, "mind-mem.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "recall": {
                            "validity_gate": {"enabled": True, "provenance_class": {"enabled": True}},
                            "temporal_hard_filter": True,
                        }
                    },
                    fh,
                )
            with clock_census() as census:
                hits = recall(td, TEMPORAL_QUERY, scoring_instant=SCORING_INSTANT)
        sentinel.assert_clock_free()
        census.assert_clock_free()
        assert hits

    def test_multi_hop_decomposition_is_clock_free(self, monkeypatch) -> None:
        """A multi-hop query re-enters ``recall`` per sub-query.

        Each sub-query is part of the answer, so a recursion that let the
        callee re-resolve its own instant would reinstate the clock read one
        level down — and a decomposition straddling UTC midnight would merge
        two differently-dated rankings into one result set.
        """
        from mind_mem._recall_detection import _INTENT_TO_QUERY_TYPE
        from mind_mem.intent_router import IntentRouter

        intent = IntentRouter().classify(MULTI_HOP_QUERY).intent
        assert _INTENT_TO_QUERY_TYPE.get(intent, "single-hop") == "multi-hop", "query no longer decomposes"

        sentinel = install_clock_sentinel(monkeypatch)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_workspace(td)
            with clock_census() as census:
                hits = recall(td, MULTI_HOP_QUERY, scoring_instant=SCORING_INSTANT)
        # Assert on the recorded reads AND on the result, never on completion.
        # The decomposition loop wraps each sub-query in `except Exception`, so
        # a clock read down there is swallowed into a `sub_query_recall_failed`
        # log line and an empty merge — the query "succeeds" with zero hits.
        sentinel.assert_clock_free()
        census.assert_clock_free()
        assert hits, "every sub-query died silently — the recursion is reading a clock"

    def test_resolver_is_the_single_boundary_clock_read(self, monkeypatch) -> None:
        """The one permitted clock read sits in the resolver, at the boundary.

        With no scoring_instant supplied, breaking *only* the resolver must
        break recall — proof that every downstream recency term is fed from it
        rather than reading a clock of its own.
        """
        import mind_mem.scoring_instant as si_mod

        def _boom(*_a: object, **_k: object) -> None:
            raise ClockRead("boundary clock read")

        monkeypatch.setattr(si_mod, "_read_utc_today", _boom)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_workspace(td)
            with pytest.raises(ClockRead, match="boundary clock read"):
                recall(td, TEMPORAL_QUERY)


# ---------------------------------------------------------------------------
# T2 — timezone independence
# ---------------------------------------------------------------------------


class TestTimezoneIndependence:
    @requires_tzset
    def test_ranking_byte_identical_across_extreme_timezones(self, restore_timezone) -> None:
        """T2: UTC-11, UTC and UTC+14 serve the identical set and order."""
        per_zone: dict[str, str] = {}
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _pin_wall_clock(mp, FROZEN_INSTANT)
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                    _write_workspace(td)
                    per_zone[tz_name] = _ranking(recall(td, TEMPORAL_QUERY, scoring_instant=SCORING_INSTANT))
        assert per_zone["UTC"] != "[]", "fixture served nothing — test would be vacuous"
        baseline = per_zone["UTC"]
        for tz_name, ranking in per_zone.items():
            assert ranking == baseline, f"ranking drifted under TZ={tz_name}: {ranking} != {baseline}"

    def test_fixture_actually_exercises_the_temporal_filter(self) -> None:
        """Guard the guard: a sliding window must genuinely drop blocks here.

        The temporal hard filter only runs for a query the intent router sends
        to ``WHEN``. If the classifier ever stops routing this phrasing there,
        the timezone test above would still pass while testing nothing.
        """
        from mind_mem._recall_detection import _INTENT_TO_QUERY_TYPE
        from mind_mem._recall_temporal import resolve_time_reference
        from mind_mem.intent_router import IntentRouter

        intent = IntentRouter().classify(TEMPORAL_QUERY).intent
        assert _INTENT_TO_QUERY_TYPE.get(intent, "single-hop") == "temporal"

        start, end = resolve_time_reference(TEMPORAL_QUERY, reference_date=SCORING_INSTANT)
        assert start is not None and end is not None, "no window resolved — nothing would be filtered"

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_workspace(td)
            served = recall(td, TEMPORAL_QUERY, scoring_instant=SCORING_INSTANT)
        assert 0 < len(served) < len(_BLOCKS), "the hard filter dropped nothing — the window is not biting"

    @requires_tzset
    def test_pin_helper_is_actually_timezone_sensitive(self, restore_timezone) -> None:
        """Guard the guard: if the pin ignored TZ these tests could not fail."""
        seen: set[str] = set()
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _pin_wall_clock(mp, FROZEN_INSTANT)
                seen.add(_recall_temporal.date.today().isoformat())
        assert len(seen) > 1, f"pinned local day is not timezone-sensitive: {seen}"


# ---------------------------------------------------------------------------
# T3 — day independence
# ---------------------------------------------------------------------------


class TestDayIndependence:
    def test_ranking_byte_identical_when_wall_clock_moves_months(self, monkeypatch) -> None:
        """T3: move "today" by six months; a pinned scoring_instant does not move."""
        rankings: list[str] = []
        for instant in (FROZEN_INSTANT, MONTHS_LATER):
            with pytest.MonkeyPatch.context() as mp:
                _pin_wall_clock(mp, instant)
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                    _write_workspace(td)
                    rankings.append(_ranking(recall(td, TEMPORAL_QUERY, scoring_instant=SCORING_INSTANT)))
        assert rankings[0] != "[]", "fixture served nothing — test would be vacuous"
        assert rankings[0] == rankings[1], f"ranking drifted with the wall clock: {rankings}"

    def test_recency_is_still_live_not_deleted(self, monkeypatch) -> None:
        """The seam preserves recency: a different instant *does* change scoring."""
        _pin_wall_clock(monkeypatch, FROZEN_INSTANT)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_workspace(td)
            near = _ranking(recall(td, PLAIN_QUERY, scoring_instant=SCORING_INSTANT))
            far = _ranking(recall(td, PLAIN_QUERY, scoring_instant=LATER_INSTANT))
        assert near != far, "recency was flattened, not parameterised"


# ---------------------------------------------------------------------------
# T6 — back-compat default
# ---------------------------------------------------------------------------


class TestDefaultsToTodayUtc:
    def test_omitting_scoring_instant_matches_today_utc(self, monkeypatch) -> None:
        """T6: no argument == passing today-in-UTC explicitly."""
        _pin_wall_clock(monkeypatch, FROZEN_INSTANT)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_workspace(td)
            implicit = _ranking(recall(td, TEMPORAL_QUERY))
            explicit = _ranking(recall(td, TEMPORAL_QUERY, scoring_instant=FROZEN_INSTANT.date()))
        assert implicit == explicit

    @requires_tzset
    def test_default_is_utc_today_not_local_today(self, restore_timezone) -> None:
        from mind_mem.scoring_instant import resolve_scoring_instant

        seen: set[date] = set()
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _pin_wall_clock(mp, FROZEN_INSTANT)
                seen.add(resolve_scoring_instant(None))
        assert seen == {FROZEN_INSTANT.date()}, f"default instant is timezone-dependent: {seen}"

    def test_datetime_is_narrowed_to_a_utc_date(self) -> None:
        """A datetime IS-A date; it must never reach the preimage at second precision."""
        from mind_mem.scoring_instant import format_scoring_instant, resolve_scoring_instant

        narrowed = resolve_scoring_instant(datetime(2026, 8, 27, 23, 59, 59, tzinfo=timezone.utc))
        assert narrowed == date(2026, 8, 27)
        assert format_scoring_instant(narrowed) == "2026-08-27"
        assert len(format_scoring_instant(narrowed)) == 10

    def test_garbage_is_rejected_at_the_boundary(self) -> None:
        from mind_mem.scoring_instant import resolve_scoring_instant

        with pytest.raises(TypeError):
            resolve_scoring_instant(20260827)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            resolve_scoring_instant("not-a-date")
        with pytest.raises(ValueError):
            resolve_scoring_instant("")

    @pytest.mark.parametrize(
        "shape",
        [
            "2026-W35-4",  # ISO week date: also exactly ten characters
            "2026-W01-1",  # ...and this one resolves EIGHT MONTHS away, to 2025-12-29
            "2026-243",  # ISO ordinal date
            " 2026-08-27 ",  # padded
            "2026-08-27T00:00:00",  # a timestamp, not a date
            "2026-8-7",  # unpadded components
        ],
    )
    def test_only_yyyy_mm_dd_is_accepted(self, shape: str) -> None:
        """A ten-character length check is not a shape check.

        ``date.fromisoformat`` accepts the whole ISO-8601 date grammar from
        Python 3.11 on, and the week form is the same width as the calendar
        form. Guarding on width alone let a client name ``2026-W01-1`` and be
        scored against ``2025-12-29`` without a word, while the error message
        the module raises for everything else insists on ``YYYY-MM-DD``.
        """
        from mind_mem.scoring_instant import resolve_scoring_instant

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            resolve_scoring_instant(shape)

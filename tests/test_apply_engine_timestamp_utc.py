"""Apply/rollback audit timestamps must be genuinely UTC, not local-time-with-a-Z.

Regression cover for the audit-trail defect where the apply engine wrote

    datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

into three durable artifacts — ``last_apply_ts`` in ``memory/intel-state.json``,
the ``<Status>: <ts>`` line stamped into a proposal block, and the
``RolledBack:`` line appended to ``APPLY_RECEIPT.md``. That formats a *naive
local* instant and then appends ``Z``, the UTC designator, so every record
actively claimed to be UTC while being local: a host at UTC+14 stamped the
audit trail 14 hours off, and a host at UTC-11 stamped it 11 hours the other
way, with nothing in the value to say so.

The clock is now :func:`mind_mem.apply_engine._utc_now`, so these tests freeze a
single instant, replay it under the widest real timezone spread available, and
assert the *same* string comes out. The wire format is unchanged, so stored
values written before the fix still parse.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone

import pytest

from mind_mem import apply_engine
from mind_mem.apply_engine import (
    AUDIT_TS_FORMAT,
    _mark_proposal_status,
    _parse_audit_ts,
    _utc_now,
    _utc_stamp,
    check_no_touch_window,
    create_snapshot,
    rollback,
    update_last_apply_ts,
)

# tzset() is POSIX-only; the Windows CI rows cannot rebind local time.
requires_tzset = pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="time.tzset() is unavailable on this platform",
)

# Deliberately just after UTC midnight: local time is still the *previous*
# day at UTC-11 and already mid-afternoon at UTC+14, so a naive local stamp
# differs from the UTC one in both the date and the time-of-day fields.
FROZEN_INSTANT = datetime(2026, 8, 27, 0, 30, 0, tzinfo=timezone.utc)
FROZEN_STAMP = "2026-08-27T00:30:00Z"

# UTC-11, UTC, UTC+14 — the widest spread of real zones available.
WIDE_ZONES = ("Pacific/Niue", "UTC", "Pacific/Kiritimati")

# The shape every emitted audit timestamp must keep.
TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``apply_engine``'s clock to FROZEN_INSTANT, honouring local TZ.

    ``now()`` is resolved through ``fromtimestamp`` exactly as CPython does,
    so a naive call still picks up the process timezone. A patch that simply
    returned a fixed UTC value would mask the very bug under test — see
    ``test_frozen_clock_helper_actually_tracks_local_time``.
    """
    epoch = FROZEN_INSTANT.timestamp()
    base = datetime

    class _Frozen(base):  # type: ignore[misc, valid-type]
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
            return base.fromtimestamp(epoch, tz)

    monkeypatch.setattr(apply_engine, "datetime", _Frozen)


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


def _make_workspace(root: str) -> None:
    """Minimal workspace with the state file the apply engine expects."""
    os.makedirs(os.path.join(root, "memory"), exist_ok=True)
    with open(os.path.join(root, "memory", "intel-state.json"), "w", encoding="utf-8") as fh:
        json.dump({}, fh)


def _last_apply_ts(root: str) -> str:
    with open(os.path.join(root, "memory", "intel-state.json"), encoding="utf-8") as fh:
        return str(json.load(fh)["last_apply_ts"])


def _stamp_under(tz_name: str) -> str:
    """``_utc_stamp()`` as produced on a host in ``tz_name``."""
    with pytest.MonkeyPatch.context() as mp:
        _set_timezone(mp, tz_name)
        _freeze(mp)
        return _utc_stamp()


def _write_proposal(root: str) -> str:
    path = os.path.join(root, "DECISIONS_PROPOSED.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("[P-20260827-001]\nProposalId: P-20260827-001\nType: edit\nTargetBlock: D-20260827-001\nStatus: staged\n")
    return path


def _status_stamp_under(tz_name: str) -> str:
    """The timestamp ``_mark_proposal_status`` stamps into a proposal block."""
    with pytest.MonkeyPatch.context() as mp:
        _set_timezone(mp, tz_name)
        _freeze(mp)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            path = _write_proposal(td)
            assert _mark_proposal_status(path, "P-20260827-001", "rejected", reason="superseded by a newer proposal")
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
    match = re.search(r"^Rejected: (\S+)$", content, re.MULTILINE)
    assert match, f"no status timestamp line written:\n{content}"
    return match.group(1)


def _rollback_stamp_under(tz_name: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """The timestamp ``rollback`` appends to APPLY_RECEIPT.md."""
    from mind_mem.init_workspace import init

    _set_timezone(monkeypatch, tz_name)
    monkeypatch.setattr(
        apply_engine,
        "check_preconditions",
        lambda ws: (True, ["validate: PASS (TOTAL 0 issues)"]),
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
        init(ws)
        proposal_path = os.path.join(ws, "intelligence", "proposed", "DECISIONS_PROPOSED.md")
        with open(proposal_path, "w", encoding="utf-8") as fh:
            fh.write("[P-20260827-001]\nProposalId: P-20260827-001\nType: edit\nTargetBlock: D-20260827-001\nStatus: applied\n")

        receipt_ts = "20260827-003000"
        snap_dir = create_snapshot(ws, receipt_ts, files_touched=["decisions/DECISIONS.md"])
        receipt_path = os.path.join(snap_dir, "APPLY_RECEIPT.md")
        with open(receipt_path, "w", encoding="utf-8") as fh:
            fh.write(f"[AR-{receipt_ts}]\nProposal: P-20260827-001\nResult: applied\n")

        # Freeze only now: create_snapshot/init do their own clock reads.
        _freeze(monkeypatch)
        assert rollback(ws, receipt_ts)
        with open(receipt_path, encoding="utf-8") as fh:
            content = fh.read()
    match = re.search(r"^RolledBack: (\S+)$", content, re.MULTILINE)
    assert match, f"no RolledBack timestamp written:\n{content}"
    return match.group(1)


class TestTimestampIsGenuinelyUTC:
    """(a) One frozen instant, three wildly different zones, identical string."""

    @requires_tzset
    def test_stamp_identical_across_timezones(self, restore_timezone) -> None:
        seen = {tz: _stamp_under(tz) for tz in WIDE_ZONES}
        assert set(seen.values()) == {FROZEN_STAMP}, f"audit timestamp is timezone-dependent: {seen}"

    @requires_tzset
    def test_last_apply_ts_identical_across_timezones(self, restore_timezone) -> None:
        seen: dict[str, str] = {}
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _freeze(mp)
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
                    _make_workspace(ws)
                    update_last_apply_ts(ws)
                    seen[tz_name] = _last_apply_ts(ws)
        assert set(seen.values()) == {FROZEN_STAMP}, f"last_apply_ts is timezone-dependent: {seen}"

    @requires_tzset
    def test_proposal_status_timestamp_identical_across_timezones(self, restore_timezone) -> None:
        seen = {tz: _status_stamp_under(tz) for tz in WIDE_ZONES}
        assert set(seen.values()) == {FROZEN_STAMP}, f"proposal status timestamp is timezone-dependent: {seen}"

    @requires_tzset
    def test_rollback_receipt_timestamp_identical_across_timezones(self, restore_timezone) -> None:
        seen: dict[str, str] = {}
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                seen[tz_name] = _rollback_stamp_under(tz_name, mp)
        assert set(seen.values()) == {FROZEN_STAMP}, f"RolledBack timestamp is timezone-dependent: {seen}"

    @requires_tzset
    def test_frozen_clock_helper_actually_tracks_local_time(self, restore_timezone) -> None:
        """Guard the guard: the freeze must still expose naive local drift.

        If ``_Frozen.now()`` ignored the process timezone, every assertion
        above would pass even with the naive call reintroduced.
        """
        naive_by_zone: set[str] = set()
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _freeze(mp)
                naive_by_zone.add(apply_engine.datetime.now().isoformat())
        assert len(naive_by_zone) == len(WIDE_ZONES), "frozen clock is not timezone-sensitive, so these tests cannot catch the bug"

    def test_stamp_matches_the_wall_clock_instant(self) -> None:
        """Unfrozen: the emitted value tracks real UTC, not local time."""
        before = _utc_now().replace(microsecond=0)
        emitted = _parse_audit_ts(_utc_stamp())
        after = _utc_now()
        assert before <= emitted <= after


class TestWireFormatUnchanged:
    """(b) Same characters on the wire — only the value became honest."""

    def test_shape_is_unchanged(self) -> None:
        stamp = _utc_stamp()
        assert stamp.endswith("Z")
        assert TS_PATTERN.match(stamp), stamp
        assert len(stamp) == 20

    def test_parses_with_the_documented_pattern(self) -> None:
        assert AUDIT_TS_FORMAT == "%Y-%m-%dT%H:%M:%SZ"
        parsed = datetime.strptime(_utc_stamp(), AUDIT_TS_FORMAT)
        assert parsed.tzinfo is None  # strptime's %Z-less parse, as before

    def test_iso_round_trip_yields_the_same_instant(self) -> None:
        stamp = FROZEN_INSTANT.strftime(AUDIT_TS_FORMAT)
        assert stamp == FROZEN_STAMP
        assert datetime.fromisoformat(stamp.replace("Z", "+00:00")) == FROZEN_INSTANT

    def test_status_and_receipt_lines_keep_their_key_shape(self, restore_timezone) -> None:
        """The governance key-line reader must still recognise the lines."""
        key_line = re.compile(r"^\s*(Status|Applied|Rejected|RolledBack|Reason|Proposal):")
        assert key_line.match(f"Rejected: {_utc_stamp()}")
        assert key_line.match(f"RolledBack: {_utc_stamp()}")


class TestReadBackStillWorks:
    """(c) Everything that parses these values back still behaves."""

    def test_parse_audit_ts_round_trips(self) -> None:
        assert _parse_audit_ts(FROZEN_STAMP) == FROZEN_INSTANT

    def test_parse_audit_ts_treats_offsetless_legacy_value_as_utc(self) -> None:
        """Values stored before the fix carried no offset once "Z" is stripped."""
        assert _parse_audit_ts("2026-08-27T00:30:00") == FROZEN_INSTANT

    def test_parse_audit_ts_converts_an_explicit_offset(self) -> None:
        assert _parse_audit_ts("2026-08-26T13:30:00-11:00") == FROZEN_INSTANT

    @requires_tzset
    def test_no_touch_window_blocks_a_just_written_stamp_in_every_zone(self, restore_timezone) -> None:
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _freeze(mp)
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
                    _make_workspace(ws)
                    update_last_apply_ts(ws)
                    ok, reason = check_no_touch_window(ws)
            assert not ok, f"cooldown ignored a fresh apply under TZ={tz_name}"
            assert "No-touch window" in reason

    @requires_tzset
    def test_no_touch_window_clears_an_hour_old_stamp_in_every_zone(self, restore_timezone) -> None:
        old_stamp = (FROZEN_INSTANT - timedelta(hours=1)).strftime(AUDIT_TS_FORMAT)
        for tz_name in WIDE_ZONES:
            with pytest.MonkeyPatch.context() as mp:
                _set_timezone(mp, tz_name)
                _freeze(mp)
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
                    _make_workspace(ws)
                    with open(os.path.join(ws, "memory", "intel-state.json"), "w", encoding="utf-8") as fh:
                        json.dump({"last_apply_ts": old_stamp}, fh)
                    ok, reason = check_no_touch_window(ws)
            assert ok, f"cooldown blocked an hour-old apply under TZ={tz_name}: {reason}"

    def test_no_touch_window_tolerates_a_corrupt_stamp(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ws:
            _make_workspace(ws)
            with open(os.path.join(ws, "memory", "intel-state.json"), "w", encoding="utf-8") as fh:
                json.dump({"last_apply_ts": "not-a-timestamp"}, fh)
            ok, _ = check_no_touch_window(ws)
        assert ok


class TestUtcClockSeam:
    """The clock helper itself is timezone-aware, so callers cannot regress."""

    def test_utc_now_is_timezone_aware(self) -> None:
        assert _utc_now().tzinfo is timezone.utc

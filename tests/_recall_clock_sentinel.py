"""Two independent instruments for proving the recall scoring path reads no clock.

Why two, and why neither of them is the obvious one.

The obvious guard — monkeypatch the clock accessors to raise, run a recall, and
assert it completed — **does not work here, and looked like it did.** ``recall``
wraps its optional legs in ``except Exception`` and degrades to a log line, so
an ``AssertionError`` signal is swallowed by the very handlers the guard has to
escape. Reverting the calibration-weight, validity-gate or trust-signal
threading left such a guard fully green while the warning it raised was printed
three times per run. A guard that reports success while its own alarm is going
off is worse than no guard.

So:

:class:`ClockSentinel` fixes the signal — it raises :class:`ClockRead`, which
derives from ``BaseException`` and is therefore invisible to every
``except Exception`` on the path, *and* it records each read in ``.reads``
before raising, so even a hypothetical ``except BaseException`` cannot hide the
event from the assertion.

:func:`clock_census` fixes the *coverage*. A monkeypatch only breaks the
accessors somebody thought to name; the census installs a ``sys.setprofile``
hook and observes every C-level call to ``datetime.now`` / ``datetime.utcnow``
/ ``date.today`` made anywhere inside :mod:`mind_mem`, whatever module makes it
and whether or not the result is swallowed. It needs no list of accessors, so a
clock read introduced tomorrow in a module nobody patched still trips it.

SCOPE, stated so the guard is not read as claiming more than it checks. Both
instruments target the *date* clocks — ``datetime.now``, ``datetime.utcnow``,
``date.today`` — because those are the reads that can decide a rank or a set
membership. ``time.time()`` / ``time.monotonic()`` are deliberately out of
scope: the cache TTL, the latency timers and the metrics counters read them,
and none of those values reaches a scoring term.

Exactly one call site is unconditionally allowlisted
(:data:`_ALWAYS_ALLOWED_CLOCK_SITES` — the log record's timestamp). The
resolver's own read is allowlisted **only** when a test opts in with
``allow_boundary_read=True``, and that distinction is load-bearing rather than
cosmetic: every dropped pass-through re-enters the resolver as ``None`` and
reads the clock *there*, so a census that always forgave the boundary reported
four separate reverts — the index-missing fallback, the hybrid BM25 hand-off,
the validity gate and the trust-signal load — as perfectly clean.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from datetime import date, datetime, timezone
from types import FrameType
from typing import Any, Callable, Iterator

import pytest

_MIND_MEM_DIR = f"{os.sep}mind_mem{os.sep}"

#: Call sites permitted to read a wall clock during *any* recall: every
#: structured log record stamps a ``ts``, which is written to a stream and
#: never read back into a score.
_ALWAYS_ALLOWED_CLOCK_SITES = frozenset({"observability.py"})

#: THE sanctioned boundary read, :func:`mind_mem.scoring_instant._read_utc_today`.
#: Allowed **only** on a run that deliberately took the default instant. On a run
#: given an explicit instant it must never be reached, and treating it as always
#: allowed is what let a whole class of defect through: every pass-through that
#: drops the instant — the index-missing fallback, the hybrid BM25 hand-off, the
#: validity gate, the trust-signal load — resurfaces here as "the boundary read",
#: and a census that waved it through reported those reverts as clean.
_BOUNDARY_CLOCK_SITE = "scoring_instant.py"

#: The clock reads that can move a ranking: ``datetime.now`` / ``datetime.utcnow``
#: / ``datetime.today`` / ``date.today``. Matched on the method name plus the
#: **identity** of the class it is bound to, because these builtins carry
#: ``__module__ is None`` — and by identity rather than by set membership,
#: because ``__self__`` on an arbitrary C call may be an unhashable object.
_CLOCK_NAMES = frozenset({"now", "utcnow", "today"})


class ClockRead(BaseException):
    """A wall-clock read on the deterministic scoring path.

    Deliberately **not** an :class:`Exception`. ``recall()`` degrades rather
    than fails on almost every optional leg — the calibration weight, the
    validity gate, the trust-signal load and each multi-hop sub-query are all
    wrapped in ``except Exception`` — so an ``Exception`` signal is caught by
    the code under test and turned into a warning. Deriving from
    ``BaseException`` makes the alarm escape all of them.
    """


def _innermost_mind_mem_frame(skip: int = 0) -> str:
    """Name the ``mind_mem`` frame that made the call, for the failure message."""
    for frame in reversed(traceback.extract_stack()[: -1 - skip]):
        if _MIND_MEM_DIR in frame.filename:
            return f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}()"
    return "<no mind_mem frame>"


class ClockSentinel:
    """Records every scoring-path clock read, then raises past the handlers."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def trip(self, *_args: object, **_kwargs: object) -> Any:
        site = _innermost_mind_mem_frame()
        self.reads.append(site)
        raise ClockRead(f"the deterministic core read a clock at {site}")

    def assert_clock_free(self) -> None:
        assert not self.reads, "the deterministic core read a clock at: " + "; ".join(self.reads)


def install_clock_sentinel(monkeypatch: pytest.MonkeyPatch) -> ClockSentinel:
    """Break every named scoring clock and return the recorder.

    Covers the accessors the recency layer reaches through: the recency ramp's
    ``_utc_now``, the calibration window's ``datetime``, the temporal filter's
    ``date``, and the sanctioned boundary read itself.
    """
    from mind_mem import _recall_scoring, _recall_temporal, calibration, scoring_instant

    sentinel = ClockSentinel()

    class _NoClockDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
            return sentinel.trip(tz)

        @classmethod
        def utcnow(cls) -> datetime:  # type: ignore[override]
            return sentinel.trip()

    class _NoClockDate(date):
        @classmethod
        def today(cls) -> date:  # type: ignore[override]
            return sentinel.trip()

    monkeypatch.setattr(_recall_scoring, "_utc_now", sentinel.trip)
    monkeypatch.setattr(_recall_scoring, "_datetime", _NoClockDateTime)
    monkeypatch.setattr(calibration, "datetime", _NoClockDateTime)
    monkeypatch.setattr(_recall_temporal, "date", _NoClockDate)
    monkeypatch.setattr(scoring_instant, "_read_utc_today", sentinel.trip)
    return sentinel


class ClockCensus:
    """Every date-clock read made inside ``mind_mem``, by call site."""

    def __init__(self, *, allow_boundary_read: bool = False) -> None:
        self.reads: list[str] = []
        self._allowed = set(_ALWAYS_ALLOWED_CLOCK_SITES)
        if allow_boundary_read:
            self._allowed.add(_BOUNDARY_CLOCK_SITE)

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        if event != "c_call":
            return
        name = getattr(arg, "__name__", "")
        if name not in _CLOCK_NAMES:
            return
        owner = getattr(arg, "__self__", None)
        if owner is not datetime and owner is not date:
            return
        filename = frame.f_code.co_filename
        if _MIND_MEM_DIR not in filename:
            return
        basename = os.path.basename(filename)
        if basename in self._allowed:
            return
        self.reads.append(f"{basename}:{frame.f_lineno} in {frame.f_code.co_name}() -> {name}")

    def assert_clock_free(self) -> None:
        assert not self.reads, "the deterministic core read a clock at: " + "; ".join(sorted(set(self.reads)))


def clock_census(*, allow_boundary_read: bool = False) -> "_CensusContext":
    """Observe (never intercept) date-clock reads for the duration of a block.

    An observer rather than an exception, so no ``except`` anywhere on the path
    can hide a read from it — which is precisely the failure mode that made the
    first version of this guard vacuous.

    Args:
        allow_boundary_read: Permit :data:`_BOUNDARY_CLOCK_SITE`. Pass ``True``
            **only** for a run that deliberately omitted ``scoring_instant``.
            The default forbids it, which is what makes a dropped pass-through
            visible: the dropped value re-enters the resolver as ``None`` and
            reads the clock right there.
    """
    return _CensusContext(allow_boundary_read=allow_boundary_read)


class _CensusContext:
    def __init__(self, *, allow_boundary_read: bool) -> None:
        self._allow_boundary_read = allow_boundary_read

    def __enter__(self) -> ClockCensus:
        self._census = ClockCensus(allow_boundary_read=self._allow_boundary_read)
        self._previous = sys.getprofile()
        # ``sys.setprofile`` is per-thread, and the hybrid backend fans its
        # BM25 and dense legs out to a ThreadPoolExecutor — so the main-thread
        # hook alone is blind to exactly the two hand-offs most likely to drop
        # the instant. ``threading.setprofile`` arms every thread started from
        # here on, which is all of the pool's.
        threading.setprofile(self._census._profile)
        sys.setprofile(self._census._profile)
        return self._census

    def __exit__(self, *_exc: object) -> None:
        sys.setprofile(self._previous)
        threading.setprofile(None)


def write_workspace(root: str, blocks: Iterator[tuple[str, str, str]] | tuple[tuple[str, str, str], ...]) -> None:
    """Write *blocks* as decisions into a minimal recall workspace."""
    for sub in ("decisions", "tasks", "entities", "intelligence"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    with open(os.path.join(root, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        for bid, statement, when in blocks:
            fh.write(f"[{bid}]\nStatement: {statement}\nStatus: active\nDate: {when}\n\n")


def seed_calibration_feedback(workspace: str, block_ids: tuple[str, ...], *, stamped: str) -> None:
    """Record rejection feedback so the rolling calibration window has content.

    Without rows the weight is a constant 1.0 and the window is not an input at
    all — a guard over the calibration leg would be measuring nothing. Rows go
    in through the sanctioned writer (:meth:`CalibrationManager.record_feedback`
    stamps them "now"); only the timestamp is then pinned to *stamped*, which is
    the one thing a clock-pinning test has to control. ``MIN_FEEDBACK_THRESHOLD``
    rows are needed before the weight moves off 1.0 at all.
    """
    from mind_mem.calibration import MIN_FEEDBACK_THRESHOLD, CalibrationManager

    mgr = CalibrationManager(workspace)
    for index in range(MIN_FEEDBACK_THRESHOLD + 1):
        mgr.record_feedback(
            query_id=f"seed-{index}",
            block_ids_useful=[],
            block_ids_not_useful=list(block_ids),
            feedback_type="rejected",
            query_text="seed",
            query_type="single-hop",
        )
    with mgr._mgr.write_lock:
        conn = mgr._mgr.get_write_connection()
        conn.execute("UPDATE calibration_feedback SET created_at = ?", (stamped,))
        conn.commit()


def scored_order(hits: list[dict]) -> list[tuple[str, float]]:
    """Serialize a served ranking so drift is a value difference, not an epsilon."""
    return [(str(h.get("_id")), round(float(h.get("score", 0.0) or 0.0), 6)) for h in hits]


__all__ = [
    "ClockCensus",
    "ClockRead",
    "ClockSentinel",
    "clock_census",
    "install_clock_sentinel",
    "scored_order",
    "seed_calibration_feedback",
    "write_workspace",
]


# Re-exported for the guard tests: a Callable alias keeps the type checker quiet
# about assigning ``sentinel.trip`` over a classmethod.
TripFn = Callable[..., Any]

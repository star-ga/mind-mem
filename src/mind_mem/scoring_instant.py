"""The recency seam — one UTC date, resolved once, threaded everywhere.

mind-mem's wedge is deterministic retrieval. That splits the recall pipeline in
two, and this module is the joint between the halves:

**Deterministic core** — BM25F, the vector leg, RRF fusion, the validity gate
and the guardrails. It reads no clock at all. Given the same corpus and config
it produces the same candidates and the same base scores forever.

**Recency layer** — ``date_score``, ``temporal_decay_score``, the calibration
manager's rolling window, and the temporal hard filter. Every one of these is a
function of *when you are asking*, and recency is load-bearing for a coding
agent: a memory store that cannot prefer last week's decision to last year's is
not useful. So the answer is not to delete recency, it is to stop letting it
read a hidden clock.

Each of those terms therefore takes an explicit instant, and every one of them
is fed from :func:`resolve_scoring_instant`. That resolver is the **only** place
the wall clock is consulted on the recall path, and it sits at the boundary —
``recall()``'s signature — not in the scoring loop. Supply ``scoring_instant``
and the whole path is clock-free; omit it and exactly one read happens, once,
before ranking begins.

The instant is a UTC **date**, not a timestamp. The underlying math is already
day-granular (``days_old``, a 30-day window), so a date is the honest unit: it
is stable for a whole day, cheap to record, replayable by hand, and it cannot
introduce spurious hash churn the way a second-resolution timestamp would.

The resulting claim is precise and true: recall is **deterministic given
(corpus, config, scoring_instant)**, and any attested run replays by passing
back its attested date.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

__all__ = [
    "resolve_scoring_instant",
    "as_utc_datetime",
    "format_scoring_instant",
    "parse_scoring_instant",
    "ISO_DATE_WIDTH",
]

#: Width of the serialized form, ``YYYY-MM-DD``. Fixed, ASCII, NUL-free.
ISO_DATE_WIDTH = 10

#: The *only* accepted input shape. Deliberately stricter than
#: :meth:`datetime.date.fromisoformat`, which since Python 3.11 accepts the
#: whole ISO-8601 date grammar — including the week form ``YYYY-Www-D``, which
#: is also exactly ten characters. A client sending ``2026-W01-1`` would
#: otherwise be silently scored against ``2025-12-29``, eight months from the
#: date it named, while this module's own error message insists on
#: ``YYYY-MM-DD``. Surrounding whitespace is rejected for the same reason: the
#: value is a hash-bound run input, so the boundary accepts one spelling of it.
_ISO_CALENDAR_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _read_utc_today() -> date:
    """THE clock read: today's date in UTC.

    The single wall-clock access on the recall scoring path. It is deliberately
    a named module-level accessor rather than an inline ``datetime.now()`` so a
    test can break it and prove nothing else reads a clock behind its back.
    """
    return datetime.now(timezone.utc).date()


def resolve_scoring_instant(value: date | str | None = None) -> date:
    """Normalise *value* to the UTC date the recency layer will score against.

    ``None`` means "now", resolved as today **in UTC** — never in the host's
    local zone, which is what made the same corpus rank differently on two
    machines at the same instant.

    A :class:`~datetime.datetime` is accepted but narrowed: it is converted to
    UTC (a naive value is read as UTC, not local) and its date taken. This is an
    explicit boundary rule, because ``datetime`` is a subclass of ``date`` and
    would otherwise slip through and serialise at second resolution.

    Args:
        value: A ``date``, a ``datetime``, an ISO-8601 ``YYYY-MM-DD`` string, or
            ``None`` for today-in-UTC.

    Returns:
        The UTC date to score against.

    Raises:
        TypeError: *value* is not a date, datetime, string or None.
        ValueError: *value* is a string that is not an ISO-8601 calendar date.
    """
    if value is None:
        return _read_utc_today()
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return aware.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return parse_scoring_instant(value)
    raise TypeError(f"scoring_instant must be a date, datetime, ISO-8601 string or None, got {type(value).__name__}")


def parse_scoring_instant(raw: str) -> date:
    """Parse a serialized instant back into a date (the replay direction).

    Accepts exactly one spelling — ``YYYY-MM-DD`` — and nothing else. See
    :data:`_ISO_CALENDAR_DATE` for why a length check over
    :meth:`~datetime.date.fromisoformat` is not sufficient here.

    Raises:
        ValueError: *raw* is not exactly a ``YYYY-MM-DD`` calendar date.
    """
    if not isinstance(raw, str):  # pragma: no cover — guarded by resolve_scoring_instant
        raise TypeError(f"scoring_instant string expected, got {type(raw).__name__}")
    if not _ISO_CALENDAR_DATE.match(raw):
        raise ValueError(f"scoring_instant must be an ISO-8601 date of the form YYYY-MM-DD, got {raw!r}")
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"scoring_instant must be an ISO-8601 date of the form YYYY-MM-DD, got {raw!r}") from exc


def format_scoring_instant(instant: date) -> str:
    """Serialize an instant for the attestation preimage and the envelope.

    Always ten ASCII characters with no time component and no offset suffix, so
    the preimage byte string is stable and the envelope value round-trips
    through :func:`parse_scoring_instant` unchanged. A ``datetime`` is narrowed
    first — its ``isoformat()`` would otherwise emit nineteen characters and
    churn the hash on every run.
    """
    return resolve_scoring_instant(instant).isoformat()


def as_utc_datetime(instant: date) -> datetime:
    """Anchor an instant at UTC midnight for the day-granular recency helpers.

    ``date_score`` / ``temporal_decay_score`` / the calibration cutoff all take
    a ``datetime``; every one of them reduces it to whole days, so midnight is
    the exact anchor that makes the day the unit of account.
    """
    resolved = resolve_scoring_instant(instant)
    return datetime(resolved.year, resolved.month, resolved.day, tzinfo=timezone.utc)

"""Recall engine temporal filtering — resolve relative time references and filter blocks."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

__all__ = [
    "resolve_time_reference",
    "apply_temporal_filter",
]

# Month name -> number mapping
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# Patterns for relative time references, ordered by specificity
_LAST_N_DAYS_RE = re.compile(
    r"\blast\s+(\d+)\s+days?\b", re.IGNORECASE,
)
_N_DAYS_AGO_RE = re.compile(
    r"\b(\d+)\s+days?\s+ago\b", re.IGNORECASE,
)
_N_WEEKS_AGO_RE = re.compile(
    r"\b(\d+)\s+weeks?\s+ago\b", re.IGNORECASE,
)
_N_MONTHS_AGO_RE = re.compile(
    r"\b(\d+)\s+months?\s+ago\b", re.IGNORECASE,
)
_LAST_WEEK_RE = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_LAST_MONTH_RE = re.compile(r"\blast\s+month\b", re.IGNORECASE)
_LAST_YEAR_RE = re.compile(r"\blast\s+year\b", re.IGNORECASE)
_THIS_WEEK_RE = re.compile(r"\bthis\s+week\b", re.IGNORECASE)
_THIS_MONTH_RE = re.compile(r"\bthis\s+month\b", re.IGNORECASE)
_THIS_YEAR_RE = re.compile(r"\bthis\s+year\b", re.IGNORECASE)
_YESTERDAY_RE = re.compile(r"\byesterday\b", re.IGNORECASE)
_TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)

# "in January 2025" or "in January"
_IN_MONTH_YEAR_RE = re.compile(
    r"\bin\s+(" + "|".join(_MONTH_NAMES.keys()) + r")(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)

# "in 2025"
_IN_YEAR_RE = re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE)

# "before/after DATE" where DATE is YYYY-MM-DD or "Month DD, YYYY" or "DD Month YYYY"
_BEFORE_DATE_RE = re.compile(
    r"\bbefore\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE,
)
_AFTER_DATE_RE = re.compile(
    r"\bafter\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE,
)

# "before/after Month DD, YYYY" or "before/after Month YYYY"
_MONTH_NAMES_PATTERN = "|".join(
    sorted(_MONTH_NAMES.keys(), key=len, reverse=True),
)
_BEFORE_MONTH_DATE_RE = re.compile(
    r"\bbefore\s+(" + _MONTH_NAMES_PATTERN + r")\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_AFTER_MONTH_DATE_RE = re.compile(
    r"\bafter\s+(" + _MONTH_NAMES_PATTERN + r")\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)


def _parse_iso_date(s: str) -> date | None:
    """Parse YYYY-MM-DD string to date, or None."""
    try:
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def _month_range(year: int, month: int) -> tuple[date, date]:
    """Return (first_day, last_day) for a given month."""
    first = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    last = date(year, month, last_day)
    return first, last


def resolve_time_reference(
    query: str,
    reference_date: date | None = None,
) -> tuple[date | None, date | None]:
    """Parse relative time references in a query and return a date range.

    Returns (start_date, end_date) or (None, None) if no temporal reference found.
    For "before X" returns (None, X). For "after X" returns (X, None).

    Args:
        query: The search query string.
        reference_date: The reference date for relative calculations.
            Defaults to today if None.
    """
    if not query:
        return None, None

    ref = reference_date or date.today()

    # "yesterday"
    if _YESTERDAY_RE.search(query):
        yesterday = ref - timedelta(days=1)
        return yesterday, yesterday

    # "today"
    if _TODAY_RE.search(query):
        return ref, ref

    # "last N days"
    m = _LAST_N_DAYS_RE.search(query)
    if m:
        n = int(m.group(1))
        return ref - timedelta(days=n), ref

    # "N days ago"
    m = _N_DAYS_AGO_RE.search(query)
    if m:
        n = int(m.group(1))
        d = ref - timedelta(days=n)
        return d, d

    # "N weeks ago"
    m = _N_WEEKS_AGO_RE.search(query)
    if m:
        n = int(m.group(1))
        d = ref - timedelta(weeks=n)
        # Give a week-wide window centered on that date
        return d - timedelta(days=3), d + timedelta(days=3)

    # "N months ago"
    m = _N_MONTHS_AGO_RE.search(query)
    if m:
        n = int(m.group(1))
        # Approximate: subtract n*30 days, return that month's range
        approx = ref - timedelta(days=n * 30)
        return _month_range(approx.year, approx.month)

    # "last week" — previous Monday through Sunday
    if _LAST_WEEK_RE.search(query):
        # Monday of this week
        this_monday = ref - timedelta(days=ref.weekday())
        last_monday = this_monday - timedelta(days=7)
        last_sunday = last_monday + timedelta(days=6)
        return last_monday, last_sunday

    # "this week" — Monday of current week through today
    if _THIS_WEEK_RE.search(query):
        this_monday = ref - timedelta(days=ref.weekday())
        return this_monday, ref

    # "last month"
    if _LAST_MONTH_RE.search(query):
        first_of_this = ref.replace(day=1)
        last_of_prev = first_of_this - timedelta(days=1)
        first_of_prev = last_of_prev.replace(day=1)
        return first_of_prev, last_of_prev

    # "this month"
    if _THIS_MONTH_RE.search(query):
        first_of_this = ref.replace(day=1)
        return first_of_this, ref

    # "last year"
    if _LAST_YEAR_RE.search(query):
        return date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31)

    # "this year"
    if _THIS_YEAR_RE.search(query):
        return date(ref.year, 1, 1), ref

    # "before Month DD, YYYY"
    m = _BEFORE_MONTH_DATE_RE.search(query)
    if m:
        month_num = _MONTH_NAMES.get(m.group(1).lower())
        if month_num:
            try:
                d = date(int(m.group(3)), month_num, int(m.group(2)))
                return None, d
            except ValueError:
                pass

    # "after Month DD, YYYY"
    m = _AFTER_MONTH_DATE_RE.search(query)
    if m:
        month_num = _MONTH_NAMES.get(m.group(1).lower())
        if month_num:
            try:
                d = date(int(m.group(3)), month_num, int(m.group(2)))
                return d, None
            except ValueError:
                pass

    # "before YYYY-MM-DD"
    m = _BEFORE_DATE_RE.search(query)
    if m:
        d = _parse_iso_date(m.group(1))
        if d:
            return None, d

    # "after YYYY-MM-DD"
    m = _AFTER_DATE_RE.search(query)
    if m:
        d = _parse_iso_date(m.group(1))
        if d:
            return d, None

    # "in January 2025" or "in January"
    m = _IN_MONTH_YEAR_RE.search(query)
    if m:
        month_num = _MONTH_NAMES.get(m.group(1).lower())
        if month_num:
            year = int(m.group(2)) if m.group(2) else ref.year
            return _month_range(year, month_num)

    # "in 2025" — but not if already matched "in Month 2025"
    m = _IN_YEAR_RE.search(query)
    if m:
        year = int(m.group(1))
        # Sanity: only match reasonable years
        if 1900 <= year <= 2100:
            return date(year, 1, 1), date(year, 12, 31)

    return None, None


def apply_temporal_filter(
    blocks: list[dict],
    start: date | None,
    end: date | None,
) -> list[dict]:
    """Filter blocks by date range. Blocks without a Date field pass through.

    Args:
        blocks: List of block/result dicts (must have optional "Date" key in YYYY-MM-DD format).
        start: Inclusive start date, or None for no lower bound.
        end: Inclusive end date, or None for no upper bound.

    Returns:
        Filtered list of blocks.
    """
    if start is None and end is None:
        return blocks

    filtered = []
    for block in blocks:
        date_str = block.get("Date", "")
        if not date_str:
            # No date — pass through (don't exclude undated blocks)
            filtered.append(block)
            continue

        block_date = _parse_iso_date(date_str[:10])
        if block_date is None:
            # Unparseable date — pass through
            filtered.append(block)
            continue

        # Apply range check
        if start is not None and block_date < start:
            continue
        if end is not None and block_date > end:
            continue

        filtered.append(block)

    return filtered

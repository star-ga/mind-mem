"""Pure parsing and filtering for optional bearer-token expiries.

Bare entries retain the existing never-expiring behaviour.  An entry may
carry an absolute epoch deadline as ``<token>|exp=<epoch>``.  Malformed
attributes fail closed, and an all-expired configured set stays empty rather
than falling through to a less-specific credential source.
"""

from __future__ import annotations

import math
from typing import Optional

ATTR_SEP = "|"
EXPIRY_KEY = "exp"


def parse_token_entry(entry: object) -> tuple[str, Optional[int]]:
    """Return ``(token_value, expiry_or_None)`` for one configured entry."""
    text = str(entry or "").strip()
    if ATTR_SEP not in text:
        return text, None
    value, _, attrs = text.partition(ATTR_SEP)
    value = value.strip()
    key, eq, raw = attrs.strip().partition("=")
    if key.strip().lower() != EXPIRY_KEY or not eq:
        return value, -1
    try:
        seconds = int(str(raw).strip())
    except (TypeError, ValueError):
        return value, -1
    if seconds < 0:
        return value, -1
    return value, seconds


def is_expired(entry: object, *, now: float | int) -> bool:
    """Whether *entry* is expired at epoch time *now*.

    The deadline is inclusive on the live side: equality remains valid.
    Avoid converting the configured integer deadline to float, because very
    large valid epoch values must not overflow during authentication.
    """
    _, expires = parse_token_entry(entry)
    if expires is None:
        return False
    if expires < 0:
        return True
    if isinstance(now, int):
        return now > expires
    try:
        current = float(now)
    except (TypeError, ValueError, OverflowError):
        return True
    if not math.isfinite(current):
        return True
    return current > expires


def active_token_values(raw: object, *, now: float | int) -> list[str]:
    """Return non-expired token values from one configured raw list.

    ``[]`` means no entry is currently usable.  Callers must preserve that
    result when the source variable was configured; it is not a fallback
    signal.
    """
    out: list[str] = []
    for part in str(raw or "").split(","):
        if not part.strip():
            continue
        value, _ = parse_token_entry(part)
        if value and not is_expired(part, now=now):
            out.append(value)
    return out


__all__ = ["ATTR_SEP", "EXPIRY_KEY", "active_token_values", "is_expired", "parse_token_entry"]

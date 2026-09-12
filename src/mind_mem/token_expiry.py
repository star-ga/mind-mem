"""Server-side expiry for rotated auth tokens. The grace window stops being advisory.

ROADMAP residual against the shipped token-rotation primitive: "the grace window is
ADVISORY. ``grace_seconds`` is printed and the operator must run the emitted
``shell_final`` export; the server never timestamps or auto-expires an old token, so
the 'then expires' half of the original line is not implemented."

MEASURED before writing this: ``MIND_MEM_TOKENS="old,new"`` yields both tokens
active with no expiry on either, so a retired credential stays valid until a human
edits an environment variable. Rotation without expiry is not rotation, it is
accumulation -- and the security property the feature advertises does not exist.

THE ENTRY GRAMMAR is additive, because every deployment today sets bare tokens and
must keep working forever:

    <token>                 no expiry, exactly today's behaviour
    <token>|exp=<epoch>     expires after that instant

FAIL-CLOSED ON A MALFORMED EXPIRY, which is the one judgement call in this module:
an entry whose expiry cannot be parsed is treated as EXPIRED, never as
never-expiring. A typo in a timestamp must not silently grant a permanent
credential; when the subject is authentication, a mistake should cost availability
rather than security. The same rule covers an unknown suffix key -- ``|ttl=3600`` is
not ``|exp=``, and guessing what an operator meant is not safe here.

AN ALL-EXPIRED SET RETURNS EMPTY, deliberately. The tempting "if nothing is left,
fall back to the raw list" would turn a fully-rotated deployment into an open one --
the exact shape of failure this codebase has been closing all session, where a check
that cannot pass returns what a passing check returns. A caller seeing an empty set
must refuse the request.

Pure: the caller passes ``now``. No clock is read here, so a token decision is
testable and replayable, and a test cannot pass merely because it ran quickly.
"""

from __future__ import annotations

from typing import Optional

#: Separator between a token value and its optional attributes.
ATTR_SEP = "|"

#: The one supported attribute: absolute expiry, epoch seconds.
EXPIRY_KEY = "exp"


def parse_token_entry(entry: object) -> tuple[str, Optional[int]]:
    """``(token_value, expiry_or_None)`` for one configured entry.

    A bare entry yields ``(value, None)`` -- no expiry, which is what every
    existing deployment means. An entry carrying an unparseable or unknown
    attribute yields ``(value, -1)``: a sentinel that :func:`is_expired` reads as
    already expired, so the failure is closed rather than open.
    """
    text = str(entry or "").strip()
    if ATTR_SEP not in text:
        return text, None
    value, _, attrs = text.partition(ATTR_SEP)
    value = value.strip()
    key, eq, raw = attrs.strip().partition("=")
    if key.strip().lower() != EXPIRY_KEY or not eq:
        # Unknown attribute. Fail closed: -1 is in the past for any sane clock.
        return value, -1
    try:
        seconds = int(str(raw).strip())
    except (TypeError, ValueError):
        return value, -1
    if seconds < 0:
        return value, -1
    return value, seconds


def is_expired(entry: object, *, now: float) -> bool:
    """Whether *entry* is no longer acceptable at *now* (epoch seconds).

    The boundary is inclusive of the live side: at exactly the expiry instant the
    token is still valid. That resolves a clock landing on the deadline toward the
    in-flight client the grace window exists to protect, rather than producing a
    spurious authentication failure mid-request.
    """
    _, expires = parse_token_entry(entry)
    if expires is None:
        return False            # bare token: no expiry, as today
    if expires < 0:
        return True             # malformed: fail closed
    return float(now) > float(expires)


def active_token_values(raw: object, *, now: float) -> list[str]:
    """The token values a server should accept at *now*, in configured order.

    Blank entries are dropped rather than becoming an empty token value -- an
    empty token would match an empty ``Authorization`` header, which is the
    accidental open door.

    Returns ``[]`` when every entry has expired. A caller MUST treat an empty
    result as "refuse everything", never as "no restriction configured".
    """
    out: list[str] = []
    for part in str(raw or "").split(","):
        if not part.strip():
            continue
        value, _ = parse_token_entry(part)
        if not value:
            continue
        if is_expired(part, now=now):
            continue
        out.append(value)
    return out


__all__ = [
    "ATTR_SEP",
    "EXPIRY_KEY",
    "active_token_values",
    "is_expired",
    "parse_token_entry",
]

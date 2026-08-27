# Copyright 2026 STARGA, Inc.
"""Pattern primitives for GUARDRAIL triggers — one deterministic matcher.

Every trigger dimension (tool / command / intent / path) is matched with the
same tiny glob grammar, so a rule author learns one syntax and the engine has
one code path to reason about:

* ``*`` matches within one path segment, ``**`` crosses segments, ``**/``
  matches zero or more leading segments, ``?`` matches one non-separator
  character.  Everything else — brackets included — is literal.
* A pattern with no metacharacter is an exact match (tool / intent) or a
  substring match (command).
* Values and patterns are normalised the same way: casefolded, whitespace
  runs collapsed, ``\\`` rewritten to ``/`` for paths.

No model call, no clock, no randomness — matching a context against a corpus
is a pure function, and the compiled-regex cache is keyed on the pattern text
alone.  This module is a leaf: it knows nothing about blocks or recall.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Sequence

__all__ = [
    "MAX_PATTERNS_PER_DIMENSION",
    "MAX_PATTERN_LEN",
    "GuardrailSpecError",
    "coerce_patterns",
    "exact_or_glob",
    "path_match",
    "substring_or_glob",
]

#: Per-dimension cap on declared patterns (bounds match work per block).
MAX_PATTERNS_PER_DIMENSION = 64

#: Per-pattern length cap (bounds regex translation work).
MAX_PATTERN_LEN = 256

_GLOB_META = ("*", "?")
_WS_RUN = re.compile(r"\s+")


class GuardrailSpecError(ValueError):
    """Raised when a ``[GR-...]`` block cannot be read as a guardrail."""


# ---------------------------------------------------------------------------
# Deterministic glob matcher (shared by every trigger dimension)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob pattern into an anchored regex.  Pure + cached."""
    out: list[str] = ["(?s)\\A"]
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if pattern[i : i + 3] == "**/":
                out.append("(?:[^/]*/)*")
                i += 3
                continue
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
            continue
        if char == "?":
            out.append("[^/]")
            i += 1
            continue
        out.append(re.escape(char))
        i += 1
    out.append("\\Z")
    return re.compile("".join(out))


def _has_glob(pattern: str) -> bool:
    return any(meta in pattern for meta in _GLOB_META)


def _norm_text(value: str) -> str:
    """Casefold + collapse whitespace runs.  Deterministic, locale-free."""
    return _WS_RUN.sub(" ", value.strip()).casefold()


def _norm_path(value: str) -> str:
    """Normalise a path for glob matching: ``/`` separators, no ``./``."""
    text = value.strip().replace("\\", "/").casefold()
    while text.startswith("./"):
        text = text[2:]
    return text


def exact_or_glob(patterns: Sequence[str], value: str) -> bool:
    if not value:
        return False
    target = _norm_text(value)
    for pattern in patterns:
        if _has_glob(pattern):
            if _glob_regex(pattern).match(target):
                return True
        elif pattern == target:
            return True
    return False


def substring_or_glob(patterns: Sequence[str], value: str) -> bool:
    if not value:
        return False
    target = _norm_text(value)
    for pattern in patterns:
        if _has_glob(pattern):
            if _glob_regex(pattern).match(target):
                return True
        elif pattern in target:
            return True
    return False


def path_match(patterns: Sequence[str], paths: Sequence[str]) -> bool:
    for raw in paths:
        target = _norm_path(raw)
        if not target:
            continue
        for pattern in patterns:
            if _glob_regex(pattern).match(target):
                return True
    return False


# ---------------------------------------------------------------------------
# Value coercion at the block boundary
# ---------------------------------------------------------------------------


def coerce_patterns(raw: Any, *, field: str, block_id: str, as_path: bool) -> tuple[str, ...]:
    """Read one trigger field into a validated, normalised pattern tuple.

    Accepts the two shapes the block parser produces: a scalar string
    (comma-separated, matching the ``Tags``/``References`` convention) or a
    markdown list.  List entries are taken verbatim — that is the escape
    hatch for a pattern that itself contains a comma.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: list[str] = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        items = []
        for entry in raw:
            if not isinstance(entry, str):
                raise GuardrailSpecError(f"{block_id}: {field} entries must be strings, got {type(entry).__name__}")
            items.append(entry)
    else:
        raise GuardrailSpecError(f"{block_id}: {field} must be a string or list, got {type(raw).__name__}")

    out: list[str] = []
    for item in items:
        text = item.strip().strip("`")
        if not text:
            continue
        if len(text) > MAX_PATTERN_LEN:
            raise GuardrailSpecError(f"{block_id}: {field} pattern exceeds {MAX_PATTERN_LEN} chars")
        out.append(_norm_path(text) if as_path else _norm_text(text))
        if len(out) > MAX_PATTERNS_PER_DIMENSION:
            raise GuardrailSpecError(f"{block_id}: {field} declares more than {MAX_PATTERNS_PER_DIMENSION} patterns")
    return tuple(out)

"""Invisible-Unicode codepoint sanitization for block ingestion (security).

Invisible codepoints are a prompt-injection vector: instructions encoded
in Unicode tag characters (U+E0000-U+E007F), zero-width characters, or
bidi controls survive copy/paste and file ingestion while staying
invisible to human reviewers — and an LLM reading the recalled block
text *does* see them. This module strips them at the ingestion boundary
so they never enter the corpus.

Stripped classes (union of explicit ranges + Unicode categories):

- Zero-width characters — U+200B ZWSP, U+200C ZWNJ, U+200D ZWJ,
  U+FEFF ZWNBSP/BOM, U+2060 WORD JOINER, U+00AD SOFT HYPHEN, U+180E.
- Unicode tag characters — U+E0000-U+E007F (the classic hidden-prompt
  smuggling channel).
- Bidi controls — U+202A-U+202E (embed/override) and U+2066-U+2069
  (isolates), plus U+200E/U+200F/U+061C marks.
- All other ``Cf`` (format) and ``Co`` (private-use) category
  codepoints, ``Cs`` surrogates, and ``Cc`` controls **except**
  ``\\t``, ``\\n``, ``\\r``.

Preserved: normal whitespace (space/tab/newline/CR), all letters,
digits, punctuation, symbols and real non-ASCII text (Cyrillic, CJK,
accented Latin, emoji — emoji ZWJ sequences degrade to their component
emoji, which is the accepted trade-off for closing the channel).

Config gate (default **ON**)::

    {"ingest": {"sanitize_codepoints": false}}   # mind-mem.json — disable

Environment override (wins over config): ``MIND_MEM_SANITIZE_CODEPOINTS``
set to ``0``/``false``/``no``/``off`` disables, ``1``/``true``/``yes``/
``on`` force-enables.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Mapping

__all__ = [
    "is_sanitize_enabled",
    "sanitize_codepoints",
    "sanitize_enabled_for_workspace",
    "sanitize_structure",
    "sanitize_text_for_ingest",
]

_log = logging.getLogger("mind_mem.codepoint_sanitize")

# ---------------------------------------------------------------------------
# Strip pattern
# ---------------------------------------------------------------------------

# Ranges are the union of Unicode categories Cf (format), Co (private
# use), Cs (surrogates) and Cc-except-\t\n\r, generated from the
# Unicode 15.0 character database and widened to whole invisible blocks
# (e.g. the full U+E0000-U+E007F tag block including its unassigned
# slots, U+2060-U+206F including reserved U+2065) so unassigned (Cn)
# invisibles in those blocks are stripped too. A category cross-check
# test (tests/test_codepoint_sanitize.py) verifies the pattern covers
# every Cf/Co/Cs codepoint of the interpreter's own Unicode database.
_STRIP_PATTERN = re.compile(
    "["
    "\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f"  # Cc except \t \n \r
    "\u00ad"  # SOFT HYPHEN
    "\u0600-\u0605\u061c\u06dd\u070f\u0890\u0891\u08e2"  # Arabic format chars
    "\u180e"  # MONGOLIAN VOWEL SEPARATOR
    "\u200b-\u200f"  # ZWSP ZWNJ ZWJ LRM RLM
    "\u202a-\u202e"  # bidi embeddings/overrides
    "\u2060-\u206f"  # WORD JOINER, invisible operators, bidi isolates
    "\ud800-\udfff"  # surrogates (Cs)
    "\ue000-\uf8ff"  # Private Use Area (Co)
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\ufff9-\ufffb"  # interlinear annotation controls
    "\U000110bd\U000110cd"  # Kaithi number signs
    "\U00013430-\U0001343f"  # Egyptian hieroglyph format controls
    "\U0001bca0-\U0001bca3"  # shorthand format controls
    "\U0001d173-\U0001d17a"  # musical symbol format controls
    "\U000e0000-\U000e007f"  # tag characters (full block)
    "\U000f0000-\U000ffffd"  # Supplementary Private Use Area-A (Co)
    "\U00100000-\U0010fffd"  # Supplementary Private Use Area-B (Co)
    "]+"
)

_MAX_STRUCTURE_DEPTH = 64


# ---------------------------------------------------------------------------
# Pure sanitizers
# ---------------------------------------------------------------------------


def sanitize_codepoints(text: str) -> str:
    """Return *text* with invisible/format codepoints removed.

    Pure function: strips zero-width characters, Unicode tag characters,
    bidi controls, and other Cf/Co/Cs/Cc-category invisibles (module
    docstring has the full list). ``\\t``, ``\\n``, ``\\r`` and all
    visible content — including non-ASCII letters — pass through
    unchanged.
    """
    if not isinstance(text, str):
        raise TypeError(f"sanitize_codepoints expects str, got {type(text).__name__}")
    return _STRIP_PATTERN.sub("", text)


def sanitize_structure(value: Any, *, _depth: int = 0) -> Any:
    """Recursively sanitize every string in a JSON-shaped structure.

    Dict keys and values, list/tuple items, and bare strings are passed
    through :func:`sanitize_codepoints`; other scalar types (numbers,
    booleans, ``None``) are returned unchanged. Always builds new
    containers — the input is never mutated. Raises :class:`ValueError`
    beyond ``64`` nesting levels so a hostile deeply-nested payload
    cannot blow the stack.
    """
    if _depth > _MAX_STRUCTURE_DEPTH:
        raise ValueError(f"structure nesting exceeds {_MAX_STRUCTURE_DEPTH} levels")
    if isinstance(value, str):
        return sanitize_codepoints(value)
    if isinstance(value, dict):
        return {sanitize_structure(k, _depth=_depth + 1): sanitize_structure(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [sanitize_structure(v, _depth=_depth + 1) for v in value]
        return items if isinstance(value, list) else tuple(items)
    return value


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------

_ENV_FLAG = "MIND_MEM_SANITIZE_CODEPOINTS"
_FALSY = frozenset({"0", "false", "no", "off"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def is_sanitize_enabled(config: Mapping[str, Any] | None = None) -> bool:
    """Return whether ingest codepoint sanitization is enabled.

    Default is **ON**. Disable via ``mind-mem.json``::

        {"ingest": {"sanitize_codepoints": false}}

    The ``MIND_MEM_SANITIZE_CODEPOINTS`` environment variable, when set
    to a recognized truthy/falsy token, overrides the config either way.
    """
    env = os.environ.get(_ENV_FLAG, "").strip().lower()
    if env in _FALSY:
        return False
    if env in _TRUTHY:
        return True
    if config is None:
        return True
    ingest = config.get("ingest")
    if not isinstance(ingest, Mapping):
        return True
    return bool(ingest.get("sanitize_codepoints", True))


def sanitize_enabled_for_workspace(workspace: str) -> bool:
    """Read ``mind-mem.json`` in *workspace* and resolve the gate.

    Missing or unreadable config resolves to the default (enabled) —
    the gate fails closed on the security side.
    """
    config_path = os.path.join(os.path.abspath(workspace), "mind-mem.json")
    config: dict[str, Any] | None = None
    try:
        with open(config_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            config = loaded
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        config = None
    return is_sanitize_enabled(config)


def sanitize_text_for_ingest(text: str, workspace: str, *, source: str = "") -> str:
    """Gated convenience for ingestion call-sites.

    Applies :func:`sanitize_codepoints` when the workspace config
    enables it (the default) and logs a warning naming *source* when
    invisible codepoints were actually removed — silent stripping would
    hide evidence of an attempted injection from the operator.
    """
    if not sanitize_enabled_for_workspace(workspace):
        return text
    clean = sanitize_codepoints(text)
    removed = len(text) - len(clean)
    if removed:
        _log.warning(
            "invisible_codepoints_stripped",
            extra={"removed": removed, "source": source or "<unknown>"},
        )
    return clean

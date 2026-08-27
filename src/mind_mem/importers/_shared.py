# Copyright 2026 STARGA, Inc.
"""Helpers shared by every importer parser.

Extracted verbatim from :mod:`mind_mem.importers.parsers` when the
note-tree / transcript parsers landed, so both parser modules bound
metadata the same way instead of growing two copies. Behaviour is
unchanged — ``parsers`` re-exports these names.

Everything here is a pure function: same input, same output, no clock,
no filesystem, no network.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .records import ImportParseError

__all__ = [
    "MAX_METADATA_KEYS",
    "MAX_METADATA_VALUE_LEN",
    "clean_text",
    "first_str",
    "flatten_metadata",
    "merge_metadata",
    "require_mapping",
]

# Metadata is metadata, not content — bound every value so a hostile dump
# can't inflate a block. Mirrors block_provenance.MAX_PROVENANCE_VALUE_LEN.
MAX_METADATA_VALUE_LEN = 256
MAX_METADATA_KEYS = 24

_META_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def flatten_metadata(raw: Any) -> dict[str, str]:
    """Return *raw* as a sorted, bounded ``str -> str`` mapping.

    Non-mapping input yields ``{}``. Nested containers are rendered as
    comma-joined scalars; anything else is dropped. Keys that are not
    plain identifiers are dropped so they can never collide with a
    block field name after rendering.
    """
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, str] = {}
    for key in sorted(str(k) for k in raw.keys()):
        if len(out) >= MAX_METADATA_KEYS:
            break
        if not _META_KEY_RE.match(key):
            continue
        value = raw[key]
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (int, float, str)):
            text = str(value)
        elif isinstance(value, (list, tuple)):
            text = ",".join(str(v) for v in value if isinstance(v, (bool, int, float, str)))
        else:
            continue
        text = " ".join(text.split())[:MAX_METADATA_VALUE_LEN]
        if text:
            out[key] = text
    return out


def merge_metadata(*layers: Mapping[str, str]) -> dict[str, str]:
    """Merge already-flattened metadata layers, later layers winning.

    Returns a NEW sorted dict bounded by :data:`MAX_METADATA_KEYS` — the
    inputs are never mutated. Layers are filled highest-priority-first so
    the cap can never evict a structural key (``path``, ``name``, ...) in
    favour of an alphabetically-earlier foreign header field.
    """
    out: dict[str, str] = {}
    for layer in reversed(layers):
        for key in sorted(layer):
            if len(out) >= MAX_METADATA_KEYS:
                break
            if key in out or not _META_KEY_RE.match(key):
                continue
            text = " ".join(str(layer[key]).split())[:MAX_METADATA_VALUE_LEN]
            if text:
                out[key] = text
    return {k: out[k] for k in sorted(out)}


def clean_text(raw: Any) -> str:
    """Return *raw* as text, or ``""`` when it is not usable content."""
    if not isinstance(raw, str):
        return ""
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def first_str(source: Mapping[str, Any], keys: Sequence[str]) -> str:
    """First non-empty string value among *keys*, else ``""``."""
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def require_mapping(payload: Any, system: str) -> Mapping[str, Any]:
    """Assert *payload* is a JSON object, else raise a named parse error."""
    if not isinstance(payload, Mapping):
        raise ImportParseError(f"{system} dump must be a JSON object, got {type(payload).__name__}")
    return payload

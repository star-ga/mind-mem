# Copyright 2026 STARGA, Inc.
"""Deterministic symbol-definition probe — "is this name still defined here?".

A grep-class check, on purpose. :mod:`mind_mem.world_staleness` needs to
know whether a cited symbol still exists in a cited file without parsing
every language it might meet, without a compiler, and without ever
calling out to anything. Per-extension regexes over the file's text give
exactly that: same file plus same name always yields the same verdict.

Two probe strengths:

``definition``
    The extension is known (Python, Rust, TypeScript/JavaScript, Go).
    The name counts as live only when it appears in a *definition*
    position — ``def recall(``, ``struct Anchor``, ``export function x``.
    A name that survives only inside a comment or a call site is
    correctly reported dead.

``presence``
    The extension is unknown. The probe degrades to a whole-word search
    anywhere in the file. Weaker, and deliberately so: an unfamiliar
    language must never manufacture a false "the world moved" flag.

The probe strength travels with the result so a reader of ``scan()``
knows how much the verdict is worth.

Stdlib only. Reads the one file it is asked about; writes nothing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Final, Mapping

__all__ = [
    "PROBE_DEFINITION",
    "PROBE_PRESENCE",
    "SymbolProbeResult",
    "probe_strength_for",
    "probe_symbol",
]

PROBE_DEFINITION: Final = "definition"
PROBE_PRESENCE: Final = "presence"

#: Default cap on the file size the probe will read, in bytes. A file
#: over the cap is reported ``readable=False`` rather than being pulled
#: into memory — the caller turns that into "unverifiable", never
#: "stale".
DEFAULT_MAX_FILE_BYTES: Final = 4_000_000

#: Definition-position patterns per file extension. ``{name}`` is
#: replaced with the escaped symbol. Every pattern is applied with
#: ``re.MULTILINE``.
_DEFINITION_PATTERNS: Final[Mapping[str, tuple[str, ...]]] = {
    "py": (
        r"^[ \t]*(?:async[ \t]+)?def[ \t]+{name}[ \t]*\(",
        r"^[ \t]*class[ \t]+{name}[ \t]*[\(:]",
        r"^[ \t]*{name}[ \t]*(?::[^=\n]+)?=(?!=)",
    ),
    "pyi": (
        r"^[ \t]*(?:async[ \t]+)?def[ \t]+{name}[ \t]*\(",
        r"^[ \t]*class[ \t]+{name}[ \t]*[\(:]",
        r"^[ \t]*{name}[ \t]*(?::[^=\n]+)?=(?!=)",
    ),
    "rs": (
        r"\bfn[ \t]+{name}[ \t]*[(<]",
        r"\b(?:struct|enum|trait|union|type|mod|const|static)[ \t]+{name}\b",
        r"\bmacro_rules![ \t]*{name}\b",
    ),
    "ts": (
        r"\bfunction[ \t]+{name}\b",
        r"\bclass[ \t]+{name}\b",
        r"\b(?:const|let|var)[ \t]+{name}\b",
        r"\b(?:interface|type|enum|namespace)[ \t]+{name}\b",
    ),
    "go": (
        r"\bfunc[ \t]+(?:\([^)\n]*\)[ \t]*)?{name}[ \t]*[(\[]",
        r"\btype[ \t]+{name}\b",
        r"\b(?:const|var)[ \t]+{name}\b",
    ),
}

#: Extensions that reuse another extension's pattern set.
_EXTENSION_ALIASES: Final[Mapping[str, str]] = {
    "tsx": "ts",
    "js": "ts",
    "jsx": "ts",
    "mjs": "ts",
    "cjs": "ts",
}


@dataclass(frozen=True)
class SymbolProbeResult:
    """Outcome of one symbol probe."""

    found: bool
    strength: str
    readable: bool
    detail: str = ""


def _extension(path: str) -> str:
    return os.path.splitext(path)[1].lstrip(".").lower()


def probe_strength_for(path: str) -> str:
    """Return the probe strength that :func:`probe_symbol` would use."""
    ext = _EXTENSION_ALIASES.get(_extension(path), _extension(path))
    return PROBE_DEFINITION if ext in _DEFINITION_PATTERNS else PROBE_PRESENCE


def _patterns_for(path: str, name: str) -> tuple[str, tuple[re.Pattern[str], ...]]:
    ext = _EXTENSION_ALIASES.get(_extension(path), _extension(path))
    escaped = re.escape(name)
    raw = _DEFINITION_PATTERNS.get(ext)
    if raw is None:
        return PROBE_PRESENCE, (re.compile(r"\b" + escaped + r"\b"),)
    compiled = tuple(re.compile(p.format(name=escaped), re.MULTILINE) for p in raw)
    return PROBE_DEFINITION, compiled


def probe_symbol(
    file_path: str,
    name: str,
    *,
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> SymbolProbeResult:
    """Report whether *name* is still defined in *file_path*.

    Args:
        file_path: Absolute path to an existing regular file.
        name:      Identifier-shaped symbol name.
        max_bytes: Refuse to read a file larger than this; the result
                   comes back ``readable=False`` so the caller can
                   report "unverifiable" instead of guessing.

    Returns:
        A :class:`SymbolProbeResult`. Unreadable files always come back
        ``found=False, readable=False`` — never a staleness verdict.
    """
    if not name:
        raise ValueError("name must be non-empty")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    try:
        size = os.path.getsize(file_path)
    except OSError as exc:
        return SymbolProbeResult(False, probe_strength_for(file_path), False, f"stat failed: {exc.strerror or exc}")
    if size > max_bytes:
        return SymbolProbeResult(
            False,
            probe_strength_for(file_path),
            False,
            f"file exceeds probe cap ({size} > {max_bytes} bytes)",
        )

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        return SymbolProbeResult(False, probe_strength_for(file_path), False, f"read failed: {exc.strerror or exc}")

    strength, patterns = _patterns_for(file_path, name)
    for pattern in patterns:
        if pattern.search(text):
            return SymbolProbeResult(True, strength, True)
    return SymbolProbeResult(False, strength, True)

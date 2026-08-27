# Copyright 2026 STARGA, Inc.
"""External-anchor extraction — the citations a block makes to the world.

``lineage_staleness`` propagates staleness *within* the corpus: a block
goes stale because another block contradicted it. Nothing in that path
notices that the **world outside the corpus** moved — the file a
decision cites was deleted, the function it names was renamed, the
commit it pins was left behind.

This module is the first half of that check: it turns a block into an
immutable tuple of :class:`Anchor` records — the external things the
block claims exist. :mod:`mind_mem.world_staleness` is the second half;
it verifies those anchors against the local filesystem and git, and
surfaces the dead ones through ``scan()``.

Two citation surfaces, both deterministic:

``Anchors:`` field (explicit, preferred)
    A list field on the block. Each entry is either scheme-qualified
    (``path:``, ``symbol:``, ``git:``) or a bare repo-relative path::

        Anchors:
        - path:src/mind_mem/recall.py
        - symbol:src/mind_mem/recall.py#recall
        - git:a1b2c3d
        - docs/ARCHITECTURE.md

``Statement`` / ``Rationale`` / … prose (inline, conservative)
    A token is an inline anchor only when it has at least one ``/``
    **and** a recognised source-file extension — ``src/mind_mem/lint.py``
    is an anchor, ``the lint module`` is not. Prose that cites nothing
    yields nothing, which is what keeps the false-positive rate at zero
    on ordinary blocks. ``Sources:`` is deliberately **not** scanned: it
    is corpus provenance with its own semantics, and on a non-Markdown
    backend those paths need not exist on disk.

Path anchors are always repo-relative: absolute paths and ``..``
segments are rejected at the boundary (reported as ``invalid``, never
touched on disk), so anchor extraction can never be steered into
probing the filesystem outside a configured root.

Stdlib only. Reads nothing — extraction is pure text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

__all__ = [
    "Anchor",
    "ANCHOR_KINDS",
    "INLINE_FIELDS",
    "KIND_GIT_REF",
    "KIND_INVALID",
    "KIND_PATH",
    "KIND_SYMBOL",
    "ORIGIN_FIELD",
    "ORIGIN_INLINE",
    "extract_anchors",
    "parse_anchor_entry",
]

KIND_PATH: Final = "path"
KIND_SYMBOL: Final = "symbol"
KIND_GIT_REF: Final = "git_ref"
KIND_INVALID: Final = "invalid"

#: Every anchor kind this module can emit, in report order.
ANCHOR_KINDS: Final[tuple[str, ...]] = (KIND_PATH, KIND_SYMBOL, KIND_GIT_REF, KIND_INVALID)

ORIGIN_FIELD: Final = "anchors_field"
ORIGIN_INLINE: Final = "inline"

#: The block field that carries explicit anchors.
ANCHORS_FIELD: Final = "Anchors"

#: Prose fields scanned for inline citations, in this fixed order.
#: ``Sources`` is excluded on purpose (see the module docstring).
INLINE_FIELDS: Final[tuple[str, ...]] = (
    "Title",
    "Statement",
    "Rationale",
    "Description",
    "Content",
    "Notes",
)

#: Source-file extensions an inline token must carry to count as a path
#: citation. Conservative by design: an unknown extension is not an
#: anchor, so it can never produce a false "the world moved" flag.
_INLINE_EXTENSIONS: Final[tuple[str, ...]] = (
    "py",
    "pyi",
    "rs",
    "ts",
    "tsx",
    "js",
    "jsx",
    "go",
    "c",
    "h",
    "cc",
    "cpp",
    "hpp",
    "java",
    "rb",
    "sh",
    "mind",
    "toml",
    "yaml",
    "yml",
    "json",
    "md",
    "sql",
    "proto",
    "cfg",
    "ini",
)

_INLINE_PATH_RE: Final = re.compile(
    r"(?<![\w/.\-])"
    r"((?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+\.(?:" + "|".join(_INLINE_EXTENSIONS) + r"))"
    r"(?:(?:::|#)([A-Za-z_][A-Za-z0-9_]*))?"
)

#: A symbol name we are willing to grep for. Deliberately identifier-shaped.
_SYMBOL_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Characters git accepts in a ref / rev expression. A leading ``-`` is
#: rejected separately so an anchor can never be read as a git option.
_GIT_REF_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/^~{}\-]*$")

_MAX_ANCHOR_LEN: Final = 400


@dataclass(frozen=True)
class Anchor:
    """One external thing a block claims exists.

    Attributes:
        kind:   One of :data:`ANCHOR_KINDS`. ``invalid`` means the entry
                was cited explicitly but is not a well-formed anchor —
                a corpus defect, not world drift.
        raw:    The citation exactly as written in the block.
        target: Repo-relative path (``path`` / ``symbol``) or git rev
                expression (``git_ref``). Empty for ``invalid``.
        symbol: Symbol name for ``symbol`` anchors, else ``""``.
        origin: :data:`ORIGIN_FIELD` or :data:`ORIGIN_INLINE`.
        reason: Why an ``invalid`` anchor was rejected, else ``""``.
    """

    kind: str
    raw: str
    target: str = ""
    symbol: str = ""
    origin: str = ORIGIN_FIELD
    reason: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity used for de-duplication within a block."""
        return (self.kind, self.target or self.raw, self.symbol)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "raw": self.raw,
            "target": self.target,
            "origin": self.origin,
        }
        if self.symbol:
            out["symbol"] = self.symbol
        if self.reason:
            out["reason"] = self.reason
        return out


def _reject(raw: str, reason: str, origin: str) -> Anchor:
    return Anchor(kind=KIND_INVALID, raw=raw, origin=origin, reason=reason)


def _path_problem(path: str) -> str | None:
    """Return why *path* is unusable as a repo-relative anchor, else None."""
    if not path:
        return "empty path"
    if len(path) > _MAX_ANCHOR_LEN:
        return "path too long"
    if path.startswith("/") or path.startswith("\\"):
        return "absolute path (anchors are repo-relative)"
    if re.match(r"^[A-Za-z]:[\\/]", path):
        return "absolute path (anchors are repo-relative)"
    normalised = path.replace("\\", "/")
    if any(seg == ".." for seg in normalised.split("/")):
        return "parent-directory traversal"
    if "\x00" in path:
        return "null byte in path"
    return None


def parse_anchor_entry(entry: str, *, origin: str = ORIGIN_FIELD) -> Anchor | None:
    """Parse one ``Anchors:`` entry into an :class:`Anchor`.

    Returns ``None`` for an entry that is blank or a comment. A
    non-blank entry that cannot be understood comes back as an
    :data:`KIND_INVALID` anchor carrying the rejection ``reason`` —
    a typo in the corpus stays visible instead of being silently
    dropped.
    """
    if not isinstance(entry, str):
        return None
    raw = entry.strip().strip("`").strip()
    if not raw or raw.startswith("#"):
        return None
    if len(raw) > _MAX_ANCHOR_LEN:
        return _reject(entry.strip()[:_MAX_ANCHOR_LEN], "anchor too long", origin)

    scheme, _, rest = raw.partition(":")
    scheme_l = scheme.strip().lower()
    body = rest.strip()

    # deferred: package-version anchors ("requests>=2.31") are not verified.
    # An honest check needs the installed distribution or a lockfile, and
    # reading the ambient environment is not deterministic across machines.
    # Released versions are already coverable today as ``git:v1.2.3`` (a tag
    # is a ref). Upgrade path: a ``version:`` scheme resolved against a
    # lockfile committed inside a configured root (uv.lock / poetry.lock /
    # Cargo.lock) — still local, still deterministic.
    if scheme_l in {"git", "ref", "tag", "commit"} and body:
        if body.startswith("-") or not _GIT_REF_RE.match(body) or ".." in body:
            return _reject(raw, "malformed git ref", origin)
        return Anchor(kind=KIND_GIT_REF, raw=raw, target=body, origin=origin)

    if scheme_l == "symbol" and body:
        return _parse_symbol_body(raw, body, origin)

    if scheme_l == "path" and body:
        return _parse_path_body(raw, body, origin)

    if scheme_l in {"git", "ref", "tag", "commit", "symbol", "path"}:
        return _reject(raw, f"empty {scheme_l} anchor", origin)

    # Bare entry: a path, optionally with a #symbol / ::symbol suffix.
    if "#" in raw or "::" in raw:
        return _parse_symbol_body(raw, raw, origin)
    return _parse_path_body(raw, raw, origin)


def _split_symbol(body: str) -> tuple[str, str]:
    """Split ``path#name`` / ``path::name`` into ``(path, name)``."""
    if "::" in body:
        path, _, name = body.rpartition("::")
        return path.strip(), name.strip()
    path, _, name = body.partition("#")
    return path.strip(), name.strip()


def _parse_symbol_body(raw: str, body: str, origin: str) -> Anchor:
    path, name = _split_symbol(body)
    problem = _path_problem(path)
    if problem is not None:
        return _reject(raw, problem, origin)
    if not _SYMBOL_RE.match(name):
        return _reject(raw, "malformed symbol name", origin)
    return Anchor(kind=KIND_SYMBOL, raw=raw, target=path, symbol=name, origin=origin)


def _parse_path_body(raw: str, body: str, origin: str) -> Anchor:
    problem = _path_problem(body)
    if problem is not None:
        return _reject(raw, problem, origin)
    return Anchor(kind=KIND_PATH, raw=raw, target=body, origin=origin)


def _field_entries(value: Any) -> list[str]:
    """Normalise a block field into a list of raw string entries."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [v for v in value if isinstance(v, str)]
    return []


def _inline_text(block: Mapping[str, Any]) -> str:
    """Concatenate the prose fields scanned for inline citations."""
    parts: list[str] = []
    for field in INLINE_FIELDS:
        value = block.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            parts.extend(v for v in value if isinstance(v, str))
    return "\n".join(parts)


def extract_anchors(block: Mapping[str, Any], *, inline: bool = True) -> tuple[Anchor, ...]:
    """Return every external anchor cited by *block*, de-duplicated.

    Order is deterministic: explicit ``Anchors:`` entries in field
    order, then inline citations in text order. The first occurrence of
    a ``(kind, target, symbol)`` triple wins, so an explicit anchor is
    never shadowed by an inline restatement of the same thing.

    Args:
        block:  A parsed block dict (``block_parser`` / store shape).
        inline: Scan the prose fields in :data:`INLINE_FIELDS` as well
                as the ``Anchors:`` field. ``False`` restricts the
                checker to explicitly-declared anchors only.

    Returns:
        An immutable tuple. Empty for a block that cites nothing —
        the common case, and the reason ordinary blocks can never be
        flagged by the world-staleness checker.
    """
    if not isinstance(block, Mapping):
        raise TypeError("block must be a mapping")

    found: list[Anchor] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(anchor: Anchor | None) -> None:
        if anchor is None:
            return
        if anchor.key in seen:
            return
        seen.add(anchor.key)
        found.append(anchor)

    for entry in _field_entries(block.get(ANCHORS_FIELD)):
        _add(parse_anchor_entry(entry, origin=ORIGIN_FIELD))

    if inline:
        for match in _INLINE_PATH_RE.finditer(_inline_text(block)):
            path, symbol = match.group(1), match.group(2)
            raw = match.group(0)
            if symbol:
                _add(_parse_symbol_body(raw, f"{path}#{symbol}", ORIGIN_INLINE))
            else:
                _add(_parse_path_body(raw, path, ORIGIN_INLINE))

    return tuple(found)

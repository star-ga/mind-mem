# Copyright 2026 STARGA, Inc.
"""OKF bundle as an import source — foreign knowledge, governed on arrival.

:mod:`mind_mem.core_export` can already read an Open Knowledge Format
bundle off disk (:func:`~mind_mem.core_export.import_okf_bundle`). What
it could not do is *land* one: the function returned block dicts and
nothing called it, so the only way to get an OKF bundle into a corpus
was to write those dicts somewhere yourself — outside the governance
gate, unquarantined, immediately recallable. That is exactly the write
path this package refuses to have.

This module makes the OKF bundle an ordinary **import source** instead,
so it inherits the whole bulk-ingest bargain documented in
:mod:`mind_mem.importers.quarantine` for free:

* every concept lands as an ``IMP-okf-…`` block with
  ``Status: quarantined`` + ``IngestTier: external-ingest``;
* recall withholds it until a governed release proposal is approved
  (``propose_import_release`` -> ``approve_apply``);
* the run is appended to the tamper-evident audit chain.

A foreign bundle therefore cannot be recalled on the strength of its own
say-so — including the trust fields OKF lets a producer self-declare.
``import_okf_bundle`` already parks ``verified`` / ``generated`` /
``status`` / ``receipt`` under ``OkfClaim*`` keys as untrusted claims;
this module keeps them there, as metadata on the imported block, and
never lets one reach a mind-mem governance field.

Gated: reachable only when the ``core_export`` v4 flag is ON (see
:func:`mind_mem.importers.resolve_system`).
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from ._shared import clean_text as _clean_text
from ._shared import flatten_metadata as _flatten_metadata
from .fs_source import (
    DEFAULT_EXCLUDED_DIRS,
    MAX_NOTE_BYTES,
    MAX_TREE_BYTES,
    MAX_TREE_FILES,
    NOTE_EXTENSIONS,
)
from .records import ImportParseError, ImportRecord

__all__ = ["OKF_SYSTEM", "load_okf_bundle", "parse_okf"]

#: Source-system slug for this importer.
OKF_SYSTEM = "okf"

#: OKF frontmatter mapped by ``import_okf_bundle`` onto a capitalised block
#: field. ``Statement`` carries the record text; the rest ride along as
#: metadata so nothing the bundle declared is dropped on the floor.
_TEXT_FIELDS = ("Statement", "Title")

#: Fields deliberately NOT copied into metadata: ``_id`` becomes the record's
#: ``external_id``, ``Date`` becomes ``created_at``, and the two text fields
#: are already the record text.
_METADATA_SKIP = frozenset({"_id", "Date", "Statement"})


def _bounds_check(root: str) -> None:
    """Refuse an OKF bundle that is too large to read safely.

    :func:`~mind_mem.core_export.import_okf_bundle` ``rglob``s every
    ``*.md`` under the root and ``read_text``s each one with no cap, so
    the bound has to be applied *before* it runs. The limits are
    :mod:`~mind_mem.importers.fs_source`'s own — one note-tree reader
    should not be an order of magnitude more permissive than the other
    just because a different function does the reading.

    Only ``stat`` is called here; the contents are read exactly once, by
    ``import_okf_bundle``.
    """
    files = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in DEFAULT_EXCLUDED_DIRS and not d.startswith("."))
        for filename in sorted(filenames):
            if not filename.endswith(NOTE_EXTENSIONS):
                continue
            full = os.path.join(dirpath, filename)
            if not os.path.isfile(full):
                continue
            size = os.path.getsize(full)
            if size > MAX_NOTE_BYTES:
                raise ImportParseError(f"OKF concept file too large: {os.path.relpath(full, root)} ({size} bytes, max {MAX_NOTE_BYTES})")
            files += 1
            total += size
            if files > MAX_TREE_FILES:
                raise ImportParseError(f"OKF bundle holds more than {MAX_TREE_FILES} concept files: {root}")
            if total > MAX_TREE_BYTES:
                raise ImportParseError(f"OKF bundle exceeds {MAX_TREE_BYTES} bytes: {root}")


def load_okf_bundle(path: str) -> tuple[dict[str, Any], ...]:
    """Read the OKF bundle directory at *path* into raw block dicts.

    Delegates the format work to
    :func:`mind_mem.core_export.import_okf_bundle` — this is the wiring,
    not a second parser — after bounding the tree.

    Raises:
        ImportParseError: the path is missing, is not a directory, or the
            bundle breaks a size bound.
    """
    if not isinstance(path, str) or not path.strip():
        raise ImportParseError("OKF bundle path must be a non-empty string")
    if not os.path.exists(path):
        raise ImportParseError(f"OKF bundle not found: {path}")
    if not os.path.isdir(path):
        raise ImportParseError(f"OKF bundle path is not a directory: {path} (this importer reads a bundle directory)")
    root = os.path.realpath(path)
    _bounds_check(root)

    from ..core_export import import_okf_bundle

    try:
        blocks = import_okf_bundle(root)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ImportParseError(f"OKF bundle is unreadable ({exc}): {path}") from exc
    return tuple(blocks)


def _scalarize(value: Any) -> Any:
    """Render a nested mapping as a scalar so metadata flattening keeps it.

    ``_shared.flatten_metadata`` drops any value that is not a scalar or a
    flat sequence. OKF's ``generated`` claim is a ``{by, at}`` mapping, and
    silently dropping the producer's self-declared provenance would lose the
    very thing the ``OkfClaim*`` keys exist to preserve — a claim we do not
    honour still has to be visible.
    """
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={value[k]}" for k in sorted(str(k) for k in value))
    return value


def parse_okf(payload: Any) -> tuple[ImportRecord, ...]:
    """Project OKF blocks onto :class:`ImportRecord` values.

    The block dicts :func:`load_okf_bundle` returns are already in
    mind-mem's field vocabulary (``Title`` / ``Statement`` / ``Tags`` /
    ``OkfClaim*``); this turns each into the neutral record shape the
    import engine writes. A concept with no text at all is dropped
    rather than written as an empty block.

    Ordering is the bundle's own sorted-path order, so the derived block
    ids — and therefore the import — stay a pure function of the bundle.
    """
    if not isinstance(payload, (list, tuple)):
        raise ImportParseError(f"OKF payload must be a sequence of concepts, got {type(payload).__name__}")

    records: list[ImportRecord] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            raise ImportParseError(f"OKF concept must be a mapping, got {type(entry).__name__}")
        text = ""
        for field in _TEXT_FIELDS:
            text = _clean_text(entry.get(field))
            if text:
                break
        if not text:
            continue
        external_id = _clean_text(entry.get("_id"))
        if not external_id:
            continue
        metadata = _flatten_metadata({k: _scalarize(v) for k, v in entry.items() if k not in _METADATA_SKIP})
        records.append(
            ImportRecord(
                system=OKF_SYSTEM,
                external_id=external_id,
                text=text,
                metadata=metadata,
                created_at=_clean_text(entry.get("Date")),
            )
        )
    return tuple(records)

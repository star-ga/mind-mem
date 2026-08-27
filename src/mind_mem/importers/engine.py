# Copyright 2026 STARGA, Inc.
"""Migration-import engine — foreign dump file to ``IMP-`` corpus blocks.

Design notes
------------
*Idempotency* is structural, not bolted on: a block's id is derived from
``(system, external_id, text)`` via :func:`mind_mem.capture.content_hash`,
so re-importing the same dump resolves to the same ids, the engine sees
them already present, and writes nothing. Nothing new is deduplicated
here — the optional near-duplicate collapse defers to the existing
:mod:`mind_mem.dedup` cosine layer.

*Recall* works because ``IMP-`` routes through ``_BLOCK_PREFIX_MAP`` to
``memory/IMPORTED.md``, which is registered in ``CORPUS_FILES`` — the
same wiring ``INBOX-`` and ``MSG-`` blocks use.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

from ..block_provenance import attach_provenance
from ..capture import content_hash
from ..observability import get_logger
from .records import ImporterError, ImportParseError, ImportRecord, ImportResult

_log = get_logger("importers")

__all__ = [
    "IMPORT_BLOCK_PREFIX",
    "IMPORTED_CORPUS_FILE",
    "IMPORT_BLOCK_TYPE",
    "MAX_DUMP_BYTES",
    "build_import_block",
    "load_dump",
    "provenance_token",
    "run_import",
]

IMPORT_BLOCK_PREFIX = "IMP"
IMPORTED_CORPUS_FILE = "memory/IMPORTED.md"
IMPORT_BLOCK_TYPE = "ImportedMemory"

# A dump is read fully into memory before parsing, so it is bounded.
# 64 MiB comfortably covers a realistic single-agent export while
# refusing a file that would exhaust a small runner.
MAX_DUMP_BYTES = 64 * 1024 * 1024


def provenance_token(system: str) -> str:
    """Canonical provenance marker every imported block carries."""
    return f"imported:{system}"


# ---------------------------------------------------------------------------
# Dump loading (boundary validation)
# ---------------------------------------------------------------------------


def load_dump(path: str) -> Any:
    """Read and JSON-decode the dump at *path*.

    Raises:
        ImportParseError: the path is missing, not a regular file, larger
            than :data:`MAX_DUMP_BYTES`, or not valid UTF-8 JSON.
    """
    if not isinstance(path, str) or not path.strip():
        raise ImportParseError("dump path must be a non-empty string")
    if not os.path.exists(path):
        raise ImportParseError(f"dump file not found: {path}")
    if not os.path.isfile(path):
        raise ImportParseError(f"dump path is not a regular file: {path}")
    size = os.path.getsize(path)
    if size > MAX_DUMP_BYTES:
        raise ImportParseError(f"dump file too large: {size} bytes (max {MAX_DUMP_BYTES})")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except UnicodeDecodeError as exc:
        raise ImportParseError(f"dump file is not valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ImportParseError(f"dump file is not valid JSON ({exc.msg} at line {exc.lineno}): {path}") from exc
    except OSError as exc:
        raise ImportParseError(f"cannot read dump file {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------


def _as_block_value(text: str) -> str:
    """Render *text* so the Markdown block parser round-trips it.

    ``block_parser`` only re-attaches a continuation line when it is
    indented by two spaces and does not start a list item, so every line
    after the first is indented and a leading ``-`` bullet is rewritten
    to ``*``. Indentation also guarantees no continuation line can be
    read back as a ``Key: value`` field or an ``[ID]`` block header.
    """
    lines = text.split("\n")
    out = [lines[0].strip()]
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            stripped = "*" + stripped[1:]
        out.append("  " + stripped)
    return "\n".join(out)


def _record_digest(record: ImportRecord) -> str:
    """Stable content digest for *record* (reuses ``capture.content_hash``)."""
    return content_hash("\x1f".join((record.system, record.external_id, record.text)))


def block_id_for(record: ImportRecord) -> str:
    """Deterministic ``IMP-<system>-<digest>`` id for *record*."""
    return f"{IMPORT_BLOCK_PREFIX}-{record.system}-{_record_digest(record)}"


def _iso_date(raw: str) -> str:
    """``YYYY-MM-DD`` prefix of an ISO-8601 timestamp, else ``""``."""
    candidate = raw[:10]
    if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
        if candidate.replace("-", "").isdigit():
            return candidate
    return ""


def build_import_block(record: ImportRecord) -> dict[str, Any]:
    """Build the immutable ``IMP-`` block dict for *record*.

    Every block carries the ``imported:<system>`` provenance token twice:
    as the ``Source`` field (visible to recall + BM25) and as the
    ``ToolId`` provenance field (structured, read back by
    :func:`mind_mem.block_provenance.extract_provenance`).
    """
    token = provenance_token(record.system)
    block: dict[str, Any] = {
        "_id": block_id_for(record),
        "Type": IMPORT_BLOCK_TYPE,
        "Statement": _as_block_value(record.text),
        "Status": "active",
        "Source": token,
        "ExternalId": record.external_id,
        "ContentHash": _record_digest(record),
    }
    date = _iso_date(record.created_at)
    if date:
        block["Date"] = date
    if record.created_at:
        block["Timestamp"] = record.created_at
    if record.metadata:
        block["Metadata"] = "; ".join(f"{k}={v}" for k, v in sorted(record.metadata.items()))
    return attach_provenance(
        block,
        actor_id=record.system,
        actor_role="importer",
        tool_id=token,
        purpose="migration-import",
    )


# ---------------------------------------------------------------------------
# Existing-block lookup
# ---------------------------------------------------------------------------


def _existing_import_ids(workspace: str) -> set[str]:
    """Ids already present in ``memory/IMPORTED.md`` (empty when absent).

    The Markdown store is the default backend and routes ``IMP-`` blocks
    to a file outside ``CORPUS_DIRS``, so ``store.get_by_id`` cannot see
    them; the canonical file is read directly instead. On any other
    backend ``write_block`` is an upsert on the block id, so a re-import
    still converges to the same block set — only the reported
    ``skipped_existing`` count degrades to 0.
    """
    path = os.path.join(workspace, IMPORTED_CORPUS_FILE)
    if not os.path.isfile(path):
        return set()
    from ..block_parser import parse_file

    try:
        blocks = parse_file(path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        _log.warning("import_existing_scan_failed", extra={"file": IMPORTED_CORPUS_FILE, "error": str(exc)})
        return set()
    return {str(b.get("_id")) for b in blocks if b.get("_id")}


def _collapse_near_duplicates(records: Iterable[ImportRecord], threshold: float) -> tuple[ImportRecord, ...]:
    """Opt-in near-duplicate collapse via the existing dedup cosine layer."""
    from ..dedup import layer_cosine_dedup

    ordered = list(records)
    shaped = [{"content": r.text, "_index": i} for i, r in enumerate(ordered)]
    kept = layer_cosine_dedup(shaped, threshold=threshold)
    keep_indices = {int(item["_index"]) for item in kept}
    return tuple(r for i, r in enumerate(ordered) if i in keep_indices)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_import(
    workspace: str,
    system: str,
    path: str,
    *,
    dedup_near: bool = False,
    dedup_threshold: float = 0.85,
    dry_run: bool = False,
) -> ImportResult:
    """Import the dump at *path* into *workspace*.

    Args:
        workspace: mind-mem workspace root.
        system: Source system slug; must be file-based and supported.
        path: Path to the JSON dump.
        dedup_near: Opt-in (default OFF) near-duplicate collapse across
            the incoming batch, using ``dedup.layer_cosine_dedup``. With
            the default the import is a pure function of the dump.
        dedup_threshold: Cosine threshold for ``dedup_near``.
        dry_run: Parse + plan without writing any block.

    Raises:
        UnsupportedSystemError: *system* has no file-based importer.
        ImportParseError: the dump is unreadable or malformed.
    """
    from . import resolve_system

    resolved = resolve_system(system)
    if not isinstance(workspace, str) or not workspace.strip():
        raise ImporterError("workspace must be a non-empty path")

    from .parsers import parse_payload

    records = tuple(_sanitized(r, workspace) for r in parse_payload(resolved, load_dump(path)))
    parsed = len(records)

    skipped_near = 0
    if dedup_near and records:
        kept = _collapse_near_duplicates(records, dedup_threshold)
        skipped_near = parsed - len(kept)
        records = kept

    existing = _existing_import_ids(workspace)
    written: list[str] = []
    skipped_existing = 0

    store = None
    transform_hash = ""
    if not dry_run and records:
        from ..pipeline_hash import current_pipeline_hash
        from ..storage import get_block_store

        store = get_block_store(workspace)
        # Computed once per run (not per block) — the pipeline hash is a
        # property of the workspace config, identical for every block.
        transform_hash = str(current_pipeline_hash(workspace))

    planned: set[str] = set()
    for record in records:
        block_id = block_id_for(record)
        if block_id in existing or block_id in planned:
            skipped_existing += 1
            continue
        planned.add(block_id)
        if dry_run or store is None:
            written.append(block_id)
            continue
        block = build_import_block(record)
        if transform_hash:
            block = {**block, "TransformHash": transform_hash}
        written.append(str(store.write_block(block)))

    result = ImportResult(
        system=resolved,
        source_path=os.path.abspath(path),
        parsed=parsed,
        imported=len(written),
        skipped_existing=skipped_existing,
        skipped_near_duplicate=skipped_near,
        block_ids=tuple(written),
        dry_run=dry_run,
    )
    _log.info("import_completed", extra=result.as_dict())
    return result


def _sanitized(record: ImportRecord, workspace: str) -> ImportRecord:
    """Strip invisible codepoints from record text (prompt-injection channel).

    Config-gated + default ON via ``ingest.sanitize_codepoints``, exactly
    as :mod:`mind_mem.inbox` does for folder ingestion. Returns a NEW
    record; the input is never mutated.
    """
    from ..codepoint_sanitize import sanitize_text_for_ingest

    clean = sanitize_text_for_ingest(record.text, workspace, source=f"import:{record.system}:{record.external_id}")
    if clean == record.text:
        return record
    return ImportRecord(
        system=record.system,
        external_id=record.external_id,
        text=clean,
        metadata=record.metadata,
        created_at=record.created_at,
    )

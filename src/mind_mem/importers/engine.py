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

*Recall* routing works because ``IMP-`` maps through
``_BLOCK_PREFIX_MAP`` to ``memory/IMPORTED.md``, which is registered in
``CORPUS_FILES`` — the same wiring ``INBOX-`` and ``MSG-`` blocks use.
Routing is not admission, though: every block this engine writes lands
**quarantined**, so recall filters it out until a governance proposal
releases it. See :mod:`mind_mem.importers.quarantine` for why bulk
ingest is one chained, ungated write plus one governed release rather
than one proposal per block.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any, Iterable

from ..block_provenance import attach_provenance
from ..capture import content_hash
from ..enums import IngestTier
from ..observability import get_logger
from .quarantine import (
    BATCH_FIELD,
    QUARANTINE_STATUS,
    QUARANTINE_TIER,
    TIER_FIELD,
    batch_id_for,
    record_import_in_chain,
)
from .records import ImporterError, ImportParseError, ImportRecord, ImportResult

_log = get_logger("importers")

__all__ = [
    "DIRECTORY_SYSTEMS",
    "IMPORT_BLOCK_PREFIX",
    "IMPORTED_CORPUS_FILE",
    "IMPORT_BLOCK_TYPE",
    "MAX_DUMP_BYTES",
    "build_import_block",
    "load_dump",
    "load_source",
    "provenance_token",
    "run_import",
]

# Systems whose source is a DIRECTORY of markdown notes rather than a
# single JSON dump. Everything else goes through :func:`load_dump`.
DIRECTORY_SYSTEMS: frozenset[str] = frozenset({"agentmem", "markdown"})

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


def load_source(system: str, path: str) -> Any:
    """Load the import source for *system* — a JSON dump or a note tree.

    One dispatch point so :func:`run_import` stays source-shape agnostic:
    the directory formats get a bounded, deterministic note walk, every
    other format gets the JSON dump reader.

    Raises:
        ImportParseError: the source is missing, the wrong kind of path,
            oversized, or malformed for its shape.
    """
    if system in DIRECTORY_SYSTEMS:
        from .fs_source import load_note_tree

        return load_note_tree(path)
    return load_dump(path)


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------


# Every codepoint ``str.splitlines`` treats as a line break, plus CR.
# The corpus is written as text and read back through Python's universal
# newline translation, so a bare CR in a field value becomes a real line
# on the way back in — CR is as dangerous as LF here, and the exotic
# breaks are neutralised for any consumer that splits with ``splitlines``.
_LINE_BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_BREAK_RUN_RE = re.compile(f"[{_LINE_BREAKS}]+")


def _as_field_value(text: str) -> str:
    """Render *text* as ONE physical line of a Markdown block.

    A single-line field is emitted verbatim by
    ``block_store._render_block``, which neutralises only ``"\n["``. Any
    other line break in the value therefore lands in the corpus as a
    real, unindented line, and ``block_parser`` reads a line shaped like
    ``Key: value`` back as a *field* — last one wins. Foreign dump
    content reaching such a field could rewrite ``Status`` and walk
    straight out of quarantine, so every break is flattened to a space
    before the value is ever rendered.

    Mirrors ``block_provenance.sanitize_provenance_value``, which already
    does this for the provenance fields, and ``_shared.flatten_metadata``,
    which already does it for parsed metadata values.
    """
    return " ".join(_BREAK_RUN_RE.sub(" ", text).split())


def _as_block_value(text: str) -> str:
    """Render *text* as a multi-line block value the parser reads back safely.

    ``block_parser`` only re-attaches a continuation line when it is
    indented by two spaces and does not start a list item, so every line
    after the first is indented and a leading ``-`` bullet is rewritten
    to ``*``. Indentation also guarantees no continuation line can be
    read back as a ``Key: value`` field or an ``[ID]`` block header.

    This is deliberately **not** a round trip, and the format is what
    forbids one. Three transforms are lossy and none is reversible:

    * blank lines are dropped — the block format cannot represent one
      inside a value at all (a blank line ends the block);
    * a leading ``-`` becomes ``*`` — a ``-`` would be re-read as a new
      list item rather than a continuation;
    * per-line indentation is stripped on the way in (``.strip()``) and
      again on the way out (``block_parser`` removes the two-space
      continuation prefix), so an indented code block comes back flush
      left.

    So ``parse_file(render(x))`` yields a normalised form of *x*, not
    *x*. Nothing recomputes a digest from the rendered value —
    ``ContentHash`` is :func:`_record_digest` over the *record*
    (system + external_id + text), taken before this function runs — so
    the losses do not break id stability or idempotent re-import; they
    cost the corpus copy its paragraph structure and its code-block
    indentation.
    """
    # Split on every break the round-trip can resurrect, not just "\n".
    lines = _BREAK_RUN_RE.split(text)
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


def build_import_block(record: ImportRecord, *, batch: str = "") -> dict[str, Any]:
    """Build the immutable ``IMP-`` block dict for *record*.

    Every block carries the ``imported:<system>`` provenance token twice:
    as the ``Source`` field (visible to recall + BM25) and as the
    ``ToolId`` provenance field (structured, read back by
    :func:`mind_mem.block_provenance.extract_provenance`).

    It also arrives **quarantined**: ``Status`` is
    :data:`~mind_mem.importers.quarantine.QUARANTINE_STATUS` and
    ``IngestTier`` is the existing ``external-ingest`` provenance class,
    never ``active``. External content is inert until a governance
    proposal releases it — see :mod:`mind_mem.importers.quarantine`.

    Args:
        record: The parsed foreign memory unit.
        batch: Optional import-batch id, stamped as ``ImportBatch`` so a
            block can be traced back to the run — and to that run's
            audit-chain entry — that wrote it. Omitted when empty.
    """
    token = provenance_token(record.system)
    # Every value below that carries dump content is rendered through
    # _as_field_value: these are single-line fields, and an un-flattened
    # break in one of them injects a sibling `Key: value` line that the
    # parser reads back as a real field (last one wins) — including a
    # second `Status:` that would silently lift the quarantine.
    block: dict[str, Any] = {
        "_id": block_id_for(record),
        "Type": IMPORT_BLOCK_TYPE,
        "Statement": _as_block_value(record.text),
        "Status": QUARANTINE_STATUS,
        TIER_FIELD: QUARANTINE_TIER,
        "Source": _as_field_value(token),
        "ExternalId": _as_field_value(record.external_id),
        "ContentHash": _record_digest(record),
    }
    if batch:
        block[BATCH_FIELD] = batch
    date = _iso_date(record.created_at)
    if date:
        block["Date"] = date
    if record.created_at:
        block["Timestamp"] = _as_field_value(record.created_at)
    if record.metadata:
        block["Metadata"] = "; ".join(f"{_as_field_value(str(k))}={_as_field_value(str(v))}" for k, v in sorted(record.metadata.items()))
    if record.links:
        # Link targets are kept as a real, readable field (never dropped
        # on the floor). Rendered bare rather than as `[[name]]` so the
        # block parser cannot mistake the value for an inline list.
        block["Links"] = ", ".join(_as_field_value(str(link)) for link in record.links)
    return attach_provenance(
        block,
        actor_id=record.system,
        actor_role="importer",
        tool_id=token,
        purpose="migration-import",
    )


def _write_batch(
    workspace: str,
    plan: list[tuple[str, ImportRecord]],
    *,
    batch: str,
    system: str,
    transform_hash: str,
    store: Any,
) -> list[str]:
    """Write a planned import batch under one governance admission.

    Bulk ingest cannot be one proposal per block, so it is one *admission*
    per run instead: a single chain entry whose content is the exact id
    set this batch may write. ``store.write_block`` refuses any id the
    entry did not name.
    """
    from ..governance_gate import get_gate

    planned_ids = [block_id for block_id, _record in plan]
    written: list[str] = []
    with get_gate(workspace).admit_batch(
        action="INGEST",
        batch_id=batch,
        block_ids=planned_ids,
        content="\n".join(planned_ids),
        tier=IngestTier.EXTERNAL_INGEST,
        actor=f"importer:{system}",
        target_file=IMPORTED_CORPUS_FILE,
        metadata={"system": system, "batch": batch, "blocks": len(planned_ids)},
    ):
        for _block_id, record in plan:
            block = build_import_block(record, batch=batch)
            if transform_hash:
                block = {**block, "TransformHash": transform_hash}
            written.append(str(store.write_block(block)))
    return written


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


def _materialize_link_edges(workspace: str, records: Iterable[ImportRecord]) -> int:
    """Materialize ``[[wikilink]]`` targets as ``cites`` lineage edges.

    Opt-in only (see ``link_edges``). Edges land in the lineage graph,
    never in the corpus, so imported blocks stay byte-identical to a
    flag-off run. Targets that do not resolve inside this batch are
    no-ops rather than dangling edges.
    """
    from .note_parsers import link_aliases

    ordered = list(records)
    # name -> block id. First writer wins, and the batch is already in a
    # deterministic order, so the mapping is a pure function of input.
    by_name: dict[str, str] = {}
    for record in ordered:
        block_id = block_id_for(record)
        for alias in link_aliases(record):
            by_name.setdefault(alias, block_id)
    if not by_name:
        return 0

    from ..block_lineage import add_block_edge

    written = 0
    for record in ordered:
        if not record.links:
            continue
        src = block_id_for(record)
        for target in record.links:
            dst = by_name.get(target)
            if dst is None or dst == src:
                continue
            try:
                add_block_edge(workspace, src, dst, "cites")
            except (ValueError, OSError) as exc:
                _log.warning("import_link_edge_failed", extra={"src": src, "dst": dst, "error": str(exc)})
                continue
            written += 1
    return written


def run_import(
    workspace: str,
    system: str,
    path: str,
    *,
    dedup_near: bool = False,
    dedup_threshold: float = 0.85,
    link_edges: bool = False,
    dry_run: bool = False,
) -> ImportResult:
    """Import the source at *path* into *workspace*, quarantined.

    Every block written lands with ``Status: quarantined`` and is
    therefore invisible to :func:`mind_mem.recall.recall` until a
    release proposal is approved
    (:func:`mind_mem.importers.quarantine.propose_import_release` ->
    ``approve_apply``). The run itself is appended to the tamper-evident
    audit chain before this function returns.

    Args:
        workspace: mind-mem workspace root.
        system: Source system slug; must be locally readable and supported.
        path: Path to the JSON dump, or the root of the note tree for the
            directory formats (``markdown`` / ``agentmem``).
        dedup_near: Opt-in (default OFF) near-duplicate collapse across
            the incoming batch, using ``dedup.layer_cosine_dedup``. With
            the default the import is a pure function of the dump.
        dedup_threshold: Cosine threshold for ``dedup_near``.
        link_edges: Opt-in (default OFF) materialization of
            ``[[wikilink]]`` targets as ``cites`` lineage edges. Link
            names are ALWAYS preserved as the ``Links`` block field; this
            flag only decides whether the lineage graph is also written,
            so flag-off output is byte-identical.
        dry_run: Parse + plan without writing any block.

    Raises:
        UnsupportedSystemError: *system* has no file-based importer.
        ImportParseError: the dump is unreadable or malformed.
        ImportQuarantineError: blocks were written but the run could not
            be recorded in the audit chain. The blocks stay quarantined
            and inert, so the corpus is safe; the import is not a
            success and must not be reported as one.
    """
    from . import resolve_system

    resolved = resolve_system(system)
    if not isinstance(workspace, str) or not workspace.strip():
        raise ImporterError("workspace must be a non-empty path")

    from .parsers import parse_payload

    records = tuple(_sanitized(r, workspace) for r in parse_payload(resolved, load_source(resolved, path)))
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

    # Plan first, then stamp: the batch id is derived from the ids this
    # run will write, so it cannot be computed per block inside the loop.
    seen: set[str] = set()
    plan: list[tuple[str, ImportRecord]] = []
    for record in records:
        block_id = block_id_for(record)
        if block_id in existing or block_id in seen:
            skipped_existing += 1
            continue
        seen.add(block_id)
        plan.append((block_id, record))

    batch = batch_id_for(resolved, (block_id for block_id, _ in plan))

    planned_ids = [block_id for block_id, _record in plan]
    if dry_run or store is None:
        written.extend(planned_ids)
    else:
        written.extend(_write_batch(workspace, plan, batch=batch, system=resolved, transform_hash=transform_hash, store=store))

    linked = 0
    if link_edges and not dry_run and records:
        linked = _materialize_link_edges(workspace, records)

    # Bulk ingest is ungated by design; the audit-chain entry is the half
    # of that bargain that keeps it honest, so it is not best-effort.
    # A dry run writes nothing, so it records nothing.
    if written and not dry_run:
        record_import_in_chain(
            workspace,
            system=resolved,
            source_path=os.path.abspath(path),
            batch=batch,
            block_ids=written,
            corpus_file=IMPORTED_CORPUS_FILE,
        )

    result = ImportResult(
        system=resolved,
        source_path=os.path.abspath(path),
        parsed=parsed,
        imported=len(written),
        skipped_existing=skipped_existing,
        skipped_near_duplicate=skipped_near,
        block_ids=tuple(written),
        dry_run=dry_run,
        linked_edges=linked,
        batch=batch,
        status=QUARANTINE_STATUS,
    )
    _log.info("import_completed", extra=result.as_dict())
    return result


def _text_size(value: Any) -> int:
    """Total characters of every string inside a JSON-shaped value.

    Used only to report how many codepoints sanitization removed, so a
    strip on a non-``text`` field is logged rather than silent.
    """
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_text_size(k) + _text_size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_text_size(item) for item in value)
    return 0


def _sanitized(record: ImportRecord, workspace: str) -> ImportRecord:
    """Strip invisible codepoints from every untrusted field of *record*.

    Config-gated + default ON via ``ingest.sanitize_codepoints``. Returns
    a NEW record; the input is never mutated.

    ``text`` is not the whole untrusted surface. :mod:`mind_mem.inbox`
    sanitizes a whole file body because that body *is* everything it
    ingests; an ``ImportRecord`` splits the same foreign dump across five
    fields, and ``metadata`` / ``links`` / ``external_id`` /
    ``created_at`` are rendered into the block's ``Metadata`` / ``Links``
    / ``ExternalId`` / ``Timestamp`` fields. ``_as_field_value`` only
    collapses whitespace there, and Cf codepoints — zero-width spaces,
    bidi overrides, Unicode tag characters — are not whitespace, so they
    reached the corpus verbatim while the ``Statement`` beside them was
    clean. :func:`~mind_mem.codepoint_sanitize.sanitize_structure` (the
    same call ``ingestion_pipeline`` already makes) covers the nested
    shapes.
    """
    from ..codepoint_sanitize import (
        sanitize_enabled_for_workspace,
        sanitize_structure,
        sanitize_text_for_ingest,
    )

    if not sanitize_enabled_for_workspace(workspace):
        return record

    source = f"import:{record.system}:{record.external_id}"
    clean_text = sanitize_text_for_ingest(record.text, workspace, source=source)
    original_fields: dict[str, Any] = {
        "external_id": record.external_id,
        "created_at": record.created_at,
        "metadata": dict(record.metadata),
        "links": list(record.links),
    }
    clean_fields: dict[str, Any] = sanitize_structure(original_fields)

    # sanitize_text_for_ingest already logged the ``text`` strip; report
    # the other fields separately so neither channel goes unrecorded.
    removed = _text_size(original_fields) - _text_size(clean_fields)
    if removed:
        _log.warning(
            "invisible_codepoints_stripped",
            extra={"removed": removed, "source": f"{source}:fields"},
        )

    if clean_text == record.text and not removed and clean_fields == original_fields:
        return record
    return ImportRecord(
        system=record.system,
        external_id=str(clean_fields["external_id"]),
        text=clean_text,
        metadata=clean_fields["metadata"],
        created_at=str(clean_fields["created_at"]),
        links=tuple(clean_fields["links"]),
    )

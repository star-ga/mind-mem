#!/usr/bin/env python3
"""Corpus → typed knowledge-graph ingestion (HITL-gated).

Wires the extraction layer to the typed knowledge graph without ever
writing the graph from an un-reviewed model call:

1. :func:`backfill` walks a corpus slice, runs relation extraction per
   block, and reports yield metrics (edges-per-block + predicate
   histogram). Dry-run by default — the measurement IS the first
   deliverable.
2. Extracted triples are staged as SIGNALS.md entries (pattern
   ``auto-capture-relation``) via the same ``append_signals`` path the
   entity ingester uses. Nothing touches the graph here.
3. :func:`approve_relation_signals` is the apply-on-approve step: for
   each operator-approved signal it calls ``KnowledgeGraph.add_edge``
   with the real ``source_block_id``, a ``valid_from`` timestamp, and
   an origin marker, then flips the signal status so approval is
   idempotent.

The extractor is injected (``extract_fn``) so the loop proves out with
zero model calls in CI; the default binds to the configured extraction
model via :func:`mind_mem.llm_extractor.extract_relations`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from .capture import append_signals
from .knowledge_graph import KnowledgeGraph, Predicate, default_db_path
from .mind_filelock import FileLock
from .observability import get_logger

_log = get_logger("graph_ingest")

RELATION_PATTERN = "auto-capture-relation"
RELATION_SIGNAL_TYPE = "auto-capture-relation"
EDGE_ORIGIN = "mind-graph:extracted:v1"

_MAX_ENTITY_NAME_LEN = 512
_TAG_PREDICATE = "predicate="
_TAG_SOURCE_BLOCK = "source-block="
_TAG_CONFIDENCE = "edge-confidence="


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelationTriple:
    """One validated extracted relation, bound to its source block."""

    subject: str
    predicate: str
    object: str
    source_block_id: str
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.subject or not self.subject.strip():
            raise ValueError("subject must be a non-empty string")
        if not self.object or not self.object.strip():
            raise ValueError("object must be a non-empty string")
        if len(self.subject) > _MAX_ENTITY_NAME_LEN or len(self.object) > _MAX_ENTITY_NAME_LEN:
            raise ValueError(f"entity names must be ≤{_MAX_ENTITY_NAME_LEN} chars")
        if not self.source_block_id or not self.source_block_id.strip():
            raise ValueError("source_block_id is required for provenance")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        # Raises ValueError on unknown predicates (honours runtime-
        # registered ones) — validation at the boundary, not at apply.
        Predicate.from_str(self.predicate)


# ---------------------------------------------------------------------------
# Staging — triples → SIGNALS.md proposals
# ---------------------------------------------------------------------------


def _encode_tag_value(value: str) -> str:
    """Escape the tag-list delimiter inside a tag VALUE.

    SIGNALS.md serializes tags as one comma-joined ``Tags:`` line and
    the reader splits it back on commas — a comma inside a value (e.g.
    an unusual ``source_block_id``) would split mid-value and silently
    corrupt the provenance anchor of every edge approved from that
    signal. Percent-escape the comma, and the escape character itself
    first so decoding is unambiguous.
    """
    return value.replace("%", "%25").replace(",", "%2C")


def _decode_tag_value(value: str) -> str:
    """Reverse :func:`_encode_tag_value` (order matters: comma first)."""
    return value.replace("%2C", ",").replace("%25", "%")


def relations_to_signals(relations: Iterable[RelationTriple]) -> list[dict]:
    """Convert triples to signal dicts for ``append_signals``.

    Mirrors ``entity_ingest.entities_to_signals``; predicate, source
    block, and confidence ride in the structured tags (values escaped
    via :func:`_encode_tag_value`) so the approve step can reconstruct
    the full edge from the SIGNALS.md block byte-exactly.
    """
    signals = []
    for rel in relations:
        signals.append(
            {
                "line": 0,
                "type": "relation",
                "text": (
                    f"Relation proposed: {rel.subject} {rel.predicate} {rel.object} "
                    f"(from {rel.source_block_id}, confidence {rel.confidence:g})"
                ),
                "pattern": RELATION_PATTERN,
                "confidence": "medium",
                "priority": "P2",
                "structure": {
                    "subject": rel.subject,
                    "object": rel.object,
                    "tags": [
                        "graph-edge",
                        "auto-ingest",
                        f"{_TAG_PREDICATE}{_encode_tag_value(rel.predicate)}",
                        f"{_TAG_SOURCE_BLOCK}{_encode_tag_value(rel.source_block_id)}",
                        f"{_TAG_CONFIDENCE}{rel.confidence:g}",
                    ],
                },
            }
        )
    return signals


def stage_relation_signals(
    workspace: str,
    relations: Iterable[RelationTriple],
    date_str: Optional[str] = None,
) -> int:
    """Stage triples as pending SIGNALS.md entries. Returns count written.

    Never writes the knowledge graph — approval is a separate,
    operator-driven step (:func:`approve_relation_signals`).
    """
    rels = list(relations)
    if not rels:
        return 0
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return int(append_signals(workspace, relations_to_signals(rels), date_str))


# ---------------------------------------------------------------------------
# Reading staged signals back
# ---------------------------------------------------------------------------


def _signals_path(workspace: str) -> str:
    return os.path.join(workspace, "intelligence", "SIGNALS.md")


def _parse_tags(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def _relation_from_block(block: dict) -> Optional[dict]:
    """Decode one SIGNALS.md block into a relation dict, or None."""
    if str(block.get("Type", "")) != RELATION_SIGNAL_TYPE:
        return None
    tags = _parse_tags(block.get("Tags"))
    predicate = source_block = None
    confidence = 0.5
    for tag in tags:
        if tag.startswith(_TAG_PREDICATE):
            predicate = _decode_tag_value(tag[len(_TAG_PREDICATE) :])
        elif tag.startswith(_TAG_SOURCE_BLOCK):
            source_block = _decode_tag_value(tag[len(_TAG_SOURCE_BLOCK) :])
        elif tag.startswith(_TAG_CONFIDENCE):
            try:
                confidence = max(0.0, min(1.0, float(tag[len(_TAG_CONFIDENCE) :])))
            except ValueError:
                confidence = 0.5
    subject = str(block.get("Subject", "")).strip()
    obj = str(block.get("Object", "")).strip()
    if not (subject and obj and predicate and source_block):
        return None
    return {
        "signal_id": str(block.get("_id", "")),
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "source_block_id": source_block,
        "confidence": confidence,
        "status": str(block.get("Status", "")).strip().lower(),
    }


def _relation_signal_blocks(workspace: str) -> list[dict]:
    path = _signals_path(workspace)
    if not os.path.isfile(path):
        return []
    from .block_parser import parse_file

    out = []
    for block in parse_file(path):
        rel = _relation_from_block(block)
        if rel is not None:
            out.append(rel)
    return out


def pending_relation_signals(workspace: str) -> list[dict]:
    """Return staged relation signals still awaiting operator approval."""
    return [r for r in _relation_signal_blocks(workspace) if r["status"] == "pending"]


_EXCERPT_MAX_LEN = 160


def attach_source_excerpts(
    workspace: str,
    relations: Iterable[dict],
    *,
    corpus: Optional[list[dict]] = None,
    max_len: int = _EXCERPT_MAX_LEN,
) -> list[dict]:
    """Return copies of *relations* with a bounded ``source_excerpt``.

    HITL review surface: without the source-block text next to a
    pending relation, the operator cannot verify the relation actually
    reflects the block (vs an injected fabrication) before approving.
    The excerpt is whitespace-collapsed and hard-capped at *max_len*
    characters; a missing source block yields an empty string rather
    than an error so the review listing never fails open-ended.

    Input dicts are not mutated — new dicts are returned.
    """
    if corpus is None:
        corpus = _load_corpus(workspace)
    id_to_text: dict[str, str] = {}
    for block in corpus:
        bid = str(block.get("_id") or "").strip()
        if bid and bid not in id_to_text:
            id_to_text[bid] = str(block.get("excerpt") or block.get("content") or block.get("Statement") or "")
    out: list[dict] = []
    for rel in relations:
        text = id_to_text.get(str(rel.get("source_block_id", "")), "")
        excerpt = " ".join(text.split())[: max(0, int(max_len))]
        out.append({**rel, "source_excerpt": excerpt})
    return out


# ---------------------------------------------------------------------------
# Apply-on-approve
# ---------------------------------------------------------------------------


def _flip_signal_status(workspace: str, signal_id: str, new_status: str) -> bool:
    """Rewrite ``Status:`` for one signal block in SIGNALS.md."""
    path = _signals_path(workspace)
    if not os.path.isfile(path):
        return False
    with FileLock(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = re.compile(
            rf"(\[{re.escape(signal_id)}\](?:(?!\n\[)[\s\S])*?\nStatus: )\w+",
        )
        new_content, n = pattern.subn(rf"\g<1>{new_status}", content, count=1)
        if n == 0:
            return False
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    return True


def approve_relation_signals(
    workspace: str,
    signal_ids: Iterable[str],
    *,
    now: Optional[str] = None,
) -> dict:
    """Apply operator-approved relation signals to the knowledge graph.

    For each signal id: validate it is a pending relation signal,
    write the edge via ``KnowledgeGraph.add_edge`` (real
    ``source_block_id``, ``valid_from`` stamped, origin marker in
    metadata), then flip the signal's status to ``applied``.

    Returns ``{"applied": [...], "errors": {signal_id: reason}}``.
    """
    valid_from = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    staged = {r["signal_id"]: r for r in _relation_signal_blocks(workspace)}
    applied: list[str] = []
    errors: dict[str, str] = {}
    kg: Optional[KnowledgeGraph] = None
    try:
        for sig_id in signal_ids:
            rel = staged.get(sig_id)
            if rel is None:
                errors[sig_id] = "no relation signal with this id"
                continue
            if rel["status"] != "pending":
                errors[sig_id] = f"signal is not pending (status={rel['status']!r})"
                continue
            if kg is None:
                kg = KnowledgeGraph(default_db_path(workspace))
            try:
                kg.add_edge(
                    rel["subject"],
                    rel["predicate"],
                    rel["object"],
                    source_block_id=rel["source_block_id"],
                    confidence=rel["confidence"],
                    valid_from=valid_from,
                    metadata={"origin": EDGE_ORIGIN, "signal_id": sig_id},
                )
            except ValueError as exc:
                errors[sig_id] = str(exc)
                continue
            if not _flip_signal_status(workspace, sig_id, "applied"):
                # Edge landed but the ledger flip failed — surface it
                # loudly; idempotent add_edge makes a retry safe.
                errors[sig_id] = "edge written but SIGNALS.md status update failed"
                continue
            applied.append(sig_id)
            _log.info(
                "relation_signal_applied",
                signal_id=sig_id,
                subject=rel["subject"],
                predicate=rel["predicate"],
                object=rel["object"],
                source_block_id=rel["source_block_id"],
            )
    finally:
        if kg is not None:
            kg.close()
    return {"applied": applied, "errors": errors}


# ---------------------------------------------------------------------------
# Backfill — corpus slice → extraction → yield metrics (+ optional staging)
# ---------------------------------------------------------------------------


def _default_extract_fn(workspace: str) -> Callable[[str], list[dict]]:
    """Bind the configured extraction model as the extractor callable."""
    from .llm_extractor import extract_relations, load_config

    config = load_config(workspace)
    model = str(config.get("model", "qwen3.5:9b"))
    backend = str(config.get("backend", "auto"))

    def _extract(text: str) -> list[dict]:
        return extract_relations(text, model=model, backend=backend, workspace=workspace)

    return _extract


def _load_corpus(workspace: str) -> list[dict]:
    """Load the workspace block corpus (markdown store)."""
    from .block_parser import parse_file
    from .block_store import MarkdownBlockStore

    store = MarkdownBlockStore(workspace)
    blocks: list[dict] = []
    for path in store.list_blocks():
        try:
            blocks.extend(parse_file(path))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            _log.debug("backfill_block_parse_skipped", error=str(exc))
    return blocks


def backfill(
    workspace: str,
    *,
    corpus: Optional[list[dict]] = None,
    extract_fn: Optional[Callable[[str], list[dict]]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    dry_run: bool = True,
) -> dict:
    """Run relation extraction over a corpus slice; report yield.

    Args:
        workspace: Workspace root.
        corpus: Block dicts to scan; loaded from the workspace when
            omitted. Each block needs ``_id`` plus text in
            ``excerpt`` / ``content`` / ``Statement``.
        extract_fn: Injected extractor ``text -> [{subject, predicate,
            object, confidence}]``. Defaults to the configured
            extraction model. Invalid triples are dropped, counted in
            ``edges_dropped_invalid``.
        limit / offset: Slice bounds over the corpus (blocks with ids).
        dry_run: When true (default) nothing is written anywhere —
            the run is purely a yield measurement. When false, valid
            triples are staged as pending SIGNALS.md entries (still
            never a direct graph write).

    Returns:
        Metrics dict: ``blocks_scanned``, ``blocks_with_edges``,
        ``edges_extracted``, ``edges_per_block``,
        ``predicate_histogram``, ``edges_dropped_invalid``,
        ``signals_written``, ``dry_run``.
    """
    if corpus is None:
        corpus = _load_corpus(workspace)
    if extract_fn is None:
        extract_fn = _default_extract_fn(workspace)

    with_ids = [b for b in corpus if str(b.get("_id") or "").strip()]
    if offset > 0:
        with_ids = with_ids[offset:]
    if limit is not None:
        with_ids = with_ids[: max(0, int(limit))]

    triples: list[RelationTriple] = []
    histogram: dict[str, int] = {}
    blocks_with_edges = 0
    dropped = 0

    for block in with_ids:
        bid = str(block["_id"]).strip()
        text = str(block.get("excerpt") or block.get("content") or block.get("Statement") or "")
        if not text.strip():
            continue
        raw = extract_fn(text)
        block_edges = 0
        for tri in raw or []:
            try:
                rel = RelationTriple(
                    subject=str(tri.get("subject", "")),
                    predicate=str(tri.get("predicate", "")),
                    object=str(tri.get("object", "")),
                    source_block_id=bid,
                    confidence=float(tri.get("confidence", 0.5)),
                )
            except (ValueError, TypeError):
                dropped += 1
                continue
            triples.append(rel)
            histogram[rel.predicate] = histogram.get(rel.predicate, 0) + 1
            block_edges += 1
        if block_edges:
            blocks_with_edges += 1

    scanned = len(with_ids)
    signals_written = 0
    if not dry_run and triples:
        signals_written = stage_relation_signals(workspace, triples)

    report = {
        "dry_run": bool(dry_run),
        "blocks_scanned": scanned,
        "blocks_with_edges": blocks_with_edges,
        "edges_extracted": len(triples),
        "edges_per_block": (len(triples) / scanned) if scanned else 0.0,
        "predicate_histogram": dict(sorted(histogram.items())),
        "edges_dropped_invalid": dropped,
        "signals_written": signals_written,
    }
    _log.info("graph_backfill_report", **report)
    return report


__all__ = [
    "RELATION_PATTERN",
    "EDGE_ORIGIN",
    "RelationTriple",
    "relations_to_signals",
    "stage_relation_signals",
    "pending_relation_signals",
    "attach_source_excerpts",
    "approve_relation_signals",
    "backfill",
]

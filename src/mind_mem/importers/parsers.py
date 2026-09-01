# Copyright 2026 STARGA, Inc.
"""Parser registry — one parser per supported source format.

Each parser takes an already-loaded payload (decoded JSON for the dump
formats, a note tuple for the directory formats) and returns an
immutable tuple of :class:`~mind_mem.importers.records.ImportRecord`.
No I/O, no network, no source-system SDK.

The note-tree / transcript parsers — the formats agent memory actually
lives in — are defined in :mod:`mind_mem.importers.note_parsers` and
registered here, so ``parse_payload`` stays the single dispatch point.

Supported shapes
----------------
``markdown``
    A directory of markdown notes (vault-style or a plain note tree).
    Front matter becomes metadata; ``[[wikilink]]`` targets are kept.

``agentmem``
    A coding-agent auto-memory directory: an index note plus sibling
    notes carrying ``name`` / ``description`` / ``metadata.type`` front
    matter, plus any root-level instruction file.

``chatjson``
    A chat-session transcript: a list of ``{role, content}`` turns, or a
    wrapper object/array holding them.

``mem0``
    The ``get_all()`` payload: ``{"results": [...]}`` where each entry
    carries ``id`` + ``memory`` + optional ``metadata`` / ``created_at``.
    A bare list of those entries is accepted too.

``letta``
    The agent-file (``.af``) payload: ``core_memory`` blocks
    (``label`` / ``value``) plus ``archival_memory`` entries. An
    ``agents: [...]`` wrapper (multi-agent file) is accepted.

``chroma``
    The ``collection.get()`` payload. Low value — see
    :func:`parse_chroma`.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ._shared import (
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LEN,
)
from ._shared import (
    clean_text as _clean_text,
)
from ._shared import (
    first_str as _first_str,
)
from ._shared import (
    flatten_metadata as _flatten_metadata,
)
from ._shared import (
    require_mapping as _require_mapping,
)
from .note_parsers import parse_agent_memory, parse_chat_json, parse_markdown_vault
from .okf_source import OKF_SYSTEM, parse_okf
from .records import ImportParseError, ImportRecord

__all__ = ["MAX_METADATA_KEYS", "MAX_METADATA_VALUE_LEN", "PARSERS", "parse_payload"]

# ---------------------------------------------------------------------------
# chroma
# ---------------------------------------------------------------------------


def _parse_chroma_collection(payload: Mapping[str, Any], prefix: str) -> list[ImportRecord]:
    documents = payload.get("documents")
    if documents is None:
        raise ImportParseError("chroma dump is missing the 'documents' array")
    if not isinstance(documents, list):
        raise ImportParseError(f"chroma 'documents' must be a list, got {type(documents).__name__}")

    ids = payload.get("ids")
    if ids is not None and not isinstance(ids, list):
        raise ImportParseError(f"chroma 'ids' must be a list, got {type(ids).__name__}")
    if isinstance(ids, list) and len(ids) != len(documents):
        raise ImportParseError(f"chroma dump is inconsistent: {len(ids)} ids vs {len(documents)} documents")

    metadatas = payload.get("metadatas")
    if metadatas is not None and not isinstance(metadatas, list):
        raise ImportParseError(f"chroma 'metadatas' must be a list, got {type(metadatas).__name__}")
    # Same malformedness as a short 'ids' array, so it gets the same
    # answer. A short 'metadatas' used to be absorbed silently: the tail
    # documents imported with {} metadata and therefore no created_at,
    # so they landed as blocks with no Date and nothing said so.
    if isinstance(metadatas, list) and len(metadatas) != len(documents):
        raise ImportParseError(f"chroma dump is inconsistent: {len(metadatas)} metadatas vs {len(documents)} documents")

    records: list[ImportRecord] = []
    for index, document in enumerate(documents):
        text = _clean_text(document)
        if not text:
            continue
        raw_id = str(ids[index]) if isinstance(ids, list) else f"{prefix}{index}"
        meta = _flatten_metadata(metadatas[index]) if isinstance(metadatas, list) and index < len(metadatas) else {}
        if prefix:
            meta = {**meta, "collection": prefix.rstrip("/")}
        records.append(
            ImportRecord(
                system="chroma",
                external_id=raw_id,
                text=text,
                metadata=meta,
                created_at=_first_str(meta, ("created_at", "timestamp")),
            )
        )
    return records


def parse_chroma(payload: Any) -> tuple[ImportRecord, ...]:
    """Parse a Chroma ``collection.get()`` / multi-collection JSON dump.

    deferred: this is a LOW-VALUE import path and is kept only because
    some exports do happen to carry usable text. A vector store persists
    *embeddings*, and an embedding is re-derived from scratch on import,
    so the vectors in a dump are worth nothing here — the only thing
    worth lifting is the source text, which survives an export only when
    the writer kept a ``documents`` array (or stashed the text in payload
    metadata). A dump whose documents are absent or blank therefore
    imports to nothing, correctly.
    upgrade path: prefer the note-tree (``markdown`` / ``agentmem``) and
    transcript (``chatjson``) importers, which read source text directly;
    if a payload-metadata text field must be supported, add an explicit
    ``--text-field`` selector rather than guessing at key names.
    """
    if isinstance(payload, list):
        docs = [{"document": entry} if isinstance(entry, str) else entry for entry in payload]
        flat = {
            "ids": [str(d.get("id", i)) if isinstance(d, Mapping) else str(i) for i, d in enumerate(docs)],
            "documents": [d.get("document", d.get("text", "")) if isinstance(d, Mapping) else "" for d in docs],
            "metadatas": [d.get("metadata") if isinstance(d, Mapping) else None for d in docs],
        }
        return tuple(_parse_chroma_collection(flat, ""))

    mapping = _require_mapping(payload, "chroma")
    collections = mapping.get("collections")
    if isinstance(collections, list):
        records: list[ImportRecord] = []
        for entry in collections:
            if not isinstance(entry, Mapping):
                raise ImportParseError("chroma 'collections' entries must be JSON objects")
            name = _first_str(entry, ("name", "collection")) or "collection"
            records.extend(_parse_chroma_collection(entry, f"{name}/"))
        return tuple(records)

    name = _first_str(mapping, ("collection", "name"))
    return tuple(_parse_chroma_collection(mapping, f"{name}/" if name else ""))


# ---------------------------------------------------------------------------
# mem0
# ---------------------------------------------------------------------------


def parse_mem0(payload: Any) -> tuple[ImportRecord, ...]:
    """Parse a mem0 ``get_all()`` JSON dump."""
    if isinstance(payload, list):
        entries: Any = payload
    else:
        mapping = _require_mapping(payload, "mem0")
        entries = mapping.get("results", mapping.get("memories"))
        if entries is None:
            raise ImportParseError("mem0 dump is missing the 'results' array")
    if not isinstance(entries, list):
        raise ImportParseError(f"mem0 'results' must be a list, got {type(entries).__name__}")

    records: list[ImportRecord] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ImportParseError(f"mem0 result #{index} must be a JSON object, got {type(entry).__name__}")
        text = _clean_text(entry.get("memory") or entry.get("text") or entry.get("content") or entry.get("data"))
        if not text:
            continue
        meta = _flatten_metadata(entry.get("metadata"))
        for passthrough in ("user_id", "agent_id", "run_id"):
            value = entry.get(passthrough)
            if isinstance(value, str) and value.strip():
                meta[passthrough] = value.strip()[:MAX_METADATA_VALUE_LEN]
        records.append(
            ImportRecord(
                system="mem0",
                external_id=_first_str(entry, ("id", "memory_id", "hash")) or f"index-{index}",
                text=text,
                metadata=dict(sorted(meta.items())),
                created_at=_first_str(entry, ("created_at", "updated_at")),
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# letta
# ---------------------------------------------------------------------------

# deferred: letta conversation `messages` are NOT imported — a message log is
# dialogue, not a durable memory block, and folding it in would flood recall
# with turn-level noise. Upgrade path: run the transcript through
# mind_mem.extractor first, then import the extracted blocks.
_LETTA_SKIPPED_SECTIONS = ("messages", "message_ids", "in_context_message_ids")


def _parse_letta_agent(agent: Mapping[str, Any]) -> list[ImportRecord]:
    agent_name = _first_str(agent, ("name", "agent_name", "id")) or "agent"
    records: list[ImportRecord] = []

    core = agent.get("core_memory")
    if core is None:
        core = agent.get("memory_blocks", agent.get("blocks"))
    if isinstance(core, Mapping):
        core = [{"label": k, "value": v} for k, v in sorted(core.items())]
    if isinstance(core, list):
        for index, block in enumerate(core):
            if not isinstance(block, Mapping):
                continue
            text = _clean_text(block.get("value") or block.get("text"))
            if not text:
                continue
            label = _first_str(block, ("label", "name")) or f"block-{index}"
            records.append(
                ImportRecord(
                    system="letta",
                    external_id=f"{agent_name}/core/{label}",
                    text=text,
                    metadata={"agent": agent_name, "label": label, "section": "core_memory"},
                    created_at=_first_str(agent, ("created_at",)),
                )
            )

    archival = agent.get("archival_memory", agent.get("archival_memories"))
    if isinstance(archival, list):
        for index, entry in enumerate(archival):
            if isinstance(entry, str):
                text, external_id, created = _clean_text(entry), f"{agent_name}/archival/{index}", ""
            elif isinstance(entry, Mapping):
                text = _clean_text(entry.get("text") or entry.get("content") or entry.get("value"))
                external_id = _first_str(entry, ("id", "memory_id")) or f"{agent_name}/archival/{index}"
                created = _first_str(entry, ("created_at", "timestamp"))
            else:
                continue
            if not text:
                continue
            records.append(
                ImportRecord(
                    system="letta",
                    external_id=external_id,
                    text=text,
                    metadata={"agent": agent_name, "section": "archival_memory"},
                    created_at=created or _first_str(agent, ("created_at",)),
                )
            )
    return records


def parse_letta(payload: Any) -> tuple[ImportRecord, ...]:
    """Parse a Letta agent-file (``.af``) JSON dump."""
    mapping = _require_mapping(payload, "letta")
    agents = mapping.get("agents")
    if isinstance(agents, list):
        records: list[ImportRecord] = []
        for entry in agents:
            if not isinstance(entry, Mapping):
                raise ImportParseError("letta 'agents' entries must be JSON objects")
            records.extend(_parse_letta_agent(entry))
        return tuple(records)

    records = _parse_letta_agent(mapping)
    if not records:
        known = ", ".join(k for k in mapping.keys() if k not in _LETTA_SKIPPED_SECTIONS)
        raise ImportParseError(f"letta dump has no 'core_memory' or 'archival_memory' content (top-level keys: {known or 'none'})")
    return tuple(records)


PARSERS: dict[str, Callable[[Any], tuple[ImportRecord, ...]]] = {
    "agentmem": parse_agent_memory,
    "chatjson": parse_chat_json,
    "chroma": parse_chroma,
    "letta": parse_letta,
    "markdown": parse_markdown_vault,
    "mem0": parse_mem0,
    # Flag-gated (v4 ``core_export``): unreachable unless resolve_system
    # lets the slug through, so a registered parser is not a live source.
    OKF_SYSTEM: parse_okf,
}


def parse_payload(system: str, payload: Any) -> tuple[ImportRecord, ...]:
    """Dispatch *payload* to the parser registered for *system*.

    Raises:
        ImportParseError: no parser is registered, or the payload is
            malformed for the claimed format.
    """
    parser = PARSERS.get(system)
    if parser is None:
        raise ImportParseError(f"no parser registered for source system {system!r}")
    return parser(payload)

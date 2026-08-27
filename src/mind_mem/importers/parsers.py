# Copyright 2026 STARGA, Inc.
"""Dump parsers for the three file-based source systems.

Each parser takes an already-decoded JSON payload and returns an
immutable tuple of :class:`~mind_mem.importers.records.ImportRecord`.
No I/O, no network, no source-system SDK — a dump is just JSON, and the
whole point of the file-based subset is that it stays that way.

Supported shapes
----------------
``chroma``
    The ``collection.get()`` payload: parallel ``ids`` / ``documents`` /
    ``metadatas`` arrays. Also accepts a ``collections: [...]`` wrapper
    (multi-collection dump) and a bare list of per-document records.

``mem0``
    The ``get_all()`` payload: ``{"results": [...]}`` where each entry
    carries ``id`` + ``memory`` + optional ``metadata`` / ``created_at``.
    A bare list of those entries is accepted too.

``letta``
    The agent-file (``.af``) payload: ``core_memory`` blocks
    (``label`` / ``value``) plus ``archival_memory`` entries. An
    ``agents: [...]`` wrapper (multi-agent file) is accepted.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from .records import ImportParseError, ImportRecord

__all__ = ["PARSERS", "parse_payload"]

# Metadata is metadata, not content — bound every value so a hostile dump
# can't inflate a block. Mirrors block_provenance.MAX_PROVENANCE_VALUE_LEN.
MAX_METADATA_VALUE_LEN = 256
MAX_METADATA_KEYS = 24

_META_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _flatten_metadata(raw: Any) -> dict[str, str]:
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


def _clean_text(raw: Any) -> str:
    """Return *raw* as text, or ``""`` when it is not usable content."""
    if not isinstance(raw, str):
        return ""
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def _first_str(source: Mapping[str, Any], keys: Sequence[str]) -> str:
    """First non-empty string value among *keys*, else ``""``."""
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _require_mapping(payload: Any, system: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ImportParseError(f"{system} dump must be a JSON object, got {type(payload).__name__}")
    return payload


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
    """Parse a Chroma ``collection.get()`` / multi-collection JSON dump."""
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
    "chroma": parse_chroma,
    "letta": parse_letta,
    "mem0": parse_mem0,
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

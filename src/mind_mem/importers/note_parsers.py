# Copyright 2026 STARGA, Inc.
"""Parsers for the three formats agent memory actually lives in today.

A vector store keeps *embeddings*, and an embedding is re-derived on
import — the only thing worth lifting out of one is the source text, and
that is present only when the exporter happened to keep it. The formats
below all hold the source text directly, which is why they are the
importers that carry weight:

``markdown``
    A note tree (vault-style or a plain folder of notes). Front matter
    becomes metadata, ``[[wikilink]]`` targets are preserved as links.

``agentmem``
    A coding-agent auto-memory directory: an index note listing entries
    by ``[[name]]`` or ``[label](name.md)``, sibling notes carrying
    ``name`` / ``description`` / ``metadata.type`` front matter, plus any
    root-level instruction file. Sections are classified structurally,
    never by vendor filename.

``chatjson``
    A session transcript — a list of ``{role, content}`` turns, or a
    wrapper object/array holding them. One block per turn.

All three are pure transforms: a tuple of
:class:`~mind_mem.importers.fs_source.SourceNote` (or decoded JSON) in,
a tuple of :class:`~mind_mem.importers.records.ImportRecord` out.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ._shared import clean_text, first_str, flatten_metadata, merge_metadata, require_mapping
from .fs_source import SourceNote, markdown_link_targets, wikilink_targets
from .records import ImportParseError, ImportRecord

__all__ = [
    "INDEX_FILENAMES",
    "LINK_KEY",
    "MESSAGE_KEYS",
    "link_aliases",
    "link_key_for",
    "parse_agent_memory",
    "parse_chat_json",
    "parse_markdown_vault",
    "resolve_links",
]

#: Metadata key holding the name a ``[[wikilink]]`` resolves against.
LINK_KEY = "name"

#: Index-note filenames, matched case-sensitively at any depth. Generic
#: layout names only — the classifier is structural, not vendor-keyed.
INDEX_FILENAMES: tuple[str, ...] = ("MEMORY.md", "INDEX.md", "index.md", "README.md")

#: Keys a transcript wrapper may use for its turn array.
MESSAGE_KEYS: tuple[str, ...] = ("messages", "turns", "conversation", "history", "chat")

_SESSION_KEYS: tuple[str, ...] = ("sessions", "conversations", "threads")
_ROLE_KEYS: tuple[str, ...] = ("role", "from", "sender", "speaker", "author")
_CONTENT_KEYS: tuple[str, ...] = ("content", "text", "message", "value")
_TIME_KEYS: tuple[str, ...] = ("created_at", "timestamp", "time", "sent_at")


def _require_notes(payload: Any, system: str) -> tuple[SourceNote, ...]:
    """Assert the loader handed us a note tree, not a JSON dump."""
    if isinstance(payload, tuple) and all(isinstance(n, SourceNote) for n in payload):
        return payload
    raise ImportParseError(f"{system} importer expects a directory of markdown notes, got {type(payload).__name__}")


def link_key_for(record: ImportRecord) -> str:
    """The name other notes use to link *record* (``""`` when unnamed)."""
    return record.metadata.get(LINK_KEY, "")


def link_aliases(record: ImportRecord) -> tuple[str, ...]:
    """Every name *record* can be linked by, most specific first.

    A note is addressable by its declared ``name`` and by its filename
    stem — the two coincide in a well-kept auto-memory directory and
    diverge in a vault whose notes carry a human title.
    """
    stem = record.external_id.rsplit("/", 1)[-1]
    for suffix in (".md", ".markdown"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return tuple(dict.fromkeys(name for name in (link_key_for(record), stem) if name))


# ---------------------------------------------------------------------------
# markdown note trees (vault-style or a plain folder of notes)
# ---------------------------------------------------------------------------


def _note_metadata(note: SourceNote, extra: Mapping[str, str]) -> dict[str, str]:
    """Front matter plus structural keys, bounded and sorted."""
    return merge_metadata(flatten_metadata(dict(note.front_matter)), dict(extra))


def parse_markdown_vault(payload: Any) -> tuple[ImportRecord, ...]:
    """Parse a markdown note tree into one record per non-empty note."""
    notes = _require_notes(payload, "markdown")
    records: list[ImportRecord] = []
    for note in notes:
        body = clean_text(note.body)
        if not body:
            continue
        front = dict(note.front_matter)
        # A vault resolves ``[[target]]`` against the FILENAME, so the
        # stem — not the human title — is the note's link identity.
        structural = {
            LINK_KEY: note.stem,
            "path": note.relative_path,
            "title": first_str(front, ("title", "name")) or note.stem,
        }
        if note.folder:
            structural["folder"] = note.folder
        records.append(
            ImportRecord(
                system="markdown",
                external_id=note.relative_path,
                text=body,
                metadata=_note_metadata(note, structural),
                created_at=first_str(front, ("created", "created_at", "date")),
                links=wikilink_targets(body),
            )
        )
    if not records:
        raise ImportParseError(f"markdown note tree has {len(notes)} notes but none carry any body text")
    return tuple(records)


# ---------------------------------------------------------------------------
# agent auto-memory directories
# ---------------------------------------------------------------------------


def _section_for(note: SourceNote) -> str:
    """Classify a note structurally: index / instructions / memory.

    An index is a known layout filename. A root-level note with no front
    matter is a standing instruction file. Everything else is a memory
    note. No vendor filename is special-cased.
    """
    name = note.relative_path.rsplit("/", 1)[-1]
    if name in INDEX_FILENAMES:
        return "index"
    if not note.folder and not note.front_matter:
        return "instructions"
    return "memory"


def parse_agent_memory(payload: Any) -> tuple[ImportRecord, ...]:
    """Parse an auto-memory directory (index + front-matter notes)."""
    notes = _require_notes(payload, "agentmem")
    records: list[ImportRecord] = []
    for note in notes:
        body = clean_text(note.body)
        if not body:
            continue
        front = dict(note.front_matter)
        section = _section_for(note)
        structural = {
            LINK_KEY: first_str(front, ("name", "title")) or note.stem,
            "path": note.relative_path,
            "section": section,
        }
        note_type = first_str(front, ("metadata.type", "type", "metadata.node_type"))
        if note_type:
            structural["type"] = note_type
        description = first_str(front, ("description", "summary"))
        if description:
            structural["description"] = description
        # An index links its entries by path; notes link each other by name.
        links = wikilink_targets(body)
        if section == "index":
            links = tuple(dict.fromkeys(links + markdown_link_targets(body)))
        records.append(
            ImportRecord(
                system="agentmem",
                external_id=note.relative_path,
                text=body,
                metadata=_note_metadata(note, structural),
                created_at=first_str(front, ("created", "created_at", "date")),
                links=links,
            )
        )
    if not records:
        raise ImportParseError(f"auto-memory directory has {len(notes)} notes but none carry any body text")
    return tuple(records)


# ---------------------------------------------------------------------------
# chat-session transcripts
# ---------------------------------------------------------------------------


def _turn_text(raw: Any) -> str:
    """Turn content to text, accepting the structured-parts shape."""
    if isinstance(raw, str):
        return clean_text(raw)
    if isinstance(raw, (list, tuple)):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                piece = first_str(part, ("text", "content", "value"))
                if piece:
                    parts.append(piece)
        return clean_text("\n".join(parts))
    return ""


def _session_records(session: Mapping[str, Any], turns: Sequence[Any], fallback: str) -> list[ImportRecord]:
    label = first_str(session, ("session_id", "id", "conversation_id", "name", "title")) or fallback
    records: list[ImportRecord] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise ImportParseError(f"chat turn #{index} of session {label!r} must be a JSON object, got {type(turn).__name__}")
        text = _turn_text(turn.get("content", turn.get("text", turn.get("message", turn.get("value")))))
        if not text:
            continue
        role = first_str(turn, _ROLE_KEYS) or "unknown"
        metadata = merge_metadata(
            flatten_metadata(turn.get("metadata")),
            {"role": role, "session": label, "turn": str(index)},
        )
        records.append(
            ImportRecord(
                system="chatjson",
                external_id=first_str(turn, ("id", "message_id", "uuid")) or f"{label}/{index}",
                text=text,
                metadata=metadata,
                created_at=first_str(turn, _TIME_KEYS) or first_str(session, _TIME_KEYS),
            )
        )
    return records


def _turns_of(session: Mapping[str, Any]) -> Sequence[Any] | None:
    for key in MESSAGE_KEYS:
        value = session.get(key)
        if isinstance(value, list):
            return value
        if value is not None:
            raise ImportParseError(f"chat transcript {key!r} must be a list, got {type(value).__name__}")
    return None


def _looks_like_turn(entry: Any) -> bool:
    return isinstance(entry, Mapping) and any(key in entry for key in _ROLE_KEYS + _CONTENT_KEYS)


def parse_chat_json(payload: Any) -> tuple[ImportRecord, ...]:
    """Parse a chat-session transcript into one record per turn.

    deferred: turns land one block each rather than being condensed into
    durable statements, so a long transcript is imported verbatim.
    upgrade path: run the imported blocks through ``mind_mem.extractor``
    (or an injected condenser) as a second, governed pass.
    """
    if isinstance(payload, list):
        entries: list[Any] = payload
        if entries and all(_looks_like_turn(e) for e in entries):
            return tuple(_session_records({}, entries, "session"))
        records: list[ImportRecord] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ImportParseError(f"chat transcript entry #{index} must be a JSON object, got {type(entry).__name__}")
            turns = _turns_of(entry)
            if turns is None:
                raise ImportParseError(f"chat transcript entry #{index} has no turn array (looked for: {', '.join(MESSAGE_KEYS)})")
            records.extend(_session_records(entry, turns, f"session-{index}"))
        if not records:
            raise ImportParseError("chat transcript is an empty list")
        return tuple(records)

    mapping = require_mapping(payload, "chatjson")
    for key in _SESSION_KEYS:
        sessions = mapping.get(key)
        if isinstance(sessions, list):
            return parse_chat_json(sessions)
        if sessions is not None:
            raise ImportParseError(f"chat transcript {key!r} must be a list, got {type(sessions).__name__}")

    turns = _turns_of(mapping)
    if turns is None:
        known = ", ".join(sorted(str(k) for k in mapping.keys())) or "none"
        raise ImportParseError(f"chat transcript has no turn array (looked for: {', '.join(MESSAGE_KEYS)}; top-level keys: {known})")
    return tuple(_session_records(mapping, turns, "session"))


def resolve_links(records: Iterable[ImportRecord]) -> dict[str, str]:
    """Map every link name (and filename alias) to an ``external_id``.

    Deterministic: on a name collision the first record in iteration
    order wins, and iteration order is the sorted note-tree order.
    """
    index: dict[str, str] = {}
    for record in records:
        for alias in link_aliases(record):
            index.setdefault(alias, record.external_id)
    return index

# Copyright 2026 STARGA, Inc.
"""Migration importers — foreign agent memory into the corpus.

``mm import --from <system> <path>`` lifts memory out of another system
and lands it in the corpus as ``IMP-`` blocks, each stamped with an
``imported:<system>`` provenance token and recallable immediately.

The set is chosen by where agent memory actually lives on disk, not by
which stores are fashionable. Three formats hold the *source text*:

``markdown``
    A directory of notes — vault-style or a plain note tree. Front
    matter becomes metadata, ``[[wikilink]]`` targets are preserved.

``agentmem``
    A coding-agent auto-memory directory: an index note plus notes
    carrying ``name`` / ``description`` / ``metadata.type`` front
    matter, plus any root-level instruction file.

``chatjson``
    A chat-session transcript — ``{role, content}`` turns.

Two agent-memory services export a usable JSON dump (``mem0``,
``letta``) and are supported as-is. The vector-store path (``chroma``)
is kept but is explicitly low value: a vector store persists embeddings,
which are re-derived on import, so only its optional source-text array
is worth anything — see :func:`mind_mem.importers.parsers.parse_chroma`.

Every supported source is a local file or a local directory, so an
import never opens a socket and never needs a credential. The
endpoint-backed stores in the roadmap line (pinecone / weaviate /
qdrant) are *deferred*, not silently missing: asking for one raises
:class:`~mind_mem.importers.records.UnsupportedSystemError` naming it
and saying why.
"""

from __future__ import annotations

from .engine import (
    DIRECTORY_SYSTEMS,
    IMPORT_BLOCK_PREFIX,
    IMPORT_BLOCK_TYPE,
    IMPORTED_CORPUS_FILE,
    MAX_DUMP_BYTES,
    block_id_for,
    build_import_block,
    load_dump,
    load_source,
    provenance_token,
    run_import,
)
from .records import (
    ImporterError,
    ImportParseError,
    ImportRecord,
    ImportResult,
    UnsupportedSystemError,
)

__all__ = [
    "SUPPORTED_SYSTEMS",
    "DEFERRED_SYSTEMS",
    "DIRECTORY_SYSTEMS",
    "ALL_SYSTEMS",
    "IMPORT_BLOCK_PREFIX",
    "IMPORT_BLOCK_TYPE",
    "IMPORTED_CORPUS_FILE",
    "MAX_DUMP_BYTES",
    "ImporterError",
    "ImportParseError",
    "ImportRecord",
    "ImportResult",
    "UnsupportedSystemError",
    "block_id_for",
    "build_import_block",
    "load_dump",
    "load_source",
    "provenance_token",
    "resolve_system",
    "run_import",
]

# Systems whose export is a local file or a local directory — the
# shipped subset. Sorted, and the CLI choice list is held in lockstep.
SUPPORTED_SYSTEMS: tuple[str, ...] = ("agentmem", "chatjson", "chroma", "letta", "markdown", "mem0")

# deferred: pinecone / weaviate / qdrant importers need a live endpoint +
# an API credential to page through vectors, which the no-network test
# gate (and every offline migration) cannot provide. Upgrade path: add a
# client-backed reader per system behind an explicit `--endpoint` /
# `--api-key-env` flag pair, keeping this file-based path as the default.
DEFERRED_SYSTEMS: dict[str, str] = {
    "pinecone": "requires a live Pinecone index endpoint + API key to page vectors",
    "qdrant": "requires a live Qdrant endpoint (or a snapshot restore) to read points",
    "weaviate": "requires a live Weaviate endpoint + schema introspection to read objects",
}

ALL_SYSTEMS: tuple[str, ...] = tuple(sorted(SUPPORTED_SYSTEMS + tuple(DEFERRED_SYSTEMS)))


def resolve_system(system: str) -> str:
    """Normalize and validate a ``--from`` value.

    Returns:
        The canonical lowercase slug of a supported, locally-readable
        system (a JSON dump or a directory of notes).

    Raises:
        UnsupportedSystemError: the system is one of the deferred
            endpoint-backed systems, or is not recognised at all. The
            message names the system and the supported set explicitly.
    """
    if not isinstance(system, str):
        raise UnsupportedSystemError(f"source system must be a string, got {type(system).__name__}")
    slug = system.strip().lower()
    if slug in SUPPORTED_SYSTEMS:
        return slug
    supported = ", ".join(SUPPORTED_SYSTEMS)
    if slug in DEFERRED_SYSTEMS:
        raise UnsupportedSystemError(
            f"import from {slug!r} is DEFERRED and not supported: {DEFERRED_SYSTEMS[slug]}. "
            f"mind-mem ships the local-file and local-directory importers only ({supported}); "
            f"deferred endpoint-backed systems: {', '.join(sorted(DEFERRED_SYSTEMS))}."
        )
    raise UnsupportedSystemError(f"unsupported source system {slug!r}; supported local importers: {supported}")

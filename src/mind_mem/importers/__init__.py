# Copyright 2026 STARGA, Inc.
"""Migration importers (roadmap Group G) — file-based subset.

``mm import --from {chroma|mem0|letta} <path>`` lifts a memory export out
of another system and lands it in the corpus as ``IMP-`` blocks, each
stamped with an ``imported:<system>`` provenance token and recallable
immediately.

Scope is deliberately the **file-based** subset. The endpoint-backed
systems in the roadmap line (pinecone / weaviate / qdrant) are *deferred*,
not silently missing: asking for one raises
:class:`~mind_mem.importers.records.UnsupportedSystemError` naming it and
saying why. Every supported format is plain JSON on disk, so an import
never opens a socket and never needs a credential.
"""

from __future__ import annotations

from .engine import (
    IMPORT_BLOCK_PREFIX,
    IMPORT_BLOCK_TYPE,
    IMPORTED_CORPUS_FILE,
    MAX_DUMP_BYTES,
    block_id_for,
    build_import_block,
    load_dump,
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
    "provenance_token",
    "resolve_system",
    "run_import",
]

# Systems whose export is a local file — the shipped subset.
SUPPORTED_SYSTEMS: tuple[str, ...] = ("chroma", "letta", "mem0")

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
        The canonical lowercase slug of a supported, file-based system.

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
            f"mind-mem ships the file-based importers only ({supported}); "
            f"deferred endpoint-backed systems: {', '.join(sorted(DEFERRED_SYSTEMS))}."
        )
    raise UnsupportedSystemError(f"unsupported source system {slug!r}; supported file-based importers: {supported}")

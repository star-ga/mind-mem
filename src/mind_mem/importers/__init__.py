# Copyright 2026 STARGA, Inc.
"""Migration importers — foreign agent memory into the corpus.

``mm import --from <system> <path>`` lifts memory out of another system
and lands it in the corpus as ``IMP-`` blocks, each stamped with an
``imported:<system>`` provenance token — and each **quarantined**.

External ingest is never authoritative on arrival: an imported block
carries ``Status: quarantined`` + ``IngestTier: external-ingest``, recall
filters it out, and it becomes recallable only when a governance
proposal releases it (``propose_import_release`` -> ``approve_apply``).
The bulk write itself is recorded in the tamper-evident audit chain.
:mod:`mind_mem.importers.quarantine` documents why bulk ingest is one
chained write plus one governed release rather than a proposal per block.

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

The supported source set remains local-file/local-directory first, so ordinary
imports never open a socket. Qdrant additionally has an explicit endpoint-only
scroll adapter: it is never selected by a plain file import, requires a
collection and optional environment-backed API key, and still lands through
the same quarantine/release path. Pinecone and Weaviate remain deferred.
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
    verify_document_anchor,
)
from .quarantine import (
    MAX_RELEASE_BLOCKS,
    QUARANTINE_STATUS,
    QUARANTINE_TIER,
    ImportQuarantineError,
    NothingToReleaseError,
    ReleaseTooLargeError,
    admitted_import_ids,
    batch_id_for,
    is_quarantined,
    propose_import_release,
    quarantined_import_ids,
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
    "ENDPOINT_SYSTEMS",
    "GATED_SYSTEMS",
    "DIRECTORY_SYSTEMS",
    "ALL_SYSTEMS",
    "IMPORT_BLOCK_PREFIX",
    "IMPORT_BLOCK_TYPE",
    "IMPORTED_CORPUS_FILE",
    "MAX_DUMP_BYTES",
    "MAX_RELEASE_BLOCKS",
    "QUARANTINE_STATUS",
    "QUARANTINE_TIER",
    "ImporterError",
    "ImportParseError",
    "ImportQuarantineError",
    "ImportRecord",
    "ImportResult",
    "NothingToReleaseError",
    "ReleaseTooLargeError",
    "UnsupportedSystemError",
    "admitted_import_ids",
    "batch_id_for",
    "block_id_for",
    "enabled_gated_systems",
    "build_import_block",
    "is_quarantined",
    "load_dump",
    "load_source",
    "propose_import_release",
    "provenance_token",
    "quarantined_import_ids",
    "resolve_system",
    "run_import",
    "verify_document_anchor",
]

# Systems whose export is a local file or a local directory — the
# shipped subset. Sorted, and the CLI choice list is held in lockstep.
SUPPORTED_SYSTEMS: tuple[str, ...] = ("agentmem", "chatjson", "chroma", "letta", "markdown", "mem0")

# deferred: Pinecone and Weaviate need a live endpoint + an API credential
# to page through vectors, which the no-network test gate (and every offline
# migration) cannot provide. Qdrant is separately exposed through the bounded
# stdlib scroll adapter below; keeping it out of this mapping's supported
# local-file set prevents accidental network access.
DEFERRED_SYSTEMS: dict[str, str] = {
    "pinecone": "requires a live Pinecone index endpoint + API key to page vectors",
    "qdrant": "requires explicit endpoint mode (a bounded REST scroll adapter is available)",
    "weaviate": "requires a live Weaviate endpoint + schema introspection to read objects",
}

# Qdrant is endpoint-backed but deliberately not part of SUPPORTED_SYSTEMS:
# keeping it out preserves the file-import choice list and makes accidental
# network access impossible. ``run_import(..., endpoint=...)`` is the sole
# explicit opt-in entry point.
ENDPOINT_SYSTEMS: dict[str, str] = {"qdrant": "requires --endpoint and --collection; reads via bounded REST scroll"}

ALL_SYSTEMS: tuple[str, ...] = tuple(sorted(SUPPORTED_SYSTEMS + tuple(DEFERRED_SYSTEMS)))

# Importers that exist but are reachable only while a feature flag is ON.
# Deliberately NOT merged into SUPPORTED_SYSTEMS / ALL_SYSTEMS: those are
# published constants (and the ``mm import --from`` choice list is pinned
# to them), so a flag-off build must report exactly the set it reported
# before this importer existed.
#
#   okf -> mind_mem.importers.okf_source, gated on v4 ``core_export``.
GATED_SYSTEMS: dict[str, str] = {"okf": "core_export"}


def enabled_gated_systems() -> tuple[str, ...]:
    """The gated importers whose flag is currently ON, sorted.

    Empty — and byte-for-byte free of side effects — when every gate is
    off: the probe is :func:`~mind_mem.v4.feature_flags.is_enabled_quiet`,
    which emits nothing on a missing or malformed config. A probe that
    decides whether a feature is on must not itself be observable when
    the answer is no.
    """
    from ..v4.feature_flags import is_enabled_quiet

    return tuple(sorted(slug for slug, flag in GATED_SYSTEMS.items() if is_enabled_quiet(flag)))


def resolve_system(system: str) -> str:
    """Normalize and validate a ``--from`` value.

    Returns:
        The canonical lowercase slug of a supported, locally-readable
        system (a JSON dump or a directory of notes), or of a
        :data:`GATED_SYSTEMS` importer whose flag is ON.

    Raises:
        UnsupportedSystemError: the system is one of the deferred
            endpoint-backed systems, is gated behind a flag that is OFF,
            or is not recognised at all. The message names the system and
            the supported set explicitly.
    """
    if not isinstance(system, str):
        raise UnsupportedSystemError(f"source system must be a string, got {type(system).__name__}")
    slug = system.strip().lower()
    if slug in SUPPORTED_SYSTEMS:
        return slug
    gated = enabled_gated_systems()
    if slug in gated:
        return slug
    # Appended, never substituted, so the flag-off message is character-for-
    # character the one this function has always produced.
    supported = ", ".join(SUPPORTED_SYSTEMS + gated)
    if slug in DEFERRED_SYSTEMS:
        raise UnsupportedSystemError(
            f"import from {slug!r} is DEFERRED and not supported: {DEFERRED_SYSTEMS[slug]}. "
            f"mind-mem ships the local-file and local-directory importers only ({supported}); "
            f"deferred endpoint-backed systems: {', '.join(sorted(DEFERRED_SYSTEMS))}."
        )
    if slug in ENDPOINT_SYSTEMS:
        raise UnsupportedSystemError(
            f"import from {slug!r} requires explicit endpoint mode: {ENDPOINT_SYSTEMS[slug]}. "
            f"Use run_import(..., endpoint=..., collection=...) or the matching CLI flags; "
            f"plain file imports remain local ({supported})."
        )
    raise UnsupportedSystemError(f"unsupported source system {slug!r}; supported local importers: {supported}")

# Copyright 2026 STARGA, Inc.
"""Graph schema versioning — the replay half of a governed edge.

An edge already carries a receipt (``add_edge`` opens with
``require_admission`` on the deterministic ``edge_id``). A receipt says
*who authorised this edge*. It does not say *what rules produced it*, and
without that an approved edge cannot be re-derived: the predicate
vocabulary, the entity-canonicalisation rule and the extraction prompt
are all inputs to the triple, and all three drift between releases. An
edge extracted under a vocabulary of ten predicates and an edge extracted
under a vocabulary of fourteen are different artefacts that look
identical in the ``edges`` table.

This module makes them distinguishable. :func:`current_version` folds
every input that decides what a triple can say into one deterministic id,
``gs1-<12 hex>``; :func:`stamp` writes it into an edge's ``metadata`` at
the one choke point every door already goes through
(``KnowledgeGraph.add_edge``), and preserves an id the triple already
carries so a proposal staged in June and approved in September records
the schema it was *extracted* under, not the one in force at approval.

Three properties the rest of the graph layer depends on:

* **Deterministic.** No clock, no counter, no randomness, no filesystem.
  The same vocabulary and the same prompt yield the same id on any
  machine, which is what lets a re-extraction be *checked* rather than
  trusted.
* **Sensitive.** Registering a predicate at runtime widens what an
  extraction may emit, so it changes the id. That is not a nuisance —
  it is the whole mechanism: every edge written before the widening is
  now visibly of an older schema.
* **Cheap.** A few string joins and one SHA-256 over ~500 bytes, on a
  path that already computes a SHA-256 and commits to SQLite.

Stdlib only. Nothing here imports :mod:`mind_mem.knowledge_graph` at
module level (that module imports *this* one), so the two compose without
a cycle.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Optional

#: Format generation of the id itself. Bump only when the *shape* of the
#: version string changes (not when its inputs do — that is what the
#: digest is for).
GRAPH_SCHEMA_TAG = "gs1"

#: Hex characters of the SHA-256 kept in the id. 48 bits is ample for
#: distinguishing a handful of vocabulary generations and keeps the stamp
#: readable in a metadata dump.
_DIGEST_LEN = 12

#: A well-formed schema id. Accepts any generation tag so a store written
#: by a future ``gs2`` build still reads as *a schema id* here rather than
#: as corruption.
SCHEMA_VERSION_RE = re.compile(r"^gs[0-9]+-[0-9a-f]{12}$")

#: The metadata key an edge, an edge proposal and a staged relation signal
#: all use for the stamp. One spelling, so a reader never has to guess.
METADATA_KEY = "schema_version"

#: Identifier for the entity-canonicalisation rule in force
#: (``knowledge_graph._canonicalise``: strip, lowercase, collapse internal
#: whitespace). Two surface forms that fold together are one entity, so
#: the folding rule decides what a triple *means* and belongs in the id.
#: Change the rule, change this string in the same commit.
ENTITY_CANONICALISATION_RULE = "strip+lower+ws-collapse:v1"

#: Cached digest of the relation-extraction prompt template. The template
#: is a module constant that cannot change within a process, so this is
#: computed once; the vocabulary it interpolates is hashed separately and
#: is *not* cached, because runtime predicate registration can widen it.
_PROMPT_DIGEST: Optional[str] = None


class SchemaVersionError(ValueError):
    """A schema stamp that is not a well-formed schema id.

    Raised rather than silently overwritten: a malformed stamp means the
    caller believes something false about where the triple came from, and
    quietly replacing it with the current id would erase the evidence.
    """


def _relation_prompt_digest() -> str:
    """SHA-256 (truncated) of the relation-extraction prompt template.

    Imported lazily and cached: :mod:`mind_mem.llm_extractor` is
    stdlib-only at import time but is not needed at all on a read path,
    and one import per process is one import.
    """
    global _PROMPT_DIGEST
    if _PROMPT_DIGEST is None:
        from .llm_extractor import _RELATION_PROMPT

        _PROMPT_DIGEST = hashlib.sha256(_RELATION_PROMPT.encode("utf-8")).hexdigest()[:16]
    return _PROMPT_DIGEST


def predicate_vocabulary() -> tuple[str, ...]:
    """Every predicate an extraction may emit right now, lex-sorted.

    Includes runtime-registered predicates: a caller that registers one
    has widened the vocabulary, and an edge written afterwards is of a
    different schema than one written before.
    """
    from .knowledge_graph import _RUNTIME_PREDICATES, Predicate

    builtin = [p.value for p in Predicate]
    runtime = list(_RUNTIME_PREDICATES.keys())
    return tuple(sorted(set(builtin) | set(runtime)))


def schema_components() -> dict[str, str]:
    """The inputs the version digest is taken over, as a readable map.

    Returned for diagnostics — ``mm graph-backfill --schema`` prints it —
    so an operator who sees two different ids can see *which* input moved
    instead of being handed an opaque hash.
    """
    return {
        "entity_canonicalisation": ENTITY_CANONICALISATION_RULE,
        "extraction_prompt": _relation_prompt_digest(),
        "predicates": ",".join(predicate_vocabulary()),
    }


def current_version() -> str:
    """The schema id in force in this process.

    Pure function of :func:`schema_components` — no clock, no randomness,
    no filesystem — so it is identical across processes and substrates
    given the same vocabulary and prompt.
    """
    components = schema_components()
    preimage = "\x00".join(f"{key}={components[key]}" for key in sorted(components)).encode("utf-8")
    return f"{GRAPH_SCHEMA_TAG}-{hashlib.sha256(preimage).hexdigest()[:_DIGEST_LEN]}"


def is_schema_version(value: Any) -> bool:
    """True iff *value* is a well-formed schema id of any generation."""
    return isinstance(value, str) and bool(SCHEMA_VERSION_RE.match(value))


def version_of(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Read the stamp out of an edge's metadata, or ``None`` when absent.

    A malformed stamp reads as ``None`` rather than raising: this is the
    *read* side, and a report over a store that predates stamping must be
    able to say "unversioned" about a row instead of crashing on it.
    """
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(METADATA_KEY)
    return raw if is_schema_version(raw) else None


def stamp(metadata: Optional[Mapping[str, Any]], *, version: Optional[str] = None) -> dict[str, Any]:
    """Return a copy of *metadata* carrying a schema id.

    Immutable: the input mapping is never modified.

    An id already present wins — that is what carries a proposal's
    extraction-time schema through to the edge its approval commits,
    months later, under a vocabulary that may have moved. It must still be
    well-formed; a junk stamp raises :class:`SchemaVersionError` rather
    than being overwritten, because overwriting it would manufacture a
    provenance claim.

    Args:
        metadata: Existing edge / proposal metadata, or ``None``.
        version: Explicit id to write when *metadata* carries none.
            Defaults to :func:`current_version`.
    """
    out = dict(metadata or {})
    existing = out.get(METADATA_KEY)
    if existing is not None:
        if not is_schema_version(existing):
            raise SchemaVersionError(f"{METADATA_KEY} must match {SCHEMA_VERSION_RE.pattern}, got {existing!r}")
        return out
    chosen = version if version is not None else current_version()
    if not is_schema_version(chosen):
        raise SchemaVersionError(f"{METADATA_KEY} must match {SCHEMA_VERSION_RE.pattern}, got {chosen!r}")
    out[METADATA_KEY] = chosen
    return out


__all__ = [
    "GRAPH_SCHEMA_TAG",
    "SCHEMA_VERSION_RE",
    "METADATA_KEY",
    "ENTITY_CANONICALISATION_RULE",
    "SchemaVersionError",
    "predicate_vocabulary",
    "schema_components",
    "current_version",
    "is_schema_version",
    "version_of",
    "stamp",
]

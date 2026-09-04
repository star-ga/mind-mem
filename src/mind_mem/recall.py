#!/usr/bin/env python3
"""mind-mem Recall Engine (BM25 + TF-IDF + Graph + Stemming). Zero external deps.

Default recall backend: BM25 scoring with Porter stemming, stopword filtering,
query expansion, field boosts, recency weighting, and optional graph-based
neighbor boosting via cross-reference traversal.

For semantic recall (embeddings), see RecallBackend interface below.
Optional vector backends (Qdrant/Pinecone) can be plugged in via config.

Usage:
    python3 -m mind_mem.recall --query "authentication" --workspace "."
    python3 -m mind_mem.recall --query "auth" --workspace "." --json --limit 5
    python3 -m mind_mem.recall --query "deadline" --active-only
    python3 -m mind_mem.recall --query "database" --graph --workspace .

Determinism: recall is a pure function of **(corpus, config, scoring_instant)**.
Pass ``scoring_instant`` (a UTC ``datetime.date``) to pin the recency layer —
the recency ramp, the rolling calibration window and the temporal hard filter —
and the served set, order and scores are identical on any host, on any day. Omit
it and it resolves to today in UTC, the single clock read on the whole path.
See :mod:`mind_mem.scoring_instant`.

This module is the facade AND the **serving entry**. All ranking lives in
_recall_*.py submodules; what is added here is the proof obligation: a call to
:func:`recall` derives the run's :class:`~mind_mem.recall_attestation.RecallAttestation`
and appends the served-set ledger row before the answer is handed back. That is
why external consumers import from ``recall`` and never from ``_recall_*``
directly — the underscore module is the engine, and the engine alone cannot
prove what was served. ``tests/test_every_serving_surface_attests.py`` fails the
build when a new module reaches past this one.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from datetime import date
from typing import Any, Iterator

# --- Constants (_recall_constants) ---
from ._recall_constants import (
    _BLOCK_ID_RE,
    _IRREGULAR_LEMMA,
    _STOPWORDS,
    _VALID_RECALL_KEYS,
    BM25_B,
    BM25_K1,
    CORPUS_FILES,
    FIELD_WEIGHTS,
    GRAPH_BOOST_FACTOR,
    MAX_BLOCKS_PER_QUERY,
    MAX_GRAPH_NEIGHBORS_PER_HOP,
    MAX_RERANK_CANDIDATES,
    SEARCH_FIELDS,
    SIG_FIELDS,
)

# --- Context Packing (_recall_context) ---
from ._recall_context import _block_to_result, _parse_dia_id, context_pack

# --- Core Engine (_recall_core) ---
from ._recall_core import (
    RecallBackend,
    _load_backend,
    main,
    prefetch_context,
)
from ._recall_core import recall as _engine_recall

# --- Detection & Utilities (_recall_detection) ---
from ._recall_detection import (
    _INTENT_TO_QUERY_TYPE,
    _QUERY_TYPE_PARAMS,
    _parse_speaker_from_tags,
    chunk_text,
    detect_query_type,
    extract_field_tokens,
    extract_text,
    get_bigrams,
    get_block_type,
    get_excerpt,
    is_skeptical_query,
    normalise_tags,
)

# --- Query Expansion (_recall_expansion) ---
from ._recall_expansion import (
    _QUERY_EXPANSIONS,
    _rm3_language_model,
    expand_months,
    expand_query,
    rm3_expand,
)

# --- Reranking (_recall_reranking) ---
from ._recall_reranking import rerank_hits

# --- Scoring (_recall_scoring) ---
from ._recall_scoring import (
    _category_match_boost,
    _classify_categories,
    _date_proximity_score,
    _detect_negation,
    _extract_bigram_phrases,
    _extract_dates,
    _extract_entities,
    _extract_speaker_names,
    _negation_penalty,
    build_xref_graph,
    date_score,
)

# --- Temporal Filtering (_recall_temporal) ---
from ._recall_temporal import apply_temporal_filter, resolve_time_reference

# --- Tokenization (_recall_tokenization) ---
from ._recall_tokenization import _stem, tokenize

# --- Guardrails (guardrails / guardrail_surface) ---
from .guardrail_surface import apply_guardrail_surfacing, guardrail_hits
from .guardrails import (
    GuardrailContext,
    GuardrailPolicy,
    GuardrailSpecError,
    GuardrailTrigger,
    load_guardrails,
    match_guardrails,
)
from .scoring_instant import (
    as_utc_datetime,
    format_scoring_instant,
    parse_scoring_instant,
    resolve_scoring_instant,
)

# ---------------------------------------------------------------------------
# The serving entry — "prove what it served" is a property of the DOOR
# ---------------------------------------------------------------------------
#
# Round 1 bound the attestation to exactly one caller, the MCP recall handler.
# Every other door — the HTTP transport, the CLI, the axis orchestrator, the
# guardrail surface, the chat evidence path — reached the engine directly and
# served content that nothing recorded. An attestation that covers one of nine
# doors is not a property of the system, it is a property of that door.
#
# So the obligation moved to the entry every consumer already imports. The
# engine (``_recall_core.recall``) still ranks and still knows nothing about a
# ledger; this wrapper is what turns a ranking into a *serve*.
#
# ONE OUTERMOST SERVE, ONE ROW. Parts of the retrieval stack call back into
# this facade as a *leg* rather than as a door: ``sqlite_index.query_index``
# falls back here when the FTS database is missing, ``hybrid_recall``'s BM25
# arm does the same, and the MCP handler re-derives its attestation post-cache
# and must not have one baked in below it. A leg is not a serve, and recording
# one would put candidate sets nobody was ever handed into the ledger the
# outcome join reads. :func:`serving_scope` is how a caller that owns the
# serve says so; inside an open scope this function ranks and returns without
# attesting.


class ServedResults(list):  # type: ignore[type-arg]
    """The ranked hits, plus the attestation that commits to them.

    A ``list`` subclass on purpose: every existing consumer indexes, slices,
    iterates and ``json.dumps``-es the engine's return value, and all of that
    keeps working unchanged. What is new is :attr:`attestation` — the
    ``RECALL_ATTEST_v2`` record for the run that produced these hits, or
    ``None`` when this call was a leg inside someone else's serve (see
    :func:`serving_scope`) or when derivation failed.
    """

    attestation: dict[str, Any] | None = None
    #: The engine's ``{leg, reason}`` degradation marker, re-presented so a
    #: consumer reading it off the engine's return type still finds it here.
    #: :func:`~mind_mem.recall_attestation.derive_legs` reads exactly this
    #: attribute, so dropping it would silently turn a degraded vector leg
    #: into a clean one in the attestation.
    degraded: dict[str, str] | None = None


_serving = threading.local()

#: The modules that reach this entry from INSIDE another recall, never as a
#: door of their own. ``sqlite_index.query_index`` falls back here when the FTS
#: database is missing; ``hybrid_recall``'s BM25 arm does the same, and it does
#: it on a **worker thread** when query expansion produces more than one
#: variant. Neither is ever entered except from a recall that is already
#: underway, so a row minted by either is always a duplicate of the serve above
#: it, naming a candidate set nobody was handed.
#:
#: This is the half of the guard :func:`serving_scope` cannot cover. The scope
#: is thread-state, and a pool worker starts with a fresh one — measured, not
#: assumed: a leg dispatched onto a worker under an open scope minted its own
#: row, while the same call on the owning thread did not. Both halves are
#: needed and neither subsumes the other: the scope covers a door that nests
#: any module underneath it, this covers a leg that crosses a thread. The set is
#: pinned by ``tests/test_every_serving_surface_attests.py``, which fails the
#: build on a module that reaches the engine and is classified as neither.
LEG_MODULES = frozenset({"mind_mem.hybrid_recall", "mind_mem.sqlite_index"})


def in_serving_scope() -> bool:
    """Is an outer caller already the door for this serve?"""
    return int(getattr(_serving, "depth", 0)) > 0


def _called_as_leg() -> bool:
    """Is the immediate caller of :func:`recall` one of :data:`LEG_MODULES`?

    Read off the caller's frame rather than off any state the caller had to
    remember to set, which is what makes it survive the thread hop that defeats
    the scope. Answers ``False`` on any interpreter that will not give up a
    frame — failing toward *recording* a serve, since an extra row is a
    correctable surplus and a missing one is unrecoverable.
    """
    try:
        frame = sys._getframe(2)
    except ValueError:  # pragma: no cover — defensive; no such frame
        return False
    return str(frame.f_globals.get("__name__", "")) in LEG_MODULES


@contextmanager
def serving_scope() -> Iterator[None]:
    """Claim ownership of the serve for the duration of the block.

    A caller that will attest the served set itself — the MCP recall handler,
    which re-derives post-cache, or an orchestrator fusing several engine
    passes into one answer — opens this so the engine calls underneath it do
    not each mint a row. Re-entrant and restored on the way out, including on
    an exception, so a raising leg cannot leave the flag stuck on.

    Thread-scoped, and that is a real boundary rather than an implementation
    detail: a leg dispatched onto a worker thread (``hybrid_recall`` fans its
    query variants out over a pool) does not inherit the scope, so it would
    attest on its own account. That half is covered by :data:`LEG_MODULES`,
    which is read off the calling frame and therefore survives the hop. Two
    mechanisms, two jobs — this one lets a door nest ANY module underneath it,
    that one catches the two known legs wherever they run — and
    ``tests/test_every_serving_surface_attests.py`` exercises both, with the
    same call from a door on a worker thread as the positive control.
    """
    depth = int(getattr(_serving, "depth", 0))
    _serving.depth = depth + 1
    try:
        yield
    finally:
        _serving.depth = depth


def resolve_vector_flags(workspace: str, backend: str, config: Any = None) -> tuple[bool, bool]:
    """Resolve the CURRENT config's ``(vector_requested, vector_available)``.

    Derived fresh from the live configuration each call so a toggle (e.g.
    ``recall.vector_enabled``) is reflected in the attestation even when the
    ranking itself came from a cache. A ``bm25`` request never runs the vector
    leg regardless of config, so it answers ``(False, False)`` without probing.

    *config* lets a caller supply the config mapping it already holds; omitted,
    the engine's own mtime-cached reader is used. One implementation, two
    loaders — the MCP handler keeps the loader it has always used and this
    module keeps the engine's, and neither grows a second copy of the rule.

    Any failure degrades to the BM25-only shape rather than raising: an
    auxiliary artifact must not break recall.
    """
    if backend == "bm25":
        return False, False
    try:
        from .hybrid_recall import HybridBackend

        if config is None:
            from ._recall_core import _get_config

            config = _get_config(workspace)
        hb = HybridBackend.from_config(config)
        return bool(getattr(hb, "vector_enabled", False)), bool(getattr(hb, "vector_available", False))
    except Exception as exc:  # pragma: no cover — defensive
        _serving_log().warning("recall_attestation_vector_flags_failed", error=str(exc))
        return False, False


def _serving_log() -> Any:
    from .observability import get_logger

    return get_logger("recall")


def default_backend_for(workspace: str) -> str:
    """Which leg name a serve through this module should attest, unasked.

    Resolved from the workspace, never assumed: :func:`_load_backend` is the
    same function :func:`mind_mem._recall_core.recall` dispatches on, so the
    name reported is the one the run really went through rather than a constant
    chosen once and hoped to still be true.

    The rule is the one the CLI door already applies — the lexical engines are
    ``bm25`` (``"sqlite"``, and the built-in BM25 scan, which resolves to
    ``None``); a configured backend *object* is ``auto``, so the config's
    vector flags — which on that path ARE the run's own — get reported instead
    of being suppressed.

    Any failure degrades to ``bm25``, the shape that claims least: an auxiliary
    artifact must not break recall, and a record that under-claims on a broken
    config is safer than one that over-claims.
    """
    try:
        resolved = _load_backend(workspace)
    except Exception as exc:  # pragma: no cover — defensive
        _serving_log().warning("recall_attestation_backend_probe_failed", error=str(exc))
        return "bm25"
    return "auto" if isinstance(resolved, RecallBackend) else "bm25"


def attest_and_record(
    workspace: str,
    query: str,
    results: Any,
    *,
    backend: str | None = None,
    scoring_instant: date | str | None = None,
) -> dict[str, Any] | None:
    """Derive this run's attestation and append its served-ledger row.

    The two halves of "prove what it served", in the only order that is honest:
    the attestation is derived from the **recorded** run state (per-hit
    provenance and the ``.degraded`` marker on *results*), and the ledger row
    then reuses that record's own ``query_hash`` / ``results_digest`` /
    ``config_hash`` / ``index_anchor`` / ``scoring_instant`` verbatim, so the
    row and the record can never be two opinions of one run.

    Called strictly AFTER the ranking is fixed, and it writes no block, so the
    store's admission gate is untouched and nothing here can feed back into the
    ranking it describes. The ledger import is function-local for the same
    reason it is in the MCP handler: a module-level one would put a ledger in
    the eager-import closure of the scoring path, which
    ``tests/test_recall_attestation_v2.py`` fails the build over.

    *backend* names the leg the run actually used. Omitted, it is RESOLVED from
    the workspace by :func:`default_backend_for` rather than assumed.

    It used to default to the literal ``"bm25"``, justified here by the claim
    that ``_recall_core.recall`` "is the lexical engine and runs no dense leg on
    any path". That claim was false. ``_load_backend`` returns a
    ``VectorBackend`` for ``recall.backend: "vector"`` (dense-only) and a
    ``PostgresRecallBackend`` for a Postgres block store (server-side BM25 +
    pgvector); on both, a serve through this module published
    ``legs_ran=['bm25']`` with ``vector_requested=False`` about a run that did
    ask for a dense leg — under-claiming, the mirror image of a record naming a
    leg that never ran.

    What the old default got right is preserved by the resolution rule: on the
    lexical engines the answer is still ``bm25``, so a workspace with vector
    recall enabled for some *other* surface does not get a vector leg marked
    DEGRADED on every scan-backed call — that would read as "the dense leg was
    requested and not served" about a run that never asked for it. A caller
    that knows its own leg still passes its own name.

    Returns the attestation dict, or ``None`` when derivation failed. Never
    raises: an answer that was computed must be served even if the proof of it
    could not be written down.

    The returned record carries
    :data:`~mind_mem.served_ledger.LEDGER_ATTESTATION_KEYS` —
    ``served_seq`` / ``served_row_hash`` / ``ledger_error`` — because the row
    and the record are published together or the caller is told which one is
    missing. Both are produced by one call to
    :func:`~mind_mem.served_ledger.attach_served_run` rather than by threading
    six fields into the ledger here and hoping the next surface threads them
    the same way.
    """
    try:
        from .recall_attestation import _served_ids, derive_recall_attestation_for_workspace

        if backend is None:
            backend = default_backend_for(workspace)
        vector_requested, vector_available = resolve_vector_flags(workspace, backend)
        attestation = derive_recall_attestation_for_workspace(
            results,
            workspace,
            vector_requested=vector_requested,
            vector_available=vector_available,
            query=query,
            scoring_instant=scoring_instant,
        )
        record = attestation.to_dict()
    except Exception as exc:  # pragma: no cover — defensive; recall must not fail on attestation
        _serving_log().warning("recall_attestation_apply_failed", error=str(exc))
        return None
    from .served_ledger import attach_served_run

    return attach_served_run(record, workspace, ids=_served_ids(results))


def _carry_degraded(raw: Any, served: "ServedResults") -> None:
    """Copy a ``degraded`` marker from the engine's result onto the served one.

    One implementation for both return paths of :func:`recall`, so a marker
    cannot be preserved on one and silently dropped on the other again.
    """
    degraded = getattr(raw, "degraded", None)
    if isinstance(degraded, dict):
        served.degraded = degraded


def recall(
    workspace: str,
    query: str,
    *args: Any,
    **kwargs: Any,
) -> ServedResults:
    """Rank *query* against *workspace*, and record what was served.

    The signature is the engine's — every argument is forwarded verbatim to
    :func:`mind_mem._recall_core.recall`, deliberately by ``*args`` /
    ``**kwargs`` so this wrapper cannot drift from the ranking function it
    fronts and cannot silently swallow an argument the engine grew.

    Two things happen that the engine does not do:

    * ``scoring_instant`` is resolved HERE, once — **including when the caller
      omits it**, which is the case that matters. Left to default, the engine
      would read the clock for the ranking and the attestation would read it
      again a moment later for the record; across a UTC midnight those are two
      different dates, and the record would then name a day the ranking never
      scored against and replay to a different answer. Resolving at the boundary
      and handing the same date to both keeps recall's "one clock read on the
      whole path" true through the serving entry as well.
    * the served set is attested and recorded, unless an outer caller has
      claimed the serve with :func:`serving_scope` or the immediate caller is
      one of the retrieval legs in :data:`LEG_MODULES`.

    Returns a :class:`ServedResults` — a list of hit dicts, as before, carrying
    the run's attestation on ``.attestation``.
    """
    kwargs["scoring_instant"] = resolve_scoring_instant(kwargs.get("scoring_instant"))
    if in_serving_scope() or _called_as_leg():
        # The marker is carried on THIS branch too. It used to be dropped here
        # and re-presented only below, which meant a degraded engine result
        # lost its marker on exactly the path that serves it: ``hybrid_recall``
        # and ``sqlite_index`` are both in ``LEG_MODULES``, so every recall
        # reached through the hybrid arm or the FTS fallback took this return.
        # A degradation that cannot survive the leg hop is not in-band.
        raw_leg = _engine_recall(workspace, query, *args, **kwargs)
        served_leg = ServedResults(raw_leg)
        _carry_degraded(raw_leg, served_leg)
        return served_leg
    with serving_scope():
        raw = _engine_recall(workspace, query, *args, **kwargs)
        served = ServedResults(raw)
        # ``derive_legs`` reads the ``.degraded`` marker off the results object;
        # re-presenting it here keeps a degraded vector leg visible through the
        # wrapper instead of being lost with the engine's own return type.
        _carry_degraded(raw, served)
        served.attestation = attest_and_record(
            workspace,
            query,
            served,
            scoring_instant=kwargs.get("scoring_instant"),
        )
    return served


__all__ = [
    # Constants
    "SEARCH_FIELDS",
    "SIG_FIELDS",
    "CORPUS_FILES",
    "BM25_K1",
    "BM25_B",
    "FIELD_WEIGHTS",
    "_STOPWORDS",
    "_IRREGULAR_LEMMA",
    "GRAPH_BOOST_FACTOR",
    "_BLOCK_ID_RE",
    "MAX_BLOCKS_PER_QUERY",
    "MAX_GRAPH_NEIGHBORS_PER_HOP",
    "MAX_RERANK_CANDIDATES",
    "_VALID_RECALL_KEYS",
    # Tokenization
    "_stem",
    "tokenize",
    # Expansion
    "_QUERY_EXPANSIONS",
    "expand_months",
    "expand_query",
    "_rm3_language_model",
    "rm3_expand",
    # Detection & Utilities
    "extract_text",
    "extract_field_tokens",
    "get_bigrams",
    "is_skeptical_query",
    "detect_query_type",
    "_QUERY_TYPE_PARAMS",
    "_INTENT_TO_QUERY_TYPE",
    "chunk_text",
    "get_excerpt",
    "_parse_speaker_from_tags",
    "normalise_tags",
    "get_block_type",
    # Scoring
    "date_score",
    "build_xref_graph",
    "_detect_negation",
    "_negation_penalty",
    "_extract_dates",
    "_date_proximity_score",
    "_classify_categories",
    "_category_match_boost",
    "_extract_entities",
    "_extract_bigram_phrases",
    "_extract_speaker_names",
    # Reranking
    "rerank_hits",
    # Guardrails
    "GuardrailContext",
    "GuardrailPolicy",
    "GuardrailSpecError",
    "GuardrailTrigger",
    "load_guardrails",
    "match_guardrails",
    "apply_guardrail_surfacing",
    "guardrail_hits",
    # Context Packing
    "context_pack",
    "_parse_dia_id",
    "_block_to_result",
    # Core
    "RecallBackend",
    "recall",
    "_load_backend",
    "prefetch_context",
    "main",
    # Serving entry
    "ServedResults",
    "attest_and_record",
    "default_backend_for",
    "in_serving_scope",
    "LEG_MODULES",
    "resolve_vector_flags",
    "serving_scope",
    # Temporal Filtering
    "resolve_time_reference",
    "resolve_scoring_instant",
    "format_scoring_instant",
    "parse_scoring_instant",
    "as_utc_datetime",
    "apply_temporal_filter",
]

if __name__ == "__main__":
    main()

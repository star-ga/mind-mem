"""Recall surface — the retrieval core of the MCP API.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, recall domain). Eight tools + one shared helper:

* :func:`_recall_impl` — the BM25/hybrid backend switchboard that
  ``recall`` + ``hybrid_search`` both delegate to.
* ``recall`` — top-level ranked retrieval.
* ``recall_with_axis`` — axis-aware ODC recall.
* ``hybrid_search`` — deprecated alias (calls ``_recall_impl``).
* ``pack_recall_budget`` — token-budget-constrained pack.
* ``prefetch`` — pre-assemble from conversation signals.
* ``intent_classify`` — 9-way query router preview.
* ``find_similar`` — co-occurrence similarity.
* ``retrieval_diagnostics`` — per-stage rejection histogram.

Kept together because every one of them participates in the
single "search the workspace" mental model, and ``_recall_impl``
is the shared choke point they all ultimately lean on.
"""

from __future__ import annotations

import json
import os
import re as _re_mod
import sqlite3
import time
from datetime import date
from typing import Any

from mind_mem.error_codes import ErrorCode
from mind_mem.recall import recall as recall_engine
from mind_mem.retrieval_graph import retrieval_diagnostics as _retrieval_diag
from mind_mem.scoring_instant import format_scoring_instant, resolve_scoring_instant
from mind_mem.sqlite_index import _db_path as fts_db_path
from mind_mem.sqlite_index import query_index as fts_query

from ..infra.config import QUERY_TIMEOUT_SECONDS, _get_limits, _load_config
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import _is_db_locked, _sqlite_busy_error, mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import (
    _context_budget_enabled,
    _retrieval_metrics_enabled,
    error_envelope,
    get_logger,
    metrics,
)

_log = get_logger("mcp_server")


_MAX_QUERY_LEN = 8192


def _resolve_chain_head(ws: str) -> str:
    """The workspace's governed-ledger head — the corpus coordinate of a recall.

    Delegates to :func:`mind_mem.prefetch.chain_head`, which reads it through the
    same resolver ``_apply_attestation`` uses, so the cached answer and the
    attested corpus state are one value rather than two opinions. Resolved once
    per recall, here at the top, and handed to both consumers.

    Degrades to the genesis anchor on any failure: a cache key that cannot be
    computed must fall back to a *constant*, so two runs at different corpus
    states still share a key only when nothing could be learned about either —
    never silently drop the coordinate and re-open the staleness hole it closes.
    """
    try:
        from mind_mem.prefetch import chain_head

        return chain_head(ws)
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("chain_head_unresolved", error=str(exc))
        from mind_mem.recall_attestation import GENESIS_ANCHOR

        return GENESIS_ANCHOR


def _anticipation_envelope(
    ws: str,
    query: str,
    limit: int,
    config: Any,
    head: str,
    instant_iso: str,
) -> str | None:
    """Answer *query* from the local bundle cache, or ``None`` to fall through.

    Group J's consumer half. The bundles consulted are only those recorded at
    *head*, so this can never serve content a governed write has superseded —
    that write moved the head and retired the generation with it. The
    :mod:`~mind_mem.novel_term_gate` makes the local-vs-source call, and a
    fall-through is the safe direction on every degenerate input.

    An anticipation-served envelope is **not attested**, deliberately. The
    attestation says "this run read the corpus at this anchor and served these
    ids"; this run read a *bundle*. Stamping it would be the exact
    stale-evidence-as-this-run's failure the post-cache attestation exists to
    avoid, so the early return below skips the attestation, explain and
    served-ledger stages, and the envelope says so in-band: every hit carries
    ``_retrieval_source: "anticipation_cache"``, the envelope carries an
    ``anticipation`` block with the gate's numbers, and a warning names the
    trade in words.

    **Named consequence, so it is not discovered later.** Skipping the
    served-ledger stage means an anticipation-served run leaves no RA.1 row, so
    the ledger under-counts what was served. That is deliberate rather than an
    oversight — ``append_served_run`` records an attestation's ``index_anchor``
    and this run has no attestation to record — but it is a real gap, and the
    way to close it is to give the ledger a row shape that says "served from a
    local bundle at head H, unattested" rather than to fabricate an attestation
    here. Until then, an operator should read the ledger as a record of
    *attested* recalls only.

    The gap got wider in 5.0.2 and is worth restating in those terms: the
    ledger is now ON by default, so this is no longer "two opt-ins that
    interact". Switching ``cache.anticipation.enabled`` on is now, on its own,
    a decision to stop recording the runs it answers locally.
    """
    try:
        from mind_mem.prefetch import anticipation_config, get_cache, observe_served
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("anticipation_cache_unavailable", error=str(exc))
        return None
    # Clamp to the SAME ceiling the store path applies. A second door that
    # serves results must enforce the same policy as the first, or the operator
    # limit is only advisory: without this an anticipation-served recall could
    # return more hits than ``limits.max_recall_results`` allows, purely by
    # being answered locally.
    limit = max(1, min(limit, _get_limits(ws)["max_recall_results"]))
    decision = get_cache().lookup(
        ws,
        query,
        limit=limit,
        gate_config=anticipation_config(config),
        head=head,
    )
    if not decision.serve_from_cache:
        return None
    results = [dict(hit) for hit in decision.served]
    observe_served(query, results)
    metrics.inc("mcp_recall_anticipation_hits")
    _log.info("mcp_recall_anticipated", query=query, count=len(results), reason=decision.reason)
    envelope: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "backend": "anticipation_cache",
        "query": query,
        "query_id": "",
        "count": len(results),
        "scoring_instant": instant_iso,
        "results": results,
        "anticipation": decision.as_dict(),
        "warnings": [
            "Served from the local anticipation cache at this workspace's current "
            "governed-ledger head — no store round-trip, and therefore no recall "
            "attestation. Set cache.anticipation.enabled to false for the "
            "attested path."
        ],
    }
    return json.dumps(envelope, indent=2, default=str)


def _record_anticipation_bundle(ws: str, origin: str, raw: str, head: str) -> None:
    """Store a served envelope's blocks as a bundle at *head*. Never raises.

    The producer half. Only reached when the feature is on, so the flag-off
    path pays none of the tokenization this does.
    """
    try:
        from mind_mem.prefetch import get_cache, observe_served

        envelope = json.loads(raw) if raw else None
        if not isinstance(envelope, dict):
            return
        results = envelope.get("results")
        if not isinstance(results, list) or not results:
            return
        hits = [r for r in results if isinstance(r, dict)]
        if not hits:
            return
        get_cache().record(ws, origin, hits, head=head)
        observe_served(str(envelope.get("query", "")), hits)
    except Exception as exc:  # pragma: no cover — a cache write must not break recall
        _log.warning("anticipation_cache_record_failed", origin=origin, error=str(exc))


def _recall_impl(
    query: str,
    limit: int = 10,
    active_only: bool = False,
    backend: str = "auto",
    format: str = "blocks",
    explain: bool = False,
    scoring_instant: date | str | None = None,
) -> str:
    """Claim the serve, then rank. The attesting entry for this surface.

    This surface derives its attestation and writes its ledger row POST-cache
    (see :func:`_apply_attestation`), which is a placement the engine entry
    cannot reproduce from underneath: the recall-cache key omits the pipeline
    hash, so a record baked in below the cache would be replayed stale on the
    next hit. So the handler claims the serve with
    :func:`mind_mem.recall.serving_scope` and the engine calls beneath it — the
    BM25 arm of the hybrid backend, the full-scan fallback when the FTS index
    is missing — rank without attesting. One serve, one row, derived where the
    live pipeline config is visible.

    The scope wraps the *whole* implementation rather than only the retrieval
    call, because the early return on an anticipation-cache hit is also a
    serve this handler owns, and it deliberately records nothing (see
    :func:`_anticipation_envelope`).
    """
    from mind_mem.recall import serving_scope

    with serving_scope():
        return _recall_impl_ranked(
            query,
            limit=limit,
            active_only=active_only,
            backend=backend,
            format=format,
            explain=explain,
            scoring_instant=scoring_instant,
        )


def _recall_impl_ranked(
    query: str,
    limit: int = 10,
    active_only: bool = False,
    backend: str = "auto",
    format: str = "blocks",
    explain: bool = False,
    scoring_instant: date | str | None = None,
) -> str:
    """Core recall implementation shared by recall() and hybrid_search().

    v3.2.1: when ``cache.redis_url`` is configured in ``mind-mem.json``
    (or the in-process LRU fallback is enabled — which is the default),
    results are served from :mod:`mind_mem.recall_cache` when a prior
    identical query hit within the TTL window. Governance events
    (``propose_update`` / ``approve_apply`` / ``rollback_proposal``)
    invalidate the namespace-wide cache.

    v3.3.0 Tier 3 #7: ``format="bundle"`` returns the structured
    :class:`~mind_mem.evidence_bundle.EvidenceBundle` shape instead of
    raw blocks — pre-digested facts / relations / timeline / entities
    for answerer co-design. Default is ``"blocks"`` so existing callers
    see no behavioural change.

    ``scoring_instant`` is the UTC date the recency layer scores against. It is
    resolved once here, threaded into the retrieval legs, folded into the cache
    key (two instants are two different answers) and bound into the recall
    attestation so the run is replayable. ``None`` resolves to today in UTC.

    v5.0.2 — the governed-ledger head is folded into the cache key alongside the
    instant, so a cache entry belongs to the corpus state it was computed
    against. See :func:`mind_mem.recall_cache.make_cache_key`: before this the
    entry outlived the corpus and the attestation stamped the *new* anchor onto
    the *old* answer whenever a write landed through a door that does not call
    ``_invalidate_recall_cache`` — the CLI, the HTTP transport, the apply
    engine, federation replication. The same head is the generation key of the
    anticipation cache below, so the two agree on what "this corpus" means.
    """
    if not isinstance(query, str):
        return json.dumps({"error": "query must be a string"})
    if len(query) > _MAX_QUERY_LEN:
        return error_envelope(
            f"query must be ≤{_MAX_QUERY_LEN} characters",
            ErrorCode.RECALL_QUERY_TOO_LONG,
        )
    if format not in ("blocks", "bundle"):
        return json.dumps({"error": f"format must be 'blocks' or 'bundle', got {format!r}"})
    try:
        resolved_instant = resolve_scoring_instant(scoring_instant)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid scoring_instant: {exc}"})
    instant_iso = format_scoring_instant(resolved_instant)
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    # v3.2.1 — cache wrap. The cache wrapper short-circuits straight
    # to the cached envelope when the key hits, so everything below
    # (limits, timeout, backend selection, telemetry) only fires on
    # cache misses. Opt-out: set ``cache.enabled: false`` in
    # ``mind-mem.json``. Default is enabled.
    from mind_mem.recall_cache import cached_recall

    _raw_config = _load_config(ws)
    _cache_cfg = _raw_config.get("cache", {}) if isinstance(_raw_config, dict) else {}

    def _inner(query, limit, backend, active_only, **kwargs):
        return _recall_impl_uncached(
            query,
            limit=limit,
            active_only=active_only,
            backend=backend,
            scoring_instant=resolved_instant,
        )

    # Attribution tracing bypasses the recall cache. A cache HIT runs none of
    # the retrieval features, so a trace replayed out of the cached envelope
    # would claim graph_expand / entity_prefetch fired on a request where they
    # never ran — the exact "stale evidence presented as this run's" failure
    # the attestation is derived post-cache to avoid. Diagnostics that lie are
    # worse than no diagnostics, so the trace flag (default OFF) buys the
    # measurement at the price of the cache.
    _trace_on = _trace_attribution_enabled(_raw_config)
    # The corpus coordinate of the cache key: the governed-ledger head, read
    # once here and handed to both consumers below. It is the same value
    # ``_apply_attestation`` binds as ``index_anchor``, read through the same
    # resolver, so the cached answer and the attested corpus state can never be
    # two different opinions of "which corpus is this".
    _index_anchor = _resolve_chain_head(ws)

    # Group J — the anticipation cache, consulted BEFORE the store round-trip.
    # Off by default; the probe is a dict lookup on the config already loaded
    # above, so a workspace that has not opted in pays no syscall, no parse and
    # no per-hit work for the feature's presence. Attribution tracing takes the
    # same exemption as the recall cache, and for the same reason: a locally
    # answered recall ran none of the retrieval features a trace would claim.
    from mind_mem.prefetch import anticipation_enabled

    _anticipation_on = anticipation_enabled(_raw_config) and not _trace_on
    # ``format="bundle"`` never takes the local answer. The early return below
    # skips the post-cache stages, and the bundle re-shaping is one of them, so
    # serving here would hand a bundle client the raw blocks envelope with no
    # facts / relations / timeline — the exact shape confusion
    # ``tests/test_recall_format_cache_isolation.py`` exists to prevent, arriving
    # through a different door. Falling through costs one round-trip and is
    # always correct, which is the trade this whole module makes everywhere else.
    if _anticipation_on and format == "blocks":
        _anticipated = _anticipation_envelope(ws, query, limit, _raw_config, _index_anchor, instant_iso)
        if _anticipated is not None:
            return _anticipated

    if isinstance(_cache_cfg, dict) and _cache_cfg.get("enabled", True) and not _trace_on:
        raw = cached_recall(
            _inner,
            query,
            limit=limit,
            backend=backend,
            active_only=active_only,
            config=_raw_config,
            ttl_seconds=int(_cache_cfg.get("ttl_seconds", 3600)),
            scoring_instant=instant_iso,
            index_anchor=_index_anchor,
        )
    else:
        raw_result = _inner(query, limit=limit, active_only=active_only, backend=backend)
        raw = str(raw_result) if raw_result is not None else ""

    # ``format`` is a PRESENTATION choice over one retrieval, so it is applied
    # POST-cache — the same rail the attestation and explain blocks below run
    # on, and for the same reason. ``recall_cache.make_cache_key`` derives the
    # key from (query, namespace, limit, backend, active_only, scoring_instant)
    # only; ``format`` is not in it and cannot be added from here. Converting
    # inside the cached region therefore stored one caller's chosen shape under
    # a key the other shape also hashes to, and the next caller was served the
    # wrong envelope for the whole TTL window — a bundle client getting a raw
    # blocks envelope with no facts/relations/timeline, or the reverse.
    # Deriving the bundle here keeps exactly one shape (blocks) in the cache
    # and re-derives the other per request.
    if format == "bundle" and raw:
        raw = _apply_bundle_format(query, raw)

    # v4.4.0 Finding 2 — derive the per-run recall attestation POST-cache,
    # mirroring the explain pattern below. The recall-cache key omits the
    # pipeline/config hash, so an attestation baked into the cached envelope
    # would replay a PAST run's legs_ran / config_hash / index_anchor on a
    # cache hit after config drift — presenting stale evidence as the current
    # recall's. Deriving it here (on both hit and miss) binds the attestation
    # to the CURRENT pipeline config + live index anchor every time, and keeps
    # the cached payload attestation-free.
    if raw:
        raw = _apply_attestation(raw, backend, instant_iso, query)

    # v3.11.0 Pattern 1 — apply explain annotation post-cache so that the
    # cached payload (explain-free) is not polluted and explain=True can
    # still operate on both cache-hit and cache-miss paths.
    if explain and raw:
        raw = _apply_explain(query, raw)

    # RA.1 — record the served set, strictly last. Everything above has already
    # decided and serialised the ranking, so nothing this does can reach it.
    # Default ON since 5.0.2; opt out per workspace with a literal
    # ``served_ledger.enabled: false`` in mind-mem.json.
    if raw:
        raw = _record_served_run(raw, ws)

    # Group J — the producer half. What this recall served becomes the bundle a
    # later, lexically-close query can be answered from without a round-trip,
    # and the served ids are reported to the co-retrieval predictor, which is
    # the loop the roadmap item flagged as starving (prefetch observations = 0
    # because nothing ever told it what a query resolved to). Recorded against
    # the head the answer was computed at, so a write that lands between now
    # and the next lookup retires this bundle rather than aging it out.
    if _anticipation_on and raw:
        _record_anticipation_bundle(ws, "recall", raw, _index_anchor)

    return raw


class _AttestationInput(list):
    """A list of result-hit dicts carrying the recorded ``.degraded`` marker.

    The attestation deriver reads its signals off two places on a results
    object: the ``.degraded`` attribute and per-hit provenance flags
    (``_retrieval_source`` / ``_graph_hop``) on the hit dicts. Post-cache both
    live in the recall envelope, so this tiny carrier re-presents them to
    :func:`derive_recall_attestation_for_workspace` without re-running recall.
    """

    degraded: dict | None = None


def _trace_attribution_enabled(config: Any) -> bool:
    """Is ``recall.retrieval.trace_attribution`` on for this workspace?

    The flag lives under the ``recall`` section, which is what
    :meth:`HybridBackend.from_config` hands the backend. Anything malformed
    reads as off — a diagnostic knob must never be able to fail a recall.
    """
    if not isinstance(config, dict):
        return False
    from mind_mem.retrieval_trace import is_trace_enabled

    recall_cfg = config.get("recall", {})
    return is_trace_enabled(recall_cfg if isinstance(recall_cfg, dict) else None)


def _current_vector_flags(ws: str, backend: str) -> tuple[bool, bool]:
    """Resolve the CURRENT config's ``(vector_requested, vector_available)``.

    Derived fresh from the live ``mind-mem.json`` each call so a config toggle
    (e.g. ``recall.vector_enabled``) is reflected in the attestation even on a
    cache hit — the whole point of Finding 2. A ``bm25`` request never runs the
    vector leg regardless of config. Any failure degrades to the BM25-only shape
    (both False) rather than raising — an auxiliary artifact must not break
    recall.

    The rule itself lives in :func:`mind_mem.recall.resolve_vector_flags`, which
    the serving entry uses too: one implementation, so the flags an MCP-served
    attestation binds and the flags an HTTP- or CLI-served one binds cannot
    drift apart. Only the config *loader* differs — this surface keeps the MCP
    config reader it has always used, and hands the mapping in.
    """
    from mind_mem.recall import resolve_vector_flags

    return resolve_vector_flags(ws, backend, _load_config(ws))


def _apply_bundle_format(query: str, raw_json: str) -> str:
    """Re-shape a blocks envelope into the ``format="bundle"`` envelope.

    Runs post-cache so the cached payload is always the blocks shape and the
    two formats cannot be served for one another: ``format`` is not part of
    the recall-cache key, so a bundle built inside the cached region is stored
    under a key a ``format="blocks"`` request hashes to identically.

    Applied *before* the attestation / explain / served-ledger blocks so their
    existing behaviour on a bundle is unchanged — each of them already inspects
    the envelope for a ``results`` list and no-ops on the bundle shape.

    Failure degrades to the blocks envelope rather than raising: the caller
    asked a question and an answer in the wrong shape beats no answer.
    """
    try:
        from mind_mem.evidence_bundle import build_bundle

        parsed = json.loads(raw_json)
        results = parsed.get("results", []) if isinstance(parsed, dict) else []
        bundle = build_bundle(query, results)
        return json.dumps(bundle.to_dict(), default=str)
    except Exception as exc:  # pragma: no cover — fallback to blocks
        _log.warning("recall_bundle_format_failed", error=str(exc))
        return raw_json


def _apply_attestation(raw_json: str, backend: str, scoring_instant: str, query: str) -> str:
    """Derive the recall attestation from *raw_json* + live config, inject it.

    *scoring_instant* is the instant the run **actually scored with**, passed in
    rather than re-resolved: re-reading it here would let the record disagree
    with the run it attests across the cache boundary — the same staleness class
    Finding 2 fixed for ``config_hash``.

    *query* is threaded in the same way, and read from the argument rather than
    from ``envelope["query"]``. Not because the two can disagree today — the
    cache key digests the query text (``recall_cache.make_cache_key``), so a
    hit is by construction the same question — but because the argument IS the
    run's input while the envelope field is a serialized copy of it, and
    deriving a hash-bound value from a re-parsed copy adds a place the two can
    drift for no benefit. ``scoring_instant`` is passed for the same reason.
    Only the :func:`~mind_mem.recall_attestation.query_hash` is bound; the text
    never enters the record.

    Runs post-cache (both cache-hit and cache-miss paths). Rebuilds the recorded
    run signals from the envelope (per-hit provenance + the ``degraded`` marker),
    resolves the CURRENT pipeline config hash / index anchor / vector flags, and
    stamps ``envelope["attestation"]`` with a freshly derived
    :class:`RecallAttestation`. Never touches the block store. Failure to derive
    must never break recall — it is logged and the envelope returned unchanged.
    """
    try:
        from mind_mem.recall_attestation import derive_recall_attestation_for_workspace

        envelope = json.loads(raw_json)
        if not isinstance(envelope, dict) or "results" not in envelope:
            # Not a blocks-shaped recall envelope (e.g. format="bundle"): skip.
            return raw_json
        results = envelope.get("results")
        if not isinstance(results, list):
            return raw_json
        ws = _workspace()
        carrier = _AttestationInput(results)
        degraded = envelope.get("degraded")
        if isinstance(degraded, dict):
            carrier.degraded = degraded
        vector_requested, vector_available = _current_vector_flags(ws, backend)
        attestation = derive_recall_attestation_for_workspace(
            carrier,
            ws,
            vector_requested=vector_requested,
            vector_available=vector_available,
            query=query,
            scoring_instant=scoring_instant,
        )
        envelope["attestation"] = attestation.to_dict()
        return json.dumps(envelope, indent=2, default=str)
    except Exception as exc:  # pragma: no cover — defensive; recall must not fail on attestation
        _log.warning("recall_attestation_apply_failed", error=str(exc))
        return raw_json


def _record_served_run(raw_json: str, ws: str) -> str:
    """Append this run to the served-set ledger (RA.1). Default ON since 5.0.2.

    Runs **after** ``recall()`` has returned and after the envelope is
    serialised — the last thing ``_recall_impl`` does. That placement is the
    rail, not a preference: the ranking is already fixed and written down, so a
    ledger that will later carry serve counts cannot feed any of them back into
    the run that produced them. The import is function-local for the same
    reason — a module-level one would widen this module's eager-import closure
    and put the ledger a static hop from the scoring path's package.

    Every field comes from the attestation this run already published, never
    re-resolved: re-reading the pipeline hash or the index anchor here would let
    the ledger row disagree with the record it is supposed to join to. The ids
    are read with the attestation's own :func:`_served_ids`, so the row's
    ``served_digest`` cross-check inside ``append_served_run`` is structural
    rather than hopeful.

    Writes no block, so the store's admission gate is untouched. Failure must
    never break recall — the envelope is returned regardless, but it is no
    longer returned *silently*: the attestation carries ``served_seq`` (the row
    this run is recorded as) or ``served_seq: null`` with a ``ledger_error``
    saying why there is none. A client holding this envelope can therefore tell
    "never recorded" from "row removed", which is the difference between the
    ledger being a record and being a hope.

    Returns the envelope to publish — the input string unchanged when there is
    no attestation to stamp.
    """
    try:
        from mind_mem.recall_attestation import _served_ids
        from mind_mem.served_ledger import attach_served_run

        envelope = json.loads(raw_json)
        if not isinstance(envelope, dict):
            return raw_json
        attestation = envelope.get("attestation")
        results = envelope.get("results")
        if not isinstance(attestation, dict) or not isinstance(results, list):
            # No attestation (format="bundle", or a derivation that failed):
            # there is no record to join to, so there is nothing to record and
            # nothing to stamp the outcome onto.
            return raw_json
        envelope["attestation"] = attach_served_run(attestation, ws, ids=_served_ids(results))
        return json.dumps(envelope, indent=2, default=str)
    except Exception as exc:  # pragma: no cover — defensive; recall must not fail on the ledger
        _log.warning("served_ledger_append_failed", error=str(exc))
        return raw_json


def _apply_explain(query: str, raw_json: str) -> str:
    """Parse *raw_json*, inject ``_explain`` on every hit, re-serialize.

    This runs post-cache so the cached payload stays explain-free and
    explain=True works on both cache-hit and cache-miss paths. The
    workspace handle is threaded through so that
    ``_explain.staleness_penalty`` surfaces persisted lineage-staleness
    values from ``block_staleness`` (v3.12 Theme C).
    """
    try:
        from mind_mem._recall_detection import detect_query_type
        from mind_mem._recall_explain import attach_explain

        envelope = json.loads(raw_json)
        if not isinstance(envelope, dict):
            return raw_json
        results = envelope.get("results")
        if not isinstance(results, list) or not results:
            return raw_json
        intent_match = detect_query_type(query)
        attach_explain(results, intent_match=intent_match, workspace=_workspace())
        return json.dumps(envelope, indent=2, default=str)
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("recall_explain_injection_failed", error=str(exc))
        return raw_json


def _recall_impl_uncached(
    query: str,
    limit: int = 10,
    active_only: bool = False,
    backend: str = "auto",
    scoring_instant: date | None = None,
) -> str:
    """The original recall body, now callable as the cache-miss branch of ``_recall_impl``.

    ``scoring_instant`` arrives already resolved from ``_recall_impl`` and is
    threaded into every leg, so all three backends score against one instant.
    """
    ws = _workspace()
    limits = _get_limits(ws)
    limit = max(1, min(limit, limits["max_recall_results"]))
    timeout_seconds = limits.get("query_timeout_seconds", QUERY_TIMEOUT_SECONDS)
    recall_start = time.monotonic()
    if backend not in ("auto", "bm25", "hybrid"):
        backend = "auto"
    warnings: list[str] = []
    config_warnings: list[str] = []
    used_backend = "scan"
    results: list = []
    hybrid_degraded: dict | None = None
    hybrid_trace: dict | None = None

    if backend in ("hybrid", "auto"):
        try:
            from mind_mem.hybrid_recall import HybridBackend, validate_recall_config

            config = _load_config(ws)
            recall_cfg = config.get("recall", {})
            if not isinstance(recall_cfg, dict):
                recall_cfg = {}
            schema_errors = validate_recall_config(recall_cfg)
            if schema_errors:
                config_warnings = schema_errors
                _log.warning("recall_config_errors", errors=schema_errors)
            hb = HybridBackend.from_config(config)
            results = hb.search(query, ws, limit=limit, active_only=active_only, scoring_instant=scoring_instant)
            used_backend = "hybrid"
            # Surface an in-band degradation marker: when the vector leg was
            # unavailable / timed out / failed, ``search`` returns BM25-only
            # results tagged with ``.degraded`` so a caller can tell the
            # "hybrid" label did NOT mean a two-leg fusion this time.
            hybrid_degraded = getattr(results, "degraded", None)
            # Per-feature attribution, when the operator opted in: which of the
            # conditional retrieval features actually fired on THIS request and
            # what each one added. ``None`` unless
            # ``recall.retrieval.trace_attribution`` is on.
            hybrid_trace = getattr(results, "trace", None)
        except ImportError:
            if backend == "hybrid":
                warnings.append("Hybrid backend unavailable — falling back to BM25.")
        except sqlite3.OperationalError as exc:
            if _is_db_locked(exc):
                return _sqlite_busy_error()
            raise
        except (OSError, ValueError, KeyError) as exc:
            _log.warning("recall_hybrid_failed", query=query, error=str(exc))
            if backend == "hybrid":
                warnings.append(f"Hybrid search failed — falling back to BM25: {exc}")

    if used_backend != "hybrid":
        try:
            if os.path.isfile(fts_db_path(ws)):
                results = fts_query(ws, query, limit=limit, active_only=active_only, scoring_instant=scoring_instant)
                used_backend = "sqlite"
            else:
                results = recall_engine(ws, query, limit=limit, active_only=active_only, scoring_instant=scoring_instant)
                used_backend = "scan"
                warnings.append("FTS5 index not found — using full scan. Run 'reindex' tool for faster queries.")
        except sqlite3.OperationalError as exc:
            if _is_db_locked(exc):
                return _sqlite_busy_error()
            raise

    # Admissibility: this tool reaches the hybrid / FTS legs directly, so it
    # does not pass through ``recall._apply_post_filters``. One funnel per
    # public surface, so no backend leg can be the one that leaks. The legs
    # themselves filter before fusion; this is the surface-level backstop.
    if results:
        from mind_mem._recall_core import _withhold_inadmissible

        results = _withhold_inadmissible(list(results), ws, status_key="status", leg="mcp")

    recall_elapsed = time.monotonic() - recall_start
    if recall_elapsed > timeout_seconds:
        _log.warning(
            "query_timeout_exceeded",
            elapsed=round(recall_elapsed, 2),
            limit=timeout_seconds,
            backend=used_backend,
        )
        warnings.append(f"Query exceeded timeout ({round(recall_elapsed, 1)}s > {timeout_seconds}s). Results may be incomplete.")

    # Surface the pgvector-degradation label (audit findings 1b + 7): when the
    # Postgres store served BM25-only because its ``embedding`` column is
    # un-backfilled or the embedder was unavailable, every hit carries
    # ``_retrieval_source == "bm25_fallback"``. The MCP ``warnings`` array is
    # the only surface a caller sees, so lift the degradation into it instead
    # of letting a "hybrid" backend label imply a two-leg fusion that never
    # happened.
    if any(isinstance(r, dict) and r.get("_retrieval_source") == "bm25_fallback" for r in results):
        warnings.append(
            "Vector recall degraded to BM25-only: the pgvector embedding column "
            "is empty (run the 'reindex' tool / backfill_embedding) or the "
            "embedder is unavailable. Results are BM25-only, not hybrid."
        )

    # Per-run recall attestation is derived POST-cache in ``_recall_impl``
    # (see ``_apply_attestation``), NOT here: the cache key omits the pipeline/
    # config hash, so an attestation embedded in this (cached) envelope would be
    # replayed stale on a later cache hit after config drift. Keeping the cached
    # payload attestation-free — exactly as it is explain-free — means the
    # surfaced attestation always reflects the CURRENT pipeline, with the vector
    # flags resolved from the live config at derivation time.
    try:
        from mind_mem.calibration import make_query_id

        query_id = make_query_id(query)
    except ImportError:
        query_id = ""

    metrics.inc("mcp_recall_queries")
    _log.info("mcp_recall", query=query, backend=used_backend, results=len(results))

    envelope: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "backend": used_backend,
        "query": query,
        "query_id": query_id,
        "count": len(results),
        "scoring_instant": format_scoring_instant(resolve_scoring_instant(scoring_instant)),
        "results": results,
    }
    # Per-feature attribution (opt-in): which conditional retrieval features
    # fired on this request and what each contributed. Sits beside ``degraded``
    # as the run's other in-band self-description. Absent by default.
    if hybrid_trace:
        envelope["trace"] = hybrid_trace
    # In-band degradation marker (local hybrid path): a "hybrid" backend that
    # actually served BM25-only because the vector leg was unavailable / timed
    # out / failed. Silent degradation is the bug — make it a first-class,
    # machine-readable envelope field, not just a log line.
    if hybrid_degraded:
        envelope["degraded"] = hybrid_degraded
        warnings.append(
            f"Recall degraded to BM25-only: the {hybrid_degraded.get('leg', 'vector')} leg "
            f"was not used (reason: {hybrid_degraded.get('reason', 'unknown')}). "
            "Results are BM25-only, not hybrid."
        )
    # NOTE: the runtime recall attestation is injected into this envelope
    # POST-cache by ``_apply_attestation`` (Finding 2) — deliberately not here,
    # so the cached payload carries no stale attestation.
    if warnings:
        envelope["warnings"] = warnings
    if config_warnings:
        envelope["config_warnings"] = config_warnings
    if not results:
        envelope["message"] = "No matching blocks found. Try broader terms or check workspace."
    return json.dumps(envelope, indent=2, default=str)


@mcp_tool_observe
def recall(
    query: str,
    limit: int = 10,
    active_only: bool = False,
    backend: str = "auto",
    explain: bool = False,
    scoring_instant: str | None = None,
) -> str:
    """Search across all memory files with ranked retrieval.

    When ``explain=True`` every hit gains an ``_explain`` field containing
    the score decomposition (bm25, vector, rrf_rank, governance_boost,
    intent_match, staleness_penalty, final).  Omitted by default to keep
    the payload compact.

    ``scoring_instant`` is an ISO-8601 UTC date (``"YYYY-MM-DD"``) pinning the
    recency layer — the recency ramp, the calibration window and the temporal
    filter. Recall is deterministic given (corpus, config, scoring_instant), so
    passing the instant from a previous run's attestation replays that run
    exactly. Omit it for today in UTC.
    """
    return _recall_impl(
        query,
        limit=limit,
        active_only=active_only,
        backend=backend,
        explain=explain,
        scoring_instant=scoring_instant,
    )


@mcp_tool_observe
def pack_recall_budget(
    query: str,
    max_tokens: int = 2000,
    limit: int = 20,
    scoring_instant: str = "",
    model: str = "",
) -> str:
    """Run a recall, then pack the result list under a token budget.

    The recall underneath is the ranked pipeline, so ``scoring_instant`` (an
    ISO-8601 UTC date, empty = today in UTC) pins its recency layer. Packing
    itself is a pure function of the ranked list.

    With ``v4.multi_modal`` on, results are priced by
    :func:`mind_mem.multi_modal.pack_cost` instead of by excerpt length. A
    text result costs exactly the same either way; an image is charged its
    tile cost and an audio clip its duration, because charging either by
    the length of its caption understates it by two orders of magnitude and
    silently overfills the window it was asked to respect. Off by default,
    and the cost function stays deterministic on both sides of the flag.

    With ``v4.context_budget`` on, ``model`` names the model this pack is
    being assembled for and the budget is sized to that model's REAL
    context window: a ``max_tokens`` larger than the window is clamped to
    it, because a pack that cannot be sent is not a pack. The decision is
    reported in a ``context_budget`` section beside the existing ``budget``
    integer, which keeps its meaning (the ceiling the pack ran under). A model id we
    have not verified a window for is NOT clamped — it is reported as
    ``model_known: false`` and the caller's number stands. Quietly sizing
    an unknown model to an assumed 32 K was the old behaviour of the
    lookup, and it is wrong in both directions: it throws away 84% of a
    200 K window, or overflows a smaller one, and in neither case does
    anyone find out. See :func:`mind_mem.tracking.resolve_pack_budget`.

    ``model`` is inert with the flag off — the budget, the pack and the
    returned JSON are then byte-for-byte what they were before the
    parameter existed.
    """
    from mind_mem.cognitive_forget import pack_to_budget
    from mind_mem.multi_modal import flag_enabled as _multimodal_enabled
    from mind_mem.multi_modal import pack_cost

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})
    if max_tokens <= 0 or max_tokens > 1_000_000:
        return json.dumps({"error": "max_tokens must be in [1, 1_000_000]"})
    if limit < 1 or limit > 500:
        return json.dumps({"error": "limit must be in [1, 500]"})

    raw = json.loads(_recall_impl(query, limit=limit, scoring_instant=scoring_instant or None))
    attestation: dict[str, Any] | None = None
    if isinstance(raw, dict):
        results = raw.get("results", []) or []
        attestation = raw.get("attestation")
    elif isinstance(raw, list):
        results = raw
    else:
        results = []

    # None keeps the char-count estimator that has always priced this pack,
    # so the flag-off call is unchanged down to the token.
    cost_fn = pack_cost if _multimodal_enabled(ws) else None

    # v4.context_budget: size the pack to the target model's real window.
    # Off, `budget` stays None and `effective_max` is the caller's number,
    # so the pack below is the one this tool has always produced. It is
    # reported under `context_budget`, NOT `budget` — `PackedBudget.as_dict`
    # already ships `budget` as the ceiling integer, and overwriting an int
    # with a dict would break every existing reader of this envelope.
    budget: dict[str, Any] | None = None
    effective_max = int(max_tokens)
    if _context_budget_enabled(ws):
        from mind_mem.tracking import resolve_pack_budget

        budget = resolve_pack_budget(int(max_tokens), model)
        # HONOUR the resolved budget. It used to be computed, reported, and
        # then discarded -- `effective_max` was reassigned the raw request --
        # so a caller asking for a pack larger than the model's context
        # window got told it had been clamped while the packer went ahead and
        # built the oversized pack anyway. A budget that is reported but not
        # applied is worse than none: it reads as a guarantee.
        #
        # `resolve_pack_budget` only ever LOWERS the number, and only for a
        # model whose window is in the verified table -- an unknown model is
        # left at the request rather than clamped to a guess. This whole leg
        # is behind `v4.context_budget`, so flag-off packing is unchanged.
        effective_max = int(budget["effective_max_tokens"])

    try:
        packed = pack_to_budget(results, max_tokens=effective_max, cost_fn=cost_fn)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    # v4.retrieval_metrics: leave a receipt so the feedback path can price
    # what was referenced against what was packed. Keyed by the query
    # FINGERPRINT, which is the clock-free half of a query id — computing a
    # full `make_query_id` here would put a clock read on the pack path.
    if _retrieval_metrics_enabled(ws):
        try:
            from mind_mem.calibration import query_fingerprint
            from mind_mem.tracking import default_pack_receipts, pack_receipt_from_included

            default_pack_receipts().record(pack_receipt_from_included(query_fingerprint(query), packed.included, packed.tokens_used))
        except Exception as exc:  # pragma: no cover - telemetry must never fail a pack
            _log.debug("pack_receipt_skipped", error=str(exc))

    # Group I item 2: sufficiency over the PACKED list (did what fit the
    # budget deliver enough for this query class), with the pre-pack
    # score alongside to expose packing loss. Absent unless Stage 3.1
    # credits are on — flag-off output is byte-identical.
    sufficiency: dict[str, Any] | None = None
    try:
        from mind_mem.intent_router import get_router
        from mind_mem.retrieval_graph import recall_sufficiency

        intent = get_router(workspace=ws).classify(query).intent
        pre = recall_sufficiency(results, intent)
        if pre is not None:
            sufficiency = recall_sufficiency(packed.included, intent) or {
                "score": 0.0,
                "effective_hits": 0.0,  # nothing fit: maximally starved
                "demand": pre["demand"],
                "intent_type": pre["intent_type"],
            }
            sufficiency["pre_pack_score"] = pre["score"]
    except Exception as exc:
        _log.debug("pack_sufficiency_skipped", error=str(exc))

    return json.dumps(
        {
            "query": query,
            "included": packed.included,
            "dropped": packed.dropped,
            **packed.as_dict(),
            **({"sufficiency": sufficiency} if sufficiency else {}),
            **({"context_budget": budget} if budget is not None else {}),
            # The record for the RECALL underneath the pack, surfaced rather
            # than dropped. It commits to the ranking the recall served, which
            # is ``included`` + ``dropped`` — packing is a budget decision
            # taken after the fact, and re-deriving a record over ``included``
            # alone would attest a ranking the engine never produced.
            "attestation": attestation,
            "_schema_version": "1.0",
        },
        indent=2,
        default=str,
    )


@mcp_tool_observe
def recall_with_axis(
    query: str,
    axes: str = "lexical,semantic",
    weights: str = "",
    limit: int = 10,
    active_only: bool = False,
    adversarial: bool = False,
    allow_rotation: bool = True,
    scoring_instant: str = "",
) -> str:
    """Axis-aware recall under the Observer-Dependent Cognition model.

    Every axis runs the ranked recall pipeline, so ``scoring_instant`` (an
    ISO-8601 UTC date, empty = today in UTC) pins the recency layer here too.
    Without it the axes would each re-resolve their own "today", and a
    multi-axis observation could straddle a UTC midnight.
    """
    from mind_mem.axis_recall import recall_with_axis as _axis_recall
    from mind_mem.observation_axis import AxisWeights, ObservationAxis

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    _MAX_ARG_LEN = 1024
    _MAX_TOKENS = 16
    _MAX_LIMIT = 500

    if len(axes) > _MAX_ARG_LEN or len(weights) > _MAX_ARG_LEN:
        return json.dumps({"error": f"axes/weights args must be ≤{_MAX_ARG_LEN} chars"})
    if limit < 1 or limit > _MAX_LIMIT:
        return json.dumps({"error": f"limit must be in [1, {_MAX_LIMIT}]"})

    axis_tokens = [tok.strip() for tok in axes.split(",") if tok.strip()]
    if not axis_tokens:
        return json.dumps({"error": "axes must include at least one axis name"})
    if len(axis_tokens) > _MAX_TOKENS:
        return json.dumps({"error": f"axes list must contain ≤{_MAX_TOKENS} entries"})
    try:
        allowed = {ObservationAxis.from_str(tok) for tok in axis_tokens}
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    if weights.strip():
        weight_entries = [kv for kv in weights.split(",") if kv.strip()]
        if len(weight_entries) > _MAX_TOKENS:
            return json.dumps({"error": f"weights list must contain ≤{_MAX_TOKENS} entries"})
        weight_map: dict[str, float] = {}
        for kv in weight_entries:
            kv = kv.strip()
            if "=" not in kv:
                return json.dumps({"error": f"weight entry must be axis=value, got {kv!r}"})
            axis_name, value = kv.split("=", 1)
            try:
                weight_map[axis_name.strip()] = float(value.strip())
            except ValueError:
                return json.dumps({"error": f"weight for {axis_name!r} is not numeric: {value!r}"})
        try:
            parsed_weights = AxisWeights.from_mapping(weight_map)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        effective: dict[str, float] = {}
        for axis in allowed:
            effective[axis.value] = parsed_weights.as_dict().get(axis.value, 0.0)
        weight_obj = AxisWeights.from_mapping(effective)
    else:
        weight_obj = AxisWeights.uniform(allowed)

    try:
        result = _axis_recall(
            ws,
            query,
            weights=weight_obj,
            limit=limit,
            active_only=active_only,
            adversarial=adversarial,
            allow_rotation=allow_rotation,
            # Every axis is a ranked recall pass, so they must all score
            # against one instant — otherwise a multi-axis observation can
            # straddle a UTC midnight and fuse two differently-dated rankings.
            recall_kwargs={"scoring_instant": scoring_instant} if scoring_instant else None,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    envelope = {
        "query": query,
        "results": result["results"],
        "weights": result["weights"],
        "rotated": result["rotated"],
        "diversity": result["diversity"],
        "attempts": result["attempts"],
        # The orchestrator attests the FUSED answer, not its per-axis passes:
        # what this tool served is the merged ranking below, and a record per
        # pass would name candidate sets no caller was handed.
        "attestation": result.get("attestation"),
        "_schema_version": "1.0",
    }
    return json.dumps(envelope, indent=2, default=str)


@mcp_tool_observe
def hybrid_search(
    query: str,
    limit: int = 10,
    active_only: bool = False,
    explain: bool = False,
    scoring_instant: str = "",
) -> str:
    """Hybrid BM25+Vector recall with RRF fusion.

    When ``explain=True`` every hit gains an ``_explain`` field containing
    the score decomposition.  See ``recall`` for the full field description.

    ``scoring_instant`` pins the recency layer to an ISO-8601 UTC date so the
    run is replayable; empty means today in UTC. Same seam as ``recall`` —
    without it a caller on this surface cannot reproduce a previous ranking.

    .. deprecated::
        Use ``recall(backend="hybrid")`` instead. This tool will be removed in a
        future release.
    """
    import warnings

    warnings.warn(
        "hybrid_search is deprecated. Use recall(backend='hybrid') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raw = _recall_impl(
        query,
        limit=limit,
        active_only=active_only,
        backend="hybrid",
        explain=explain,
        scoring_instant=scoring_instant or None,
    )
    try:
        envelope = json.loads(raw)
        envelope["_deprecation_notice"] = "hybrid_search is deprecated. Use recall with backend='hybrid' instead."
        return json.dumps(envelope, indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw


def _kind_neighbours(ws: str, block_id: str, kind: str, limit: int) -> dict | None:
    """The ``v4.hnsw_kind_index`` neighbourhood, or ``None`` to fall through.

    ``None`` means "this tool behaves exactly as it did before": the flag is
    OFF, the v4 surface is absent, or the workspace has no registered
    embedding for this block. Falling through rather than erroring keeps the
    default answer available to a caller who passed ``kind`` on a workspace
    that was never backfilled.

    ONE quiet flag read, before any database work, so the OFF path parses no
    config twice, opens nothing and logs nothing.

    Every returned id is re-checked against the LIVE corpus admission set.
    The registered partition was admission-filtered when it was written, but
    a block quarantined since then would still have its row -- an index that
    has outrun the corpus is ordinary, and the fix is to resolve against the
    corpus rather than to trust the cache.
    """
    try:
        from mind_mem.v4.feature_flags import is_enabled_quiet

        if not is_enabled_quiet("hnsw_kind_index"):
            return None
        from mind_mem.v4.hnsw_kind_index import get_block_embedding, knn_by_kind
    except Exception as exc:  # noqa: BLE001 - v4 surface absent is a fall-through
        _log.debug("find_similar_kind_leg_unavailable", error=str(exc))
        return None

    try:
        query = get_block_embedding(ws, block_id)
        if not query:
            return None
        # limit + 1: the block is its own nearest neighbour at distance 0.
        hits = knn_by_kind(ws, kind, query, k=limit + 1)
    except Exception as exc:  # noqa: BLE001 - never take down the default leg
        _log.warning("find_similar_kind_leg_failed", block_id=block_id, kind=kind, error=str(exc))
        return None

    from mind_mem.admissibility import admissible
    from mind_mem.storage import iter_blocks

    servable = admissible(iter_blocks(ws, active_only=False))
    similar = [{"block_id": bid, "distance": round(dist, 6)} for bid, dist in hits if bid != block_id and bid in servable][:limit]
    metrics.inc("mcp_find_similar_kind_queries")
    return {
        "_schema_version": MCP_SCHEMA_VERSION,
        "source": block_id,
        "kind": kind,
        "similar": similar,
        # Named for what runs, not for the module. See the module docstring
        # of v4/hnsw_kind_index: there is no ANN backend behind this yet.
        "method": "kind-partition-brute-force-cosine",
    }


@mcp_tool_observe
def find_similar(block_id: str, limit: int = 5, kind: str = "") -> str:
    """Find blocks co-retrieved with a given block (co-occurrence, not embeddings).

    This is the one-line description agents route on, so it states the actual
    method: the ranking comes from ``block_meta.db`` co-occurrence counts, and
    a block that has never been co-retrieved returns an empty list even when
    semantically near neighbours exist. For semantic nearest-neighbour search
    use ``recall`` with ``backend="hybrid"``.

    ``kind`` switches to the v4 kind-partitioned vector neighbourhood
    (``v4.hnsw_kind_index``, default OFF): neighbours of the same block KIND
    ranked by cosine distance over the embeddings ``mm kinds backfill``
    registered. It is a **brute-force scan of the kind partition**, not an
    HNSW graph -- the module ships no ANN backend yet and the reported
    ``method`` says so, because a caller told "HNSW" would reasonably assume
    a complexity guarantee nothing here provides.
    """
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps({"error": f"Invalid block_id format: {block_id}"})
    ws = _workspace()
    limits = _get_limits(ws)
    limit = max(1, min(limit, limits["max_similar_results"]))
    if kind:
        kind_payload = _kind_neighbours(ws, block_id, kind, limit)
        if kind_payload is not None:
            return json.dumps(kind_payload, indent=2)
    try:
        from mind_mem.block_metadata import BlockMetadataManager, block_meta_db_path

        # The canonical store -- same file the recall writer records
        # co-occurrence into. This used to read ``memory/block_meta.db``,
        # which nothing writes, so "similar" was always empty.
        db_path = block_meta_db_path(ws)
        mgr = BlockMetadataManager(db_path)
        co_blocks = mgr.get_co_occurring_blocks(block_id, limit=limit)
        metrics.inc("mcp_find_similar_queries")
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "source": block_id,
                "similar": co_blocks,
                "method": "co-occurrence",
            },
            indent=2,
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "find_similar requires block_metadata module",
                "block_id": block_id,
            },
            indent=2,
        )
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("find_similar_failed", block_id=block_id, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Failed to find similar blocks. The co-occurrence index may not be initialized.",
                "block_id": block_id,
            },
            indent=2,
        )


@mcp_tool_observe
def intent_classify(query: str) -> str:
    """Show the routing strategy for a query."""
    if not isinstance(query, str) or len(query) > _MAX_QUERY_LEN:
        return json.dumps({"error": f"query must be a string of ≤{_MAX_QUERY_LEN} characters"})
    try:
        from mind_mem.intent_router import IntentRouter

        router = IntentRouter()
        result = router.classify(query)
        metrics.inc("mcp_intent_classify")
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "query": query,
                "intent": result.intent,
                "confidence": result.confidence,
                "sub_intents": result.sub_intents,
                "params": result.params,
            },
            indent=2,
        )
    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "intent_router module not available",
                "query": query,
            },
            indent=2,
        )
    except (ValueError, KeyError, AttributeError) as exc:
        _log.warning("intent_classify_failed", query=query, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Intent classification failed",
                "query": query,
            },
            indent=2,
        )


@mcp_tool_observe
def retrieval_diagnostics(last_n: int = 50, max_age_days: int = 7) -> str:
    """Pipeline diagnostics: per-stage rejection rates, intent distribution, and hard negative summary."""
    ws = _workspace()
    try:
        result = _retrieval_diag(ws, last_n=last_n, max_age_days=max_age_days)
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    # Whether the vector leg is contributing at all. An operator debugging
    # "why is recall bad" has no other way to learn that the leg is inert:
    # nothing errors, every query returns results, and the answer is still
    # labelled hybrid. Reported unconditionally -- a healthy verdict with its
    # measured spread is as useful as the warning, because it dates the check.
    from mind_mem.vector_inertness import inertness_for

    result["vector_leg"] = inertness_for(ws).as_dict()

    # Group J — the anticipation cache's own counters, on the operator surface
    # that already answers "why is recall behaving like this". Reported
    # unconditionally and for the same reason as the vector leg above: a cache
    # whose hit rate nobody can see is a cache nobody can tune, and a run of
    # zeroes is itself the answer when the feature is off or never warm. The
    # counters are process-local integers — no clock, no I/O, no store read.
    from mind_mem.prefetch import get_cache as _anticipation_cache

    result["anticipation_cache"] = _anticipation_cache().stats()

    result["_schema_version"] = MCP_SCHEMA_VERSION
    metrics.inc("mcp_retrieval_diagnostics")
    return json.dumps(result, indent=2)


@mcp_tool_observe
def prefetch(signals: str, limit: int = 5) -> str:
    """Pre-assembles likely-needed context from recent conversation signals."""
    ws = _workspace()
    signal_list = [s.strip() for s in signals.split(",") if s.strip()]
    if not signal_list:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "No signals provided. Pass comma-separated keywords.",
            }
        )

    limits = _get_limits(ws)
    limit = max(1, min(limit, limits["max_prefetch_results"]))
    # Resolved here, once, and handed to both the assembly and the record it
    # gets attested with. Left to default, the N+1 passes inside
    # ``prefetch_context`` would each read their own "today" and the
    # attestation below would read a further one — across a UTC midnight that
    # is a record naming a day none of the passes scored against.
    instant = resolve_scoring_instant(None)
    try:
        from mind_mem.recall import prefetch_context

        results = prefetch_context(ws, signal_list, limit=limit, scoring_instant=instant)
        metrics.inc("mcp_prefetch_queries")
        _log.info("mcp_prefetch", signals=signal_list, results=len(results))
        # Group J — this is the tool the roadmap item calls "idle": it
        # assembled context and then nothing consumed it. Its results now land
        # in the anticipation cache, at the current chain head, so the next
        # recall can be answered from them without a round-trip. Gated on the
        # same flag as the consumer, read from the workspace config already on
        # hand, so an opted-out workspace pays nothing for the wiring.
        from mind_mem.prefetch import anticipation_enabled, get_cache

        if anticipation_enabled(_load_config(ws)):
            hits = [r for r in results if isinstance(r, dict)]
            if hits:
                get_cache().record(ws, "prefetch", hits, head=_resolve_chain_head(ws))
        # This tool is a door: it hands assembled block content back to a
        # caller, so it owes the same proof every other door owes. It cannot
        # inherit one from underneath — ``prefetch_context`` fans its signals
        # out over a thread pool and calls the ENGINE per signal, so no single
        # inner run describes what was served. The serve is the merged,
        # deduplicated list this function returns, and that is what is attested
        # here: one record, one row, over the answer the caller actually got.
        from mind_mem.recall import attest_and_record

        attestation = attest_and_record(ws, ",".join(signal_list), results, scoring_instant=instant)
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "signals": signal_list,
                "count": len(results),
                "results": results,
                "attestation": attestation,
            },
            indent=2,
            default=str,
        )
    except Exception:
        import traceback

        _log.warning("prefetch_failed", signals=signal_list, traceback=traceback.format_exc())
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Prefetch failed",
                "signals": signal_list,
            },
            indent=2,
        )


def register(mcp) -> None:
    """Wire the recall tools onto *mcp*."""
    mcp.tool(recall)
    mcp.tool(pack_recall_budget)
    mcp.tool(recall_with_axis)
    mcp.tool(hybrid_search)
    mcp.tool(find_similar)
    mcp.tool(intent_classify)
    mcp.tool(retrieval_diagnostics)
    mcp.tool(prefetch)

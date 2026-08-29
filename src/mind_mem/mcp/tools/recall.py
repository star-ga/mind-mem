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

from mind_mem.recall import recall as recall_engine
from mind_mem.retrieval_graph import retrieval_diagnostics as _retrieval_diag
from mind_mem.scoring_instant import format_scoring_instant, resolve_scoring_instant
from mind_mem.sqlite_index import _db_path as fts_db_path
from mind_mem.sqlite_index import query_index as fts_query

from ..infra.config import QUERY_TIMEOUT_SECONDS, _get_limits, _load_config
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import _is_db_locked, _sqlite_busy_error, mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")


_MAX_QUERY_LEN = 8192


def _recall_impl(
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
    """
    if not isinstance(query, str):
        return json.dumps({"error": "query must be a string"})
    if len(query) > _MAX_QUERY_LEN:
        return json.dumps({"error": f"query must be ≤{_MAX_QUERY_LEN} characters"})
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

    def _inner_with_format(query, limit, backend, active_only, **kwargs):
        raw_json = _recall_impl_uncached(
            query,
            limit=limit,
            active_only=active_only,
            backend=backend,
            scoring_instant=resolved_instant,
        )
        if format == "blocks":
            return raw_json
        # format="bundle": re-parse JSON → build_bundle → re-serialize.
        try:
            from mind_mem.evidence_bundle import build_bundle

            parsed = json.loads(raw_json)
            results = parsed.get("results", []) if isinstance(parsed, dict) else []
            bundle = build_bundle(query, results)
            return json.dumps(bundle.to_dict(), default=str)
        except Exception as exc:  # pragma: no cover — fallback to blocks
            _log.warning("recall_bundle_format_failed", error=str(exc))
            return raw_json

    if isinstance(_cache_cfg, dict) and _cache_cfg.get("enabled", True):
        raw = cached_recall(
            _inner_with_format,
            query,
            limit=limit,
            backend=backend,
            active_only=active_only,
            config=_raw_config,
            ttl_seconds=int(_cache_cfg.get("ttl_seconds", 3600)),
            scoring_instant=instant_iso,
        )
    else:
        raw_result = _inner_with_format(query, limit=limit, active_only=active_only, backend=backend)
        raw = str(raw_result) if raw_result is not None else ""

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


def _current_vector_flags(ws: str, backend: str) -> tuple[bool, bool]:
    """Resolve the CURRENT config's ``(vector_requested, vector_available)``.

    Derived fresh from the live ``mind-mem.json`` each call so a config toggle
    (e.g. ``recall.vector_enabled``) is reflected in the attestation even on a
    cache hit — the whole point of Finding 2. A ``bm25`` request never runs the
    vector leg regardless of config. Any failure degrades to the BM25-only shape
    (both False) rather than raising — an auxiliary artifact must not break
    recall.
    """
    if backend == "bm25":
        return False, False
    try:
        from mind_mem.hybrid_recall import HybridBackend

        hb = HybridBackend.from_config(_load_config(ws))
        return bool(getattr(hb, "vector_enabled", False)), bool(getattr(hb, "vector_available", False))
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("recall_attestation_vector_flags_failed", error=str(exc))
        return False, False


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
def pack_recall_budget(query: str, max_tokens: int = 2000, limit: int = 20, scoring_instant: str = "") -> str:
    """Run a recall, then pack the result list under a token budget.

    The recall underneath is the ranked pipeline, so ``scoring_instant`` (an
    ISO-8601 UTC date, empty = today in UTC) pins its recency layer. Packing
    itself is a pure function of the ranked list.
    """
    from mind_mem.cognitive_forget import pack_to_budget

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
    if isinstance(raw, dict):
        results = raw.get("results", []) or []
    elif isinstance(raw, list):
        results = raw
    else:
        results = []

    try:
        packed = pack_to_budget(results, max_tokens=int(max_tokens))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

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


@mcp_tool_observe
def find_similar(block_id: str, limit: int = 5) -> str:
    """Find blocks similar to a given block using vector similarity."""
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps({"error": f"Invalid block_id format: {block_id}"})
    ws = _workspace()
    limits = _get_limits(ws)
    limit = max(1, min(limit, limits["max_similar_results"]))
    try:
        from mind_mem.block_metadata import BlockMetadataManager

        db_path = os.path.join(ws, "memory", "block_meta.db")
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
    try:
        from mind_mem.recall import prefetch_context

        results = prefetch_context(ws, signal_list, limit=limit)
        metrics.inc("mcp_prefetch_queries")
        _log.info("mcp_prefetch", signals=signal_list, results=len(results))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "signals": signal_list,
                "count": len(results),
                "results": results,
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

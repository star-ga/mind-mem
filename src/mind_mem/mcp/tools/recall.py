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
from typing import Any

from mind_mem.observability import get_logger, metrics
from mind_mem.recall import recall as recall_engine
from mind_mem.retrieval_graph import retrieval_diagnostics as _retrieval_diag
from mind_mem.sqlite_index import _db_path as fts_db_path
from mind_mem.sqlite_index import query_index as fts_query

from ..infra.config import QUERY_TIMEOUT_SECONDS, _get_limits, _load_config
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import _is_db_locked, _sqlite_busy_error, mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace

_log = get_logger("mcp_server")


_MAX_QUERY_LEN = 8192


def _recall_impl(
    query: str,
    limit: int = 10,
    active_only: bool = False,
    backend: str = "auto",
    format: str = "blocks",
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
    """
    if not isinstance(query, str):
        return json.dumps({"error": "query must be a string"})
    if len(query) > _MAX_QUERY_LEN:
        return json.dumps({"error": f"query must be ≤{_MAX_QUERY_LEN} characters"})
    if format not in ("blocks", "bundle"):
        return json.dumps({"error": f"format must be 'blocks' or 'bundle', got {format!r}"})
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
        raw_json = _recall_impl_uncached(query, limit=limit, active_only=active_only, backend=backend)
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
        return cached_recall(
            _inner_with_format,
            query,
            limit=limit,
            backend=backend,
            active_only=active_only,
            config=_raw_config,
            ttl_seconds=int(_cache_cfg.get("ttl_seconds", 3600)),
        )
    return _inner_with_format(query, limit=limit, active_only=active_only, backend=backend)


def _recall_impl_uncached(query: str, limit: int = 10, active_only: bool = False, backend: str = "auto") -> str:
    """The original recall body, now callable as the cache-miss branch of ``_recall_impl``."""
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
            results = hb.search(query, ws, limit=limit, active_only=active_only)
            used_backend = "hybrid"
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
                results = fts_query(ws, query, limit=limit, active_only=active_only)
                used_backend = "sqlite"
            else:
                results = recall_engine(ws, query, limit=limit, active_only=active_only)
                used_backend = "scan"
                warnings.append("FTS5 index not found — using full scan. Run 'reindex' tool for faster queries.")
        except sqlite3.OperationalError as exc:
            if _is_db_locked(exc):
                return _sqlite_busy_error()
            raise

    recall_elapsed = time.monotonic() - recall_start
    if recall_elapsed > timeout_seconds:
        _log.warning(
            "query_timeout_exceeded",
            elapsed=round(recall_elapsed, 2),
            limit=timeout_seconds,
            backend=used_backend,
        )
        warnings.append(f"Query exceeded timeout ({round(recall_elapsed, 1)}s > {timeout_seconds}s). Results may be incomplete.")

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
        "results": results,
    }
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
) -> str:
    """Search across all memory files with ranked retrieval."""
    return _recall_impl(query, limit=limit, active_only=active_only, backend=backend)


@mcp_tool_observe
def pack_recall_budget(query: str, max_tokens: int = 2000, limit: int = 20) -> str:
    """Run a recall, then pack the result list under a token budget."""
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

    raw = json.loads(_recall_impl(query, limit=limit))
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

    return json.dumps(
        {
            "query": query,
            "included": packed.included,
            "dropped": packed.dropped,
            **packed.as_dict(),
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
) -> str:
    """Axis-aware recall under the Observer-Dependent Cognition model."""
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
def hybrid_search(query: str, limit: int = 10, active_only: bool = False) -> str:
    """Hybrid BM25+Vector recall with RRF fusion.

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
    raw = _recall_impl(query, limit=limit, active_only=active_only, backend="hybrid")
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

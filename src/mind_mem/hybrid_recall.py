#!/usr/bin/env python3
"""mind-mem Hybrid Recall -- BM25 + Vector + RRF fusion.

Orchestrates BM25 (via recall.py or sqlite_index.py) and vector search
(via recall_vector.py) in parallel, then fuses rankings using Reciprocal
Rank Fusion (RRF).  Falls back gracefully to BM25-only when vector
dependencies (sentence-transformers) are not installed.

Configuration (mind-mem.json):
    {
      "recall": {
        "backend": "hybrid",
        "rrf_k": 60,
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "vector_model": "all-MiniLM-L6-v2",
        "vector_enabled": false
      }
    }
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import Any

from .observability import get_logger, metrics, timed

_log = get_logger("hybrid_recall")

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_NUMERIC_POSITIVE_KEYS = ("bm25_weight", "vector_weight", "rrf_k")


def validate_recall_config(cfg: dict[str, Any]) -> list[str]:
    """Validate recall config section. Returns list of error strings (empty = valid).

    Checks that bm25_weight, vector_weight, and rrf_k are numeric and positive.
    """
    errors: list[str] = []
    for key in _NUMERIC_POSITIVE_KEYS:
        if key not in cfg:
            continue
        val = cfg[key]
        try:
            numeric = float(val)
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric, got {type(val).__name__}: {val!r}")
            continue
        if numeric <= 0:
            errors.append(f"{key} must be positive, got {numeric}")
    return errors


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    ranked_lists: list[list[dict]],
    weights: list[float],
    k: int = 60,
    id_key: str = "_id",
) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    For each document, RRF score = sum_i( weight_i / (k + rank_i) )
    where rank_i is the 1-based rank of the document in list i.

    Audit R-2 — weight semantics: ``weights`` are RAW multipliers
    applied to each list's contribution. The output ``rrf_score``
    therefore scales linearly with the weight magnitudes — passing
    ``[1.0, 1.0]`` and ``[0.5, 0.5]`` produce identical RANKINGS but
    different absolute scores. Callers comparing absolute scores
    across requests must keep weights stable; callers comparing only
    RANKINGS are unaffected.

    Audit R-3: if only ONE non-empty list is present (the others
    are empty), ``hybrid_single_list_degenerate`` is incremented on
    the metrics registry so dashboards can flag silent BM25-only or
    vector-only fallbacks.

    Audit R-1: when two lists report the same document with different
    metadata, the entry whose ``Date`` field is the most recent is
    retained rather than blindly preferring the first-seen copy.

    Args:
        ranked_lists: List of ranked result lists. Each result is a dict
            that must contain ``id_key`` for dedup.
        weights: Per-list raw weight multipliers (same length as
            ranked_lists). NOT normalized — see R-2 above.
        k: RRF smoothing constant (default 60). Higher values dampen the
            advantage of top-ranked documents.
        id_key: Dict key used to identify unique documents.

    Returns:
        Fused list sorted by descending RRF score. Each item is a copy of
        the freshest (by Date metadata) dict for that ID, with
        ``rrf_score`` and ``fusion`` fields injected.
    """
    if not ranked_lists:
        return []

    # Audit R-3: count non-empty source lists; if only one survived,
    # emit a metric so dashboards can flag degenerate fusion.
    non_empty = sum(1 for r in ranked_lists if r)
    if non_empty <= 1 and len(ranked_lists) > 1:
        try:
            from .observability import metrics as _metrics

            _metrics.inc("hybrid_single_list_degenerate")
        except Exception:  # nosec B110 — optional observability metric; import or inc failure is non-fatal
            pass

    scores: dict[str, float] = {}
    block_data: dict[str, dict] = {}

    for list_idx, results in enumerate(ranked_lists):
        w = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank_0, item in enumerate(results):
            bid = _get_block_id(item, id_key)
            scores[bid] = scores.get(bid, 0.0) + w / (k + rank_0 + 1)
            existing = block_data.get(bid)
            if existing is None:
                block_data[bid] = item
                continue
            # Audit R-1: prefer the dict whose Date metadata is more
            # recent. Date comes from frontmatter and is typically
            # ISO-8601. We compare as strings (ISO ordering is correct
            # lexicographically) and fall back to first-seen on ties
            # or when either side is missing.
            new_date = item.get("Date") or item.get("date")
            old_date = existing.get("Date") or existing.get("date")
            if isinstance(new_date, str) and isinstance(old_date, str):
                if new_date > old_date:
                    block_data[bid] = item
            elif new_date and not old_date:
                block_data[bid] = item

    # Total-order tie-break (score, block_id): equal fused scores are common
    # (RRF sums are coarse w/(k+rank)), and without the block_id secondary key
    # ties fall back to dict-insertion = BM25-vs-vector arrival order =
    # non-reproducible recall. Matches the (score, _id) discipline used
    # throughout _recall_core / _recall_reranking, so the fused order is a pure
    # function of the input ranked-list multiset.
    sorted_ids = sorted(scores, key=lambda x: (scores[x], x), reverse=True)
    fused = []
    for bid in sorted_ids:
        item = block_data[bid].copy()
        item["rrf_score"] = round(scores[bid], 6)
        item["fusion"] = "rrf"
        fused.append(item)

    return fused


def _get_block_id(item: dict, id_key: str) -> str:
    """Extract a stable block identifier from a result dict.

    Audit R-4: emit a ``rrf_fallback_id_used`` warning + metric when
    the file:line fallback path is taken, so silent ID collisions
    (two distinct blocks at the same file:line) show up in logs
    rather than producing wrong merge results.
    """
    bid = item.get(id_key)
    if bid:
        return str(bid)
    for alt in ("id", "block_id", "_id"):
        val = item.get(alt)
        if val:
            return str(val)
    fallback = f"{item.get('file', '?')}:{item.get('line', 0)}"
    try:
        from .observability import get_logger as _get_logger
        from .observability import metrics as _metrics

        _get_logger("mind_mem.hybrid_recall").warning(
            "rrf_fallback_id_used",
            fallback_id=fallback,
            advice="result dict lacks _id / id / block_id; collisions may merge distinct blocks",
        )
        _metrics.inc("rrf_fallback_id_used")
    except Exception:  # nosec B110 — best-effort warning + metric; fallback id is always returned regardless
        pass
    return fallback


# ---------------------------------------------------------------------------
# HybridBackend
# ---------------------------------------------------------------------------


class HybridBackend:
    """Orchestrates BM25 and vector search with RRF fusion.

    When vector search is unavailable (no sentence-transformers or
    ``vector_enabled`` is False), transparently falls back to BM25-only.

    Supports optional multi-query expansion: when ``query_expansion.enabled``
    is True in config, generates alternative query phrasings and fuses
    results across all variants for improved recall.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}

        # Validate numeric fields; fall back to defaults on bad values
        errors = validate_recall_config(cfg)
        if errors:
            _log.warning(
                "hybrid_config_validation_failed",
                errors=errors,
                fallback="bm25_only",
            )
            # Reset bad values to defaults so constructor doesn't raise
            for key in _NUMERIC_POSITIVE_KEYS:
                if any(key in e for e in errors):
                    cfg.pop(key, None)

        self.rrf_k: int = int(cfg.get("rrf_k", 60))
        self.bm25_weight: float = float(cfg.get("bm25_weight", 1.0))
        self.vector_weight: float = float(cfg.get("vector_weight", 1.0))
        self.vector_enabled: bool = bool(cfg.get("vector_enabled", False))
        self.vector_model: str = cfg.get("vector_model", "all-MiniLM-L6-v2")
        self._config = cfg
        self._config_errors: list[str] = errors

        # Query expansion config (opt-in: adds ~3x query latency when enabled)
        qe_cfg = cfg.get("query_expansion", {})
        if not isinstance(qe_cfg, dict):
            qe_cfg = {}
        self._query_expansion_enabled: bool = bool(qe_cfg.get("enabled", False))
        self._query_expansion_config: dict[str, Any] = qe_cfg

        # Probe vector availability once at init. When the operator
        # explicitly enabled vector recall (recall.vector_enabled=true) but
        # the backend is not importable / not serviceable, FAIL LOUD
        # (warning + metric) instead of silently degrading to BM25 — a
        # misconfigured or missing embedder must be visible, never invisible
        # (#139: silent BM25-only fallback).
        if self.vector_enabled:
            self._vector_available = self._check_vector()
            if not self._vector_available:
                _log.warning(
                    "hybrid_vector_requested_but_unavailable",
                    hint=(
                        "recall.vector_enabled=true but the vector backend is not "
                        "importable; recall is degraded to BM25-only. Install the "
                        "vector extras and verify the embedder is reachable."
                    ),
                    fallback="bm25_only",
                )
                metrics.inc("hybrid_vector_requested_but_unavailable")
        else:
            self._vector_available = False

        _log.info(
            "hybrid_backend_init",
            rrf_k=self.rrf_k,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
            vector_available=self._vector_available,
            query_expansion=self._query_expansion_enabled,
        )

    # -- capability probing ------------------------------------------------

    def _check_vector(self) -> bool:
        """Return True if recall_vector + sentence-transformers are importable."""
        try:
            from . import recall_vector  # noqa: F401

            return True
        except ImportError:
            _log.info("vector_backend_unavailable", reason="import failed")
            return False

    @property
    def vector_available(self) -> bool:
        return self._vector_available

    def _vector_deadline_seconds(self) -> float:
        """Wall-clock bound (seconds) on the parallel vector leg.

        Guarantees recall degrades to BM25-only fusion rather than
        blocking when the embedder or vector store stalls. Defaults to a
        margin above the per-request embed timeout; override via
        ``recall.vector_deadline_seconds``. Clamped to a sane range.
        """
        default = 14.0
        try:
            val = float(self._config.get("vector_deadline_seconds", default))
        except (TypeError, ValueError):
            return default
        if val <= 0:
            return default
        return max(1.0, min(val, 120.0))

    # -- search entry point -------------------------------------------------

    def search(
        self,
        query: str,
        workspace: str,
        limit: int = 10,
        active_only: bool = False,
        graph_boost: bool = False,
        retrieve_wide_k: int = 200,
        rerank: bool = True,
        _skip_auto_features: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        """Run BM25 and (optionally) vector search, fuse via RRF.

        When vector search is unavailable the method returns BM25
        results directly (no fusion overhead).

        Args:
            query: Search query string.
            workspace: Workspace root path.
            limit: Maximum results to return.
            active_only: Only return active blocks.
            graph_boost: Enable cross-reference graph boosting (BM25).
            retrieve_wide_k: Candidate pool size per backend.
            rerank: Enable BM25 reranker (passed through).
            **kwargs: Forwarded to underlying backends.

        Returns:
            Ranked list of result dicts.
        """
        if not query or not query.strip():
            return []

        # Audit R-6: detect_query_type is called from the expansion
        # path, the decomposition path, and the cross-encoder path on
        # every search() invocation. The detector itself is regex-only
        # and cheap, but the import + dispatch shows up in profiles
        # when called 3× per request. Memoize per request so the same
        # query string is classified once.
        _qt_cache: dict[str, str] = {}

        def _qt() -> str | None:
            if query in _qt_cache:
                return _qt_cache[query]
            try:
                from ._recall_detection import detect_query_type as _detect

                value = _detect(query)
            except Exception as exc:  # pragma: no cover — defensive
                _log.debug("query_type_detect_skipped", error=str(exc))
                return None
            _qt_cache[query] = value
            return value

        # --- Multi-query expansion ---
        # When enabled, expand the query into alternative phrasings and
        # run a search for each variant.  Results are fused via RRF so
        # documents matching multiple phrasings rank higher.
        # v3.3.0 Tier 2 #4: auto-enable on multi-hop/temporal queries
        # even when operator hasn't flipped ``query_expansion.enabled``,
        # unless ``query_expansion.auto_enable: false`` is set.
        # Thread-safety: _search_expanded recurses into search() with
        # _skip_auto_features=True to avoid re-entering expansion /
        # decomposition loops. Previous version mutated
        # ``self._query_expansion_enabled`` which races between
        # concurrent requests (python-reviewer 2026-04-20).
        if _skip_auto_features:
            expansion_active = False
        else:
            expansion_active = self._query_expansion_enabled
        if not expansion_active and self._query_expansion_config.get("auto_enable", True):
            qt = _qt()
            if qt in ("multi-hop", "temporal"):
                expansion_active = True
                _log.info(
                    "query_expansion_auto_enabled",
                    query_type=qt,
                    reason="v3.3.0_tier2_ambiguous_query",
                )
        if expansion_active:
            try:
                from .query_expansion import expand_queries

                expanded = expand_queries(
                    query,
                    config=self._query_expansion_config,
                )
                if len(expanded) > 1:
                    _log.info(
                        "multi_query_expansion",
                        original=query,
                        variants=len(expanded),
                    )
                    metrics.inc("query_expansion_used")
                    return self._search_expanded(
                        queries=expanded,
                        workspace=workspace,
                        limit=limit,
                        active_only=active_only,
                        graph_boost=graph_boost,
                        retrieve_wide_k=retrieve_wide_k,
                        rerank=rerank,
                        **kwargs,
                    )
            except Exception as exc:
                _log.warning(
                    "query_expansion_failed",
                    error=str(exc),
                    fallback="single_query",
                )

        # v3.3.0 Tier 1 #1 — query decomposition for multi-hop queries.
        # Split compound questions ("A after B") into independent
        # sub-queries, run retrieval on each, RRF-fuse. Same opt-out
        # shape as expansion: ``retrieval.query_decomposition.auto_enable
        # = false`` to skip.
        decomp_cfg = self._config.get("retrieval", {}).get("query_decomposition", {})
        if not isinstance(decomp_cfg, dict):
            decomp_cfg = {}
        decomp_active = False if _skip_auto_features else bool(decomp_cfg.get("enabled", False))
        if not decomp_active and decomp_cfg.get("auto_enable", True):
            if _qt() == "multi-hop":
                decomp_active = True
                _log.info(
                    "query_decomposition_auto_enabled",
                    reason="v3.3.0_tier1_multi_hop",
                )
        if decomp_active:
            try:
                from .query_planner import decompose_query

                decomposed = decompose_query(
                    query,
                    config=self._config,
                    max_subqueries=int(decomp_cfg.get("max_subqueries", 4)),
                )
                if len(decomposed) > 1:
                    _log.info(
                        "multi_query_decomposition",
                        original=query,
                        sub_queries=len(decomposed),
                    )
                    metrics.inc("query_decomposition_used")
                    return self._search_expanded(
                        queries=decomposed,
                        workspace=workspace,
                        limit=limit,
                        active_only=active_only,
                        graph_boost=graph_boost,
                        retrieve_wide_k=retrieve_wide_k,
                        rerank=rerank,
                        **kwargs,
                    )
            except Exception as exc:
                _log.warning(
                    "query_decomposition_failed",
                    error=str(exc),
                    fallback="single_query",
                )

        # Postgres workspaces fuse BM25 + pgvector SERVER-SIDE: the local
        # "BM25" leg here (recall -> PostgresRecallBackend.search) is itself
        # the store's ``hybrid_search`` (BM25 + pgvector RRF, labeled
        # ``hybrid_pgvector`` / ``bm25_fallback``). Running HybridBackend's
        # OWN second local vector leg on top would (a) double-count the
        # vector contribution and (b) drive the provider=postgres path in
        # ``search_batch`` (audit 1a). So for postgres we take the single-leg
        # local path — which is already server-side hybrid — and let the
        # cross-encoder rerank below still apply, instead of fusing twice.
        pg_server_side = isinstance(self._config, dict) and self._config.get("provider") == "postgres"
        with timed("hybrid_search"):
            if not self._vector_available or pg_server_side:
                _log.info("hybrid_bm25_only", query=query, pg_server_side=pg_server_side)
                results = self._bm25_search(
                    query,
                    workspace,
                    limit=limit,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=retrieve_wide_k,
                    rerank=rerank,
                    **kwargs,
                )
                metrics.inc("hybrid_searches_bm25_only")
                # v3.3.0 Tier 2: cross-encoder rerank also applies to
                # BM25-only deployments (previously only post-fusion).
                results = self._maybe_cross_encoder_rerank(query, results, limit)
                return results

            # Run BM25 + vector in parallel
            _log.info("hybrid_parallel_search", query=query)
            bm25_results: list[dict] = []
            vec_results: list[dict] = []

            # Manual pool lifecycle — NOT ``with ThreadPoolExecutor(...) as
            # pool``. The context manager's __exit__ calls shutdown(wait=True),
            # which re-joins a still-running vector worker even after the
            # deadline below fired — so ``timeout=`` bounded the RESULT wait
            # but NOT the wall-clock of an unbounded leg (e.g. a provider=local
            # sentence-transformers embed that hangs on model download). We
            # therefore shut the pool down with wait=False + cancel_futures so
            # recall returns at the deadline and abandons the leaked worker
            # instead of blocking on it (audit finding 4). cancel_futures drops
            # any not-yet-started task; an already-running embed thread cannot
            # be force-killed, but it no longer holds up the response.
            pool = ThreadPoolExecutor(max_workers=2)
            try:
                bm25_future: Future = pool.submit(
                    self._bm25_search,
                    query,
                    workspace,
                    limit=retrieve_wide_k,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=retrieve_wide_k,
                    rerank=False,  # defer reranking to post-fusion
                    **kwargs,
                )
                vec_future: Future = pool.submit(
                    self._vector_search,
                    query,
                    workspace,
                    limit=retrieve_wide_k,
                    active_only=active_only,
                )
                bm25_results = bm25_future.result()
                # Hard bound on the vector leg: if the embedder is cold /
                # slow / down, degrade to BM25-only fusion instead of
                # blocking the whole recall request (the vector work
                # includes an embedding HTTP round-trip, itself bounded).
                try:
                    vec_results = vec_future.result(timeout=self._vector_deadline_seconds())
                except _FutureTimeout:
                    _log.warning(
                        "hybrid_vector_leg_timeout",
                        deadline=self._vector_deadline_seconds(),
                        fallback="bm25_only",
                    )
                    vec_future.cancel()
                    vec_results = []
            finally:
                # wait=False so a hung vector leg cannot re-block the response
                # here (the whole point of the deadline above).
                pool.shutdown(wait=False, cancel_futures=True)

            _log.info(
                "hybrid_results_pre_fusion",
                bm25_count=len(bm25_results),
                vector_count=len(vec_results),
            )

            fused = rrf_fuse(
                ranked_lists=[bm25_results, vec_results],
                weights=[self.bm25_weight, self.vector_weight],
                k=self.rrf_k,
            )

            metrics.inc("hybrid_searches_fused")
            result = fused[:limit]

            # Cross-encoder reranking (post-fusion) — v3.3.0 Tier 2
            # extracted into a helper so the BM25-only early-return path
            # also benefits from auto-enable on multi-hop/temporal queries.
            result = self._maybe_cross_encoder_rerank(query, result, limit)

            # v3.3.0 Tier 1 #2 + Tier 3 #8 — multi-hop graph expansion +
            # entity prefetch. Corpus is loaded once and shared across
            # both helpers (was O(2N) disk reads, now O(N); architect +
            # python-reviewer 2026-04-20).
            corpus = self._load_corpus_if_needed(query, workspace)
            result = self._maybe_graph_expand(query, workspace, result, corpus=corpus)
            result = self._maybe_entity_prefetch(query, workspace, result, corpus=corpus)

            # v3.3.0 Tier 2 #5 — session-boundary preservation.
            result = self._maybe_session_boost(result)

            # v3.3.0 — temporal half-life decay (opt-in hot-path).
            result = self._maybe_temporal_decay(result)

            # v3.3.0 — probabilistic truth_score annotation.
            result = self._maybe_truth_score(result)

            # Enforce the caller's limit AFTER expansions — previous code
            # truncated before the graph/entity expansions appended
            # blocks, so the final list could exceed ``limit``. Dedup
            # runs next, then we slice to the requested size.
            # (python-reviewer 2026-04-20)

            # 4-layer dedup filter (post-fusion, post-rerank)
            dedup_cfg = self._config.get("dedup")
            if dedup_cfg is None or (isinstance(dedup_cfg, dict) and dedup_cfg.get("enabled", True)):
                try:
                    from .dedup import DedupConfig, deduplicate_results

                    dc = DedupConfig(dedup_cfg if isinstance(dedup_cfg, dict) else None)
                    result = deduplicate_results(result, config=dc)
                except Exception as e:
                    _log.warning("hybrid_dedup_failed", error=str(e))

            # Final slice so callers never receive more than they asked for.
            result = result[:limit]

            _log.info(
                "hybrid_search_complete",
                query=query,
                results=len(result),
                top_rrf=result[0].get("rrf_score", 0) if result else 0,
            )
            return result

    # -- multi-query expansion search ----------------------------------------

    def _search_expanded(
        self,
        queries: list[str],
        workspace: str,
        limit: int = 10,
        active_only: bool = False,
        graph_boost: bool = False,
        retrieve_wide_k: int = 200,
        rerank: bool = True,
        **kwargs: Any,
    ) -> list[dict]:
        """Search with multiple query variants and fuse results via RRF.

        Each query variant is searched independently using the standard
        single-query pipeline.  Results from all variants are then fused
        using RRF with equal weights, ensuring documents that match
        multiple phrasings rank higher.

        Args:
            queries: List of query variant strings (original + expansions).
            workspace: Workspace root path.
            limit: Maximum results to return.
            active_only: Only return active blocks.
            graph_boost: Enable cross-reference graph boosting.
            retrieve_wide_k: Candidate pool size per backend per query.
            rerank: Enable BM25 reranker.
            **kwargs: Forwarded to underlying search.

        Returns:
            RRF-fused ranked list of result dicts.
        """
        # Pass _skip_auto_features=True so the recursion into search()
        # doesn't re-trigger expansion/decomposition. Previous code
        # mutated self._query_expansion_enabled which raced under
        # concurrent calls (python-reviewer 2026-04-20 → commit
        # b31e862 follow-up).
        #
        # Audit R-5: each variant is an independent BM25 + vector
        # search. Sequential dispatch dominates wall-clock latency
        # when expansion is enabled (default 3 variants → 3× latency).
        # Use a ThreadPoolExecutor to fan out — the SQLite backend is
        # WAL-mode and safe for concurrent reads, and the vector
        # backend is read-only at query time. Single-query callers
        # take the in-line path to avoid pool-spinup overhead.
        per_query_results: list[list[dict]] = []
        if len(queries) <= 1:
            for q in queries:
                per_query_results.append(
                    self.search(
                        q,
                        workspace,
                        limit=retrieve_wide_k,
                        active_only=active_only,
                        graph_boost=graph_boost,
                        retrieve_wide_k=retrieve_wide_k,
                        rerank=rerank,
                        _skip_auto_features=True,
                        **kwargs,
                    )
                )
        else:
            from concurrent.futures import ThreadPoolExecutor

            def _one(q: str) -> list[dict]:
                return self.search(
                    q,
                    workspace,
                    limit=retrieve_wide_k,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=retrieve_wide_k,
                    rerank=rerank,
                    _skip_auto_features=True,
                    **kwargs,
                )

            max_workers = min(len(queries), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                per_query_results = list(ex.map(_one, queries))

        if not per_query_results:
            return []

        # Fuse all query variant results with equal weights
        weights = [1.0] * len(per_query_results)
        fused = rrf_fuse(
            ranked_lists=per_query_results,
            weights=weights,
            k=self.rrf_k,
        )

        _log.info(
            "multi_query_fusion_complete",
            query_variants=len(queries),
            total_fused=len(fused),
            limit=limit,
        )

        return fused[:limit]

    def _maybe_session_boost(self, results: list[dict]) -> list[dict]:
        """Apply session-boundary preservation (v3.3.0 Tier 2 #5)."""
        if not results:
            return results
        try:
            from .session_boost import (
                apply_session_boost,
                is_session_boost_enabled,
                resolve_session_boost_config,
            )

            if not is_session_boost_enabled(self._config, results):
                return results
            params = resolve_session_boost_config(self._config)
            return apply_session_boost(results, **params)
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("session_boost_failed", error=str(exc))
            return results

    def _maybe_truth_score(self, results: list[dict]) -> list[dict]:
        """Annotate results with probabilistic truth_score (v3.3.0)."""
        if not results:
            return results
        try:
            from .truth_score import annotate_results, is_truth_score_enabled

            if not is_truth_score_enabled(self._config):
                return results
            # Contradiction graph is passed through when the caller
            # supplies one via config; otherwise just Status/age/access
            # signals feed the score.
            return annotate_results(results)
        except Exception as exc:  # pragma: no cover
            _log.warning("truth_score_failed", error=str(exc))
            return results

    def _maybe_temporal_decay(self, results: list[dict]) -> list[dict]:
        """Apply half-life decay to every result's score (v3.3.0 Tier 1 #3).

        Opt-in via ``retrieval.temporal_decay_hot_path`` — the raw
        function is always available (Tier 1 #3) but the hot-path
        wiring is gated because it changes ranking. With the gate
        off, the function stays a standalone helper callers invoke
        explicitly.
        """
        if not results:
            return results
        cfg = self._config.get("retrieval", {}) if isinstance(self._config, dict) else {}
        if not isinstance(cfg, dict) or not cfg.get("temporal_decay_hot_path", False):
            return results
        try:
            from ._recall_scoring import _resolve_half_life_days, temporal_decay_score

            half_life = _resolve_half_life_days(self._config)
            # Audit R-10: copy-on-write so we don't mutate dicts the
            # caller still holds a reference to. Two upstream paths
            # (cross-encoder rerank, session boost) reuse the input
            # list, and in-place score mutation corrupted their views
            # when temporal_decay_hot_path was enabled mid-request.
            decayed: list[dict] = []
            for r in results:
                mult = temporal_decay_score(r, half_life_days=half_life)
                current = float(r.get("score", 0.0) or 0.0)
                copy = dict(r)
                copy["score"] = current * mult
                copy["_temporal_decay"] = round(mult, 4)
                decayed.append(copy)
            decayed.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            _log.info("temporal_decay_applied", count=len(decayed), half_life_days=half_life)
            return decayed
        except Exception as exc:  # pragma: no cover
            _log.warning("temporal_decay_failed", error=str(exc))
        return results

    def _load_corpus_if_needed(self, query: str, workspace: str) -> list[dict] | None:
        """Return the workspace block corpus — shared by graph + entity
        helpers so we load once per request rather than twice.

        Returns None when no feature needs the corpus — avoids paying
        the disk cost at all when both auto-enables are off.
        """
        try:
            from .entity_prefetch import is_entity_prefetch_enabled
            from .graph_recall import is_graph_expand_enabled
        except ImportError:  # pragma: no cover
            return None
        if not (is_graph_expand_enabled(self._config, query) or is_entity_prefetch_enabled(self._config)):
            return None
        try:
            from .block_parser import parse_file
            from .block_store import MarkdownBlockStore

            store = MarkdownBlockStore(workspace)
            blocks: list[dict] = []
            for path in store.list_blocks():
                try:
                    blocks.extend(parse_file(path))
                except Exception as exc:  # pragma: no cover
                    _log.debug("corpus_block_parse_skipped", error=str(exc))
                    continue
            return blocks
        except Exception as exc:  # pragma: no cover
            _log.warning("corpus_load_failed", error=str(exc))
            return None

    def _maybe_entity_prefetch(
        self,
        query: str,
        workspace: str,
        results: list[dict],
        *,
        corpus: list[dict] | None = None,
    ) -> list[dict]:
        """Inject entity-graph prefetched blocks (v3.3.0 Tier 3 #8).

        When the query mentions a Person/Project/Tool/Incident, fetch
        the entity block + 1-hop neighbourhood and merge into the
        result set. ``corpus`` — when passed — skips a workspace reload
        (shared with graph_expand). Fails open on any error.
        """
        try:
            from .entity_prefetch import (
                is_entity_prefetch_enabled,
                prefetch_entity_blocks,
                resolve_entity_prefetch_config,
            )

            if not is_entity_prefetch_enabled(self._config):
                return results
            params = resolve_entity_prefetch_config(self._config)
            prefetched = prefetch_entity_blocks(
                query,
                workspace,
                max_entities=params["max_entities"],
                max_hops=params["max_hops"],
                entity_score=params["entity_score"],
                corpus=corpus,
            )
            if not prefetched:
                return results
            # Merge: keep original order, append prefetched blocks that
            # aren't already in the result set. Downstream dedup catches
            # any ID collisions.
            seen_ids = {str(r.get("_id")) for r in results if r.get("_id")}
            merged = list(results)
            for b in prefetched:
                bid = str(b.get("_id") or "")
                if not bid or bid in seen_ids:
                    continue
                seen_ids.add(bid)
                merged.append(b)
            if len(merged) > len(results):
                _log.info(
                    "entity_prefetch_merged",
                    seeds=len(results),
                    added=len(merged) - len(results),
                )
            return merged
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("entity_prefetch_failed", error=str(exc))
            return results

    def _maybe_graph_expand(
        self,
        query: str,
        workspace: str,
        results: list[dict],
        *,
        corpus: list[dict] | None = None,
    ) -> list[dict]:
        """Append graph-walked blocks when enabled (v3.3.0 Tier 1 #2).

        ``corpus`` — when provided — skips a workspace reload (shared
        with entity_prefetch). Fails open on any error so recall
        never blocks on graph issues.
        """
        if not results:
            return results
        try:
            from .graph_recall import (
                graph_expand,
                is_graph_expand_enabled,
                resolve_graph_config,
            )

            if not is_graph_expand_enabled(self._config, query):
                return results
            if corpus is not None:
                all_blocks = corpus
            else:
                # Legacy path: caller didn't pre-load the corpus.
                from .block_parser import parse_file
                from .block_store import MarkdownBlockStore

                store = MarkdownBlockStore(workspace)
                all_blocks = []
                for path in store.list_blocks():
                    try:
                        all_blocks.extend(parse_file(path))
                    except Exception as exc:  # pragma: no cover
                        _log.debug("graph_expand_block_parse_skipped", error=str(exc))
                        continue
            params = resolve_graph_config(self._config)
            expanded = graph_expand(results, all_blocks, **params)
            if len(expanded) > len(results):
                _log.info(
                    "graph_expand_applied",
                    seeds=len(results),
                    final=len(expanded),
                    max_hops=params["max_hops"],
                )
            return expanded
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("graph_expand_failed", error=str(exc))
            return results

    def _maybe_cross_encoder_rerank(self, query: str, result: list[dict], limit: int) -> list[dict]:
        """Apply cross-encoder rerank when appropriate.

        v3.3.0 Tier 2 #6 — auto-enables on multi-hop / temporal queries
        (per detect_query_type) even when ``cross_encoder.enabled`` is
        false, unless operator sets ``cross_encoder.auto_enable: false``.
        Returns ``result`` unchanged on any failure.
        """
        if not result:
            return result
        ce_cfg = self._config.get("cross_encoder", {})
        ce_enabled = bool(ce_cfg.get("enabled", False))
        if not ce_enabled and ce_cfg.get("auto_enable", True):
            try:
                from ._recall_detection import detect_query_type

                qt = detect_query_type(query)
                if qt in ("multi-hop", "temporal"):
                    ce_enabled = True
                    _log.info(
                        "cross_encoder_auto_enabled",
                        query_type=qt,
                        reason="v3.3.0_tier2_ambiguous_query",
                    )
            except Exception as exc:  # pragma: no cover — defensive
                _log.debug("ce_query_type_detect_skipped", error=str(exc))
        if not ce_enabled:
            return result
        # v3.3.0 Tier 4 #9 — prefer reranker_ensemble when configured,
        # fall back to single-model CE. The ensemble's single-member
        # degenerate case is also the same as CE alone, so wiring this
        # in doesn't regress existing CE-only deployments.
        try:
            from .rerank_ensemble import create_ensemble

            ensemble = create_ensemble(self._config)
            if ensemble is not None:
                for r in result:
                    if "content" not in r:
                        r["content"] = r.get("excerpt", "")
                result = ensemble.rerank(
                    query,
                    result,
                    top_k=ce_cfg.get("top_k", limit),
                    blend_weight=ce_cfg.get("blend_weight", 0.6),
                )
                _log.info("reranker_ensemble_applied", candidates=len(result))
                return result
        except Exception as exc:
            _log.warning("reranker_ensemble_failed", error=str(exc))
        try:
            from .cross_encoder_reranker import CrossEncoderReranker

            if CrossEncoderReranker.is_available():
                ce = CrossEncoderReranker()
                for r in result:
                    if "content" not in r:
                        r["content"] = r.get("excerpt", "")
                result = ce.rerank(
                    query,
                    result,
                    top_k=ce_cfg.get("top_k", limit),
                    blend_weight=ce_cfg.get("blend_weight", 0.6),
                )
                _log.info(
                    "cross_encoder_rerank",
                    candidates=len(result),
                    blend_weight=ce_cfg.get("blend_weight", 0.6),
                )
        except ImportError as ie:
            _log.warning("cross_encoder_import_failed", error=str(ie))
        except Exception as e:
            _log.warning("cross_encoder_unavailable", error=str(e))
        return result

    # -- backend wrappers ---------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        workspace: str,
        limit: int = 200,
        **kwargs: Any,
    ) -> list[dict]:
        """BM25 search via the existing recall engine.

        Tries sqlite_index first (O(log N)), then falls back to recall.py
        (O(corpus)).
        """
        try:
            from .sqlite_index import _db_path, query_index

            db = _db_path(workspace)
            if os.path.isfile(db):
                return query_index(workspace, query, limit=limit, **kwargs)
        except ImportError:
            _log.debug("sqlite_index_not_available")
        except Exception as exc:
            _log.warning("sqlite_index_fallback", error=str(exc))

        try:
            from .recall import recall

            return recall(workspace, query, limit=limit, **kwargs)
        except Exception as exc:
            _log.error("bm25_search_failed", error=str(exc))
            return []

    def _vector_search(
        self,
        query: str,
        workspace: str,
        limit: int = 200,
        active_only: bool = False,
    ) -> list[dict]:
        """Vector search via recall_vector.search_batch (for RRF) or .search."""
        try:
            from . import recall_vector

            # Prefer search_batch (returns all results for RRF)
            if hasattr(recall_vector, "search_batch"):
                return list(
                    recall_vector.search_batch(
                        workspace,
                        query,
                        limit=limit,
                        active_only=active_only,
                        config=self._config,
                    )
                )

            # Fallback: VectorBackend.search
            backend = recall_vector.VectorBackend(self._config)
            return list(backend.search(workspace, query, limit=limit, active_only=active_only))
        except ImportError:
            _log.warning("vector_search_import_failed")
            return []
        except Exception as exc:
            _log.error("vector_search_failed", error=str(exc))
            return []

    # -- factory ------------------------------------------------------------

    @staticmethod
    def from_config(config: dict[str, Any]) -> "HybridBackend":
        """Create HybridBackend from a full mind-mem.json config dict.

        Validates that ``config`` contains a ``recall`` section.  When
        the section is missing or not a dict, logs a warning and falls
        back to BM25-only defaults.
        """
        recall_cfg = config.get("recall")
        if recall_cfg is None:
            _log.warning(
                "hybrid_config_missing_recall_section",
                hint="Expected 'recall' key in config. Using BM25-only defaults.",
            )
            recall_cfg = {}
        elif not isinstance(recall_cfg, dict):
            _log.warning(
                "hybrid_config_recall_not_dict",
                type=type(recall_cfg).__name__,
                hint="'recall' must be a dict. Using BM25-only defaults.",
            )
            recall_cfg = {}
        return HybridBackend(config=recall_cfg)

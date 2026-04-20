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

    Args:
        ranked_lists: List of ranked result lists. Each result is a dict
            that must contain ``id_key`` for dedup.
        weights: Per-list weight multipliers (same length as ranked_lists).
        k: RRF smoothing constant (default 60). Higher values dampen the
            advantage of top-ranked documents.
        id_key: Dict key used to identify unique documents.

    Returns:
        Fused list sorted by descending RRF score. Each item is a copy of
        the first-seen dict for that ID, with ``rrf_score`` and ``fusion``
        fields injected.
    """
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    block_data: dict[str, dict] = {}

    for list_idx, results in enumerate(ranked_lists):
        w = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank_0, item in enumerate(results):
            bid = _get_block_id(item, id_key)
            scores[bid] = scores.get(bid, 0.0) + w / (k + rank_0 + 1)
            if bid not in block_data:
                block_data[bid] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for bid in sorted_ids:
        item = block_data[bid].copy()
        item["rrf_score"] = round(scores[bid], 6)
        item["fusion"] = "rrf"
        fused.append(item)

    return fused


def _get_block_id(item: dict, id_key: str) -> str:
    """Extract a stable block identifier from a result dict."""
    bid = item.get(id_key)
    if bid:
        return str(bid)
    # Fallback: try common key variants
    for alt in ("id", "block_id", "_id"):
        val = item.get(alt)
        if val:
            return str(val)
    # Last resort: file:line
    return f"{item.get('file', '?')}:{item.get('line', 0)}"


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

        # Probe vector availability once at init
        self._vector_available = self._check_vector() if self.vector_enabled else False

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

        # --- Multi-query expansion ---
        # When enabled, expand the query into alternative phrasings and
        # run a search for each variant.  Results are fused via RRF so
        # documents matching multiple phrasings rank higher.
        if self._query_expansion_enabled:
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

        with timed("hybrid_search"):
            if not self._vector_available:
                _log.info("hybrid_bm25_only", query=query)
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

            with ThreadPoolExecutor(max_workers=2) as pool:
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
                vec_results = vec_future.result()

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

            # 4-layer dedup filter (post-fusion, post-rerank)
            dedup_cfg = self._config.get("dedup")
            if dedup_cfg is None or (isinstance(dedup_cfg, dict) and dedup_cfg.get("enabled", True)):
                try:
                    from .dedup import DedupConfig, deduplicate_results

                    dc = DedupConfig(dedup_cfg if isinstance(dedup_cfg, dict) else None)
                    result = deduplicate_results(result, config=dc)
                except Exception as e:
                    _log.warning("hybrid_dedup_failed", error=str(e))

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
        # Temporarily disable expansion to avoid infinite recursion
        orig_enabled = self._query_expansion_enabled
        self._query_expansion_enabled = False
        try:
            per_query_results: list[list[dict]] = []
            for q in queries:
                results = self.search(
                    q,
                    workspace,
                    limit=retrieve_wide_k,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=retrieve_wide_k,
                    rerank=rerank,
                    **kwargs,
                )
                per_query_results.append(results)
        finally:
            self._query_expansion_enabled = orig_enabled

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
            except Exception:  # pragma: no cover — defensive
                pass
        if not ce_enabled:
            return result
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

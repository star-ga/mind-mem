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
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from observability import get_logger, metrics, timed

_log = get_logger("hybrid_recall")

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
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.rrf_k: int = int(cfg.get("rrf_k", 60))
        self.bm25_weight: float = float(cfg.get("bm25_weight", 1.0))
        self.vector_weight: float = float(cfg.get("vector_weight", 1.0))
        self.vector_enabled: bool = bool(cfg.get("vector_enabled", False))
        self.vector_model: str = cfg.get("vector_model", "all-MiniLM-L6-v2")
        self._config = cfg

        # Probe vector availability once at init
        self._vector_available = self._check_vector() if self.vector_enabled else False

        _log.info(
            "hybrid_backend_init",
            rrf_k=self.rrf_k,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
            vector_available=self._vector_available,
        )

    # -- capability probing ------------------------------------------------

    def _check_vector(self) -> bool:
        """Return True if recall_vector + sentence-transformers are importable."""
        try:
            import recall_vector  # noqa: F401

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

        with timed("hybrid_search"):
            if not self._vector_available:
                _log.info("hybrid_bm25_only", query=query)
                results = self._bm25_search(
                    query, workspace,
                    limit=limit,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=retrieve_wide_k,
                    rerank=rerank,
                    **kwargs,
                )
                metrics.inc("hybrid_searches_bm25_only")
                return results

            # Run BM25 + vector in parallel
            _log.info("hybrid_parallel_search", query=query)
            bm25_results: list[dict] = []
            vec_results: list[dict] = []

            with ThreadPoolExecutor(max_workers=2) as pool:
                bm25_future: Future = pool.submit(
                    self._bm25_search, query, workspace,
                    limit=retrieve_wide_k,
                    active_only=active_only,
                    graph_boost=graph_boost,
                    retrieve_wide_k=retrieve_wide_k,
                    rerank=False,  # defer reranking to post-fusion
                    **kwargs,
                )
                vec_future: Future = pool.submit(
                    self._vector_search, query, workspace,
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

            _log.info(
                "hybrid_search_complete",
                query=query,
                results=len(result),
                top_rrf=result[0]["rrf_score"] if result else 0,
            )
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
            from sqlite_index import _db_path, query_index

            db = _db_path(workspace)
            if os.path.isfile(db):
                return query_index(workspace, query, limit=limit, **kwargs)
        except ImportError:
            _log.debug("sqlite_index_not_available")
        except Exception as exc:
            _log.warning("sqlite_index_fallback", error=str(exc))

        try:
            from recall import recall

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
            import recall_vector

            # Prefer search_batch (returns all results for RRF)
            if hasattr(recall_vector, "search_batch"):
                return recall_vector.search_batch(
                    workspace, query, limit=limit, active_only=active_only,
                )

            # Fallback: VectorBackend.search
            backend = recall_vector.VectorBackend(self._config)
            return backend.search(workspace, query, limit=limit, active_only=active_only)
        except ImportError:
            _log.warning("vector_search_import_failed")
            return []
        except Exception as exc:
            _log.error("vector_search_failed", error=str(exc))
            return []

    # -- factory ------------------------------------------------------------

    @staticmethod
    def from_config(config: dict[str, Any]) -> "HybridBackend":
        """Create HybridBackend from a full mind-mem.json config dict."""
        recall_cfg = config.get("recall", {})
        return HybridBackend(config=recall_cfg)

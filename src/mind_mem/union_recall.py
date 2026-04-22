"""Union-style retrieval for decomposed queries (v3.4.0).

The v3.3.0 feature combo RRF-fused sub-query retrievals, which
attenuated the seed question's best hits and caused a multi-hop
regression on LoCoMo.

This module implements the 4-LLM-consensus fix: **UNION + dedup**, not
RRF. Each sub-query's top-k is added to a single pool keyed by
``_id``; the best (lowest rank) copy of each block is kept. Result
order preserves the earliest appearance across sub-queries, with the
original question always processed first.

Why union beats RRF for multi-hop:

* RRF penalises blocks that appear in only one sub-query's top-k, even
  if that block is the critical bridge evidence for a multi-hop
  question.
* Union preserves every high-ranked block from every sub-query's
  retrieval and lets the downstream reranker / answerer see the full
  joint context.

Public entry: :func:`union_retrieve`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .observability import get_logger

_log = get_logger("union_recall")


def _block_id(block: dict[str, Any]) -> str:
    """Extract a stable identifier from a retrieval hit."""
    bid = block.get("_id") or block.get("id") or block.get("block_id")
    if bid:
        return str(bid)
    # Fall back to a content fingerprint so unrelated results don't collide.
    excerpt = (block.get("excerpt") or block.get("Statement") or "")[:80]
    return excerpt or repr(block)[:80]


def _min_rank(block: dict[str, Any], fallback: int) -> int:
    """Recover the best rank any retriever assigned to this block."""
    for key in ("_rank", "rank", "_rrf_rank"):
        if key in block and isinstance(block[key], (int, float)):
            return int(block[key])
    return fallback


def union_retrieve(
    sub_queries: Iterable[str],
    retrieve_fn: Callable[[str], list[dict[str, Any]]],
    top_k_per_query: int = 18,
    max_total: int = 50,
) -> list[dict[str, Any]]:
    """Run ``retrieve_fn`` on each sub-query and union the results.

    Args:
        sub_queries: iterable of (decomposed) queries. The caller is
            expected to put the original question first — the union
            preserves earliest-seen order, so the original question's
            top hits naturally lead the output.
        retrieve_fn: callable taking a query string, returning a list
            of block dicts. Each dict should carry at least ``_id``
            (or ``id`` / ``block_id``); a score field is optional.
        top_k_per_query: truncate each sub-query's retrieval to this
            many hits before unioning. Keeps the pool size bounded
            even with many sub-queries.
        max_total: upper bound on the returned list — protects the
            downstream reranker from explosion when a query
            decomposes into many similar sub-queries.

    Returns:
        A flat list of unique block dicts, ordered by first appearance
        across sub-queries, with each block's ``_union_first_seen``
        index recorded for downstream observability. No RRF scoring,
        no positional attenuation.
    """
    seen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    inspected = 0
    q_idx = -1  # bound even if the loop body never runs (empty iterable)

    for q_idx, query in enumerate(sub_queries):
        if not query or not query.strip():
            continue
        try:
            hits = retrieve_fn(query) or []
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("union_retrieve_query_failed", query=query[:80], error=str(exc))
            continue
        for rank_in_q, block in enumerate(hits[:top_k_per_query]):
            bid = _block_id(block)
            if bid in seen:
                # Keep the better rank of the two copies.
                prev = seen[bid]
                if rank_in_q < _min_rank(prev, fallback=top_k_per_query):
                    merged = dict(prev)
                    merged.update(block)
                    merged.setdefault("_union_first_seen", prev.get("_union_first_seen", q_idx))
                    merged["_union_best_rank"] = rank_in_q
                    seen[bid] = merged
                continue
            enriched = dict(block)
            enriched["_union_first_seen"] = q_idx
            enriched["_union_best_rank"] = rank_in_q
            seen[bid] = enriched
            order.append(bid)
            inspected += 1
            if len(order) >= max_total:
                break
        if len(order) >= max_total:
            break

    result = [seen[bid] for bid in order[:max_total]]
    _log.info(
        "union_retrieve",
        n_sub_queries=max(q_idx + 1, 0),
        inspected=inspected,
        unique=len(result),
    )
    return result


__all__ = ["union_retrieve"]

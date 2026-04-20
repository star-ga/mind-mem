"""Multi-hop graph traversal for recall (v3.3.0 Tier 1 #2).

Given an initial set of retrieved blocks, walk the cross-reference
graph up to ``max_hops`` steps and append newly-discovered blocks
with a decayed score. Fuses multi-hop evidence into the same result
set that the BM25/hybrid pipeline produces.

The graph is built from the existing ``build_xref_graph`` helper in
``_recall_scoring`` — no new ingestion required. Block IDs mentioned
anywhere in a block's text field become edges; ``supersedes``,
``supersededBy``, ``relates_to`` fields are treated as explicit edges.

Opt-in via:

    {
      "retrieval": {
        "multi_hop": {
          "enabled": false,
          "auto_enable": true,
          "max_hops": 2,
          "decay": 0.5,
          "max_neighbors_per_hop": 5
        }
      }
    }

Auto-enabled for multi-hop query types (per ``detect_query_type``)
unless ``auto_enable`` is explicitly false.
"""

from __future__ import annotations

from typing import Any, Iterable

from .observability import get_logger

_log = get_logger("graph_recall")


def graph_expand(
    results: list[dict],
    all_blocks: list[dict],
    *,
    max_hops: int = 2,
    decay: float = 0.5,
    max_neighbors_per_hop: int = 5,
    score_field: str = "score",
) -> list[dict]:
    """Walk the block cross-reference graph from each seed.

    Args:
        results: Initial ranked recall results. Each must carry
            ``_id`` and a numeric ``score_field``.
        all_blocks: Full block corpus (needed to build the xref graph
            and resolve neighbour-ID → block-dict). Not walked by ID;
            the function reads ``build_xref_graph`` once.
        max_hops: Maximum graph distance from seed blocks.
        decay: Multiplicative score decay per hop (``score_at_hop_n =
            seed_score * decay ** n``).
        max_neighbors_per_hop: Cap on how many new blocks are added
            per seed per hop — guards against pathological fan-out.
        score_field: Field used for seed scores + new-block scores.

    Returns:
        ``results`` with appended graph-walked blocks (new entries
        carry ``_graph_hop`` and ``_graph_parent`` fields). Order
        preserved: original results first, newly-added blocks after,
        sorted by descending score within the appended block.
    """
    if not results or not all_blocks or max_hops <= 0:
        return results

    from ._recall_scoring import build_xref_graph

    id_to_block: dict[str, dict] = {str(b.get("_id")): b for b in all_blocks if b.get("_id")}
    if not id_to_block:
        return results

    graph = build_xref_graph(all_blocks)
    seen_ids: set[str] = {str(r.get("_id")) for r in results if r.get("_id")}

    appended: list[dict] = []
    # BFS frontier: list of (block_id, ORIGINAL seed_score, hop, parent_id).
    # The "seed_score" stays constant across hops — decay is computed
    # relative to the original seed so a 2-hop node gets decay**2, not
    # decay**2 * decay (which would compound on the already-decayed score).
    frontier: list[tuple[str, float, int, str]] = []
    for r in results:
        bid = str(r.get("_id") or "")
        if not bid:
            continue
        base_score = float(r.get(score_field, 0.0) or 0.0)
        frontier.append((bid, base_score, 0, bid))

    while frontier:
        bid, seed_score, hop, parent = frontier.pop(0)
        if hop >= max_hops:
            continue
        neighbors: Iterable[str] = graph.get(bid, set())
        added_this_hop = 0
        for nid in neighbors:
            if nid in seen_ids:
                continue
            block = id_to_block.get(nid)
            if block is None:
                continue
            next_hop = hop + 1
            decayed = seed_score * (decay**next_hop)
            new = dict(block)
            new[score_field] = decayed
            new["_graph_hop"] = next_hop
            new["_graph_parent"] = parent
            appended.append(new)
            seen_ids.add(nid)
            # Pass the ORIGINAL seed_score so deeper hops compute
            # decay ** hop against the starting score, not the decayed.
            frontier.append((nid, seed_score, next_hop, parent))
            added_this_hop += 1
            if added_this_hop >= max_neighbors_per_hop:
                break

    if appended:
        _log.info(
            "graph_expanded",
            seeds=len(results),
            added=len(appended),
            max_hops=max_hops,
        )
        # Keep appended list sorted so the first graph-walked neighbour
        # (highest decayed score) appears before weaker ones.
        appended.sort(key=lambda b: b.get(score_field, 0.0), reverse=True)
    return list(results) + appended


def is_graph_expand_enabled(
    config: dict[str, Any] | None,
    query: str | None = None,
) -> bool:
    """Resolve whether graph expansion should fire for this call.

    Priority:
      1. ``retrieval.multi_hop.enabled: true`` — always on.
      2. ``retrieval.multi_hop.auto_enable: false`` — always off.
      3. Auto-enable when the query classifies as multi-hop.
    """
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    mh = retrieval.get("multi_hop", {})
    if not isinstance(mh, dict):
        return False
    if mh.get("enabled", False):
        return True
    if not mh.get("auto_enable", True):
        return False
    if not query:
        return False
    try:
        from ._recall_detection import detect_query_type

        return detect_query_type(query) == "multi-hop"
    except Exception:  # pragma: no cover
        return False


def resolve_graph_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract graph-expansion parameters from config with defaults."""
    defaults: dict[str, Any] = {
        "max_hops": 2,
        "decay": 0.5,
        "max_neighbors_per_hop": 5,
    }
    if not config or not isinstance(config, dict):
        return defaults
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return defaults
    mh = retrieval.get("multi_hop", {})
    if not isinstance(mh, dict):
        return defaults
    out = dict(defaults)
    if isinstance(mh.get("max_hops"), int) and mh["max_hops"] > 0:
        out["max_hops"] = int(mh["max_hops"])
    if isinstance(mh.get("decay"), (int, float)) and 0 < mh["decay"] <= 1:
        out["decay"] = float(mh["decay"])
    if isinstance(mh.get("max_neighbors_per_hop"), int) and mh["max_neighbors_per_hop"] > 0:
        out["max_neighbors_per_hop"] = int(mh["max_neighbors_per_hop"])
    return out


__all__ = [
    "graph_expand",
    "is_graph_expand_enabled",
    "resolve_graph_config",
]

"""Typed knowledge-graph fusion into recall (opt-in, default OFF).

The block-xref path (``graph_recall.graph_expand``) walks free-text
cross-references; this module walks the **typed** knowledge graph
(``knowledge_graph.KnowledgeGraph``): query terms resolve to entities
through the registry (read-only), edges are traversed up to two hops,
and each traversed edge's ``source_block_id`` pulls its backing block
into the result set with a decayed score — the same decay shape as
``graph_recall.graph_expand``.

Opt-in via:

    {
      "retrieval": {
        "kg_fusion": {
          "enabled": false,
          "max_hops": 2,
          "decay": 0.5,
          "max_neighbors_per_hop": 5,
          "max_total_added": 25
        }
      }
    }

Default OFF: until the graph is populated (see ``graph_ingest``),
enabling this would only add per-query work with nothing to walk —
and existing installs must replay byte-identical.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Optional

from .admissibility import admit_corpus, count_unresolved
from .knowledge_graph import EntityRegistry, KnowledgeGraph
from .observability import get_logger

_log = get_logger("kg_fusion")

_MAX_HOPS_CEILING = 2
_WORD_RE = re.compile(r"[\w.-]+")

_DEFAULTS: dict[str, Any] = {
    "max_hops": 2,
    "decay": 0.5,
    "max_neighbors_per_hop": 5,
    "max_total_added": 25,
}


def is_kg_fusion_enabled(config: Optional[dict]) -> bool:
    """Resolve the ``retrieval.kg_fusion.enabled`` gate. Default False."""
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    kf = retrieval.get("kg_fusion", {})
    if not isinstance(kf, dict):
        return False
    return bool(kf.get("enabled", False))


def resolve_kg_fusion_config(config: Optional[dict]) -> dict[str, Any]:
    """Extract fusion parameters with validated defaults."""
    out = dict(_DEFAULTS)
    if not config or not isinstance(config, dict):
        return out
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return out
    kf = retrieval.get("kg_fusion", {})
    if not isinstance(kf, dict):
        return out
    if isinstance(kf.get("max_hops"), int) and kf["max_hops"] > 0:
        out["max_hops"] = min(_MAX_HOPS_CEILING, int(kf["max_hops"]))
    if isinstance(kf.get("decay"), (int, float)) and 0 < kf["decay"] <= 1:
        out["decay"] = float(kf["decay"])
    if isinstance(kf.get("max_neighbors_per_hop"), int) and kf["max_neighbors_per_hop"] > 0:
        out["max_neighbors_per_hop"] = int(kf["max_neighbors_per_hop"])
    if isinstance(kf.get("max_total_added"), int) and kf["max_total_added"] > 0:
        out["max_total_added"] = int(kf["max_total_added"])
    return out


def resolve_query_entities(query: str, registry: EntityRegistry) -> list[str]:
    """Map query terms to known entity ids — strictly read-only.

    Tries individual tokens plus adjacent bigrams (multi-word entity
    names like "starga inc"). Order is deterministic (first mention
    wins); unknown terms resolve to nothing and are never minted into
    the registry.
    """
    tokens = [t for t in _WORD_RE.findall(query.lower()) if len(t) > 1]
    candidates: list[str] = []
    for i, tok in enumerate(tokens):
        if i + 1 < len(tokens):
            candidates.append(f"{tok} {tokens[i + 1]}")
        candidates.append(tok)
    found: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        entity_id = registry.lookup(cand)
        if entity_id is not None and entity_id not in seen:
            seen.add(entity_id)
            found.append(entity_id)
    return found


def kg_expand(
    results: list[dict],
    corpus: list[dict],
    kg: KnowledgeGraph,
    query: str,
    *,
    max_hops: int = 2,
    decay: float = 0.5,
    max_neighbors_per_hop: int = 5,
    max_total_added: int = 25,
    score_field: str = "score",
) -> list[dict]:
    """Append blocks reachable through typed edges from query entities.

    Args:
        results: Ranked recall results (each carries ``_id`` + score).
        corpus: Full block corpus for source_block_id → block lookup.
        kg: Open knowledge graph.
        query: The recall query — resolved to entities via the registry.
        max_hops: Edge-walk depth from each query entity (ceiling 2).
        decay: Multiplicative score decay per hop, applied to the top
            seed score (``score = seed * decay ** hop``).
        max_neighbors_per_hop: Edge cap per entity per hop.
        max_total_added: Cap on appended blocks across the whole walk.
        score_field: Field carrying the numeric score.

    Returns:
        ``results`` (unchanged object when nothing fuses) or a new list
        with appended blocks carrying ``_kg_hop`` / ``_kg_entity`` /
        ``_kg_predicate`` markers. Deterministic for a given graph +
        corpus + query.
    """
    if not results or not corpus or max_hops <= 0:
        return results
    max_hops = min(int(max_hops), _MAX_HOPS_CEILING)

    entities = resolve_query_entities(query, kg.entities)
    if not entities:
        return results

    # Same rule as the cross-reference walk: this leg appends raw corpus
    # blocks, so only admissible ones may be resolvable from an edge.
    corpus = admit_corpus(corpus)
    id_to_block = {str(b.get("_id")): b for b in corpus if b.get("_id")}
    if not id_to_block:
        return results

    seed_score = max(float(r.get(score_field, 0.0) or 0.0) for r in results)
    seen_ids = {str(r.get("_id")) for r in results if r.get("_id")}
    seen_entities = set(entities)
    appended: list[dict] = []

    frontier: deque[tuple[str, int]] = deque((e, 0) for e in entities)
    while frontier and len(appended) < max_total_added:
        entity, hop = frontier.popleft()
        if hop >= max_hops:
            continue
        next_hop = hop + 1
        edges = list(kg.edges_from(entity)) + list(kg.edges_to(entity))
        walked = 0
        for edge in edges:
            if walked >= max_neighbors_per_hop or len(appended) >= max_total_added:
                break
            other = edge.object if edge.subject == entity else edge.subject
            if other not in seen_entities:
                seen_entities.add(other)
                frontier.append((other, next_hop))
            bid = edge.source_block_id
            if bid in seen_ids:
                continue
            block = id_to_block.get(bid)
            if block is None:
                # THE unresolvable-id case: ``source_block_id`` comes out of
                # the graph database, not the corpus, so it can name a block
                # the corpus no longer answers for — deleted, or withheld by
                # the filter above. Dropped and counted, and deliberately NOT
                # a reason to mark the run degraded: an index that has outrun
                # its corpus is ordinary, and ``served`` staying a subset of
                # ``resolved`` staying a subset of ``admissible`` is the
                # invariant that matters.
                count_unresolved()
                continue
            new = dict(block)
            new[score_field] = seed_score * (decay**next_hop)
            new["_kg_hop"] = next_hop
            new["_kg_entity"] = entity
            new["_kg_predicate"] = edge.predicate.value
            appended.append(new)
            seen_ids.add(bid)
            walked += 1

    if not appended:
        return results
    appended.sort(key=lambda b: (-float(b.get(score_field, 0.0) or 0.0), str(b.get("_id", ""))))
    _log.info(
        "kg_expand_applied",
        query_entities=len(entities),
        seeds=len(results),
        added=len(appended),
        max_hops=max_hops,
    )
    return list(results) + appended


__all__ = [
    "is_kg_fusion_enabled",
    "resolve_kg_fusion_config",
    "resolve_query_entities",
    "kg_expand",
]

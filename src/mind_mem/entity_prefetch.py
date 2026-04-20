"""Entity-graph prefetch for recall (v3.3.0 Tier 3 #8).

When a query mentions a Person, Project, Tool, or Incident, we can
pre-fetch their entity block and its 1-hop graph neighbourhood
*before* BM25 runs. The prefetched blocks are injected into the
RRF fusion pool so they rank alongside the token-match results.

This addresses a common LoCoMo failure mode: a multi-hop question
like "What did Alice say about the outage?" may not hit the block
about Alice directly via BM25 (her name is only one token), but her
entity block's cross-references will surface the outage block.

The implementation is intentionally conservative:
* Entity lookup is pattern-based (PER-NNN / PRJ-NNN / TOOL-NNN /
  INC-NNN block IDs) — no LLM required.
* Entity name matching uses stemming-compatible substring search
  against entity block fields (``Name``, ``Statement``, ``Aliases``).
* When no entity matches, returns ``[]`` — the normal BM25/hybrid
  flow continues unchanged.
* Capped at ``max_entities`` (default 3) and ``max_hops`` (default 1)
  so pathological workspaces can't blow up latency.

Opt-in via:

    {
      "retrieval": {
        "entity_prefetch": {
          "enabled": false,
          "auto_enable": true,
          "max_entities": 3,
          "max_hops": 1,
          "entity_score": 5.0
        }
      }
    }
"""

from __future__ import annotations

import os
import re
from typing import Any

from .observability import get_logger

_log = get_logger("entity_prefetch")


# Entity types that ship with mind-mem's canonical block-ID prefixes.
# Matches the keys in ``_BLOCK_PREFIX_MAP`` (block_store.py) without
# importing — this module stays independent of storage concerns.
_ENTITY_TYPES: dict[str, str] = {
    "PER": "people",
    "PRJ": "projects",
    "TOOL": "tools",
    "INC": "incidents",
}


# Heuristic: capitalised words of 3+ chars are likely entity candidates.
# Real entities get confirmed against the block corpus; false positives
# drop out at the lookup step.
_CANDIDATE_TOKEN_RE = re.compile(r"\b[A-Z][a-zA-Z][a-zA-Z]+\b")


def _tokenize_lower(s: str) -> set[str]:
    """Lower-cased word set for substring matching."""
    return set(re.findall(r"\w+", s.lower()))


def extract_entity_candidates(query: str) -> list[str]:
    """Return likely-entity tokens from a query.

    Doesn't try to distinguish between "alice" and "PostgreSQL" — that
    filtering happens when we check the token against the entity
    corpus. Pure-capitalised tokens in the query plus known entity-ID
    patterns make the candidate list.
    """
    if not query:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _CANDIDATE_TOKEN_RE.finditer(query):
        tok = match.group(0)
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return out


def _load_entity_blocks(workspace: str) -> list[dict]:
    """Load every block from the ``entities/`` directory.

    Returns an empty list when the workspace lacks an entities dir
    (e.g., fresh install). No exception is raised — the caller
    handles the empty result naturally.
    """
    ent_dir = os.path.join(workspace, "entities")
    if not os.path.isdir(ent_dir):
        return []
    blocks: list[dict] = []
    try:
        from .block_parser import parse_file
    except Exception:
        return []
    for name in sorted(os.listdir(ent_dir)):
        if not name.endswith(".md"):
            continue
        path = os.path.join(ent_dir, name)
        try:
            blocks.extend(parse_file(path))
        except Exception:  # pragma: no cover
            continue
    return blocks


def _entity_matches_query(block: dict, query_tokens: set[str]) -> bool:
    """True when any of the block's name/alias tokens appears in query."""
    fields = ("Name", "Statement", "Aliases", "Type")
    for field in fields:
        val = block.get(field, "")
        if isinstance(val, list):
            val = " ".join(str(v) for v in val)
        if not val:
            continue
        block_tokens = _tokenize_lower(str(val))
        if block_tokens & query_tokens:
            return True
    return False


def prefetch_entity_blocks(
    query: str,
    workspace: str,
    *,
    max_entities: int = 3,
    max_hops: int = 1,
    entity_score: float = 5.0,
) -> list[dict]:
    """Return entity blocks + 1-hop neighbours that match ``query``.

    Args:
        query: Search query.
        workspace: Workspace root.
        max_entities: Maximum number of entity blocks to seed from.
        max_hops: Hops to walk from each matched entity block.
        entity_score: Score to assign each prefetched block (fed into
            RRF at the fusion layer).

    Returns:
        Ranked list of prefetched block dicts. Empty when no entity
        matches or the workspace has no ``entities/`` directory.
        Every returned dict carries ``_prefetch: "entity"`` so the
        downstream pipeline can distinguish prefetched evidence.
    """
    if not query or not query.strip():
        return []
    query_tokens = _tokenize_lower(query)
    candidates = extract_entity_candidates(query)
    if not candidates:
        return []

    entity_blocks = _load_entity_blocks(workspace)
    if not entity_blocks:
        return []

    matched: list[dict] = []
    for block in entity_blocks:
        if _entity_matches_query(block, query_tokens):
            bid = block.get("_id")
            if not bid:
                continue
            prefix = str(bid).split("-", 1)[0]
            if prefix not in _ENTITY_TYPES:
                continue
            matched.append(block)
            if len(matched) >= max_entities:
                break

    if not matched:
        return []

    # Annotate every matched block as prefetched so downstream callers
    # can tell it came from the graph rather than BM25.
    out: list[dict] = []
    for b in matched:
        annotated = dict(b)
        annotated["_prefetch"] = "entity"
        annotated["score"] = float(entity_score)
        out.append(annotated)

    # Walk 1 hop from each matched entity, reusing graph_expand so the
    # traversal respects the same decay + cap semantics as Tier 1 #2.
    if max_hops > 0:
        try:
            from .graph_recall import graph_expand

            # graph_expand needs the full block corpus for neighbour
            # resolution. Load it lazily.
            from .block_store import MarkdownBlockStore

            store = MarkdownBlockStore(workspace)
            from .block_parser import parse_file

            all_blocks: list[dict] = []
            for path in store.list_blocks():
                try:
                    all_blocks.extend(parse_file(path))
                except Exception:  # pragma: no cover
                    continue
            out = graph_expand(
                out,
                all_blocks,
                max_hops=max_hops,
                decay=0.5,
                max_neighbors_per_hop=3,
            )
            # Mark the graph-walked neighbours too so callers can trace.
            for b in out:
                b.setdefault("_prefetch", "entity_neighbour")
        except Exception as exc:  # pragma: no cover — defensive
            _log.warning("entity_prefetch_graph_expand_failed", error=str(exc))

    _log.info(
        "entity_prefetch",
        query_candidates=len(candidates),
        matched_entities=len(matched),
        returned=len(out),
    )
    return out


def is_entity_prefetch_enabled(config: dict[str, Any] | None) -> bool:
    """Whether entity prefetch should fire for the current call."""
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    ep = retrieval.get("entity_prefetch", {})
    if not isinstance(ep, dict):
        return False
    if ep.get("enabled", False):
        return True
    return bool(ep.get("auto_enable", True))


def resolve_entity_prefetch_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Pull prefetch parameters from config with safe defaults."""
    defaults: dict[str, Any] = {
        "max_entities": 3,
        "max_hops": 1,
        "entity_score": 5.0,
    }
    if not config or not isinstance(config, dict):
        return defaults
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return defaults
    ep = retrieval.get("entity_prefetch", {})
    if not isinstance(ep, dict):
        return defaults
    out = dict(defaults)
    if isinstance(ep.get("max_entities"), int) and ep["max_entities"] > 0:
        out["max_entities"] = int(ep["max_entities"])
    if isinstance(ep.get("max_hops"), int) and ep["max_hops"] >= 0:
        out["max_hops"] = int(ep["max_hops"])
    if isinstance(ep.get("entity_score"), (int, float)) and ep["entity_score"] > 0:
        out["entity_score"] = float(ep["entity_score"])
    return out


__all__ = [
    "extract_entity_candidates",
    "prefetch_entity_blocks",
    "is_entity_prefetch_enabled",
    "resolve_entity_prefetch_config",
]

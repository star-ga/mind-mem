"""Granularity / abstraction alignment — named merge operation (Group H, v4.0.x).

*Granularity alignment* addresses the "duplicate-memory pain": as a workspace
grows, semantically equivalent or strongly overlapping blocks accumulate at
different levels of abstraction (one block says "use Q16.16 for determinism",
another says "all scoring must use fixed-point arithmetic for bit-identity").
Neither is wrong, but both pollute recall ranking and make consolidation
harder.

This module provides a **named merge operation** that:

1. Detects pairs of blocks whose content overlaps above a configurable
   similarity threshold (``min_similarity``, default 0.75).
2. Returns ``GranularityMergeCandidate`` records describing *why* a pair
   is a merge candidate, which block is the "source" vs "detail", and a
   suggested merged representation.
3. Never writes to the workspace — all proposals are returned as data so
   callers can route them through the existing HITL ``propose_update`` /
   ``approve_apply`` governance gate.

**Design constraints**

- Optional and additive — no existing recall path is affected.  Callers
  opt in by calling :func:`find_merge_candidates`.
- Stateless per block — similarity is computed from fields already
  present in each block dict.  No extra DB round-trip.
- Deterministic — given the same set of blocks and the same threshold,
  the candidate list is always produced in the same order (sorted by
  descending similarity, then by ``(id_a, id_b)``).
- HITL-gated — the module never modifies the workspace; it only proposes.

**Similarity metric**

Term-frequency cosine similarity over the union of ``excerpt`` /
``content`` / ``tags`` fields (same tokenizer as ``dedup.py`` Layer 2)
is used as the primary signal.  This is fast, deterministic, and requires
no external dependencies.

**Merge strategies**

Three strategies control how :func:`merge_blocks` synthesises the
merged representation:

``keep_longer``
    Keep the block whose combined text is longer (richer detail).
    Metadata (status, lifecycle, tags) is merged from both.

``keep_higher_maturity``
    Keep the block whose maturity score is higher; fall back to
    ``keep_longer`` on a tie.

``concatenate``
    Concatenate both blocks' content, separated by a blank line.
    Useful when both blocks contribute non-overlapping detail.

Usage::

    from mind_mem.granularity_align import (
        GranularityMergeCandidate,
        find_merge_candidates,
        merge_blocks,
    )

    candidates = find_merge_candidates(blocks, min_similarity=0.75)
    for c in candidates:
        merged = merge_blocks(c.block_a, c.block_b, strategy="keep_longer")
        # route merged through propose_update for human approval
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_MIN_SIMILARITY",
    "MERGE_STRATEGIES",
    "GranularityMergeCandidate",
    "find_merge_candidates",
    "merge_blocks",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default similarity threshold.  Pairs below this value are not returned.
DEFAULT_MIN_SIMILARITY: float = 0.75

#: Supported merge strategies.
MERGE_STRATEGIES: frozenset[str] = frozenset(
    {"keep_longer", "keep_higher_maturity", "concatenate"}
)

# ---------------------------------------------------------------------------
# Text utilities (mirror of dedup.py tokenizer — no shared import to avoid
# coupling the two modules at import time)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
        "and", "or", "but", "not", "no", "if", "then", "so", "up", "out",
        "that", "this", "these", "those", "he", "she", "they", "we", "you",
        "i", "me", "my", "your", "his", "her", "our", "their",
    }
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS and len(t) > 1]


def _term_vec(tokens: list[str]) -> dict[str, int]:
    vec: dict[str, int] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def _cosine(va: dict[str, int], vb: dict[str, int]) -> float:
    if not va or not vb:
        return 0.0
    if len(va) > len(vb):
        va, vb = vb, va
    dot = sum(va[t] * vb[t] for t in va if t in vb)
    if dot == 0:
        return 0.0
    mag_a = math.sqrt(sum(v * v for v in va.values()))
    mag_b = math.sqrt(sum(v * v for v in vb.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _block_text(block: dict[str, Any]) -> str:
    """Extract searchable text from a block dict."""
    parts: list[str] = []
    for key in ("excerpt", "content", "tags"):
        val = block.get(key, "")
        if val:
            parts.append(str(val))
    return " ".join(parts)


def _block_id(block: dict[str, Any]) -> str:
    return str(block.get("_id") or block.get("id") or "")


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GranularityMergeCandidate:
    """A detected pair of blocks that are candidates for merging.

    Attributes:
        block_a: The first block dict (higher score / picked deterministically).
        block_b: The second block dict.
        similarity: Term-frequency cosine similarity in [0.0, 1.0].
        reason: Human-readable explanation of why this pair was flagged.
        suggested_strategy: Recommended merge strategy for this pair.
    """

    block_a: dict[str, Any]
    block_b: dict[str, Any]
    similarity: float
    reason: str
    suggested_strategy: str = "keep_longer"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id_a": _block_id(self.block_a),
            "id_b": _block_id(self.block_b),
            "similarity": round(self.similarity, 4),
            "reason": self.reason,
            "suggested_strategy": self.suggested_strategy,
        }


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Compute text-overlap similarity between two block dicts."""
    ta = _tokenize(_block_text(a))
    tb = _tokenize(_block_text(b))
    return _cosine(_term_vec(ta), _term_vec(tb))


def _suggest_strategy(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Heuristically pick a merge strategy for the pair.

    - When both blocks have an explicit ``Maturity`` field that differs,
      prefer ``keep_higher_maturity``.
    - Otherwise, prefer ``keep_longer`` (safe default).
    """
    has_maturity_a = "Maturity" in a or "maturity" in a
    has_maturity_b = "Maturity" in b or "maturity" in b
    if has_maturity_a or has_maturity_b:
        return "keep_higher_maturity"
    return "keep_longer"


def _build_reason(similarity: float) -> str:
    if similarity >= 0.95:
        return f"near-identical content (similarity={similarity:.2f}); likely duplicated block"
    if similarity >= 0.85:
        return f"high overlap (similarity={similarity:.2f}); same claim at different abstraction levels"
    return f"moderate overlap (similarity={similarity:.2f}); may be redundant"


def find_merge_candidates(
    blocks: list[dict[str, Any]],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_candidates: int = 50,
) -> list[GranularityMergeCandidate]:
    """Detect block pairs that are candidates for granularity alignment.

    Computes pairwise cosine similarity between all block pairs.  Pairs
    whose similarity meets ``min_similarity`` are returned as
    :class:`GranularityMergeCandidate` records, sorted by descending
    similarity, then by ``(id_a, id_b)`` for determinism.

    Blocks with empty combined text are skipped (no signal to compare).

    Args:
        blocks: List of block dicts as returned by the recall pipeline or
            ``block_parser``.
        min_similarity: Minimum cosine similarity (in [0, 1]) for a pair
            to be flagged.  Default 0.75.  Set to 0.0 to return all pairs;
            set to 1.0 to return only exact duplicates.
        max_candidates: Safety cap on the number of returned candidates.
            Default 50.  Set to ``0`` to disable the cap.

    Returns:
        List of :class:`GranularityMergeCandidate` records, sorted
        descending by similarity (deterministic tie-breaking by id pair).
    """
    threshold = max(0.0, min(1.0, float(min_similarity)))
    cap = int(max_candidates) if max_candidates > 0 else 0

    # Pre-compute term vectors (skip blocks with no text signal)
    indexed: list[tuple[dict[str, Any], dict[str, int]]] = []
    for blk in blocks:
        tokens = _tokenize(_block_text(blk))
        if tokens:
            indexed.append((blk, _term_vec(tokens)))

    candidates: list[GranularityMergeCandidate] = []

    for i in range(len(indexed)):
        a_blk, va = indexed[i]
        for j in range(i + 1, len(indexed)):
            b_blk, vb = indexed[j]
            sim = _cosine(va, vb)
            if sim < threshold:
                continue
            strategy = _suggest_strategy(a_blk, b_blk)
            reason = _build_reason(sim)
            candidates.append(
                GranularityMergeCandidate(
                    block_a=a_blk,
                    block_b=b_blk,
                    similarity=round(sim, 6),
                    reason=reason,
                    suggested_strategy=strategy,
                )
            )

    # Deterministic sort: descending similarity, then by id pair
    candidates.sort(
        key=lambda c: (-c.similarity, _block_id(c.block_a), _block_id(c.block_b))
    )

    if cap:
        candidates = candidates[:cap]

    return candidates


# ---------------------------------------------------------------------------
# Merge helper (pure — never writes)
# ---------------------------------------------------------------------------


def _maturity_score_simple(block: dict[str, Any]) -> float:
    """Lightweight maturity score for merge-strategy selection.

    Avoids importing ``block_maturity`` at module load time (keeps this
    module dependency-free).  Reads only the ``Maturity`` frontmatter
    field; if absent, returns 0.5 (neutral).
    """
    # Use a sentinel so a numeric Maturity=0 / 0.0 (falsy) is not confused
    # with an absent key.  A plain `or` would skip legitimate zero values.
    _MISSING = object()
    raw = block.get("Maturity", _MISSING)
    if raw is _MISSING:
        raw = block.get("maturity", _MISSING)
    if raw is not _MISSING:
        try:
            return max(0.0, min(1.0, float(raw)))
        except (ValueError, TypeError):
            pass
    return 0.5


def _merge_tags(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Union of tags from both blocks, deduped and comma-joined."""
    raw_a = str(a.get("tags") or a.get("Tags") or "")
    raw_b = str(b.get("tags") or b.get("Tags") or "")
    parts: list[str] = []
    seen: set[str] = set()
    for raw in (raw_a, raw_b):
        for t in re.split(r"[,\s]+", raw):
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                parts.append(t)
    return ", ".join(parts)


def merge_blocks(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    strategy: str = "keep_longer",
) -> dict[str, Any]:
    """Produce a merged block dict from two candidates.

    This function is **purely functional** — it never writes to a
    workspace.  The caller is responsible for routing the result through
    the ``propose_update`` / ``approve_apply`` governance gate before
    any persistent write.

    Args:
        a: First block dict (e.g. ``candidate.block_a``).
        b: Second block dict.
        strategy: One of ``"keep_longer"``, ``"keep_higher_maturity"``,
            or ``"concatenate"``.  Defaults to ``"keep_longer"``.

    Returns:
        A new dict representing the merged block.  The ``_id`` field is
        set to the id of the block that "won" (for ``keep_*`` strategies)
        or the id of *a* (for ``concatenate``).  A ``_merged_from`` key
        records the pair of source ids for audit purposes.

    Raises:
        ValueError: If ``strategy`` is not a recognised value.
    """
    if strategy not in MERGE_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(MERGE_STRATEGIES)}, got {strategy!r}"
        )

    id_a = _block_id(a)
    id_b = _block_id(b)
    merged_tags = _merge_tags(a, b)

    if strategy == "concatenate":
        content_a = str(a.get("content") or a.get("excerpt") or "").strip()
        content_b = str(b.get("content") or b.get("excerpt") or "").strip()
        merged_content = f"{content_a}\n\n{content_b}".strip()
        base = dict(a)
        base["_id"] = id_a
        base["content"] = merged_content
        base["excerpt"] = merged_content[:500] if len(merged_content) > 500 else merged_content
        base["tags"] = merged_tags
        base["_merged_from"] = [id_a, id_b]
        return base

    # keep_longer or keep_higher_maturity: pick a "winner" block
    if strategy == "keep_higher_maturity":
        score_a = _maturity_score_simple(a)
        score_b = _maturity_score_simple(b)
        winner, loser = (a, b) if score_a >= score_b else (b, a)
    else:  # keep_longer
        len_a = len(_block_text(a))
        len_b = len(_block_text(b))
        winner, loser = (a, b) if len_a >= len_b else (b, a)

    merged = dict(winner)
    # Enrich the winner's tags from the loser
    merged["tags"] = merged_tags
    # Preserve the loser's id for audit trail
    merged["_merged_from"] = [_block_id(winner), _block_id(loser)]
    return merged

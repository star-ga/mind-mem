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

from .fact_slots import canonical_tokens

__all__ = [
    "DEFAULT_MIN_SIMILARITY",
    "MERGE_STRATEGIES",
    "GranularityMergeCandidate",
    "SimilarPair",
    "find_similar_pairs",
    "find_merge_candidates",
    "scan_pairs",
    "merge_blocks",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default similarity threshold.  Pairs below this value are not returned.
DEFAULT_MIN_SIMILARITY: float = 0.75

#: Supported merge strategies.
MERGE_STRATEGIES: frozenset[str] = frozenset({"keep_longer", "keep_higher_maturity", "concatenate"})

# ---------------------------------------------------------------------------
# Text utilities (mirror of dedup.py tokenizer — no shared import to avoid
# coupling the two modules at import time)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "it",
        "its",
        "and",
        "or",
        "but",
        "not",
        "no",
        "if",
        "then",
        "so",
        "up",
        "out",
        "that",
        "this",
        "these",
        "those",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "me",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
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


#: Block fields that are bookkeeping, not content: they may differ between exact
#: duplicates. Everything else (Date, Scope, speaker, Tags, Supersedes, ...) is
#: content and must be identical for a merge to be proposed.
_BOOKKEEPING = frozenset(
    {
        "_id",
        "id",
        "_source_file",
        "file",
        "line",
        "score",
        "status",
        "Status",
        "importance",
        "Importance",
        "Maturity",
        "maturity",
        "access_count",
        "AccessCount",
        "lifecycle",
        "Lifecycle",
        "created",
        "updated",
        "Created",
        "Updated",
        "last_access",
        "LastAccess",
        "Sources",
        "sources",
        # Handled by _marked_superseded: a real successor blocks the merge; an empty one is noise.
        "SupersededBy",
        "Superseded_by",
        "superseded_by",
        "supersededBy",
        "History",
        "history",
        "Evidence",
        "evidence",
        "fields",
    }
)


#: Provenance fields: they may differ between exact duplicates, and a merge
#: keeps the union of both blocks' values (see :func:`merge_blocks`).
_PROVENANCE = ("Sources", "sources", "Evidence", "evidence", "History", "history")

#: Statuses whose blocks may be merged. A superseded, archived or retracted copy
#: is history, not a duplicate: merging it would erase the record of the change.
_MERGEABLE_STATUSES = frozenset({"", "active"})


def _as_text(value: Any) -> str:
    """A field value as text: list items one per line, anything else via str()."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(v) for v in value)
    return str(value)


def _key_order(key: Any) -> tuple[str, str]:
    """Sort key for mapping keys of any type (str, int, tuple, ...): never compares across types."""
    return (type(key).__name__, repr(key))


def _canon_value(value: Any) -> Any:
    """A hashable, order-stable normal form of a field value of any shape.

    Strings use the merge normal form (fact_slots.canonical_tokens). Lists keep
    their order (``Participants: - Ilona`` vs ``- the contractor`` differ);
    mappings are compared key by key, whatever the key types.
    """
    if isinstance(value, str):
        return ("s", canonical_tokens(value))
    if isinstance(value, (list, tuple)):
        return ("l", tuple(_canon_value(v) for v in value))
    if isinstance(value, dict):
        return ("d", tuple((_key_order(k), _canon_value(v)) for k, v in sorted(value.items(), key=lambda kv: _key_order(kv[0]))))
    if isinstance(value, (set, frozenset)):
        return ("S", tuple(sorted((_canon_value(v) for v in value), key=repr)))
    return ("v", type(value).__name__, repr(value))


def _content_fields(block: dict[str, Any]) -> dict[Any, Any]:
    def keep(k: Any, v: Any) -> bool:
        return k not in _BOOKKEEPING and not str(k).startswith("_") and v not in (None, "", [], {})

    out = {k: v for k, v in block.items() if keep(k, v)}
    nested = block.get("fields")
    if isinstance(nested, dict):  # full frontmatter carried by the plan_consolidation projection
        out.update({k: v for k, v in nested.items() if keep(k, v)})
    return out


def _status(block: dict[str, Any]) -> str:
    raw = block.get("Status", block.get("status"))
    if raw is None and isinstance(block.get("fields"), dict):
        raw = block["fields"].get("Status", block["fields"].get("status"))
    return str(raw or "").strip().lower()


_ID_RE = re.compile(r"\b[A-Z]+-[A-Za-z0-9_.-]+\b")


def _superseded_by_reference(blocks: Any) -> set[str]:
    """IDs named in any block's Supersedes field (top level or carried frontmatter).

    Such a block is history even if its own Status was never updated, so it is
    never a merge survivor.
    """
    out: set[str] = set()
    for blk in blocks:
        nested = blk.get("fields") if isinstance(blk.get("fields"), dict) else {}
        for raw in (blk.get("Supersedes"), nested.get("Supersedes")):
            for value in raw if isinstance(raw, list) else [raw]:
                if isinstance(value, str) and value.strip().lower() not in ("", "none", "n/a"):
                    out.update(
                        t.casefold()
                        for t in re.findall(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?", value)
                        if t.casefold() not in ("none", "n/a")
                    )
    return out


def _empty_ref(value: Any) -> bool:
    """None, '', 'none', 'null', 'n/a', '-' and whitespace are placeholders, not a successor."""
    return value is None or str(value).strip().lower() in ("", "none", "null", "n/a", "na", "-", "—")


def _marked_superseded(block: dict[str, Any]) -> bool:
    """A block whose own SupersededBy / Superseded_by field names a successor is history."""
    raw_fields = block.get("fields")
    nested: dict[str, Any] = raw_fields if isinstance(raw_fields, dict) else {}
    for key in ("SupersededBy", "Superseded_by", "superseded_by", "supersededBy"):
        for raw in (block.get(key), nested.get(key)):
            if isinstance(raw, list) and any(not _empty_ref(v) for v in raw):
                return True
            if isinstance(raw, str) and not _empty_ref(raw):
                return True
            if raw is not None and not isinstance(raw, (str, list)) and raw not in (0, {}, False):
                return True  # a store may hold an int or a mapping
    return False


def _identity(block: dict[str, Any]) -> tuple:
    """Everything that makes a block the memory it is, normalised (see fact_slots.canonical_tokens).

    Two blocks are the SAME memory only when their identities are equal. Every
    content field counts, whatever its shape (string, list, mapping), so a
    who/when/zone value held in a list field still separates two blocks.
    Computed once per block per call, so the pairwise scan compares cheap tuples.
    """
    text = " ".join(_as_text(block.get(k)) for k in ("excerpt", "content") if block.get(k))
    fields = _content_fields(block)
    for key in ("excerpt", "content"):
        fields.pop(key, None)
    # Lines the block parser did not map to a field still make the block what it is.
    raw_fields = block.get("fields")
    nested: dict[str, Any] = raw_fields if isinstance(raw_fields, dict) else {}
    unparsed = block.get("_unparsed") or nested.get("_unparsed") or []
    ordered = sorted(fields.items(), key=lambda kv: _key_order(kv[0]))
    return (
        canonical_tokens(text),
        tuple((_key_order(k), _canon_value(v)) for k, v in ordered),
        _canon_value(list(unparsed)),
    )


def _facts_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True unless the two blocks are the SAME memory (every content field equal after normalisation).

    A merge proposal is only ever made for exact duplicates: deciding that two
    differently worded statements mean the same thing is a semantic judgement
    that text overlap cannot make safely ("call with Ilona Tue 10am Arizona"
    vs "call with the contractor Tue 10am Pacific" share most words, yet
    merging them loses a memory). Differently worded look-alikes are reported
    as ``similar_pairs`` for a human, never as merge candidates.
    """
    return _identity(a) != _identity(b)


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


@dataclass(frozen=True)
class SimilarPair:
    """Two look-alike blocks that are NOT exact duplicates: shown to a reviewer, never proposed for merge."""

    id_a: str
    id_b: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_a": self.id_a,
            "id_b": self.id_b,
            "similarity": round(self.similarity, 4),
            "reason": "similar wording but the statements differ; not proposed for merge (review by hand)",
        }


def scan_pairs(
    blocks: list[dict[str, Any]],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_candidates: int = 50,
    max_pairs: int = 50,
    superseded_ids: set[str] | None = None,
) -> tuple[list[GranularityMergeCandidate], list[SimilarPair]]:
    """One pairwise pass returning (merge candidates, advisory look-alike pairs).

    ``superseded_ids`` adds ids superseded by blocks outside ``blocks`` (a caller
    that scans a window of a larger corpus passes the corpus-wide set).

    A merge candidate is a pair of EXACT duplicates (equal :func:`_identity`),
    both active. ``min_similarity`` bounds the look-alike report; exact
    duplicates score ~1.0, so it never removes a merge candidate.
    Blocks with empty combined text are skipped (no signal to compare).
    """
    threshold = max(0.0, min(1.0, float(min_similarity)))
    # Non-mapping entries carry no fields to compare: skipped, never a crash.
    indexed = [(blk, _term_vec(toks)) for blk in blocks if isinstance(blk, dict) and (toks := _tokenize(_block_text(blk)))]
    identities = [_identity(blk) for blk, _ in indexed]
    # Every block's Supersedes counts, including one with no comparable text of its own.
    replaced = _superseded_by_reference(blk for blk in blocks if isinstance(blk, dict))
    replaced |= {str(bid).casefold() for bid in (superseded_ids or ())}
    mergeable = [
        _status(blk) in _MERGEABLE_STATUSES and str(_block_id(blk)).casefold() not in replaced and not _marked_superseded(blk)
        for blk, _ in indexed
    ]
    candidates: list[GranularityMergeCandidate] = []
    pairs: list[SimilarPair] = []
    for i in range(len(indexed)):
        a_blk, va = indexed[i]
        for j in range(i + 1, len(indexed)):
            b_blk, vb = indexed[j]
            same = identities[i] == identities[j]
            sim = _cosine(va, vb)
            if same and mergeable[i] and mergeable[j]:
                candidates.append(
                    GranularityMergeCandidate(
                        block_a=a_blk,
                        block_b=b_blk,
                        similarity=round(sim, 6),
                        reason=_build_reason(sim),
                        suggested_strategy=_suggest_strategy(a_blk, b_blk),
                    )
                )
            elif not same and sim >= threshold:
                pairs.append(SimilarPair(_block_id(a_blk), _block_id(b_blk), round(sim, 6)))
    candidates.sort(key=lambda c: (-c.similarity, _block_id(c.block_a), _block_id(c.block_b)))
    pairs.sort(key=lambda p: (-p.similarity, p.id_a, p.id_b))
    if max_candidates > 0:
        candidates = candidates[: int(max_candidates)]
    if max_pairs > 0:
        pairs = pairs[: int(max_pairs)]
    return candidates, pairs


def find_merge_candidates(
    blocks: list[dict[str, Any]],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_candidates: int = 50,
) -> list[GranularityMergeCandidate]:
    """Exact-duplicate, active block pairs, sorted by descending similarity then id pair.

    Only blocks whose every content field is equal after normalisation are
    proposed (see :func:`_facts_conflict`). ``min_similarity`` is kept for API
    compatibility and bounds only :func:`find_similar_pairs`. ``max_candidates``
    caps the result (``0`` disables the cap).
    """
    return scan_pairs(blocks, min_similarity=min_similarity, max_candidates=max_candidates, max_pairs=1)[0]


def find_similar_pairs(
    blocks: list[dict[str, Any]],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_pairs: int = 50,
) -> list[SimilarPair]:
    """Look-alike pairs above ``min_similarity`` that are not exact duplicates (advisory only)."""
    return scan_pairs(blocks, min_similarity=min_similarity, max_candidates=1, max_pairs=max_pairs)[1]


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


def _union_values(a: Any, b: Any) -> Any:
    """Order-preserving union of two provenance values (strings or lists)."""
    items: list[Any] = []
    for value in (a, b):
        if value in (None, ""):
            continue
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if item not in items:
                items.append(item)
    if all(isinstance(i, str) for i in items) and not isinstance(a, (list, tuple)) and not isinstance(b, (list, tuple)):
        return "; ".join(items)
    return items


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
        raise ValueError(f"strategy must be one of {sorted(MERGE_STRATEGIES)}, got {strategy!r}")

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
    # Exact duplicates may carry different provenance: keep both blocks' records.
    for key in _PROVENANCE:
        if key in winner or key in loser:
            merged[key] = _union_values(winner.get(key), loser.get(key))
    w_fields, l_fields = winner.get("fields"), loser.get("fields")
    if isinstance(w_fields, dict) and isinstance(l_fields, dict):
        merged["fields"] = dict(w_fields)
        for key in _PROVENANCE:
            if key in w_fields or key in l_fields:
                merged["fields"][key] = _union_values(w_fields.get(key), l_fields.get(key))
    # Preserve the loser's id for audit trail
    merged["_merged_from"] = [_block_id(winner), _block_id(loser)]
    return merged

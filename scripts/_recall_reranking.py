"""Recall engine reranking — deterministic feature-based re-scoring of BM25 hits."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from ._recall_constants import MONTH_TOKEN_SET
from ._recall_scoring import (
    _category_match_boost,
    _date_proximity_score,
    _detect_negation,
    _extract_bigram_phrases,
    _extract_entities,
    _extract_speaker_names,
    _negation_penalty,
)
from .observability import get_logger

_log = get_logger("reranking")

__all__ = ["rerank_hits", "llm_rerank"]


# ---------------------------------------------------------------------------
# v7: Deterministic Reranker — wider retrieve + feature-based re-scoring
# ---------------------------------------------------------------------------

# Regex for detecting time-intent in queries
_TIME_INTENT_RE = re.compile(
    r"\b(what month|what day|when|what date|what year|what week|how long ago"
    r"|which month|which year|which day|what time"
    r"|as mentioned on|mentioned on|on\s+(?:january|february|march|april|may|june"
    r"|july|august|september|october|november|december)"
    r"|on\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june"
    r"|july|august|september|october|november|december)"
    r"|in\s+(?:january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s+\d{4}"
    r"|on\s+\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)

_DAY_TOKENS = frozenset(
    {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
    }
)
_TEMPORAL_CONTENT_TOKENS = (
    MONTH_TOKEN_SET
    | _DAY_TOKENS
    | frozenset(
        {
            "yesterday",
            "tomorrow",
            "tonight",
            "weekend",
            "spring",
            "summer",
            "fall",
            "winter",
            "autumn",
            "holiday",
            "birthday",
            "anniversary",
            "christmas",
            "planned",
            "planning",
            "going",
            "trip",
            "visit",
            "travel",
            "vacation",
        }
    )
)

# Reranker feature weights (tuned for LoCoMo coverage)
# Reduced speaker dominance (was 0.40), increased entity/phrase overlap
# to avoid "right speaker, wrong topic" over-preference.
_RERANK_W_ENTITY = 0.30
_RERANK_W_TIME = 0.15
_RERANK_W_BIGRAM = 0.15
_RERANK_W_RECENCY = 0.10
_RERANK_W_SPEAKER = 0.25
_RERANK_W_SPEAKER_MISMATCH = -0.10


def rerank_hits(
    query: str,
    hits: list[dict],
    debug: bool = False,
) -> list[dict]:
    """Deterministic reranker: rescores BM25 hits with entity/time/phrase/recency/speaker features.

    Each feature is in [0,1]. Final score = bm25_score + weighted sum of features.
    The top-1 BM25 hit is always preserved in the final set (anchor rule).

    Args:
        query: The original search query.
        hits: BM25-scored results (must have 'score', 'excerpt', 'speaker', optionally 'DiaID').
        debug: If True, attach '_rerank_features' dict to each hit.

    Returns:
        Hits with updated scores, sorted descending.
    """
    if not hits:
        return hits

    # Preserve BM25 top-1 anchor
    bm25_top1_id = hits[0]["_id"] if hits else None

    # --- Extract query signals ---
    q_entities = _extract_entities(query)
    q_phrases = _extract_bigram_phrases(query)
    q_lower = query.lower()
    has_time_intent = bool(_TIME_INTENT_RE.search(q_lower))
    q_mentioned_speakers = _extract_speaker_names(query, hits)
    has_neg, negated_terms = _detect_negation(query)

    # Plan-related boosting: if query mentions plan/going/trip, boost those verbs
    plan_intent = bool(re.search(r"\b(plan|going|trip|visit|travel|vacation)\b", q_lower))

    # Find max DiaID for recency normalization
    max_dia = 0
    for h in hits:
        dia = h.get("DiaID", "")
        if dia:
            try:
                # DiaID format varies: could be numeric or "D123" etc.
                dia_num = int(re.sub(r"[^0-9]", "", dia) or "0")
                max_dia = max(max_dia, dia_num)
            except (ValueError, TypeError):
                pass
    if max_dia == 0:
        # Fallback: use position in list
        max_dia = len(hits)

    for idx, h in enumerate(hits):
        bm25_score = h["score"]
        excerpt = h.get("excerpt", "")
        excerpt_lower = excerpt.lower()
        tags_str = h.get("tags", "")
        speaker = h.get("speaker", "").lower()

        # (a) Entity overlap
        h_entities = _extract_entities(excerpt)
        # Also extract from tags
        if tags_str:
            h_entities |= _extract_entities(tags_str)
        if q_entities:
            entity_overlap = len(q_entities & h_entities) / max(1, len(q_entities))
        else:
            entity_overlap = 0.0

        # (b) Time overlap
        time_overlap = 0.0
        if has_time_intent:
            # Check if hit contains temporal content tokens
            hit_tokens = set(re.findall(r"[a-z]+", excerpt_lower))
            temporal_matches = hit_tokens & _TEMPORAL_CONTENT_TOKENS
            if temporal_matches:
                time_overlap = min(1.0, len(temporal_matches) / 2.0)
            # Also check for date patterns (YYYY-MM-DD, "Month Day", etc.)
            if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", excerpt):
                time_overlap = max(time_overlap, 0.5)
            month_pat = (
                r"\b(?:January|February|March|April|May|June|July|August"
                r"|September|October|November|December)\s+\d{1,2}\b"
            )
            if re.search(month_pat, excerpt, re.IGNORECASE):
                time_overlap = max(time_overlap, 0.8)
            # Plan-intent boost
            if plan_intent and re.search(r"\b(plan|going|trip|visit|travel|vacation)\b", excerpt_lower):
                time_overlap = max(time_overlap, 0.6)

        # (c) Bigram phrase bonus
        bigram_bonus = 0.0
        if q_phrases:
            for phrase in q_phrases:
                if phrase in excerpt_lower:
                    bigram_bonus = 1.0
                    break

        # (d) Recency bonus (later turns preferred)
        recency_bonus = 0.5  # default mid-range
        dia = h.get("DiaID", "")
        if dia:
            try:
                dia_num = int(re.sub(r"[^0-9]", "", dia) or "0")
                if max_dia > 0:
                    recency_bonus = dia_num / max_dia
            except (ValueError, TypeError):
                pass
        else:
            # Fallback: use line number if available
            line = h.get("line", 0)
            if line > 0:
                recency_bonus = min(1.0, line / 1000.0)

        # (e) Speaker boost / mismatch penalty
        speaker_bonus = 0.0
        if q_mentioned_speakers:
            if speaker and speaker in q_mentioned_speakers:
                speaker_bonus = 1.0
            elif speaker and speaker not in q_mentioned_speakers:
                # Block belongs to a different speaker than the one asked about
                speaker_bonus = _RERANK_W_SPEAKER_MISMATCH / max(abs(_RERANK_W_SPEAKER), 0.01)
            # No speaker tag -> neutral (0.0)

        # --- Combine ---
        feature_sum = (
            _RERANK_W_ENTITY * entity_overlap
            + _RERANK_W_TIME * time_overlap
            + _RERANK_W_BIGRAM * bigram_bonus
            + _RERANK_W_RECENCY * recency_bonus
            + _RERANK_W_SPEAKER * speaker_bonus
        )
        h["score"] = round(bm25_score + feature_sum, 4)

        # --- Phase 4: multiplicative reranking features ---
        block_text = h.get("excerpt", "")
        if has_neg:
            h["score"] *= _negation_penalty(block_text, negated_terms)
        h["score"] *= _date_proximity_score(query, block_text)
        h["score"] *= _category_match_boost(query, block_text)
        h["score"] = round(h["score"], 4)

        if debug:
            h["_rerank_features"] = {
                "entity_overlap": round(entity_overlap, 3),
                "time_overlap": round(time_overlap, 3),
                "bigram_bonus": round(bigram_bonus, 3),
                "recency_bonus": round(recency_bonus, 3),
                "speaker_bonus": round(speaker_bonus, 3),
                "feature_sum": round(feature_sum, 4),
                "bm25_original": round(bm25_score, 4),
            }

    # Sort by reranked score
    hits.sort(key=lambda r: (r["score"], r.get("_id", "")), reverse=True)

    # Anchor rule: ensure BM25 top-1 is in final set
    if bm25_top1_id:
        if bm25_top1_id not in {h["_id"] for h in hits[:10]}:
            # Find it and swap into position
            for i, h in enumerate(hits):
                if h["_id"] == bm25_top1_id:
                    # Insert at position 1 (keep reranked #1, add anchor at #2)
                    anchor = hits.pop(i)
                    hits.insert(min(1, len(hits)), anchor)
                    break

    return hits


# ---------------------------------------------------------------------------
# v8: Optional LLM Reranking — config-gated, uses urllib.request (stdlib)
# ---------------------------------------------------------------------------


def llm_rerank(
    query: str,
    hits: list[dict],
    *,
    url: str = "http://localhost:11434/api/generate",
    model: str = "qwen3-coder:30b",
    weight: float = 0.3,
    timeout: float = 10.0,
) -> list[dict]:
    """Optional LLM-based reranking via local Ollama (or compatible API).

    Sends query + candidate excerpts to the LLM, asks it to score relevance
    as a JSON array of floats in [0,1]. Merges LLM scores with existing scores
    using the given weight.

    Falls back silently (returns original hits) on any failure.

    Args:
        query: The original search query.
        hits: Candidates already scored by deterministic reranker.
        url: Ollama-compatible generate endpoint.
        model: Model name to use.
        weight: Blend weight for LLM scores (0=ignore, 1=replace).
        timeout: HTTP request timeout in seconds.

    Returns:
        Hits with blended scores, sorted descending.
    """
    if not hits:
        return hits

    # Build scoring prompt with numbered candidates
    candidates = []
    for i, h in enumerate(hits):
        excerpt = h.get("excerpt", "")[:200]
        candidates.append(f"{i}: {excerpt}")

    prompt = (
        f"Score the relevance of each candidate to the query.\n"
        f"Query: {query}\n\n"
        f"Candidates:\n" + "\n".join(candidates) + "\n\n"
        f"Return ONLY a JSON array of {len(hits)} floats in [0,1], "
        f"one per candidate. Example: [0.95, 0.3, 0.7]\n"
        f"No explanation, just the JSON array."
    )

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        }
    ).encode()

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())

        response_text = body.get("response", "")
        # Extract JSON array from response (may have surrounding text)
        match = re.search(r"\[[\d.,\s]+\]", response_text)
        if not match:
            _log.warning("llm_rerank_parse_failed", response=response_text[:200])
            return hits

        scores = json.loads(match.group())
        if len(scores) != len(hits):
            _log.warning("llm_rerank_length_mismatch", expected=len(hits), got=len(scores))
            return hits

        # Validate and clamp scores
        for i, s in enumerate(scores):
            if not isinstance(s, (int, float)):
                scores[i] = 0.5
            else:
                scores[i] = max(0.0, min(1.0, float(s)))

        # Blend: final_score = (1 - weight) * existing + weight * llm_score * max_existing
        max_existing = max((h["score"] for h in hits), default=1.0) or 1.0
        for i, h in enumerate(hits):
            blended = (1 - weight) * h["score"] + weight * scores[i] * max_existing
            h["score"] = round(blended, 4)

        hits.sort(key=lambda r: (r["score"], r.get("_id", "")), reverse=True)
        _log.info("llm_rerank_applied", model=model, candidates=len(hits), weight=weight)

    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        _log.warning("llm_rerank_network_failed", error=str(e))
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
        _log.warning("llm_rerank_failed", error=str(e))
    except Exception as e:
        _log.warning("llm_rerank_unexpected_error", error=str(e))

    return hits

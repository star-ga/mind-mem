"""Recall engine scoring — BM25F helper, date scores, graph boosting, negation, date proximity, categories."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime as _datetime

from ._recall_constants import _BLOCK_ID_RE, BM25_B, BM25_K1, FIELD_WEIGHTS, SEARCH_FIELDS

__all__ = [
    "bm25f_score_terms",
    "compute_weighted_tf",
    "date_score",
    "build_xref_graph",
    "_detect_negation",
    "_negation_penalty",
    "_extract_dates",
    "_date_proximity_score",
    "_classify_categories",
    "_category_match_boost",
    "_extract_entities",
    "_extract_bigram_phrases",
    "_extract_speaker_names",
]


# ---------------------------------------------------------------------------
# BM25F scoring helper — single source of truth for all BM25 computations
# ---------------------------------------------------------------------------


def compute_weighted_tf(
    field_tokens: dict[str, list[str]],
    field_weights: dict[str, float] | None = None,
) -> tuple[Counter, float]:
    """Compute field-weighted term frequency and weighted document length.

    Args:
        field_tokens: {field_name: [tokens]} for a single document.
        field_weights: Per-field weight multipliers (defaults to FIELD_WEIGHTS).

    Returns:
        (weighted_tf Counter, weighted_doc_length float).
    """
    fw = field_weights or FIELD_WEIGHTS
    weighted_tf: Counter[str] = Counter()
    wdl = 0.0
    for field, tokens in field_tokens.items():
        w = fw.get(field, 1.0)
        wdl += len(tokens) * w
        for t in tokens:
            weighted_tf[t] += w  # type: ignore[assignment]
    return weighted_tf, wdl


def bm25f_score_terms(
    query_terms: list[str],
    weighted_tf: Counter,
    wdl: float,
    idf_cache: dict[str, float],
    avg_wdl: float,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    """Score a document against query terms using the BM25F formula.

    This is the single implementation of BM25F used by the main scoring loop,
    RM3 re-scoring, PRF re-scoring, and bridge (chain-of-retrieval) re-scoring.

    BM25F: sum over query terms of  idf(t) * (wtf * (k1+1)) / (wtf + k1*(1-b+b*wdl/avgdl))

    Args:
        query_terms: Tokenized query terms to score against.
        weighted_tf: Field-weighted term frequency counter for the document.
        wdl: Weighted document length (sum of field_len * field_weight).
        idf_cache: Pre-computed {term: idf_value} for query terms.
        avg_wdl: Average weighted document length across the corpus.
        k1: BM25 term frequency saturation parameter.
        b: BM25 document length normalization parameter.

    Returns:
        BM25F score (0.0 if no query terms match).
    """
    score = 0.0
    for qt in query_terms:
        wtf = weighted_tf.get(qt, 0)
        if wtf > 0:
            idf = idf_cache.get(qt, 0)
            numerator = wtf * (k1 + 1)
            denominator = wtf + k1 * (1 - b + b * wdl / avg_wdl)
            score += idf * numerator / denominator
    return score


def date_score(block: dict) -> float:
    """Boost recent blocks. Returns 0.0-1.0."""
    date_str = block.get("Date", "")
    if not date_str:
        return 0.5
    try:
        d = _datetime.strptime(date_str[:10], "%Y-%m-%d")
        now = _datetime.now()
        days_old = (now - d).days
        if days_old <= 0:
            return 1.0
        return max(0.1, 1.0 - (days_old / 365))
    except (ValueError, TypeError):
        return 0.5


# ---------------------------------------------------------------------------
# Temporal half-life decay (v3.3.0 Tier 1 #3)
# ---------------------------------------------------------------------------
#
# Pre-v3.3.0 ``date_score`` used a linear 1.0..0.1 ramp over a fixed
# 365-day window — fine as a coarse filter but too flat to meaningfully
# rank within a recall result set. The half-life decay below is
# multiplicatively compatible (still 0..1) and configurable via
# ``retrieval.temporal_half_life_days`` in ``mind-mem.json``.


def _resolve_half_life_days(config: dict | None) -> int:
    """Resolve the decay half-life from config, defaulting to 90 days."""
    default = 90
    if not isinstance(config, dict):
        return default
    retrieval = config.get("retrieval")
    if not isinstance(retrieval, dict):
        return default
    value = retrieval.get("temporal_half_life_days", default)
    if not isinstance(value, int) or value <= 0:
        return default
    return value


def temporal_decay_score(block: dict, half_life_days: int = 90) -> float:
    """Exponential half-life decay on a block's ``Created`` / ``Date`` field.

    ``score = 0.5 ** (age_days / half_life_days)``. Returns 1.0 for a
    same-day or future-dated block, 0.5 at one half-life, 0.25 at two
    half-lives, asymptotically approaches 0. Blocks without a parseable
    date return 0.5 (neutral — avoids penalising undated content).

    Half-life (default 90 days) is tunable via
    ``retrieval.temporal_half_life_days`` in ``mind-mem.json``. Used as a
    multiplicative ranking feature in the recall scorer, so an older
    block still ranks above a brand-new irrelevant one when BM25 strongly
    favours it.
    """
    raw = block.get("Created") or block.get("Date") or ""
    if not raw:
        return 0.5
    try:
        d = _datetime.strptime(str(raw)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return 0.5
    now = _datetime.now()
    age_days = (now - d).days
    if age_days <= 0:
        return 1.0
    hl = max(1, int(half_life_days))
    return float(0.5 ** (age_days / hl))


# ---------------------------------------------------------------------------
# Graph-based recall — cross-reference neighbor boosting
# ---------------------------------------------------------------------------


def build_xref_graph(all_blocks: list[dict]) -> dict[str, set[str]]:
    """Build bidirectional adjacency graph from cross-references.

    Scans every block's text fields for mentions of other block IDs.
    Returns {block_id: set(neighbor_ids)} with edges in both directions.
    """
    block_ids: set[str] = {str(b.get("_id")) for b in all_blocks if b.get("_id")}
    graph: dict[str, set[str]] = {bid: set() for bid in block_ids}

    # Fields to scan for cross-references
    xref_fields = SEARCH_FIELDS + [
        "Supersedes",
        "SupersededBy",
        "AlignsWith",
        "Dependencies",
        "Next",
        "Sources",
        "Evidence",
        "Rollback",
        "History",
    ]

    for block in all_blocks:
        bid_raw = block.get("_id")
        if not bid_raw:
            continue
        bid: str = str(bid_raw)

        # Collect all text from the block
        texts = []
        for field in xref_fields:
            val = block.get(field, "")
            if isinstance(val, str):
                texts.append(val)
            elif isinstance(val, list):
                texts.extend(str(v) for v in val)

        # Also scan ConstraintSignature scope.projects
        for sig in block.get("ConstraintSignatures", []):
            scope = sig.get("scope", {})
            if isinstance(scope, dict):
                for v in scope.values():
                    if isinstance(v, list):
                        texts.extend(str(x) for x in v)
                    elif isinstance(v, str):
                        texts.append(v)

        # Find all referenced block IDs
        full_text = " ".join(texts)
        for match in _BLOCK_ID_RE.finditer(full_text):
            ref_id = match.group(1)
            if ref_id != bid and ref_id in block_ids:
                graph[bid].add(ref_id)
                graph[ref_id].add(bid)  # bidirectional

    return graph


# ---------------------------------------------------------------------------
# Negation awareness
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bdidn't\b",
    r"\bdoesn't\b",
    r"\bwasn't\b",
    r"\bisn't\b",
    r"\bwon't\b",
    r"\bcan't\b",
    r"\bcannot\b",
    r"\bno\b",
    r"\bdon't\b",
    r"\bhasn't\b",
    r"\bhaven't\b",
    r"\bwouldn't\b",
]


def _detect_negation(query: str) -> tuple[bool, list[str]]:
    """Detect negation in query. Returns (has_negation, negated_terms)."""
    query_lower = query.lower()
    has_neg = any(re.search(p, query_lower) for p in _NEGATION_PATTERNS)
    if not has_neg:
        return False, []
    # Extract terms near negation words
    negated = []
    for pat in _NEGATION_PATTERNS:
        for m in re.finditer(pat, query_lower):
            # Get the next 1-3 words after negation
            rest = query_lower[m.end() :].strip().split()[:3]
            negated.extend(rest)
    return True, negated


def _negation_penalty(block_text: str, negated_terms: list[str], penalty: float = 0.3) -> float:
    """Penalize blocks that affirm what the query negates.
    Returns multiplier in [1-penalty, 1.0]."""
    if not negated_terms:
        return 1.0
    text_lower = block_text.lower()
    affirm_count = sum(1 for t in negated_terms if t in text_lower)
    if affirm_count == 0:
        return 1.0
    # More affirmed terms = bigger penalty
    return max(1.0 - penalty * min(affirm_count / len(negated_terms), 1.0), 1.0 - penalty)


# ---------------------------------------------------------------------------
# Date proximity scoring
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _extract_dates(text: str) -> list:
    """Extract YYYY-MM-DD dates from text."""
    dates = []
    for m in _DATE_PATTERN.finditer(text):
        try:
            dates.append(_datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    return dates


def _date_proximity_score(query: str, block_text: str, sigma: float = 30.0) -> float:
    """Gaussian decay based on date distance. Returns [0.5, 1.5] multiplier."""
    query_dates = _extract_dates(query)
    if not query_dates:
        return 1.0  # No temporal signal
    block_dates = _extract_dates(block_text)
    if not block_dates:
        return 0.8  # Mild penalty for no date when query has date

    # Use closest date pair
    min_delta = float("inf")
    for qd in query_dates:
        for bd in block_dates:
            delta = abs((qd - bd).days)
            min_delta = min(min_delta, delta)

    # Gaussian decay
    score = math.exp(-(min_delta**2) / (2 * sigma**2))
    # Map to [0.5, 1.5] range
    return 0.5 + score * 1.0


# ---------------------------------------------------------------------------
# Category match (20-category taxonomy)
# ---------------------------------------------------------------------------

_CATEGORIES = {
    "IDENTITY": ["name", "who", "person", "identity", "called"],
    "PREFERENCE": ["prefer", "like", "favorite", "enjoy", "hate", "dislike", "love"],
    "EVENT": ["happened", "event", "occurred", "when", "took place", "attended"],
    "RELATION": ["friend", "family", "married", "partner", "colleague", "relationship"],
    "MEDICAL": ["health", "doctor", "medical", "allergy", "medication", "diagnosis"],
    "WORK": ["job", "work", "company", "career", "position", "employed", "boss"],
    "HOBBY": ["hobby", "interest", "sport", "play", "collect", "practice"],
    "LOCATION": ["live", "city", "country", "address", "moved", "located", "where"],
    "OPINION": ["think", "believe", "opinion", "view", "feel", "consider"],
    "PLAN": ["plan", "going to", "will", "intend", "schedule", "future"],
    "FOOD": ["eat", "food", "diet", "restaurant", "cook", "meal", "vegetarian"],
    "EDUCATION": ["school", "university", "degree", "study", "learn", "course"],
    "TRAVEL": ["travel", "trip", "visit", "vacation", "flew", "destination"],
    "FINANCE": ["money", "salary", "invest", "budget", "cost", "price"],
    "TECHNOLOGY": ["computer", "software", "app", "code", "program", "tech"],
    "PETS": ["pet", "dog", "cat", "animal", "breed"],
    "FAMILY": ["child", "parent", "sibling", "mother", "father", "daughter", "son"],
    "SOCIAL": ["party", "gathering", "meeting", "social", "community"],
    "APPEARANCE": ["wear", "look", "style", "clothes", "appearance"],
    "HABIT": ["always", "usually", "routine", "habit", "every day", "morning"],
}


def _classify_categories(text: str) -> set[str]:
    """Classify text into categories based on keyword matching."""
    text_lower = text.lower()
    cats = set()
    for cat, keywords in _CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            cats.add(cat)
    return cats


def _category_match_boost(query: str, block_text: str, boost: float = 0.15) -> float:
    """Boost blocks matching query's category. Returns [1.0, 1.0+boost]."""
    query_cats = _classify_categories(query)
    if not query_cats:
        return 1.0
    block_cats = _classify_categories(block_text)
    if not block_cats:
        return 1.0
    overlap = len(query_cats & block_cats)
    return 1.0 + boost * min(overlap / len(query_cats), 1.0)


def _extract_entities(text: str) -> set[str]:
    """Extract likely entity tokens: capitalized words + multi-word proper nouns."""
    entities = set()
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        entities.add(m.group(0).lower())
    # Also grab individual capitalized tokens
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
        entities.add(m.group(0).lower())
    return entities


def _extract_bigram_phrases(text: str) -> set[str]:
    """Extract 2+ word proper nouns / quoted phrases for exact matching."""
    phrases = set()
    # Quoted phrases
    for m in re.finditer(r'"([^"]{3,})"', text):
        phrases.add(m.group(1).lower())
    # Multi-word proper nouns (2-4 capitalized words in sequence)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text):
        phrases.add(m.group(0).lower())
    return phrases


def _extract_speaker_names(query: str, all_results: list[dict]) -> set[str]:
    """Find speaker names mentioned in the query by cross-referencing known speakers."""
    known_speakers = set()
    for r in all_results:
        sp = r.get("speaker", "")
        if sp:
            known_speakers.add(sp.lower())
    query_lower = query.lower()
    mentioned = set()
    for sp in known_speakers:
        if sp in query_lower:
            mentioned.add(sp)
    return mentioned

"""Recall engine scoring — date scores, graph boosting, negation, date proximity, categories, entity extraction."""

from __future__ import annotations

import math
import re
from datetime import datetime as _datetime

from _recall_constants import _BLOCK_ID_RE, SEARCH_FIELDS

__all__ = [
    "date_score", "build_xref_graph",
    "_detect_negation", "_negation_penalty",
    "_extract_dates", "_date_proximity_score",
    "_classify_categories", "_category_match_boost",
    "_extract_entities", "_extract_bigram_phrases", "_extract_speaker_names",
]


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
# Graph-based recall — cross-reference neighbor boosting
# ---------------------------------------------------------------------------

def build_xref_graph(all_blocks: list[dict]) -> dict[str, set[str]]:
    """Build bidirectional adjacency graph from cross-references.

    Scans every block's text fields for mentions of other block IDs.
    Returns {block_id: set(neighbor_ids)} with edges in both directions.
    """
    block_ids = {b.get("_id") for b in all_blocks if b.get("_id")}
    graph = {bid: set() for bid in block_ids}

    # Fields to scan for cross-references
    xref_fields = SEARCH_FIELDS + [
        "Supersedes", "SupersededBy", "AlignsWith", "Dependencies",
        "Next", "Sources", "Evidence", "Rollback", "History",
    ]

    for block in all_blocks:
        bid = block.get("_id")
        if not bid:
            continue

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
    r'\bnot\b', r'\bnever\b', r"\bdidn't\b", r"\bdoesn't\b", r"\bwasn't\b",
    r"\bisn't\b", r"\bwon't\b", r"\bcan't\b", r"\bcannot\b", r'\bno\b',
    r"\bdon't\b", r"\bhasn't\b", r"\bhaven't\b", r"\bwouldn't\b",
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
            rest = query_lower[m.end():].strip().split()[:3]
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

_DATE_PATTERN = re.compile(r'(\d{4})-(\d{2})-(\d{2})')


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
    min_delta = float('inf')
    for qd in query_dates:
        for bd in block_dates:
            delta = abs((qd - bd).days)
            min_delta = min(min_delta, delta)

    # Gaussian decay
    score = math.exp(-(min_delta ** 2) / (2 * sigma ** 2))
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

"""Recall engine detection — query type classification, text extraction, block utilities."""

from __future__ import annotations

import re

from _recall_constants import SEARCH_FIELDS, SIG_FIELDS
from _recall_tokenization import tokenize

__all__ = [
    "extract_text", "extract_field_tokens", "get_bigrams",
    "is_skeptical_query", "detect_query_type", "decompose_query",
    "_QUERY_TYPE_PARAMS", "_INTENT_TO_QUERY_TYPE",
    "chunk_text", "get_excerpt", "_parse_speaker_from_tags", "get_block_type",
]


def extract_text(block: dict) -> str:
    """Extract searchable text from a block."""
    parts = []
    # Include block ID so users can search by ID
    bid = block.get("_id", "")
    if bid:
        parts.append(bid)
    for field in SEARCH_FIELDS:
        val = block.get(field, "")
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            parts.extend(str(v) for v in val)

    # Extract from ConstraintSignatures
    for sig in block.get("ConstraintSignatures", []):
        for sf in SIG_FIELDS:
            val = sig.get(sf, "")
            if isinstance(val, str):
                parts.append(val)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# BM25F — Field-weighted term extraction
# ---------------------------------------------------------------------------

def extract_field_tokens(block: dict) -> dict[str, list[str]]:
    """Extract and tokenize per-field for BM25F scoring.

    Returns {field_name: [stemmed_tokens]} for each non-empty field.
    Also includes _id tokens under a synthetic '_id' key.
    """
    field_tokens = {}

    bid = block.get("_id", "")
    if bid:
        field_tokens["_id"] = tokenize(bid)

    for field in SEARCH_FIELDS:
        val = block.get(field, "")
        if isinstance(val, str) and val:
            tokens = tokenize(val)
            if tokens:
                field_tokens[field] = tokens
        elif isinstance(val, list):
            combined = " ".join(str(v) for v in val)
            tokens = tokenize(combined)
            if tokens:
                field_tokens[field] = tokens

    # ConstraintSignature fields
    sig_parts = []
    for sig in block.get("ConstraintSignatures", []):
        for sf in SIG_FIELDS:
            val = sig.get(sf, "")
            if isinstance(val, str):
                sig_parts.append(val)
    if sig_parts:
        tokens = tokenize(" ".join(sig_parts))
        if tokens:
            field_tokens["_sig"] = tokens

    return field_tokens


# ---------------------------------------------------------------------------
# Bigram phrase matching
# ---------------------------------------------------------------------------

def get_bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Generate set of adjacent token pairs (bigrams)."""
    return set(zip(tokens, tokens[1:]))


# ---------------------------------------------------------------------------
# Skeptical Mode Detection — distractor-prone query identification
# ---------------------------------------------------------------------------

_SKEPTICAL_TRIGGERS_RE = re.compile(
    r"\b(favorite|favourite|least|best|worst|most|one of|"
    r"as mentioned on|mentioned on \w+|according to|"
    r"on \d{1,2}(?:st|nd|rd|th)?\s+\w+)\b",
    re.IGNORECASE,
)


def is_skeptical_query(query: str) -> bool:
    """Detect if a query is distractor-prone and needs skeptical retrieval.

    Triggers: superlatives, embedded date references, very low lexical specificity.
    """
    if _SKEPTICAL_TRIGGERS_RE.search(query):
        return True
    # Very short query with broad terms = low specificity
    words = query.split()
    if len(words) <= 5:
        specific = [w for w in words if len(w) > 4 and w.lower() not in
                     {"what", "which", "where", "about", "their", "there", "these", "those"}]
        if len(specific) <= 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Query Type Detection — category-specific retrieval tuning
# ---------------------------------------------------------------------------

# Temporal signal words/patterns
_TEMPORAL_PATTERNS = re.compile(
    r"\b(when|before|after|during|while|since|until|first|last|latest|earliest"
    r"|recent|previous|next|ago|year|month|week|day|date|time|period"
    r"|january|february|march|april|may|june|july|august|september"
    r"|october|november|december|spring|summer|fall|winter|autumn"
    r"|morning|evening|night|weekend|holiday|birthday|anniversary"
    r"|how long|how often|what time|what date|what year|what month"
    r"|started|ended|began|finished|happened|occurred|changed"
    r"|earlier|later|prior|subsequent|meanwhile|eventually)\b",
    re.IGNORECASE,
)

# Adversarial negation patterns
_ADVERSARIAL_PATTERNS = re.compile(
    r"\b(never|not|no one|nobody|nothing|nowhere|neither|nor"
    r"|didn.t|doesn.t|don.t|wasn.t|weren.t|isn.t|aren.t|hasn.t"
    r"|haven.t|hadn.t|won.t|wouldn.t|can.t|couldn.t|shouldn.t"
    r"|false|incorrect|wrong|untrue|lie|fake|fabricat"
    r"|no interest|no desire|no intention|no plan"
    r"|contrary|opposite|instead|rather than|unlike|different from)\b",
    re.IGNORECASE,
)

# Broader verification-intent patterns — queries expecting yes/no confirmation.
# These need precision over recall, so semantic synonym expansion is suppressed.
_VERIFICATION_INTENT_RE = re.compile(
    r"(?:"
    r"^(?:did|was|were|is|has|have|does|do)\s+"   # Yes/no question openers
    r"|(?:ever|actually|really)\b"                  # Certainty qualifiers
    r"|(?:is it true|is that true|is this true)"    # Truth verification
    r"|(?:no mention|not mention|never mention)"    # Absence checks
    r"|(?:deny|denied|denial)\b"                    # Denial patterns
    r")",
    re.IGNORECASE,
)

# Multi-hop indicators (questions requiring info from multiple sources)
_MULTIHOP_PATTERNS = re.compile(
    r"\b(both|and also|as well as|in addition|together|combined"
    r"|relationship|connection|between|compare|comparison|versus|vs|both"
    r"|how many|how much|total|count|sum|all|every|each"
    r"|who else|what else|where else|which other"
    r"|same|similar|different|shared|common|overlap"
    r"|because|caused|resulted|led to|due to|reason|why did)\b",
    re.IGNORECASE,
)


def detect_query_type(query: str) -> str:
    """Classify query into temporal/adversarial/multi-hop/single-hop.

    Returns one of: 'temporal', 'adversarial', 'multi-hop', 'single-hop'.
    Uses pattern-based heuristics with scoring to handle ambiguous queries.
    """
    query_lower = query.lower()

    temporal_hits = len(_TEMPORAL_PATTERNS.findall(query_lower))
    adversarial_hits = len(_ADVERSARIAL_PATTERNS.findall(query_lower))
    multihop_hits = len(_MULTIHOP_PATTERNS.findall(query_lower))

    # Question mark count can indicate complexity
    qmarks = query.count("?")

    # Score each category
    scores = {
        "temporal": temporal_hits * 2,
        "adversarial": adversarial_hits * 2.5,
        "multi-hop": multihop_hits * 1.5 + (1 if qmarks > 1 else 0),
        "single-hop": 1,  # default baseline
    }

    # Strong negation at start is a very strong adversarial signal
    if re.match(r"^(did|was|were|is|has|have|does|do)\b.+\b(not|never|n.t)\b", query_lower):
        scores["adversarial"] += 3
    # "Did X ever" is also adversarial (expects yes/no about something that may not have happened)
    if re.search(r"\bever\b", query_lower):
        scores["adversarial"] += 2
    # Broader verification intent (yes/no confirmation queries)
    if _VERIFICATION_INTENT_RE.search(query_lower):
        scores["adversarial"] += 1.5

    # Temporal + entity action = temporal-multi-hop
    # "When did X do Y?" requires finding the event + deriving the date
    word_count = len(query.split())
    if temporal_hits > 0 and re.search(
        r"\b(when did|when was|when is|when does)\b.*\b(do|did|go|went|start|began|"
        r"get|got|buy|bought|sell|sold|move|moved|meet|met|visit|attend|join|leave|"
        r"finish|complete|create|make|paint|write|run|play|watch|find|lose|give)\b",
        query_lower,
    ):
        scores["multi-hop"] += 1.5  # Boost multi-hop for temporal reasoning chains

    # Long queries with conjunctions are more likely multi-hop
    if word_count > 15 and multihop_hits > 0:
        scores["multi-hop"] += 2

    # Pick the highest-scoring category
    best = max(scores, key=scores.get)

    # Only override single-hop if signal is strong enough
    if best == "single-hop" or scores[best] < 1.5:
        # Distinguish single-hop from open-domain:
        # Open-domain = short query with no specific temporal/adversarial/multi-hop signal
        # and broad topic words (what, who, describe, tell me about)
        if word_count <= 10 and re.search(
            r"\b(what|who|describe|tell me|identity|about|background)\b",
            query_lower,
        ):
            return "open-domain"
        return "single-hop"

    return best


# Category-specific retrieval parameters
_QUERY_TYPE_PARAMS = {
    "temporal": {
        "recency_weight": 0.6,     # Higher recency matters more
        "date_boost": 2.0,         # Boost blocks with dates
        "expand_query": True,      # Keep expansions
        "extra_limit_factor": 2.0, # Retrieve more — date-bearing blocks are scattered
    },
    "adversarial": {
        "recency_weight": 0.3,     # Standard — adversarial needs same broad recall
        "date_boost": 1.0,         # No date boost
        "expand_query": "morph_only",  # Lemma + months only, no semantic synonyms
        "extra_limit_factor": 1.0, # Standard — handled at answerer level, not retrieval
    },
    "multi-hop": {
        "recency_weight": 0.3,     # Standard
        "date_boost": 1.0,
        "expand_query": True,
        "extra_limit_factor": 3.0, # Need more blocks to find all hops
        "graph_boost_override": True,  # Force graph traversal
    },
    "single-hop": {
        "recency_weight": 0.3,     # Standard
        "date_boost": 1.0,
        "expand_query": True,
        "extra_limit_factor": 1.0,
    },
    "open-domain": {
        "recency_weight": 0.2,     # Less recency bias for broad questions
        "date_boost": 1.0,
        "expand_query": True,
        "extra_limit_factor": 2.0, # Retrieve more for diversity
    },
}

# IntentRouter -> legacy query type mapping (backward compatible with _QUERY_TYPE_PARAMS)
_INTENT_TO_QUERY_TYPE = {
    "WHY": "multi-hop",
    "WHEN": "temporal",
    "ENTITY": "single-hop",
    "WHAT": "single-hop",
    "HOW": "single-hop",
    "LIST": "open-domain",
    "VERIFY": "adversarial",
    "COMPARE": "multi-hop",
    "TRACE": "multi-hop",
}


# ---------------------------------------------------------------------------
# Multi-hop Query Decomposition
# ---------------------------------------------------------------------------

# Conjunctions that indicate separate information needs
_CONJUNCTION_SPLIT_RE = re.compile(
    r"\s+(?:and\s+(?:also\s+)?|but\s+|also\s+|as\s+well\s+as\s+|plus\s+)",
    re.IGNORECASE,
)

# Wh-word pattern for detecting question boundaries
_WH_WORD_RE = re.compile(
    r"\b(who|what|when|where|why|how)\b",
    re.IGNORECASE,
)

# Minimum tokens for a valid sub-query (after splitting)
_MIN_SUBQUERY_TOKENS = 3

# Maximum sub-queries to prevent explosion
_MAX_SUBQUERIES = 4


_WH_WORDS_LOWER = {"what", "when", "where", "which", "who", "how", "why"}


def _extract_entities(text: str) -> list[str]:
    """Extract likely entity/topic words from text.

    Returns capitalized words and multi-word noun phrases that serve as
    shared context anchors for sub-queries.  Excludes wh-words (What, When, etc.)
    which are interrogative, not content entities.
    """
    entities = []
    # Capitalized words (proper nouns, project names, etc.)
    for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", text):
        word = m.group(1)
        if word.lower() not in _WH_WORDS_LOWER:
            entities.append(word)
    # Version patterns (v1.0.6, etc.)
    for m in re.finditer(r"\bv?\d+\.\d+(?:\.\d+)?\b", text):
        entities.append(m.group(0))
    # Quoted strings
    for m in re.finditer(r'"([^"]+)"', text):
        entities.append(m.group(1))
    return entities


def _token_count(text: str) -> int:
    """Count meaningful tokens (words) in text."""
    return len([w for w in text.split() if len(w) > 1])


def decompose_query(query: str) -> list[str]:
    """Decompose a multi-hop query into sub-queries for independent retrieval.

    Uses deterministic heuristics (no LLM calls):
    1. Split on conjunctions: "and", "but", "or", "also", "as well as", "plus"
    2. Split on question boundaries: multiple "?" or wh-words
    3. Each sub-query must have at least 3 tokens
    4. Maximum 4 sub-queries
    5. Shared context (entities/topics) from the first clause is carried into
       subsequent sub-queries that lack it.

    Returns:
        List of sub-query strings. Single-item list if no decomposition is possible.
    """
    query = query.strip()
    if not query:
        return []

    # Strategy 1: Split on multiple question marks (explicit question boundaries)
    if query.count("?") > 1:
        parts = [p.strip() for p in query.split("?") if p.strip()]
        # Re-append "?" for completeness (cosmetic, does not affect tokenization)
        parts = [p + "?" if not p.endswith("?") else p for p in parts]
        if len(parts) > 1:
            parts = [p for p in parts if _token_count(p) >= _MIN_SUBQUERY_TOKENS]
            if len(parts) > 1:
                return _preserve_context(parts[:_MAX_SUBQUERIES], query)

    # Strategy 2: Split on wh-word boundaries within a single sentence.
    # Detect patterns like "When did X happen and how long did Y take?"
    wh_positions = [m.start() for m in _WH_WORD_RE.finditer(query)]
    if len(wh_positions) >= 2:
        parts = []
        for i, pos in enumerate(wh_positions):
            end = wh_positions[i + 1] if i + 1 < len(wh_positions) else len(query)
            segment = query[pos:end].strip().rstrip("?").rstrip(",").strip()
            # Remove trailing conjunction words (word-boundary safe)
            segment = re.sub(r"\s+(?:and|but|or|plus)\s*$", "", segment, flags=re.IGNORECASE)
            if segment:
                parts.append(segment)
        parts = [p for p in parts if _token_count(p) >= _MIN_SUBQUERY_TOKENS]
        if len(parts) > 1:
            return _preserve_context(parts[:_MAX_SUBQUERIES], query)

    # Strategy 3: Split on conjunctions
    parts = _CONJUNCTION_SPLIT_RE.split(query)
    parts = [p.strip().rstrip("?").strip() for p in parts if p.strip()]
    parts = [p for p in parts if _token_count(p) >= _MIN_SUBQUERY_TOKENS]
    if len(parts) > 1:
        return _preserve_context(parts[:_MAX_SUBQUERIES], query)

    # No decomposition possible — return original query
    return [query]


def _preserve_context(parts: list[str], original_query: str) -> list[str]:
    """Carry shared entities/topics from the first clause into later sub-queries.

    If the first clause mentions an entity (e.g., "the auth migration") and
    a later clause does not, prepend the entity so each sub-query is
    self-contained for retrieval.
    """
    if len(parts) <= 1:
        return parts

    # Extract entities from the full original query and the first clause
    first_entities = _extract_entities(parts[0])
    # Also extract key noun-like tokens from the first part (lowercased, len > 4,
    # not wh-words or common verbs)
    _skip = {
        "what", "when", "where", "which", "who", "how", "why", "that", "this",
        "there", "their", "these", "those", "about", "were", "have", "does",
        "been", "being", "would", "could", "should", "will", "with", "from",
        "they", "them", "than", "into", "also", "just", "some", "each",
        # Common short function words (2-3 chars)
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "did", "get", "has",
        "him", "his", "she", "too", "use", "may", "its", "let", "say",
        "any", "new", "now", "old", "see", "way", "own", "boy", "did",
        "man", "run", "set", "try", "ask", "men", "ran", "few",
        "it", "is", "in", "on", "at", "to", "up", "so", "we", "an",
        "do", "if", "my", "no", "he", "by", "or", "as", "be", "go",
    }
    first_keywords = [
        w for w in re.findall(r"[a-zA-Z0-9_./-]+", parts[0])
        if len(w) >= 2 and w.lower() not in _skip
    ]

    # Combine entities and keywords for context candidates
    context_tokens = first_entities + first_keywords
    if not context_tokens:
        return parts

    # Deduplicate while preserving order
    seen = set()
    context_unique = []
    for t in context_tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            context_unique.append(t)

    result = [parts[0]]
    for part in parts[1:]:
        part_lower = part.lower()
        # Check if this sub-query already contains any of the context tokens
        has_context = any(t.lower() in part_lower for t in context_unique)
        if not has_context and context_unique:
            # Prepend the most relevant context tokens (up to 3)
            prefix = " ".join(context_unique[:3])
            result.append(f"{prefix} {part}")
        else:
            result.append(part)

    return result


def chunk_text(text: str, chunk_size: int = 3, overlap: int = 1) -> list[str]:
    """Split text into overlapping sentence chunks.

    Args:
        text: Full text to chunk.
        chunk_size: Sentences per chunk.
        overlap: Overlapping sentences between chunks.

    Returns list of chunk strings. Returns [text] if <=chunk_size sentences.
    """
    # Simple sentence splitting on .!? followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(sentences):
            break
    return chunks if chunks else [text]


def get_excerpt(block: dict, max_len: int = 300) -> str:
    """Get a short excerpt from a block."""
    for field in ("Statement", "Title", "Summary", "Description", "Name", "Context"):
        val = block.get(field, "")
        if isinstance(val, str) and val:
            return val[:max_len]
    return block.get("_id", "?")


def _parse_speaker_from_tags(tags_str: str) -> str:
    """Extract speaker name from Tags field. Tags format: 'FACT, Caroline'."""
    if not tags_str:
        return ""
    parts = [t.strip() for t in tags_str.split(",")]
    # First tag is the card type (FACT/EVENT/etc), rest are speaker/metadata
    for p in parts[1:]:
        if p and p[0].isupper() and p not in (
            "FACT", "EVENT", "PREFERENCE", "RELATION", "NEGATION", "PLAN",
        ):
            return p
    return ""


def get_block_type(block_id: str) -> str:
    """Infer block type from ID prefix."""
    prefixes = {
        "D-": "decision", "T-": "task", "PRJ-": "project",
        "PER-": "person", "TOOL-": "tool", "INC-": "incident",
        "C-": "contradiction", "DREF-": "drift", "SIG-": "signal",
        "P-": "proposal", "I-": "impact",
    }
    for prefix, btype in prefixes.items():
        if block_id.startswith(prefix):
            return btype
    return "unknown"

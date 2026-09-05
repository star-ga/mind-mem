"""Recall engine scoring — BM25F helper, date scores, graph boosting, negation, date proximity, categories.

Clock discipline
----------------
Every recency signal in this module reads its "now" from :func:`_utc_now`,
which returns a **timezone-aware UTC** instant. Before this was enforced the
recency helpers used a naive local ``datetime.now()``, so the day boundary
that decides ``days_old`` moved with the host's ``TZ`` setting and the same
corpus scored differently on two machines at the same instant. Recency
scoring is now machine-independent: identical corpus + identical instant
yields identical scores anywhere.

For deterministic replay every recency helper also accepts an optional
keyword-only ``now`` argument. Pass a fixed instant to pin the clock; leave
it out and the helper reads UTC now. Naive values passed in are interpreted
as UTC, aware values are converted to UTC.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime as _datetime
from datetime import timezone as _timezone

from ._recall_constants import _BLOCK_ID_RE, BM25_B, BM25_K1, FIELD_WEIGHTS, SEARCH_FIELDS

__all__ = [
    "bm25_idf",
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


def bm25_idf(n_docs: int, df: int) -> float:
    """BM25 probabilistic inverse document frequency for one term.

    ``log((N - df + 0.5) / (df + 0.5) + 1)`` — the same expression the main
    scoring loop, the RM3 / PRF expansions and the bridge re-score all use.
    It lives here, beside :func:`bm25f_score_terms`, because this module is
    the single source of truth for BM25 arithmetic and a second spelling of
    the IDF would let two legs disagree about how rare a term is.

    The ``+ 1`` inside the log is what keeps the value non-negative for a term
    that appears in more than half the corpus; without it a very common term
    scores negatively and *subtracts* from a document that contains it.

    Args:
        n_docs: Corpus size. Values below 1 are treated as 1 so a
            single-document corpus (or an empty one) cannot divide by zero or
            take the log of a non-positive number.
        df: Document frequency of the term. Clamped into ``[0, n_docs]``, so a
            caller that counted df against a larger corpus than it passed
            cannot drive the argument of the log negative.

    Returns:
        The IDF weight, always finite and >= 0.
    """
    n = max(1, int(n_docs))
    d = min(max(0, int(df)), n)
    return math.log((n - d + 0.5) / (d + 0.5) + 1)


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


# ---------------------------------------------------------------------------
# Clock seam — one UTC source for every recency signal in this module
# ---------------------------------------------------------------------------


def _utc_now() -> _datetime:
    """Timezone-aware UTC ``now`` — the single clock source for recency math.

    Recency scoring must not depend on the host's timezone: a naive
    ``datetime.now()`` puts the day boundary at local midnight, so the same
    corpus scored at the same instant produced different ``days_old`` values
    (and therefore different recall scores) on hosts in different zones.
    """
    return _datetime.now(_timezone.utc)


def _as_utc(moment: _datetime | None) -> _datetime:
    """Normalise an injected ``now`` to aware UTC; ``None`` means "read the clock".

    A naive value is interpreted as UTC rather than local time, so an
    injected instant never re-introduces host-timezone dependence.
    """
    if moment is None:
        return _utc_now()
    if moment.tzinfo is None:
        return moment.replace(tzinfo=_timezone.utc)
    return moment.astimezone(_timezone.utc)


# A leading calendar day, accepting either separator: ``2023-05-20`` and
# ``2023/05/20`` are the same day. Slash-dated corpora are common enough that
# rejecting them silently costs the whole temporal signal: an unparseable date
# scores 0.5, which is exactly the score for having no date at all, so the
# failure is invisible at every layer above this one.
_DAY_SEPARATORS = ("%Y-%m-%d", "%Y/%m/%d")


def _parse_utc_day(raw: object) -> _datetime | None:
    """Parse a leading calendar day into UTC midnight, or ``None`` if unparseable.

    Accepts ``YYYY-MM-DD`` and ``YYYY/MM/DD``. Anything trailing the day is
    ignored, so ``2023/05/20 (Sat) 02:21`` parses to 2023-05-20.
    """
    if not isinstance(raw, str):
        return None
    head = raw[:10]
    for fmt in _DAY_SEPARATORS:
        try:
            return _datetime.strptime(head, fmt).replace(tzinfo=_timezone.utc)
        except ValueError:
            continue
    return None


def date_score(block: dict, *, now: _datetime | None = None) -> float:
    """Boost recent blocks. Returns 0.0-1.0.

    Args:
        block: Block dict; the ``Date`` field is read as ``YYYY-MM-DD``.
        now: Optional instant to score against (deterministic replay).
            Defaults to UTC now; naive values are read as UTC.
    """
    date_str = block.get("Date", "")
    if not date_str:
        return 0.5
    d = _parse_utc_day(date_str)
    if d is None:
        return 0.5
    days_old = (_as_utc(now) - d).days
    if days_old <= 0:
        return 1.0
    return max(0.1, 1.0 - (days_old / 365))


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


def temporal_decay_score(
    block: dict,
    half_life_days: int = 90,
    *,
    now: _datetime | None = None,
) -> float:
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

    Age is measured against UTC (see :func:`_utc_now`) so the decay is
    machine-independent. Pass ``now`` to pin the instant for replay.
    """
    raw = block.get("Created") or block.get("Date") or ""
    if not raw:
        return 0.5
    d = _parse_utc_day(str(raw))
    if d is None:
        return 0.5
    age_days = (_as_utc(now) - d).days
    if age_days <= 0:
        return 1.0
    hl = max(1, int(half_life_days))
    # Clamp to ``1e-6`` so a century-old block still multiplies its
    # BM25 score to a non-zero ranking value. Hard zero would suppress
    # the block entirely regardless of query match strength — review
    # by python-reviewer flagged this as a correctness bug (2026-04-20).
    return max(float(0.5 ** (age_days / hl)), 1e-6)


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
    """Extract YYYY-MM-DD dates from text.

    Clock-free: the values below are only ever differenced against each
    other (query date vs. block date), never against ``now``, so they stay
    naive and carry no timezone dependence.
    """
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

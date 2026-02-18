"""Deterministic adversarial abstention classifier for Mind-Mem.

Pre-LLM confidence gate: examines retrieval results and decides whether
there is enough direct evidence to answer a query. If confidence is below
threshold, forces abstention without calling the LLM at all.

This targets LoCoMo adversarial questions which are *designed* to be
unanswerable — the gold answer is typically "No, X never mentioned Y" or
"Not found." The current 30.7% accuracy is because the LLM fabricates
answers from tangentially related context.

Reference: "Building a best-in-class agent memory system in 2026" (§ Roadmap #1)
           LongMemEval (ICLR 2025) — abstention as first-class behavior

Integration points:
  - locomo_judge.py: between pack_evidence() and answer_question()
  - evidence_packer.py: exposed for production MCP recall path
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Stop words excluded from entity/noun extraction ──────────────────

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "need",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "where", "when", "why", "how", "if", "then", "than", "but", "and",
    "or", "not", "no", "nor", "so", "at", "by", "for", "from", "in",
    "into", "of", "on", "to", "with", "about", "as", "up", "out",
    "any", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "too", "very",
    "just", "because", "also", "over", "after", "before", "between",
    "through", "during", "above", "below", "again", "further",
    "ever", "never", "always", "sometimes", "often", "still",
    "already", "even", "really", "quite", "rather",
    "don", "doesn", "didn", "won", "wouldn", "shouldn",
    "couldn", "hasn", "haven", "hadn", "isn", "aren", "wasn", "weren",
    "mention", "say", "said", "tell", "told", "talk", "discuss",
    "point", "time", "conversation", "actually", "true",
})

# ── Regex for extracting content words ───────────────────────────────

_WORD_RE = re.compile(r"[a-z]{2,}")

# Speaker patterns in queries like "Did Emma ever mention..."
_SPEAKER_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)


# ── Result dataclass ─────────────────────────────────────────────────

@dataclass
class AbstentionResult:
    """Result of the abstention classifier."""
    should_abstain: bool
    confidence: float
    features: dict = field(default_factory=dict)
    forced_answer: str = ""


# ── Feature extractors ───────────────────────────────────────────────

def _extract_query_entities(question: str) -> set[str]:
    """Extract key entities/nouns from a question (lowercased).

    Filters out stop words and common question words to get the
    meaningful content tokens that should appear in evidence.
    """
    tokens = set(_WORD_RE.findall(question.lower()))
    return tokens - _STOP_WORDS


def _extract_speaker_from_query(question: str) -> str | None:
    """Extract a proper name from the query if present.

    LoCoMo questions often ask about specific speakers:
    "Did Emma ever mention wanting to adopt a dog?"
    """
    matches = _SPEAKER_RE.findall(question)
    # Filter out common non-name capitalized words
    skip = {"Did", "Does", "Was", "Were", "Has", "Have", "Had",
            "Can", "Could", "Would", "Should", "Will", "May",
            "Is", "Are", "Do", "Not", "The", "This", "That",
            "What", "Which", "Who", "Where", "When", "Why", "How",
            "Yes", "No", "Any", "All", "Some", "Many", "Much",
            "Never", "Ever", "Also", "Just", "Only", "Very",
            "Question", "Answer", "Evidence", "During",
            "If", "But", "And", "Or", "So", "Then", "Than",
            "About", "After", "Before", "Between", "Both",
            "Each", "Every", "For", "From", "Into", "Over",
            "Such", "Through", "Under", "Until", "With"}
    cleaned = []
    for m in matches:
        # For multi-word matches like "Did Emma", strip leading skip words
        words = m.split()
        while words and words[0] in skip:
            words = words[1:]
        if words:
            cleaned.append(" ".join(words))
    return cleaned[0].lower() if cleaned else None


def _term_overlap(excerpt: str, query_entities: set[str]) -> float:
    """Fraction of query entities found in a single hit excerpt."""
    if not query_entities:
        return 0.0
    excerpt_lower = excerpt.lower()
    found = sum(1 for t in query_entities if t in excerpt_lower)
    return found / len(query_entities)


def _speaker_in_hit(hit: dict, speaker: str) -> bool:
    """Check if the queried speaker appears in this hit."""
    if not speaker:
        return False
    sp = (hit.get("speaker", "") or "").lower()
    excerpt = (hit.get("excerpt", "") or "").lower()
    return speaker in sp or speaker in excerpt


# ── Main classifier ──────────────────────────────────────────────────

# Default abstention threshold — conservative start, tune upward
# after benchmark runs. See docstring for tuning guidance.
DEFAULT_THRESHOLD = 0.20

# Input length bounds (defense against oversized inputs)
_MAX_QUESTION_LEN = 4096
_MAX_TOP_K = 200

# Abstention answer used when classifier fires
ABSTENTION_ANSWER = (
    "Not enough direct evidence in memory to answer this question."
)


def classify_abstention(
    question: str,
    hits: list[dict],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 5,
) -> AbstentionResult:
    """Determine whether to abstain from answering based on retrieval quality.

    Computes a confidence score (0.0–1.0) from multiple deterministic
    features. If confidence < threshold, recommends abstention.

    Features:
        1. entity_overlap: Do top-K hits contain the query's key entities?
        2. top1_score: Is the best BM25 match actually relevant?
        3. speaker_coverage: If query asks about a person, are they in results?
        4. evidence_density: How many top-K hits have >50% entity overlap?
        5. negation_asymmetry: For "did X ever" questions, is there positive
           evidence or only tangential context?

    Args:
        question: The user query.
        hits: Recall results (with score, excerpt, speaker fields).
        threshold: Confidence below this → abstain. Default 0.20 (conservative).
        top_k: Number of top hits to examine.

    Returns:
        AbstentionResult with should_abstain, confidence, features, forced_answer.
    """
    if not hits or top_k <= 0:
        return AbstentionResult(
            should_abstain=True,
            confidence=0.0,
            features={"reason": "no_hits"},
            forced_answer=ABSTENTION_ANSWER,
        )

    # Bound inputs to prevent resource exhaustion
    question = question[:_MAX_QUESTION_LEN]
    top_k = min(top_k, _MAX_TOP_K)

    query_entities = _extract_query_entities(question)
    speaker = _extract_speaker_from_query(question)
    top_hits = hits[:top_k]

    # ── Feature 1: Entity overlap (mean across top-K) ────────────
    overlaps = [_term_overlap(h.get("excerpt", "") or "", query_entities) for h in top_hits]
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # ── Feature 2: Top-1 BM25 score (normalized) ─────────────────
    # BM25 scores vary by corpus; normalize relative to a reasonable
    # threshold. Scores below ~2.0 are typically noise in mind-mem.
    top1_raw = (top_hits[0].get("score", 0.0) or 0.0) if top_hits else 0.0
    # Sigmoid-like normalization: maps [0, 10] → [0, ~1.0]
    top1_norm = min(1.0, top1_raw / 10.0) if top1_raw > 0 else 0.0

    # ── Feature 3: Speaker coverage ──────────────────────────────
    if speaker:
        speaker_hits = sum(1 for h in top_hits if _speaker_in_hit(h, speaker))
        speaker_cov = speaker_hits / len(top_hits)
    else:
        # No speaker in query → neutral (don't penalize)
        speaker_cov = 0.5

    # ── Feature 4: Evidence density (hits with >50% overlap) ─────
    dense_count = sum(1 for o in overlaps if o > 0.5)
    evidence_density = dense_count / len(top_hits) if top_hits else 0.0

    # ── Feature 5: Negation asymmetry ────────────────────────────
    # For "did X ever..." questions: if all top hits are tangential
    # (low overlap) and none contain positive evidence, that's a signal
    # the question is unanswerable.
    has_ever_pattern = bool(re.search(
        r"\b(ever|never|at any point|at some point)\b", question, re.IGNORECASE
    ))
    if has_ever_pattern:
        # Penalize: if asking "did X ever" and no strong evidence
        negation_penalty = 1.0 - mean_overlap  # higher when overlap is low
    else:
        negation_penalty = 0.0

    # ── Weighted combination ─────────────────────────────────────
    # Weights reflect feature importance for adversarial detection.
    # entity_overlap is most important — if the query terms aren't
    # in the results, the results aren't about the question.
    # Note: negation_penalty intentionally double-counts low overlap
    # with entity_overlap — "did X ever" questions with poor results
    # should be penalized more aggressively than regular questions.
    weights = {
        "entity_overlap": 0.35,
        "top1_score": 0.20,
        "speaker_coverage": 0.15,
        "evidence_density": 0.20,
        "negation_penalty": -0.10,
    }

    confidence = (
        weights["entity_overlap"] * mean_overlap
        + weights["top1_score"] * top1_norm
        + weights["speaker_coverage"] * speaker_cov
        + weights["evidence_density"] * evidence_density
        + weights["negation_penalty"] * negation_penalty
    )
    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    features = {
        "entity_overlap": round(mean_overlap, 4),
        "top1_score_raw": round(top1_raw, 4),
        "top1_score_norm": round(top1_norm, 4),
        "speaker_coverage": round(speaker_cov, 4),
        "evidence_density": round(evidence_density, 4),
        "negation_penalty": round(negation_penalty, 4),
        "query_entities": sorted(query_entities)[:10],  # truncate for logging
        "speaker_detected": speaker,
        "has_ever_pattern": has_ever_pattern,
        "top_k_examined": len(top_hits),
        "threshold": threshold,
    }

    should_abstain = confidence < threshold

    return AbstentionResult(
        should_abstain=should_abstain,
        confidence=round(confidence, 4),
        features=features,
        forced_answer=ABSTENTION_ANSWER if should_abstain else "",
    )

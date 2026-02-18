"""Deterministic evidence packer for Mind-Mem.

Builds structured, speaker-attributed evidence context from recall hits.
No LLM dependency — prevents starvation and hallucination in adversarial
and verification queries.

v1.0.2: ALL query types now use structured [SPEAKER=...] [DATE=...] [DiaID=...]
format with category-specific ordering:
  - temporal: chronological by DiaID
  - multi-hop: hop-clustered (group by entity/topic)
  - adversarial (true): overlap-first with denial separation
  - single-hop / open-domain: score-descending (default)

This module is the "answer view" builder for mind-mem recall results:
  recall(query) -> hits -> pack_evidence(hits, query_type) -> packed_context
"""

from __future__ import annotations

import re

_DENIAL_RE = re.compile(
    r"\b(didn't|did not|never|not|no|denied|refused|won't|can't|cannot|"
    r"doesn't|does not|hasn't|has not|wasn't|isn't)\b",
    re.IGNORECASE,
)

_SEMANTIC_PREFIX_RE = re.compile(r"^\([^)]{1,80}\)\s*")

# Patterns that indicate a question is truly adversarial/verification
# vs a normal factual question misclassified as adversarial
_ADVERSARIAL_SIGNAL_RE = re.compile(
    r"\b(ever|never|deny|denied|not\s+mention|was\s+said|"
    r"at\s+any\s+point|reject|refuse|contradict|false|untrue)\b",
    re.IGNORECASE,
)


def strip_semantic_prefix(text: str) -> str:
    """Remove leading semantic label prefix e.g. '(identity description) '."""
    return _SEMANTIC_PREFIX_RE.sub("", text)


def is_true_adversarial(question: str) -> bool:
    """Check if a question is truly adversarial/verification vs misclassified.

    Many LoCoMo 'adversarial' questions are normal factual questions.
    Only apply strict adversarial policy when the question actually
    contains verification/negation language.
    """
    return bool(_ADVERSARIAL_SIGNAL_RE.search(question))


def _format_structured_line(r: dict) -> str:
    """Build a single structured evidence line with metadata tags."""
    text = r.get("excerpt", "")
    if not text:
        return ""
    clean = strip_semantic_prefix(text.strip())
    speaker = r.get("speaker", "") or "UNKNOWN"
    date = r.get("Date", "") or ""
    dia_id = r.get("DiaID", "") or ""

    parts = [f"[SPEAKER={speaker}]"]
    if date:
        parts.append(f"[DATE={date}]")
    if dia_id:
        parts.append(f"[DiaID={dia_id}]")
    parts.append(clean)
    return " ".join(parts)


def _dia_sort_key(r: dict) -> tuple:
    """Sort key for chronological ordering by DiaID (D{session}:{turn})."""
    dia = r.get("DiaID", "")
    m = re.match(r"D(\d+):(\d+)", dia)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999999, 999999)


def _overlap_score(r: dict, query_tokens: set) -> float:
    """Compute query token overlap ratio for a hit (for overlap-first ordering)."""
    excerpt = r.get("excerpt", "").lower()
    if not query_tokens:
        return 0.0
    return sum(1 for t in query_tokens if t in excerpt) / max(1, len(query_tokens))


def check_abstention(
    question: str,
    hits: list[dict],
    threshold: float = 0.20,
) -> tuple[bool, str, float]:
    """Production abstention gate for MCP recall path.

    Thin wrapper around abstention_classifier.classify_abstention().
    Returns (should_abstain, forced_answer, confidence).
    """
    from abstention_classifier import classify_abstention
    result = classify_abstention(question, hits, threshold=threshold)
    return result.should_abstain, result.forced_answer, result.confidence


def pack_evidence(
    hits: list[dict],
    question: str = "",
    query_type: str = "",
    max_chars: int = 6000,
) -> str:
    """Build structured evidence context from recall hits.

    ALL query types now use structured [SPEAKER=...] [DATE=...] [DiaID=...]
    format, with category-specific ordering and adversarial-specific
    denial separation.

    Args:
        hits: Recall results with excerpt, speaker, tags, score.
        question: The query (used for adversarial classification).
        query_type: Category hint (adversarial, temporal, etc.).
        max_chars: Maximum context length.

    Returns:
        Formatted context string ready for LLM consumption.
    """
    if query_type == "adversarial" and is_true_adversarial(question):
        return _pack_adversarial(hits, question, max_chars)
    elif query_type == "temporal":
        return _pack_temporal(hits, max_chars)
    elif query_type == "multi-hop":
        return _pack_multihop(hits, max_chars)
    else:
        return _pack_structured(hits, max_chars)


def _pack_structured(hits: list[dict], max_chars: int = 6000) -> str:
    """Universal structured packing — score-descending with metadata tags."""
    lines = []
    total = 0
    for r in hits:
        line = _format_structured_line(r)
        if not line:
            continue
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _pack_temporal(hits: list[dict], max_chars: int = 6000) -> str:
    """Temporal packing — chronological order by DiaID."""
    sorted_hits = sorted(hits, key=_dia_sort_key)
    lines = []
    total = 0
    for r in sorted_hits:
        line = _format_structured_line(r)
        if not line:
            continue
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _pack_multihop(hits: list[dict], max_chars: int = 6000) -> str:
    """Multi-hop packing — group by speaker/session, interleave for hop coverage.

    Ensures at least two different sources (speakers or sessions) get budget,
    preventing single-source starvation.
    """
    # Split by speaker
    by_speaker: dict[str, list[dict]] = {}
    for r in hits:
        sp = r.get("speaker", "") or "UNKNOWN"
        by_speaker.setdefault(sp, []).append(r)

    # Interleave: round-robin across speakers
    lines = []
    total = 0
    speakers = list(by_speaker.keys())
    max_per_speaker = max(len(v) for v in by_speaker.values()) if by_speaker else 0

    for idx in range(max_per_speaker):
        for sp in speakers:
            sp_hits = by_speaker[sp]
            if idx >= len(sp_hits):
                continue
            line = _format_structured_line(sp_hits[idx])
            if not line:
                continue
            if total + len(line) > max_chars:
                return "\n".join(lines)
            lines.append(line)
            total += len(line)

    return "\n".join(lines)


def _pack_adversarial(hits: list[dict], question: str, max_chars: int = 6000) -> str:
    """Structured evidence packing for adversarial/verification questions.

    Deterministic — no LLM, no starvation risk.
    Overlap-first ordering, then groups evidence vs denial.
    """
    # Compute query tokens for overlap scoring
    query_tokens = set(re.findall(r"[a-z]{2,}", question.lower()))

    # Sort by overlap with query (most relevant first)
    sorted_hits = sorted(hits, key=lambda r: _overlap_score(r, query_tokens), reverse=True)

    evidence_lines = []
    denial_lines = []
    total = 0

    for r in sorted_hits:
        line = _format_structured_line(r)
        if not line:
            continue
        if total + len(line) > max_chars:
            break

        clean = strip_semantic_prefix(r.get("excerpt", "").strip())
        if _DENIAL_RE.search(clean):
            denial_lines.append(line)
        else:
            evidence_lines.append(line)
        total += len(line)

    has_evidence = bool(evidence_lines or denial_lines)

    parts = []
    parts.append(f"EVIDENCE_FOUND: {'YES' if has_evidence else 'NO'}")
    parts.append("EVIDENCE:")
    if evidence_lines:
        parts.extend(f"- {ln}" for ln in evidence_lines)
    else:
        parts.append("- (none)")
    parts.append("DENIAL_EVIDENCE:")
    if denial_lines:
        parts.extend(f"- {ln}" for ln in denial_lines)
    else:
        parts.append("- (none)")

    return "\n".join(parts)

"""Deterministic evidence packer for Mind-Mem.

Builds structured, speaker-attributed evidence context from recall hits.
No LLM dependency — prevents starvation and hallucination in adversarial
and verification queries.

Supports two packing formats (configured via ``evidence_packing`` in config):
  - **chain_of_note** (default): Chain-of-Note style structured evidence with
    per-block key facts and relevance notes.
  - **raw**: Legacy [SPEAKER=...] [DATE=...] [DiaID=...] flat format.

ALL query types use category-specific ordering:
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


def _interleave_by_speaker(hits: list[dict]) -> list[dict]:
    """Interleave hits round-robin across speakers for multi-hop coverage."""
    by_speaker: dict[str, list[dict]] = {}
    for r in hits:
        sp = r.get("speaker", "") or "UNKNOWN"
        by_speaker.setdefault(sp, []).append(r)
    speakers = list(by_speaker.keys())
    max_per = max((len(v) for v in by_speaker.values()), default=0)
    ordered: list[dict] = []
    for idx in range(max_per):
        for sp in speakers:
            sp_hits = by_speaker[sp]
            if idx < len(sp_hits):
                ordered.append(sp_hits[idx])
    return ordered


_FACT_TAG_RE = re.compile(r"(?:Statement|Tags):\s*(.+)", re.IGNORECASE)


def _extract_key_facts(r: dict) -> list[str]:
    """Extract key facts from a hit's excerpt via Statement/Tags fields.

    Deterministic regex extraction — no LLM.
    """
    excerpt = r.get("excerpt", "")
    facts = []
    for m in _FACT_TAG_RE.finditer(excerpt):
        val = m.group(1).strip()
        if val:
            facts.append(val)
    # If no Statement/Tags fields found, extract a short summary from content
    if not facts:
        clean = strip_semantic_prefix(excerpt.strip())
        # Take first sentence or first 120 chars as a fact summary
        dot = clean.find(".")
        if 0 < dot < 120:
            facts.append(clean[: dot + 1])
        elif clean:
            facts.append(clean[:120])
    return facts


def _compute_relevance_note(r: dict, question: str) -> str:
    """Compute a brief keyword-overlap relevance note (deterministic)."""
    if not question:
        return "general context"
    query_tokens = set(re.findall(r"[a-z]{2,}", question.lower()))
    excerpt_lower = r.get("excerpt", "").lower()
    matched = sorted(t for t in query_tokens if t in excerpt_lower)
    if matched:
        return f"matches query terms: {', '.join(matched)}"
    return "background context"


def format_chain_of_note(
    hits: list[dict],
    question: str = "",
    max_chars: int = 6000,
) -> str:
    """Format hits as Chain-of-Note structured evidence.

    Each block is formatted as:
        [Note N] Source: <block_id> (score: X.XX)
        Content: <block content>
        Key facts: <extracted facts>
        Relevance: <keyword overlap note>
    """
    notes = []
    total = 0
    note_num = 0
    for r in hits:
        text = r.get("excerpt", "")
        if not text:
            continue
        note_num += 1
        clean = strip_semantic_prefix(text.strip())
        score = r.get("score", 0.0)
        block_id = r.get("DiaID", "") or r.get("block_id", "") or f"hit-{note_num}"
        speaker = r.get("speaker", "") or "UNKNOWN"
        date = r.get("Date", "") or ""

        facts = _extract_key_facts(r)
        relevance = _compute_relevance_note(r, question)

        lines = [f"[Note {note_num}] Source: {block_id} (score: {score:.2f})"]
        if speaker != "UNKNOWN" or date:
            meta = f"  Speaker: {speaker}"
            if date:
                meta += f" | Date: {date}"
            lines.append(meta)
        lines.append(f"  Content: {clean}")
        lines.append(f"  Key facts: {'; '.join(facts)}")
        lines.append(f"  Relevance: {relevance}")

        block = "\n".join(lines)
        if total + len(block) + 1 > max_chars:
            break
        notes.append(block)
        total += len(block) + 1  # +1 for separator newline
    return "\n".join(notes)


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
    config: dict | None = None,
) -> str:
    """Build structured evidence context from recall hits.

    By default uses Chain-of-Note format with per-block key facts and
    relevance notes.  Set ``config={"evidence_packing": "raw"}`` to use
    the legacy flat format.

    Args:
        hits: Recall results with excerpt, speaker, tags, score.
        question: The query (used for adversarial classification).
        query_type: Category hint (adversarial, temporal, etc.).
        max_chars: Maximum context length.
        config: Optional config dict; checks ``evidence_packing`` key.

    Returns:
        Formatted context string ready for LLM consumption.
    """
    packing_mode = (config or {}).get("evidence_packing", "chain_of_note")

    # Adversarial always uses its own specialized format
    if query_type == "adversarial" and is_true_adversarial(question):
        return _pack_adversarial(hits, question, max_chars)

    if packing_mode == "chain_of_note":
        # Apply category-specific ordering before formatting
        if query_type == "temporal":
            ordered = sorted(hits, key=_dia_sort_key)
        elif query_type == "multi-hop":
            ordered = _interleave_by_speaker(hits)
        else:
            ordered = hits
        return format_chain_of_note(ordered, question=question, max_chars=max_chars)

    # Raw / legacy format
    if query_type == "temporal":
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

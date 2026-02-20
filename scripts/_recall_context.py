"""Recall engine context packing — post-retrieval augmentation rules."""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recall_constants import _STOPWORDS
from _recall_detection import _parse_speaker_from_tags, get_block_type, get_excerpt
from observability import get_logger, metrics

__all__ = ["context_pack", "_parse_dia_id", "_block_to_result"]

_log = get_logger("recall.context")


# ---------------------------------------------------------------------------
# v8: Context Packing — post-retrieval augmentation rules
# ---------------------------------------------------------------------------

# Question-turn cue patterns (triggers adjacency expansion)
_QUESTION_CUE_RE = re.compile(
    r"(?:\?\s*$|any tips|do you|what should|how do|how about|what's your|"
    r"have you|can you|could you|would you|tell me|what kind|you think)",
    re.IGNORECASE,
)

# Multi-entity / plural question patterns (triggers diversity enforcement)
_MULTI_ENTITY_RE = re.compile(
    r"[A-Z][a-z]+\s+and\s+[A-Z][a-z]+",
)
_PLURAL_CUE_RE = re.compile(
    r"\b(ways|scares|reasons|things|hobbies|activities|events|gifts|tips|"
    r"strategies|memories|experiences|goals|plans|problems|both|each|"
    r"meals|snacks|books|games|songs|friends|times|items)\b",
    re.IGNORECASE,
)

# Words unlikely to be speaker names (filter for multi-entity detection)
_NOT_NAMES = frozenset({
    "what", "which", "when", "where", "who", "how", "the", "did", "does",
    "will", "has", "had", "was", "were", "are", "can", "could", "would",
    "should", "may", "might", "shall",
})

# Pronouns that signal coreference (Rule 3 trigger)
_PRONOUN_RE = re.compile(r"\b(it|this|that|these|those)\b", re.IGNORECASE)


def _parse_dia_id(dia: str) -> tuple[str, int] | None:
    """Parse 'D{session}:{turn}' -> (session_str, turn_int)."""
    m = re.match(r"D(\d+):(\d+)", dia)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _block_to_result(block: dict, score: float = 0.0) -> dict:
    """Convert a raw parsed block to a result dict."""
    tags_str = block.get("Tags", "")
    return {
        "_id": block.get("_id", "?"),
        "type": get_block_type(block.get("_id", "")),
        "score": round(score, 4),
        "excerpt": get_excerpt(block),
        "speaker": _parse_speaker_from_tags(tags_str),
        "tags": tags_str,
        "file": block.get("_source_file", "?"),
        "line": block.get("_line", 0),
        "status": block.get("Status", ""),
        "DiaID": block.get("DiaID", ""),
    }


def context_pack(
    query: str,
    top_results: list[dict],
    all_blocks: list[dict],
    wider_pool: list[dict],
    limit: int = 10,
) -> list[dict]:
    """Post-retrieval context packing. Augments top-K with deterministic rules.

    Rules:
    1. Dialog adjacency — if a hit is a question turn, add next 1-2 answer turns.
    2. Multi-entity diversity — for plural/multi-speaker queries, ensure distinct
       speakers and DiaIDs in the context.
    3. Pronoun-target rescue — if top hits use pronouns but query has a concrete
       noun, pull +/-3 neighbor turns to recover the explicit mention.

    Args:
        query: Original search query.
        top_results: Reranked top-K results.
        all_blocks: All parsed blocks (for neighbor lookup).
        wider_pool: Larger deduped candidate pool (for diversity fallback).
        limit: Original requested limit.

    Returns:
        Augmented result list. May exceed limit by a few context-pack blocks.
    """
    if not top_results or not all_blocks:
        return top_results

    # Build DiaID -> block lookup (dialog turns only, not fact cards)
    dia_lookup: dict[str, dict] = {}
    for block in all_blocks:
        dia = block.get("DiaID", "")
        bid = block.get("_id", "")
        if dia and bid.startswith("DIA-"):
            dia_lookup[dia] = block

    augmented = list(top_results)
    existing_ids = {r.get("_id", "") for r in augmented}
    existing_dias = {r.get("DiaID", "") for r in augmented}

    adjacency_added = 0
    diversity_forced = 0
    pronoun_rescue = 0

    # --- Rule 1: Dialog adjacency expansion ---
    for r in list(augmented):
        excerpt = r.get("excerpt", "")
        dia = r.get("DiaID", "")
        bid = r.get("_id", "")
        if not dia:
            continue
        # Only expand dialog turns, not fact cards
        if not bid.startswith("DIA-"):
            continue

        is_question = excerpt.rstrip().endswith("?") or bool(_QUESTION_CUE_RE.search(excerpt))
        if not is_question:
            continue

        parsed = _parse_dia_id(dia)
        if not parsed:
            continue
        session, turn_num = parsed

        for offset in [1, 2]:
            next_dia = f"D{session}:{turn_num + offset}"
            if next_dia in existing_dias:
                continue
            block = dia_lookup.get(next_dia)
            if not block:
                break  # end of session segment

            result = _block_to_result(block, score=r["score"] * 0.8)
            result["via_adjacency"] = True
            augmented.append(result)
            existing_dias.add(next_dia)
            existing_ids.add(result["_id"])
            adjacency_added += 1

    # --- Rule 2: Multi-entity diversity enforcement ---
    is_multi_entity = bool(_MULTI_ENTITY_RE.search(query))
    is_plural = bool(_PLURAL_CUE_RE.search(query))

    if is_multi_entity or is_plural:
        # Extract names from query
        query_names = set()
        for m_name in re.finditer(r"\b([A-Z][a-z]{2,})\b", query):
            name = m_name.group(1)
            if name.lower() not in _NOT_NAMES:
                query_names.add(name.lower())

        # Current speaker coverage
        current_speakers = {
            r.get("speaker", "").lower()
            for r in augmented if r.get("speaker")
        }
        # Current session coverage
        current_sessions = set()
        for r in augmented:
            parsed = _parse_dia_id(r.get("DiaID", ""))
            if parsed:
                current_sessions.add(parsed[0])

        # Check if diversity is needed
        missing_speakers = query_names - current_speakers if query_names else set()
        needs_diversity = bool(missing_speakers) or len(current_sessions) < 2

        if needs_diversity:
            max_extra = 5
            added_this_rule = 0
            for r in wider_pool:
                if added_this_rule >= max_extra:
                    break
                rid = r.get("_id", "")
                dia = r.get("DiaID", "")
                if rid in existing_ids:
                    continue

                sp = r.get("speaker", "").lower()
                parsed = _parse_dia_id(dia)
                new_session = parsed[0] if parsed else ""

                adds_speaker = sp in missing_speakers
                adds_session = new_session and new_session not in current_sessions

                if adds_speaker or adds_session:
                    r_copy = dict(r)
                    r_copy["via_diversity"] = True
                    augmented.append(r_copy)
                    existing_ids.add(rid)
                    if dia:
                        existing_dias.add(dia)
                    if sp:
                        current_speakers.add(sp)
                        missing_speakers.discard(sp)
                    if new_session:
                        current_sessions.add(new_session)
                    diversity_forced += 1
                    added_this_rule += 1

    # --- Rule 3: Pronoun-target rescue ---
    # Extract salient nouns from query (concrete nouns, not common words)
    query_nouns = set()
    for tok in re.findall(r"[a-z]+", query.lower()):
        if tok not in _STOPWORDS and len(tok) > 3 and tok not in _NOT_NAMES:
            query_nouns.add(tok)

    if query_nouns:
        # Check if top hits use pronouns without the salient nouns
        for r in list(augmented[:5]):
            excerpt = r.get("excerpt", "")
            excerpt_lower = excerpt.lower()
            dia = r.get("DiaID", "")
            bid = r.get("_id", "")

            if not dia or not bid.startswith("DIA-"):
                continue

            has_pronoun = bool(_PRONOUN_RE.search(excerpt_lower))
            # Check if any query noun is missing from this hit
            nouns_missing = {n for n in query_nouns if n not in excerpt_lower}

            if has_pronoun and len(nouns_missing) >= len(query_nouns) * 0.5:
                parsed = _parse_dia_id(dia)
                if not parsed:
                    continue
                session, turn_num = parsed

                # Search +/-3 turns for explicit noun mentions
                for offset in range(-3, 4):
                    if offset == 0:
                        continue
                    neighbor_dia = f"D{session}:{turn_num + offset}"
                    if neighbor_dia in existing_dias:
                        continue
                    block = dia_lookup.get(neighbor_dia)
                    if not block:
                        continue
                    block_text = get_excerpt(block).lower()
                    # Check if this neighbor contains any missing noun
                    if any(n in block_text for n in nouns_missing):
                        result = _block_to_result(block, score=r["score"] * 0.6)
                        result["via_pronoun_rescue"] = True
                        augmented.append(result)
                        existing_dias.add(neighbor_dia)
                        existing_ids.add(result["_id"])
                        pronoun_rescue += 1
                        if pronoun_rescue >= 3:
                            break
            if pronoun_rescue >= 3:
                break

    if adjacency_added or diversity_forced or pronoun_rescue:
        _log.info("context_pack",
                  adjacency_added=adjacency_added,
                  diversity_forced=diversity_forced,
                  pronoun_rescue=pronoun_rescue)
        metrics.inc("pack_adjacency", adjacency_added)
        metrics.inc("pack_diversity", diversity_forced)
        metrics.inc("pack_pronoun_rescue", pronoun_rescue)

    return augmented

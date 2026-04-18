# Copyright 2026 STARGA, Inc.
"""Project intelligence profile aggregator (v2.6.0).

Given a workspace's blocks + entity mentions, build a structured
``ProjectProfile`` summarising the project's shape: top concepts,
most-touched files, block-type distribution, recent activity. The
profile is cheap to recompute and is intended to be injected at the
top of an agent's system prompt so the agent starts with relevant
context instead of blank slate.

No LLM calls — the "convention extraction" bullet from the roadmap
stays deferred because it needs an LLM.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "will",
        "with",
        "this",
        "that",
        "these",
        "those",
        "we",
        "us",
        "our",
    }
)


@dataclass(frozen=True)
class ProjectProfile:
    """Aggregated intelligence for a single workspace / project."""

    name: str
    total_blocks: int
    block_types: dict[str, int] = field(default_factory=dict)
    top_concepts: list[str] = field(default_factory=list)
    top_files: list[str] = field(default_factory=list)
    top_entities: list[str] = field(default_factory=list)
    recent_block_count: int = 0
    generated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_blocks": self.total_blocks,
            "block_types": dict(self.block_types),
            "top_concepts": list(self.top_concepts),
            "top_files": list(self.top_files),
            "top_entities": list(self.top_entities),
            "recent_block_count": self.recent_block_count,
            "generated_at": self.generated_at,
        }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) >= 3]


def build_profile(
    blocks: Iterable[Mapping[str, Any]],
    *,
    name: str,
    top_k: int = 10,
    recent_window_days: int = 14,
    now: Optional[datetime] = None,
) -> ProjectProfile:
    """Summarise a project from its block collection.

    Inputs are intentionally generic dicts so the caller can feed
    either raw parsed blocks or projected rows from an index.
    Recognised keys per block:

        ``_id`` / ``id`` / ``block_id``  — identifier (logged only)
        ``type``                          — block type for histogram
        ``text`` / ``statement`` /
            ``excerpt`` / ``content``     — free text, mined for concepts
        ``file`` / ``path``               — source file for top_files
        ``date`` / ``created_at`` /
            ``timestamp``                 — ISO 8601 for recency counter
        ``entities`` (list of str) OR
            ``mentions`` (list of str)    — typed entities to aggregate

    Args:
        name: Human-readable project name recorded on the profile.
        top_k: Size cap on every "top N" list; 0 disables (all).
        recent_window_days: Window for ``recent_block_count``.
        now: Override clock for deterministic tests.

    Returns:
        A :class:`ProjectProfile`. Absent inputs degrade gracefully —
        e.g., a project with no timestamps just reports
        ``recent_block_count = 0``.
    """
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if recent_window_days < 0:
        raise ValueError("recent_window_days must be >= 0")
    current = now or _now_utc()
    cutoff = current - timedelta(days=recent_window_days)

    type_counter: Counter[str] = Counter()
    file_counter: Counter[str] = Counter()
    entity_counter: Counter[str] = Counter()
    concept_counter: Counter[str] = Counter()
    recent = 0
    total = 0

    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        total += 1
        block_type = str(block.get("type", "")).strip()
        if block_type:
            type_counter[block_type] += 1
        source_file = block.get("file") or block.get("path")
        if isinstance(source_file, str) and source_file.strip():
            file_counter[source_file.strip()] += 1

        # Entity aggregation — be permissive about field name + type.
        for entity_field in ("entities", "mentions"):
            value = block.get(entity_field)
            if isinstance(value, list):
                for ent in value:
                    if isinstance(ent, str) and ent.strip():
                        entity_counter[ent.strip()] += 1
            elif isinstance(value, str) and value.strip():
                entity_counter[value.strip()] += 1

        # Free text → bag-of-words with stopword + length filter.
        for text_field in ("text", "statement", "excerpt", "content"):
            value = block.get(text_field)
            if isinstance(value, str) and value:
                for tok in _tokens(value):
                    concept_counter[tok] += 1
                break  # one field is enough; avoid double-counting

        # Recency counter.
        for ts_field in ("date", "created_at", "timestamp"):
            dt = _parse_iso(block.get(ts_field))
            if dt is not None:
                if dt >= cutoff:
                    recent += 1
                break

    def _top(counter: Counter[str]) -> list[str]:
        items = counter.most_common(top_k or None)
        return [k for k, _ in items]

    return ProjectProfile(
        name=name,
        total_blocks=total,
        block_types=dict(type_counter),
        top_concepts=_top(concept_counter),
        top_files=_top(file_counter),
        top_entities=_top(entity_counter),
        recent_block_count=recent,
        generated_at=current.isoformat().replace("+00:00", "Z"),
    )


__all__ = ["ProjectProfile", "build_profile"]

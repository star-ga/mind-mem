# Copyright 2026 STARGA, Inc.
"""MRR tracker + packing-quality metric + convention extraction (v2.1.0, v2.4.0, v2.6.0).

Small grab-bag of observability + heuristics called out by the
roadmap:

- :class:`MRRTracker` — per-week MRR accumulator so
  ``index_stats`` can report retrieval quality drift.
- :class:`PackingQualityMeter` — records which packed blocks the
  answerer actually referenced; powers the "% of packed tokens the
  model used" metric.
- :func:`extract_conventions` — regex-level convention mining
  (naming patterns, test markers, error-handling idioms) without
  calling an LLM.
- :func:`model_context_window` — string → max tokens for the common
  hosted models so ``pack_to_budget`` can pick sensible defaults.

Pure stdlib.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional


# ---------------------------------------------------------------------------
# MRR tracker
# ---------------------------------------------------------------------------


@dataclass
class MRRWeek:
    iso_week: str
    mrr_sum: float = 0.0
    queries: int = 0

    @property
    def mean_mrr(self) -> float:
        return (self.mrr_sum / self.queries) if self.queries else 0.0


class MRRTracker:
    """Weekly MRR tracker for the signal-capture → model-quality loop."""

    def __init__(self, *, window_weeks: int = 52) -> None:
        self._window = int(window_weeks)
        self._weeks: dict[str, MRRWeek] = {}
        self._order: deque = deque()

    def record(
        self,
        ranked_ids: Iterable[str],
        relevant_ids: Iterable[str],
        *,
        at: Optional[datetime] = None,
    ) -> float:
        current = at or datetime.now(timezone.utc)
        iso_year, iso_week, _ = current.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        relevant = {str(x) for x in relevant_ids if x}
        if not relevant:
            return 0.0
        mrr = 0.0
        for rank, rid in enumerate(ranked_ids, start=1):
            if str(rid) in relevant:
                mrr = 1.0 / rank
                break
        week = self._weeks.setdefault(key, MRRWeek(iso_week=key))
        week.mrr_sum += mrr
        week.queries += 1
        if key not in self._order:
            self._order.append(key)
            while len(self._order) > self._window:
                victim = self._order.popleft()
                self._weeks.pop(victim, None)
        return mrr

    def weeks(self) -> list[dict[str, Any]]:
        return [
            {
                "iso_week": w.iso_week,
                "mean_mrr": round(w.mean_mrr, 6),
                "queries": w.queries,
            }
            for w in self._weeks.values()
        ]

    def delta(self) -> Optional[float]:
        """Signed change in mean MRR vs the previous week."""
        if len(self._order) < 2:
            return None
        last, prev = self._order[-1], self._order[-2]
        return self._weeks[last].mean_mrr - self._weeks[prev].mean_mrr


# ---------------------------------------------------------------------------
# Packing quality meter
# ---------------------------------------------------------------------------


class PackingQualityMeter:
    """Tracks how much of the packed context the user actually referenced."""

    def __init__(self) -> None:
        self._packed_tokens = 0
        self._referenced_tokens = 0
        self._events = 0

    def observe(self, packed: int, referenced: int) -> None:
        if packed < 0 or referenced < 0:
            raise ValueError("packed/referenced must be >= 0")
        self._packed_tokens += int(packed)
        self._referenced_tokens += int(min(referenced, packed))
        self._events += 1

    def ratio(self) -> float:
        return (
            (self._referenced_tokens / self._packed_tokens)
            if self._packed_tokens
            else 0.0
        )

    def stats(self) -> dict[str, float]:
        return {
            "packed_tokens": self._packed_tokens,
            "referenced_tokens": self._referenced_tokens,
            "events": self._events,
            "ratio": round(self.ratio(), 4),
        }


# ---------------------------------------------------------------------------
# Convention extraction (regex-only)
# ---------------------------------------------------------------------------


_NAMING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("snake_case", re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")),
    ("camelCase", re.compile(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b")),
    ("PascalCase", re.compile(r"\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b")),
    ("SCREAMING_SNAKE", re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")),
)

_TEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"def\s+test_[a-z0-9_]+"),
    re.compile(r"class\s+Test[A-Z][A-Za-z0-9]*"),
    re.compile(r"@pytest\.fixture"),
)

_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"raise\s+[A-Z][A-Za-z]+Error"),
    re.compile(r"except\s*\(\s*[A-Za-z, ]+Error\s*\)"),
    re.compile(r"logger\.(error|exception|warning)\b"),
)


_CONVENTION_MAX_SAMPLES: int = 100_000
_CONVENTION_MAX_BYTES_PER_SAMPLE: int = 2_097_152  # 2 MiB


def extract_conventions(samples: Iterable[str]) -> dict[str, Any]:
    """Roll up naming / testing / error-handling signals from raw code.

    Enforces per-sample and total-sample caps to keep regex scanning
    bounded on pathological inputs.
    """
    naming: Counter[str] = Counter()
    test_hits = 0
    error_hits = 0
    scanned = 0
    truncated_samples = 0
    for sample in samples:
        if scanned >= _CONVENTION_MAX_SAMPLES:
            break
        if not isinstance(sample, str) or not sample:
            continue
        scanned += 1
        if len(sample) > _CONVENTION_MAX_BYTES_PER_SAMPLE:
            sample = sample[:_CONVENTION_MAX_BYTES_PER_SAMPLE]
            truncated_samples += 1
        for label, pat in _NAMING_PATTERNS:
            naming[label] += len(pat.findall(sample))
        for pat in _TEST_PATTERNS:
            test_hits += len(pat.findall(sample))
        for pat in _ERROR_PATTERNS:
            error_hits += len(pat.findall(sample))
    dominant = naming.most_common(1)
    return {
        "dominant_naming": dominant[0][0] if dominant else None,
        "naming_histogram": dict(naming),
        "test_pattern_hits": test_hits,
        "error_handling_hits": error_hits,
        "samples_scanned": scanned,
        "samples_truncated": truncated_samples,
    }


# ---------------------------------------------------------------------------
# Model context windows
# ---------------------------------------------------------------------------


_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.4": 1_000_000,
    "gpt-5.4-2026-03-05": 1_000_000,
    "gpt-4o": 128_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gemini-3-pro": 1_000_000,
    "grok-4-1-fast-reasoning": 200_000,
    "mistral-large-latest": 128_000,
    "deepseek-reasoner": 64_000,
    "sonar-pro": 200_000,
}

_DEFAULT_CONTEXT_WINDOW: int = 32_000


def model_context_window(model: str) -> int:
    """Best-effort lookup of a model's context-window size in tokens."""
    if not model:
        return _DEFAULT_CONTEXT_WINDOW
    return _CONTEXT_WINDOWS.get(model.strip().lower(), _DEFAULT_CONTEXT_WINDOW)


__all__ = [
    "MRRTracker",
    "MRRWeek",
    "PackingQualityMeter",
    "extract_conventions",
    "model_context_window",
]

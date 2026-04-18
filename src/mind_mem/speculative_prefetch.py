# Copyright 2026 STARGA, Inc.
"""Speculative prefetch for multi-hop recall and repeated access patterns.

Multi-hop decomposition walks a chain of sub-queries whose result sets
often share blocks with the parent query. Rather than wait for the next
sub-query to execute, :class:`PrefetchPredictor` watches access history
and predicts likely next-blocks from the current query context, so the
orchestration layer can warm them ahead of time.

The predictor is a simple first-order Markov model over block ids,
conditioned on a query signature bucket. Pure-Python stdlib; no
embeddings, no LLM calls. Accuracy is bounded but the overhead is
negligible and any hit is strictly a latency win over the no-prefetch
path.

Pure-Python, stdlib only. Process-local — persistence across runs is
out of scope for v2.0.0b1.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Signature + stop-word handling
# ---------------------------------------------------------------------------


# Stripped aggressively so "How do I reset my JWT token?" and "Reset the
# JWT token for me" collapse onto the same signature bucket. The
# predictor is only useful when different phrasings share a signature.
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
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Cap on input length before tokenisation. A hostile MCP caller could
# otherwise submit a multi-megabyte query and burn CPU in the regex
# engine. Real queries are well under this; truncation is benign.
_MAX_SIGNATURE_INPUT: int = 4096


def signature(query: str) -> str:
    """Deterministic signature bucket for a free-form query.

    Lowercases, extracts word tokens (Unicode-aware so non-English
    queries don't all collapse onto the ``signature:empty`` bucket),
    strips English stop-words, sorts, joins with single spaces, and
    SHA-256 hashes for a bounded-length key.

    Queries whose tokenised-and-filtered form is empty are distinguished
    by hashing their raw lowered text so two noise queries land in
    different buckets. Inputs longer than :data:`_MAX_SIGNATURE_INPUT`
    are truncated before tokenisation to bound CPU usage.
    """
    truncated = query[:_MAX_SIGNATURE_INPUT].lower()
    # Unique-then-sort so "jwt jwt jwt" collapses to the same signature
    # as "jwt". The predictor treats repeated tokens as the same intent.
    tokens = {t for t in _TOKEN_RE.findall(truncated) if t not in _STOPWORDS}
    if not tokens:
        # Fall back to a per-query sentinel so distinct noise queries do
        # not pool together — pooling breaks prediction accuracy for
        # whichever caller happens to observe first.
        if not truncated.strip():
            return "signature:empty"
        return "empty:" + hashlib.sha256(truncated.encode("utf-8")).hexdigest()[:12]
    canonical = " ".join(sorted(tokens))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefetchStats:
    """Immutable snapshot of predictor statistics."""

    signatures: int
    observations: int
    predictions: int
    prefetch_checks: int
    prefetch_hits: int
    prefetch_misses: int

    @property
    def hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        if total == 0:
            return 0.0
        return self.prefetch_hits / total

    def as_dict(self) -> dict[str, Any]:
        return {
            "signatures": self.signatures,
            "observations": self.observations,
            "predictions": self.predictions,
            "prefetch_checks": self.prefetch_checks,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "hit_rate": round(self.hit_rate, 4),
        }


# ---------------------------------------------------------------------------
# PrefetchPredictor
# ---------------------------------------------------------------------------


@dataclass
class _Bucket:
    """Per-signature access histogram.

    Bucket-level LRU is handled by the parent :class:`PrefetchPredictor`
    via ``OrderedDict.move_to_end``; the bucket itself only tracks
    observation counts.
    """

    counts: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))

    def observe(self, block_id: str) -> None:
        self.counts[block_id] += 1

    def top(self, limit: int) -> list[str]:
        # Stable tiebreaker by block_id keeps rankings reproducible.
        return [bid for bid, _ in sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


class PrefetchPredictor:
    """Access-history predictor for speculative block warming.

    Args:
        max_signatures: Maximum number of signature buckets to retain.
            Exceeding this trims the least-recently-updated bucket so
            bursty workloads don't exhaust memory.
        per_bucket_cap: Maximum number of distinct block ids kept per
            signature. Trimmed by count (keeping the most-observed) on
            every observe call.

    The class is intentionally conservative. Predictions are a hint to
    the caller; the predictor never reaches into the retrieval layer
    itself and cannot stall the hot path.
    """

    def __init__(
        self,
        *,
        max_signatures: int = 1024,
        per_bucket_cap: int = 64,
    ) -> None:
        if max_signatures < 1:
            raise ValueError("max_signatures must be >= 1")
        if per_bucket_cap < 1:
            raise ValueError("per_bucket_cap must be >= 1")

        self._max_signatures = int(max_signatures)
        self._per_bucket_cap = int(per_bucket_cap)
        self._buckets: "OrderedDict[str, _Bucket]" = OrderedDict()
        self._lock = threading.RLock()
        self._tick = 0

        # Counters
        self._observations = 0
        self._predictions = 0
        self._prefetch_checks = 0
        self._prefetch_hits = 0
        self._prefetch_misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, query: str, block_ids: Iterable[str]) -> None:
        """Record that *query* ended up returning these block ids.

        Skips falsy block ids defensively so callers can pass raw
        result dicts without a pre-filter pass.
        """
        sig = signature(query)
        with self._lock:
            self._tick += 1
            bucket = self._buckets.get(sig)
            if bucket is None:
                bucket = _Bucket()
                self._buckets[sig] = bucket
            # Move to end to mark as most-recently used.
            self._buckets.move_to_end(sig, last=True)
            for bid in block_ids:
                if not bid:
                    continue
                bucket.observe(str(bid))
                self._observations += 1
            # Trim the bucket — prune the tail while preserving the
            # survivors' existing counts. Resetting all counts to 1
            # (the previous behaviour) erased frequency history and
            # froze the bucket on the initial top-N under equal-count
            # churn.
            if len(bucket.counts) > self._per_bucket_cap:
                kept_ids = set(bucket.top(self._per_bucket_cap))
                bucket.counts = defaultdict(
                    int,
                    {bid: cnt for bid, cnt in bucket.counts.items() if bid in kept_ids},
                )
            # Trim the signature registry.
            while len(self._buckets) > self._max_signatures:
                self._buckets.popitem(last=False)

    def predict(self, query: str, limit: int = 8) -> list[str]:
        """Return up to *limit* block ids to warm for this query.

        An unseen signature returns an empty list rather than raising,
        so callers can blindly prefetch whatever comes back without
        branching.
        """
        if limit < 1:
            return []
        sig = signature(query)
        with self._lock:
            bucket = self._buckets.get(sig)
            if bucket is None:
                return []
            self._buckets.move_to_end(sig, last=True)
            self._predictions += 1
            return bucket.top(limit)

    def evaluate(
        self,
        query: str,
        actual_block_ids: Iterable[str],
        *,
        predicted: Optional[Iterable[str]] = None,
    ) -> int:
        """Compare a prediction against the actual result set.

        Args:
            query: The same query that was fed to :meth:`predict`. Used
                only as a fallback when ``predicted`` is not supplied —
                in that mode a fresh prediction is run with
                ``limit=per_bucket_cap``, which is coarser than whatever
                the caller may have actually warmed.
            actual_block_ids: The block ids the recall actually returned.
            predicted: The exact block ids the caller prefetched. When
                supplied, ``evaluate`` measures efficacy against THAT
                set and does not record a new prediction. Callers on the
                hot path should pass this so the reported hit rate
                reflects real warming, not a re-derived one.

        Returns:
            Number of predicted blocks that were actually needed.
        """
        if predicted is None:
            # Fallback for simple call sites; uses the predictor's own
            # view of what it would have produced.
            predicted_set = set(self.predict(query, limit=self._per_bucket_cap))
        else:
            predicted_set = {str(bid) for bid in predicted if bid}
        if not predicted_set:
            return 0
        actual = {str(bid) for bid in actual_block_ids if bid}
        hits = len(predicted_set & actual)
        misses = len(predicted_set) - hits
        with self._lock:
            self._prefetch_checks += 1
            self._prefetch_hits += hits
            self._prefetch_misses += misses
        return hits

    def clear(self) -> None:
        """Drop all buckets and reset counters."""
        with self._lock:
            self._buckets.clear()
            self._observations = 0
            self._predictions = 0
            self._prefetch_checks = 0
            self._prefetch_hits = 0
            self._prefetch_misses = 0
            self._tick = 0

    def stats(self) -> PrefetchStats:
        with self._lock:
            return PrefetchStats(
                signatures=len(self._buckets),
                observations=self._observations,
                predictions=self._predictions,
                prefetch_checks=self._prefetch_checks,
                prefetch_hits=self._prefetch_hits,
                prefetch_misses=self._prefetch_misses,
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_default_predictor: Optional[PrefetchPredictor] = None
_default_lock = threading.RLock()


def get_default_predictor() -> PrefetchPredictor:
    """Return the process-wide default predictor, lazy-constructed."""
    global _default_predictor
    with _default_lock:
        if _default_predictor is None:
            _default_predictor = PrefetchPredictor()
        return _default_predictor


def reset_default_predictor() -> None:
    """Tear down the default predictor (primarily for tests)."""
    global _default_predictor
    with _default_lock:
        _default_predictor = None


__all__ = [
    "PrefetchPredictor",
    "PrefetchStats",
    "signature",
    "get_default_predictor",
    "reset_default_predictor",
]

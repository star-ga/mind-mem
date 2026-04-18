# Copyright 2026 STARGA, Inc.
"""Interaction signals for self-improving retrieval (v2.1.0).

Captures lightweight feedback from normal user interaction so the
retrieval pipeline can improve over time without any explicit rating UI.

Three signal types (from the OpenClaw-RL "Train Any Agent Simply by
Talking" pattern):

    RE_QUERY     — the user repeated the same intent with different
                   phrasing within a short window. The prior recall did
                   not satisfy them.
    REFINEMENT   — the user rephrased with added constraints (a narrower
                   or different filter). The prior recall was
                   directionally right but incomplete.
    CORRECTION   — the user said "no, I meant X" or similar. The prior
                   recall was wrong.

The signal store is append-only JSONL — auditable, easy to replay, no
DB dependency. Ships with a simple A/B eval harness (``evaluate_ab``)
that replays signals against a retrieval function to compare a
baseline vs candidate on MRR.

Zero new dependencies — stdlib only. Fine-tuning hooks are intentionally
deferred: the signal store is the foundation; training loops arrive in
later releases once the toolchain can host them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# Signal taxonomy
# ---------------------------------------------------------------------------


class SignalType(str, Enum):
    """Three interaction-signal categories tracked for retrieval learning."""

    RE_QUERY = "re_query"
    REFINEMENT = "refinement"
    CORRECTION = "correction"


# ---------------------------------------------------------------------------
# Tokenisation helpers — shared with speculative_prefetch's signature spirit
# but deliberately lighter since we only need bag-of-words similarity here.
# ---------------------------------------------------------------------------


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
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "their",
        "them",
        "to",
        "was",
        "were",
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


# Cap on input length before tokenising. A hostile MCP caller could
# otherwise submit a multi-MB query and burn CPU on regex matching; the
# similarity ranking is unchanged on realistic queries since prefixes
# dominate.
_MAX_TOKEN_INPUT: int = 8192


def _tokens(text: str) -> set[str]:
    capped = text[:_MAX_TOKEN_INPUT].lower()
    return {t for t in _TOKEN_RE.findall(capped) if t and t not in _STOPWORDS}


def jaccard_similarity(a: str, b: str) -> float:
    """Bag-of-words Jaccard between two query strings.

    Returns a value in [0, 1]. Empty-on-both-sides returns 0.0 so the
    classifier never flags empty queries as "same intent".
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union)


# Correction markers use word-boundary regexes so phrases like "no, i
# mean business" or "wrongdoing" don't false-positive. Patterns cover
# the common colloquial forms without over-fitting to one locale.
_CORRECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bno[, ]*i\s+(?:meant?|mean)\b", re.IGNORECASE),
    re.compile(r"\bactually[, ]*i\s+(?:meant?|mean)\b", re.IGNORECASE),
    re.compile(r"\bi\s+meant\b", re.IGNORECASE),
    re.compile(r"\bthat('?s| is| was)?\s+not\s+(?:right|what|correct)\b", re.IGNORECASE),
    re.compile(r"\bthats\s+not\s+(?:right|what|correct)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+that\b", re.IGNORECASE),
    re.compile(r"\bthat'?s\s+wrong\b", re.IGNORECASE),
)


def _correction_markers(query: str) -> bool:
    """Detect explicit "no / not / actually / I meant" corrections.

    Uses word-boundary regexes so we don't false-positive on
    ``wrongdoing``, ``business``, or similar substrings that share the
    same letters as a correction phrase.
    """
    capped = query[:_MAX_TOKEN_INPUT]
    return any(p.search(capped) for p in _CORRECTION_PATTERNS)


def classify(
    previous_query: str,
    new_query: str,
    *,
    similarity_threshold: float = 0.55,
    refinement_threshold: float = 0.35,
) -> Optional[SignalType]:
    """Decide whether *new_query* relates to *previous_query* and how.

    Returns ``None`` when the two queries share no meaningful overlap —
    the user moved on rather than iterating.

    - Similarity >= ``similarity_threshold`` and no correction markers →
      ``RE_QUERY`` (they repeated the same intent).
    - Correction markers present → ``CORRECTION``.
    - Similarity in ``[refinement_threshold, similarity_threshold)`` →
      ``REFINEMENT`` (narrower or shifted angle on the same topic).
    - Otherwise ``None``.
    """
    if not new_query.strip() or not previous_query.strip():
        return None
    if _correction_markers(new_query):
        return SignalType.CORRECTION
    sim = jaccard_similarity(previous_query, new_query)
    if sim >= similarity_threshold:
        return SignalType.RE_QUERY
    if sim >= refinement_threshold:
        return SignalType.REFINEMENT
    return None


# ---------------------------------------------------------------------------
# Signal record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """Immutable interaction-feedback record.

    Stored as a JSONL line. ``signal_id`` is a stable SHA-256 hash of
    the query pair + timestamp so duplicate captures collapse under
    :meth:`SignalStore.observe`.
    """

    signal_id: str
    timestamp: str
    session_id: str
    signal_type: SignalType
    previous_query: str
    new_query: str
    previous_results: tuple[str, ...]
    similarity: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "signal_type": self.signal_type.value,
            "previous_query": self.previous_query,
            "new_query": self.new_query,
            "previous_results": list(self.previous_results),
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Signal":
        return cls(
            signal_id=str(data["signal_id"]),
            timestamp=str(data["timestamp"]),
            session_id=str(data["session_id"]),
            signal_type=SignalType(data["signal_type"]),
            previous_query=str(data["previous_query"]),
            new_query=str(data["new_query"]),
            previous_results=tuple(str(x) for x in data.get("previous_results", [])),
            similarity=float(data.get("similarity", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


def _signal_id(
    session_id: str,
    previous_query: str,
    new_query: str,
    timestamp: str,
) -> str:
    payload = f"{session_id}\x1f{previous_query}\x1f{new_query}\x1f{timestamp}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Signal store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalStats:
    """Snapshot of aggregated signal counts — used for MCP observability."""

    total: int
    re_query: int
    refinement: int
    correction: int
    unique_sessions: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "re_query": self.re_query,
            "refinement": self.refinement,
            "correction": self.correction,
            "unique_sessions": self.unique_sessions,
        }


class SignalStore:
    """Append-only JSONL store for :class:`Signal` records.

    Thread-safe: writes are lock-protected and the file is flushed /
    fsync'd per record so a crash cannot lose the tail of a session's
    feedback. Duplicate `signal_id`s are skipped silently so idempotent
    replay is safe.
    """

    def __init__(self, path: str) -> None:
        self._path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._lock = threading.RLock()
        self._seen_ids: set[str] = set()
        if os.path.isfile(self._path):
            self._load_ids()

    @property
    def path(self) -> str:
        return self._path

    def _load_ids(self) -> None:
        # errors="replace" keeps the loader robust against binary
        # corruption at the tail of the JSONL file (e.g., a partial
        # write during a crash). Bad bytes decode to the replacement
        # character, json.loads then skips the line cleanly.
        with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    continue
                sid = data.get("signal_id") if isinstance(data, dict) else None
                if isinstance(sid, str):
                    self._seen_ids.add(sid)

    def observe(
        self,
        *,
        session_id: str,
        previous_query: str,
        new_query: str,
        signal_type: SignalType,
        similarity: float,
        previous_results: Iterable[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[Signal]:
        """Append a signal. Returns the record, or ``None`` on duplicate.

        Callers that already ran :func:`classify` usually want to pass
        its result straight in; a small convenience wrapper
        (:meth:`observe_pair`) bundles classify + observe for the common
        path.
        """
        ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sid = _signal_id(session_id, previous_query, new_query, ts)
        with self._lock:
            if sid in self._seen_ids:
                return None
            record = Signal(
                signal_id=sid,
                timestamp=ts,
                session_id=session_id,
                signal_type=signal_type,
                previous_query=previous_query,
                new_query=new_query,
                previous_results=tuple(previous_results),
                similarity=float(similarity),
                metadata=dict(metadata or {}),
            )
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._seen_ids.add(sid)
            return record

    def observe_pair(
        self,
        *,
        session_id: str,
        previous_query: str,
        new_query: str,
        previous_results: Iterable[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Signal]:
        """Classify + append in one call. Returns the signal or ``None``.

        ``None`` indicates either the two queries were unrelated (not a
        feedback event) or the derived signal collided with an existing
        one (idempotent replay).
        """
        sig_type = classify(previous_query, new_query)
        if sig_type is None:
            return None
        return self.observe(
            session_id=session_id,
            previous_query=previous_query,
            new_query=new_query,
            signal_type=sig_type,
            similarity=jaccard_similarity(previous_query, new_query),
            previous_results=previous_results,
            metadata=metadata,
        )

    def all_signals(self) -> list[Signal]:
        """Load and return every signal, oldest first.

        Corrupt or non-UTF-8 lines are skipped silently so a single bad
        record never blocks verification of the rest of the ledger.
        """
        if not os.path.isfile(self._path):
            return []
        signals: list[Signal] = []
        with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    signals.append(Signal.from_dict(json.loads(stripped)))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
        return signals

    def stats(self) -> SignalStats:
        signals = self.all_signals()
        counts = {t: 0 for t in SignalType}
        sessions: set[str] = set()
        for s in signals:
            counts[s.signal_type] += 1
            sessions.add(s.session_id)
        return SignalStats(
            total=len(signals),
            re_query=counts[SignalType.RE_QUERY],
            refinement=counts[SignalType.REFINEMENT],
            correction=counts[SignalType.CORRECTION],
            unique_sessions=len(sessions),
        )


# ---------------------------------------------------------------------------
# A/B evaluation
# ---------------------------------------------------------------------------


# Type alias: a retrieval function takes a query and returns a ranked
# list of block ids. The A/B harness doesn't care what produced the
# ranking — it only scores the output.
RetrievalFn = Callable[[str], list[str]]


@dataclass(frozen=True)
class ABResult:
    """Summary of a baseline vs candidate A/B evaluation."""

    signals_scored: int
    baseline_mrr: float
    candidate_mrr: float

    @property
    def delta(self) -> float:
        return self.candidate_mrr - self.baseline_mrr

    @property
    def winner(self) -> str:
        if self.candidate_mrr > self.baseline_mrr:
            return "candidate"
        if self.candidate_mrr < self.baseline_mrr:
            return "baseline"
        return "tie"

    def as_dict(self) -> dict:
        return {
            "signals_scored": self.signals_scored,
            "baseline_mrr": round(self.baseline_mrr, 6),
            "candidate_mrr": round(self.candidate_mrr, 6),
            "delta": round(self.delta, 6),
            "winner": self.winner,
        }


def _mrr(ranked: list[str], target_ids: Iterable[str]) -> float:
    """Mean reciprocal rank of the first target id in the ranked list."""
    targets = {str(t) for t in target_ids if t}
    if not targets:
        return 0.0
    for rank, block_id in enumerate(ranked, start=1):
        if str(block_id) in targets:
            return 1.0 / rank
    return 0.0


def evaluate_ab(
    signals: Iterable[Signal],
    *,
    baseline: RetrievalFn,
    candidate: RetrievalFn,
    limit: int = 10,
) -> ABResult:
    """Replay signals against two retrieval functions and compare MRR.

    Each :data:`RE_QUERY` and :data:`REFINEMENT` signal encodes "the
    user re-asked — the ideal answer contained what they saw before
    plus whatever they really wanted". We score by whether the
    previously-returned results (``previous_results``) reappear in the
    candidate's output, which is a weak proxy but works as a regression
    gate: a change that drops known-relevant blocks fails fast.
    ``CORRECTION`` signals are excluded because the previous results
    were wrong by the user's own admission.
    """
    baseline_score = 0.0
    candidate_score = 0.0
    scored = 0
    for sig in signals:
        if sig.signal_type == SignalType.CORRECTION:
            continue
        if not sig.previous_results:
            continue
        base_rank = baseline(sig.new_query)[:limit]
        cand_rank = candidate(sig.new_query)[:limit]
        baseline_score += _mrr(base_rank, sig.previous_results)
        candidate_score += _mrr(cand_rank, sig.previous_results)
        scored += 1
    if scored == 0:
        return ABResult(signals_scored=0, baseline_mrr=0.0, candidate_mrr=0.0)
    return ABResult(
        signals_scored=scored,
        baseline_mrr=baseline_score / scored,
        candidate_mrr=candidate_score / scored,
    )


__all__ = [
    "SignalType",
    "Signal",
    "SignalStats",
    "SignalStore",
    "ABResult",
    "RetrievalFn",
    "classify",
    "jaccard_similarity",
    "evaluate_ab",
]

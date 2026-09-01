# Copyright 2026 STARGA, Inc.
"""MRR tracker + packing-quality metric + convention extraction (v2.1.0, v2.4.0, v2.6.0).

Small grab-bag of observability + heuristics called out by the
roadmap:

- :class:`MRRTracker` — per-week MRR accumulator so
  ``index_stats`` can report retrieval quality drift, and so the
  promotion gate in :mod:`mind_mem.online_trainer` has a *measured*
  baseline to compare a candidate against rather than a hand-passed one.
- :class:`PackingQualityMeter` — records which packed blocks the
  answerer actually referenced; powers the "% of packed tokens the
  model used" metric.
- :class:`PackReceiptRegistry` — the join between the two halves of
  that metric. The pack path knows what it packed and what each block
  cost; only the feedback path knows what was referenced. The registry
  is what lets one tell the other, in-process, with no new file and no
  clock.
- :func:`extract_conventions` — regex-level convention mining
  (naming patterns, test markers, error-handling idioms) without
  calling an LLM.
- :func:`context_window` / :func:`model_context_window` — model id →
  context-window size so ``pack_recall_budget`` can respect the window
  a caller is actually packing for.

Pure stdlib, plus one intra-package import: the reciprocal-rank kernel
is :func:`mind_mem.interaction_signals.reciprocal_rank`, the tested
incumbent that ``evaluate_ab`` already scores with. This module used to
carry a second copy of that loop inline. One metric, one implementation
— a tracker that drifted from the A/B harness would report a "delta"
against a baseline computed by different arithmetic.

DETERMINISM. Nothing here is on the scored recall path, and nothing here
may put a clock on it. :meth:`MRRTracker.record` takes the instant as an
argument; :func:`mrr_from_events` requires one per event. The only clock
read in the module is :func:`_utcnow`, reached solely by the convenience
default of ``record(at=None)``, which the wired paths never take — they
inject the timestamp the signal ledger already recorded, so a replay of
the same ledger yields the same weeks and the same delta forever.
"""

from __future__ import annotations

import re
import threading
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Final, Iterable, Mapping, Optional, Sequence

from .interaction_signals import reciprocal_rank


def _utcnow() -> datetime:
    """The module's only clock read — see the determinism note above.

    Factored out so a test can prove a wired path never reaches it: swap
    this for something that raises and the MRR block must still compute.
    """
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# MRR tracker
# ---------------------------------------------------------------------------


def iso_week_key(moment: datetime) -> str:
    """``YYYY-Www`` bucket key for *moment* (ISO-8601 week numbering)."""
    iso_year, iso_week, _ = moment.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


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
        """Score one ranked list against its relevant set into *at*'s week.

        ``at=None`` reads the clock and is the unwired convenience path
        only; every caller that feeds this from stored data passes the
        instant that data was recorded at, which is what keeps the weekly
        series a pure function of the ledger.
        """
        current = at or _utcnow()
        key = iso_week_key(current)
        relevant = [str(x) for x in relevant_ids if x]
        if not relevant:
            return 0.0
        mrr = reciprocal_rank([str(r) for r in ranked_ids], relevant)
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

    def baseline_mrr(self) -> Optional[float]:
        """Mean MRR of the most recent week with any scored query.

        This is the number :func:`mind_mem.online_trainer.promote_candidate`
        gates on: a candidate must beat the *measured* current quality by
        ``min_improvement``, not a figure typed into the call. ``None``
        when nothing has been scored — a promotion gate with no baseline
        must decline to gate, never silently treat "no data" as 0.0 and
        wave every candidate through.
        """
        for key in reversed(self._order):
            week = self._weeks.get(key)
            if week is not None and week.queries:
                return week.mean_mrr
        return None

    def scored(self) -> int:
        """Total queries scored across every retained week."""
        return sum(w.queries for w in self._weeks.values())

    def as_dict(self) -> dict[str, Any]:
        delta = self.delta()
        baseline = self.baseline_mrr()
        return {
            "weeks": self.weeks(),
            "queries_scored": self.scored(),
            "delta": None if delta is None else round(delta, 6),
            "baseline_mrr": None if baseline is None else round(baseline, 6),
        }


@dataclass(frozen=True)
class MRREvent:
    """One scoreable retrieval outcome: a ranked list, its labels, its instant.

    ``at`` is required and is a ``datetime`` — the event carries the
    instant it happened at, so the weekly series never depends on when it
    is read.
    """

    ranked_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    at: datetime


def mrr_from_events(events: Iterable[MRREvent], *, window_weeks: int = 52) -> MRRTracker:
    """Fold scoreable events into a tracker. Pure — no clock, no I/O.

    Events are sorted by their own instant so the week ordering (and
    therefore :meth:`MRRTracker.delta`) does not depend on the order the
    caller happened to load them in.
    """
    tracker = MRRTracker(window_weeks=window_weeks)
    for event in sorted(events, key=lambda e: e.at):
        tracker.record(event.ranked_ids, event.relevant_ids, at=event.at)
    return tracker


def mrr_events_from_signals(
    signals: Iterable[Any],
    labels: Mapping[str, Iterable[str]],
    *,
    fingerprint: Callable[[str], str],
) -> list[MRREvent]:
    """Join the ranked lists in the signal ledger to the relevance labels.

    Both halves of a real MRR already exist in the workspace and have never
    been put together:

    * ``observe_signal`` stores ``previous_results`` — the ranked block ids
      a recall actually returned — plus the instant it happened at.
    * ``calibration_feedback`` stores which block ids a caller then marked
      **accepted** for a query.

    An event is the first scored against the second. A signal with no
    ranked list, no labels for either phrasing of its intent, or an
    unparseable timestamp is skipped rather than scored as zero — an
    unlabelled query is *unmeasured*, and counting it as a miss would drag
    the series toward zero in exact proportion to how little feedback a
    workspace gives, which is the opposite of what the metric is for.

    The relevant set is the union of the labels for the previous and the
    new phrasing. A re-query, a refinement and a correction are all one
    intent asked twice; the blocks the caller accepted under either
    phrasing are the ones the earlier ranking should have surfaced. For a
    CORRECTION that union is exactly what makes the miss visible — the
    previous ranking is scored against what the user turned out to want.

    ``fingerprint`` is injected so this module keeps its dependency
    footprint (stdlib + the reciprocal-rank kernel) and does not drag the
    calibration store's sqlite layer in behind a metric helper. Pure: no
    clock, no I/O.
    """
    events: list[MRREvent] = []
    for signal in signals:
        ranked = tuple(str(r) for r in getattr(signal, "previous_results", ()) or () if r)
        if not ranked:
            continue
        moment = parse_signal_timestamp(getattr(signal, "timestamp", ""))
        if moment is None:
            continue
        relevant: set[str] = set()
        for query in (getattr(signal, "previous_query", ""), getattr(signal, "new_query", "")):
            if not query:
                continue
            for block_id in labels.get(fingerprint(query), ()) or ():
                relevant.add(str(block_id))
        if not relevant:
            continue
        events.append(
            MRREvent(
                ranked_ids=ranked,
                relevant_ids=tuple(sorted(relevant)),
                at=moment,
            )
        )
    return events


def parse_signal_timestamp(value: str) -> Optional[datetime]:
    """Parse a signal-ledger timestamp (``%Y-%m-%dT%H:%M:%SZ``) as UTC.

    Returns ``None`` rather than raising on anything unparseable: a single
    corrupt ledger line must not take the whole metric down, and an event
    with no trustworthy instant cannot be placed in a week at all.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    try:
        moment = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Packing quality meter
# ---------------------------------------------------------------------------


class PackingQualityMeter:
    """Tracks how much of the packed context the user actually referenced.

    Thread-safe: the wired instance is a process-local singleton shared by
    the pack path and the feedback path, which in an MCP server are two
    different request threads.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._packed_tokens = 0
        self._referenced_tokens = 0
        self._events = 0

    def observe(self, packed: int, referenced: int) -> None:
        if packed < 0 or referenced < 0:
            raise ValueError("packed/referenced must be >= 0")
        with self._lock:
            self._packed_tokens += int(packed)
            self._referenced_tokens += int(min(referenced, packed))
            self._events += 1

    def ratio(self) -> float:
        with self._lock:
            return (self._referenced_tokens / self._packed_tokens) if self._packed_tokens else 0.0

    def reset(self) -> None:
        with self._lock:
            self._packed_tokens = 0
            self._referenced_tokens = 0
            self._events = 0

    def stats(self) -> dict[str, float]:
        with self._lock:
            packed = self._packed_tokens
            referenced = self._referenced_tokens
            events = self._events
        return {
            "packed_tokens": packed,
            "referenced_tokens": referenced,
            "events": events,
            "ratio": round((referenced / packed) if packed else 0.0, 4),
        }


# ---------------------------------------------------------------------------
# Pack receipts — the join between "what we packed" and "what was used"
# ---------------------------------------------------------------------------


#: How many pack receipts to retain. Feedback arrives one turn after the
#: pack, so a handful would do; 512 covers an agent that fans out many
#: recalls before reporting on any of them, and caps the memory a hostile
#: caller can make the server hold.
PACK_RECEIPT_CAPACITY: Final[int] = 512


@dataclass(frozen=True)
class PackReceipt:
    """What one pack put in front of the model, priced per block."""

    fingerprint: str
    packed_tokens: int
    block_tokens: dict[str, int]

    def referenced_tokens(self, block_ids: Iterable[str]) -> int:
        """Tokens attributable to *block_ids* among the blocks we packed.

        Ids that were not in the pack contribute nothing — the metric is
        "of what we packed, how much was used", so crediting a block the
        pack never included would inflate the ratio above what happened.
        """
        wanted = {str(b) for b in block_ids if b}
        return sum(cost for bid, cost in self.block_tokens.items() if bid in wanted)


class PackReceiptRegistry:
    """Bounded, in-process, FIFO store of recent :class:`PackReceipt`s.

    Deliberately memory-only. A pack receipt is telemetry about one turn;
    persisting it would put a writer on the recall path, and a timestamped
    one would put a clock there too.
    """

    def __init__(self, *, capacity: int = PACK_RECEIPT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = int(capacity)
        self._lock = threading.RLock()
        self._receipts: dict[str, PackReceipt] = {}
        self._order: deque = deque()

    def record(self, receipt: PackReceipt) -> None:
        with self._lock:
            if receipt.fingerprint not in self._receipts:
                self._order.append(receipt.fingerprint)
            self._receipts[receipt.fingerprint] = receipt
            while len(self._order) > self._capacity:
                self._receipts.pop(self._order.popleft(), None)

    def get(self, fingerprint: str) -> Optional[PackReceipt]:
        with self._lock:
            return self._receipts.get(fingerprint)

    def clear(self) -> None:
        with self._lock:
            self._receipts.clear()
            self._order.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._receipts)


_default_meter = PackingQualityMeter()
_default_receipts = PackReceiptRegistry()


def default_packing_meter() -> PackingQualityMeter:
    """Process-local meter the pack + feedback paths share.

    Same shape as the ``prefix_cache`` / ``speculative_prefetch``
    singletons ``index_stats`` already reports: in-process counters for
    the life of the server, not a persisted series.
    """
    return _default_meter


def default_pack_receipts() -> PackReceiptRegistry:
    """Process-local pack-receipt registry the feedback path joins against."""
    return _default_receipts


def reset_packing_state() -> None:
    """Clear both process-local singletons. For tests and for `mm` reruns."""
    _default_meter.reset()
    _default_receipts.clear()


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


#: Context windows we have actually verified, keyed by lowercase model id.
#:
#: VERIFIED-ONLY, and it stays that way. The temptation on this table is to
#: fill it out with plausible numbers for every id anyone might pass; that is
#: the same failure as the silent default below, moved one line up. A guessed
#: window is worse than no window, because a *wrong* window is applied without
#: anyone being told it was invented, and the pack it sizes overflows the
#: window it was asked to respect.
#:
#: Add-don't-replace on rename: an id that has been superseded stays, so a
#: caller still on the old string keeps getting the right number instead of
#: falling off the table.
_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.5": 1_000_000,
    "gpt-5.5-2026-03-05": 1_000_000,
    "gpt-4o": 128_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-haiku-4-5": 200_000,
    "gemini-3.5-flash": 1_000_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gemini-3-pro": 1_000_000,
    "grok-4.3": 200_000,
    "grok-4-1-fast-reasoning": 200_000,
    "mistral-large-latest": 128_000,
    "deepseek-v4-pro": 64_000,
    "deepseek-reasoner": 64_000,
    "glm-5.1": 1_000_000,
    "kimi-k2.6": 200_000,
    "sonar-pro": 200_000,
}

#: What an unlisted model used to get, silently, as if it were a fact.
#:
#: It is still the value :func:`model_context_window` returns, because that
#: function's contract is "an int, always" and callers depend on it. What has
#: changed is that the number no longer travels anonymously: :func:`context_window`
#: reports ``known=False`` beside it, and the pack path refuses to *clamp* a
#: budget to a window it had to invent. Sizing a 200 K-window model's context
#: to 32 K wastes 84% of it; sizing a 32 K one to 200 K overflows it. Both were
#: previously indistinguishable from a measured answer.
UNKNOWN_MODEL_WINDOW: Final[int] = 32_000
_DEFAULT_CONTEXT_WINDOW: Final[int] = UNKNOWN_MODEL_WINDOW


@dataclass(frozen=True)
class ContextWindow:
    """A context-window answer that says whether it is known or assumed."""

    model: str
    tokens: int
    known: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "context_window": self.tokens if self.known else None,
            "model_known": self.known,
        }


def known_models() -> tuple[str, ...]:
    """Every model id with a verified context window, sorted."""
    return tuple(sorted(_CONTEXT_WINDOWS))


def _normalise_model(model: str) -> str:
    return model.strip().lower() if isinstance(model, str) else ""


def is_known_model(model: str) -> bool:
    """True iff *model* has a verified window rather than the fallback."""
    return _normalise_model(model) in _CONTEXT_WINDOWS


def context_window(model: str) -> ContextWindow:
    """Resolve *model*'s context window, flagging an unverified answer.

    Prefer this over :func:`model_context_window` anywhere the answer
    changes a decision. An unknown model yields ``known=False`` and the
    fallback token count, and it is then the caller's job to decide what
    to do about not knowing — not to spend the invented number as if it
    had been looked up.
    """
    key = _normalise_model(model)
    tokens = _CONTEXT_WINDOWS.get(key)
    if tokens is None:
        return ContextWindow(model=key, tokens=UNKNOWN_MODEL_WINDOW, known=False)
    return ContextWindow(model=key, tokens=tokens, known=True)


def model_context_window(model: str) -> int:
    """Best-effort lookup of a model's context-window size in tokens.

    Always an int. Unknown ids get :data:`UNKNOWN_MODEL_WINDOW`, which is
    an assumption, not a measurement — use :func:`context_window` when you
    need to know which one you got.
    """
    return context_window(model).tokens


def resolve_pack_budget(requested_max_tokens: int, model: str) -> dict[str, Any]:
    """Decide the token budget a pack should actually use for *model*.

    The rule, and the whole point of the three-way distinction:

    * **known model, request over the window** → clamp to the window. The
      caller asked for a pack that cannot be sent.
    * **known model, request within the window** → honour the request. A
      caller who asks for less than the window means it.
    * **unknown model** → do not clamp, and say so. Clamping to a guess is
      how a 200 K window silently became 32 K.

    Pure: a function of its two arguments only.
    """
    window = context_window(model)
    requested = int(requested_max_tokens)
    if window.known and requested > window.tokens:
        effective, clamped = window.tokens, True
    else:
        effective, clamped = requested, False
    out: dict[str, Any] = {
        **window.as_dict(),
        "requested_max_tokens": requested,
        "effective_max_tokens": effective,
        "clamped": clamped,
    }
    if not window.known:
        out["note"] = (
            (
                "unknown model id — budget left at the requested value rather than "
                "clamped to an assumed window; pass a listed id to have it enforced"
            )
            if model
            else "no model id given — budget not clamped"
        )
    return out


def pack_receipt_from_included(
    fingerprint: str,
    included: Sequence[dict],
    tokens_used: int,
) -> PackReceipt:
    """Build a :class:`PackReceipt` from a ``pack_to_budget`` result.

    Reads only the ``_token_cost`` the packer already attached and the
    block id — never block text, so this carries no corpus content and
    needs no admission decision of its own.
    """
    block_tokens: dict[str, int] = {}
    for item in included:
        bid = str(item.get("id") or item.get("block_id") or "")
        if not bid:
            continue
        block_tokens[bid] = block_tokens.get(bid, 0) + max(0, int(item.get("_token_cost", 0) or 0))
    return PackReceipt(
        fingerprint=str(fingerprint),
        packed_tokens=max(0, int(tokens_used)),
        block_tokens=block_tokens,
    )


__all__ = [
    "ContextWindow",
    "MRREvent",
    "MRRTracker",
    "MRRWeek",
    "PACK_RECEIPT_CAPACITY",
    "PackReceipt",
    "PackReceiptRegistry",
    "PackingQualityMeter",
    "UNKNOWN_MODEL_WINDOW",
    "context_window",
    "default_pack_receipts",
    "default_packing_meter",
    "extract_conventions",
    "is_known_model",
    "iso_week_key",
    "known_models",
    "model_context_window",
    "mrr_events_from_signals",
    "mrr_from_events",
    "pack_receipt_from_included",
    "parse_signal_timestamp",
    "reset_packing_state",
    "resolve_pack_budget",
]

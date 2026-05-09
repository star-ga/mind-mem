"""Deterministic block quality gate (v3.11.0, Pattern 2).

A pre-storage filter that inspects a candidate block and returns a
structured verdict. Eight rules, all deterministic and content-based:

    1. ``empty``           — the block is whitespace-only.
    2. ``too_short``       — fewer than 32 non-whitespace chars.
    3. ``oversize``        — exceeds 64 KiB of UTF-8 bytes.
    4. ``malformed_utf8``  — contains lone surrogates or otherwise
                             cannot round-trip through UTF-8.
    5. ``stopwords_only``  — every token is a stopword (no semantic
                             content).
    6. ``near_duplicate``  — Levenshtein-style similarity to a recent
                             block (within 24h) is >= 0.97.
    7. ``injection_marker``— matches a known prompt-injection pattern.
    8. ``ok``              — no rule fired (the happy path).

Default mode is *advisory*: every rule is logged but the verdict still
``accept``\\ s. Hard-fail is opt-in via ``strict=True`` keyword OR a
workspace ``QualityGateConfig(mode="strict")`` OR the workspace config
``mind-mem.json`` setting ``quality_gate_mode = "strict"``.

The rules are STARGA-native — chosen from first principles to address
the failure modes mind-mem has actually hit in production (empty
proposals from CLI typos, oversize log dumps from agents that pasted
megabytes, near-duplicates from re-runs). The rules are *additive*;
adding a ninth rule means appending to ``_RULES`` here. Existing
callers see no signature change.

The module is dependency-free (stdlib + ``mind_mem._recall_constants``)
so it is safe to import in cold-path code paths like
``propose_update`` without dragging in heavyweight retrieval state.
"""

from __future__ import annotations

import datetime as _dt
import re as _re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from ._recall_constants import _STOPWORDS

__all__ = [
    "QualityGateConfig",
    "QualityGateVerdict",
    "similarity_ratio",
    "validate_block",
]

_MIN_CHARS = 32
_MAX_BYTES = 64 * 1024
_DUP_RATIO = 0.97
_DUP_WINDOW = _dt.timedelta(hours=24)

_INJECTION_MARKERS: tuple[_re.Pattern[str], ...] = (
    _re.compile(r"\bignore\s+(?:(?:all|the|prior|previous|above)\s+)+instructions\b", _re.I),
    _re.compile(r"\bdisregard\s+(?:(?:all|the|prior|previous|above)\s+)+instructions\b", _re.I),
    _re.compile(r"\bsystem\s*:\s*you\s+are\s+now\b", _re.I),
    _re.compile(r"<\|im_start\|>", _re.I),
    _re.compile(r"\[\[\s*INST\s*\]\]", _re.I),
    _re.compile(r"\bjailbreak\b", _re.I),
)


@dataclass(frozen=True)
class QualityGateConfig:
    """Workspace-level quality-gate configuration.

    ``mode`` is the only knob today. ``"advisory"`` (default) means a
    failed rule is recorded under :pyattr:`QualityGateVerdict.advisory`
    but :pyattr:`QualityGateVerdict.accept` stays ``True``. ``"strict"``
    flips a failed rule into a hard reject.
    """

    mode: str = "advisory"

    def __post_init__(self) -> None:
        if self.mode not in ("advisory", "strict"):
            raise ValueError(f"mode must be 'advisory' or 'strict', got {self.mode!r}")


@dataclass(frozen=True)
class QualityGateVerdict:
    """Structured outcome of a quality-gate check."""

    accept: bool
    reasons: list[str] = field(default_factory=list)
    advisory: list[str] = field(default_factory=list)
    checked_rules: list[str] = field(default_factory=list)
    forced: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "accept": self.accept,
            "reasons": list(self.reasons),
            "advisory": list(self.advisory),
            "checked_rules": list(self.checked_rules),
            "forced": self.forced,
        }


def similarity_ratio(a: str, b: str) -> float:
    """Return a normalized similarity ratio in ``[0.0, 1.0]``.

    Uses stdlib :class:`difflib.SequenceMatcher`'s ratio. Identical
    strings (including the both-empty case) return ``1.0``; an empty
    string vs anything non-empty returns ``0.0``.
    """

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_valid_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_stopwords_only(text: str) -> bool:
    tokens = _re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not tokens:
        return False  # empty is handled by the empty rule, not this one
    for tok in tokens:
        if tok not in _STOPWORDS:
            return False
    return True


def _has_injection_marker(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_MARKERS)


def _near_duplicate(
    text: str,
    recent: Iterable[tuple[str, _dt.datetime]],
    *,
    now: _dt.datetime | None = None,
) -> tuple[bool, float | None]:
    if not recent:
        return (False, None)
    cutoff = (now or _dt.datetime.now(_dt.timezone.utc)) - _DUP_WINDOW
    best = 0.0
    for prior_text, prior_ts in recent:
        if prior_ts < cutoff:
            continue
        ratio = similarity_ratio(text, prior_text)
        if ratio > best:
            best = ratio
            if best >= _DUP_RATIO:
                return (True, best)
    return (False, best if best > 0.0 else None)


def validate_block(
    text: str,
    *,
    strict: bool = False,
    force: bool = False,
    config: QualityGateConfig | None = None,
    recent: Sequence[tuple[str, _dt.datetime]] | None = None,
    now: _dt.datetime | None = None,
) -> QualityGateVerdict:
    """Inspect ``text`` against the quality-gate rules.

    Args:
        text: candidate block content.
        strict: if ``True``, every fired rule rejects. Overrides the
            workspace config.
        force: escape hatch — if ``True``, accept regardless of fired
            rules. Caller takes responsibility for the consequences and
            the verdict is annotated ``forced=True``.
        config: workspace config; ``mode="strict"`` is equivalent to
            ``strict=True``.
        recent: iterable of ``(text, timestamp)`` tuples to check the
            near-duplicate rule against. ``timestamp`` must be
            timezone-aware.
        now: override "now" for testing the duplicate window.

    Returns:
        A :class:`QualityGateVerdict` describing the outcome.
    """

    cfg = config or QualityGateConfig()
    is_strict = strict or cfg.mode == "strict"

    reasons: list[str] = []
    advisory: list[str] = []
    checked: list[str] = []

    def _fail(rule: str, message: str) -> None:
        checked.append(rule)
        line = f"{rule}: {message}"
        if is_strict:
            reasons.append(line)
        else:
            advisory.append(line)

    def _pass(rule: str) -> None:
        checked.append(rule)

    # 1. empty
    stripped = text.strip()
    if not stripped:
        _fail("empty", "block is empty or whitespace-only")
    else:
        _pass("empty")

    non_ws_count = sum(1 for c in text if not c.isspace())

    # 2. too_short
    if non_ws_count < _MIN_CHARS:
        _fail(
            "too_short",
            f"block has {non_ws_count} non-whitespace chars; min is {_MIN_CHARS}",
        )
    else:
        _pass("too_short")

    # 3. oversize
    try:
        size_bytes = len(text.encode("utf-8", errors="replace"))
    except UnicodeEncodeError:
        size_bytes = len(text)
    if size_bytes > _MAX_BYTES:
        _fail(
            "oversize",
            f"block is {size_bytes} bytes; max is {_MAX_BYTES}",
        )
    else:
        _pass("oversize")

    # 4. malformed_utf8
    if not _is_valid_utf8(text):
        _fail("malformed_utf8", "block contains lone surrogates / cannot encode as UTF-8")
    else:
        _pass("malformed_utf8")

    # 5. stopwords_only
    if stripped and _is_stopwords_only(stripped):
        _fail("stopwords_only", "block has no content tokens (all stopwords)")
    else:
        _pass("stopwords_only")

    # 6. near_duplicate
    if recent:
        is_dup, ratio = _near_duplicate(text, recent, now=now)
        if is_dup:
            assert ratio is not None
            _fail(
                "near_duplicate",
                f"similar to a recent block (ratio={ratio:.3f} >= {_DUP_RATIO})",
            )
        else:
            _pass("near_duplicate")

    # 7. injection_marker
    if _has_injection_marker(text):
        _fail("injection_marker", "block contains a known prompt-injection marker")
    else:
        _pass("injection_marker")

    if force:
        return QualityGateVerdict(
            accept=True,
            reasons=[],
            advisory=advisory + reasons,
            checked_rules=checked,
            forced=True,
        )

    accept = not reasons
    return QualityGateVerdict(
        accept=accept,
        reasons=reasons,
        advisory=advisory,
        checked_rules=checked,
        forced=False,
    )

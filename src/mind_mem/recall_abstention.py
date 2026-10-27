# Copyright 2026 STARGA, Inc.
"""Computed abstention: "no record found" as a measurement, not a convention.

``recall`` returns ranked candidates and leaves one question to the caller's
discipline: *is any of this a clear match?* A caller that skips the judgement
gets no warning, so the hallucination guardrail the standing rule ("recall
first; if nothing comes back, say so -- never guess") exists to provide is
exactly as strong as the care of whoever wrote the calling code.

This module derives the judgement from a distribution the engine already
computed and used to discard: the scores of the served set.

The statistic
-------------
A single property of the **set**, not a new score on each block::

    separation = (top - runner_up) / top        # in [0, 1], rounded to 4 places

How far the best candidate stands clear of the second best, relative to its own
size. It is scale-free on purpose: a BM25 number (``24.9``), an RRF number
(``0.033``) and a cosine describe the same shape the same way, and the measured
engine really does mix those scales from one query to the next (a one-hit
hybrid answer scores ``0.033`` where a three-hit answer on the same corpus
scores ``16.4``). An absolute floor would therefore be wrong in both directions.

* nothing recalled, or nothing with a positive score  -> ``0.0``
* one candidate -> :data:`UNOPPOSED_SEPARATION`. A lone hit has nothing to be
  separated *from*, so the distribution is silent: it is evidence neither of a
  clear win nor of ambiguity. The prior is deliberately neutral, and the default
  thresholds put it at *caution* for an answer or a governed write -- measured,
  a lone strong hit and a lone weak hit are indistinguishable from scores alone.
* two or more -> the gap above. Only the top two scores enter; a long tail
  below the runner-up cannot move it. An exact tie is ``0.0``: two candidates
  neither of which is a clear winner.

Under RRF fusion the ``score`` column is rank-derived, so the statistic there
measures *agreement structure* (both legs rank one hit first while the
runner-up appears in a single leg) rather than raw similarity. That is a real
signal, but it is a different one, so the verdict records its ``basis`` and the
calibration loop scores each basis separately.

Three bands, scaled by use
--------------------------
``act`` / ``proceed_with_caution`` / ``do_not_act``. The boundary is not one
number: acting without confirmation should demand more separation the more the
answer will be trusted. Uses run least to most strict --
``summary`` < ``answer`` < ``governed_write`` -- and a policy that makes a
stricter use *looser* at either edge is rejected, so a recall feeding a
governed write can never be easier to act on than one feeding a summary.

The shipped thresholds are PROVISIONAL. They were set from a deterministic
labelled fixture (40 present / 40 absent queries over an FTS-indexed corpus:
separation AUC 0.92, absent p90 = 0.30, present median = 0.59), which is
evidence that the statistic carries signal on a ranker that finds the right
block, and nothing more. On the unindexed full-scan fallback the same fixture
did NOT separate (AUC 0.48) because that ranker did not put the right block
first -- the statistic was faithfully reporting that no clear winner had been
found. Real workspaces should measure their own (``mm abstention-eval``) and
let the outcome loop score the predictions
(:mod:`mind_mem.abstention_store`).

What this is not
----------------
No model, no network, no clock, no randomness and no I/O: a pure function of
the score list and the policy, so recall stays reproducible. Not a second
memory subsystem and not a per-block score. It never filters, reorders or
rescores the served set -- it describes it.

Off by default and invisible when off
-------------------------------------
:func:`abstention_requested` is a dictionary lookup on a mapping the caller
already holds. It logs nothing, reads nothing and does not touch a clock; with
the feature off the recall response is byte-identical to the build that never
had this module.

Leaf module: standard library only, no first-party imports.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = [
    "BANDS",
    "BAND_ACT",
    "BAND_CAUTION",
    "BAND_DO_NOT_ACT",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_USE",
    "INTENDED_USES",
    "UNOPPOSED_SEPARATION",
    "USE_ANSWER",
    "USE_GOVERNED_WRITE",
    "USE_SUMMARY",
    "AbstentionPolicy",
    "abstention_requested",
    "assess",
    "resolve_policy",
    "separation_statistic",
    "validate_intended_use",
]

# --- vocabulary --------------------------------------------------------------

#: The top candidate is clearly separated: act without asking.
BAND_ACT = "act"
#: Separated, but not enough for this use: confirm before relying on it.
BAND_CAUTION = "proceed_with_caution"
#: No clear winner (or nothing recalled): do not act on this recall.
BAND_DO_NOT_ACT = "do_not_act"
BANDS: tuple[str, ...] = (BAND_ACT, BAND_CAUTION, BAND_DO_NOT_ACT)

#: Feeding a summary: the reader sees the sources and can discount a miss.
USE_SUMMARY = "summary"
#: Answering a question directly from the recalled blocks.
USE_ANSWER = "answer"
#: Feeding a governed write (a proposal, a decision): the strictest use.
USE_GOVERNED_WRITE = "governed_write"
#: Least strict first. The order is load-bearing: policy validation requires
#: every edge to be non-decreasing along this tuple.
INTENDED_USES: tuple[str, ...] = (USE_SUMMARY, USE_ANSWER, USE_GOVERNED_WRITE)
DEFAULT_USE = USE_ANSWER

#: The statistic for a set of one. See the module docstring.
UNOPPOSED_SEPARATION = 0.5

#: ``use -> (act_min, caution_min)``. Provisional; see the module docstring.
DEFAULT_THRESHOLDS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        USE_SUMMARY: (0.35, 0.10),
        USE_ANSWER: (0.55, 0.20),
        USE_GOVERNED_WRITE: (0.70, 0.35),
    }
)

_BASIS_RRF = "rrf"
_BASIS_UNSPECIFIED = "unspecified"
_PRECISION = 4


# --- the statistic -----------------------------------------------------------


def _score_of(value: Any) -> float | None:
    """A finite, non-negative real from *value*, or ``None`` when it is not one.

    ``bool`` is excluded explicitly (it is an ``int``): ``True`` is a flag, not
    a score of ``1``. Negative scores clamp to ``0.0`` rather than inflating the
    gap, and a string such as ``"9.9"`` is not coerced -- a score column that
    arrives as text is a bug upstream, and counting it would hide the bug.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return max(0.0, number)


def _top_two(scores: Iterable[Any] | None) -> tuple[int, float, float]:
    """``(usable_count, top, runner_up)`` in one linear pass.

    Selection rather than a sort: recall sets are small, but this is also the
    function a caller can hand 50k scores, and the top two are all it needs.
    """
    count = 0
    top = 0.0
    second = 0.0
    if scores is None:
        return 0, top, second
    for raw in scores:
        score = _score_of(raw)
        if score is None:
            continue
        count += 1
        if score > top:
            top, second = score, top
        elif score > second:
            second = score
    return count, top, second


def _statistic(count: int, top: float, second: float) -> float:
    if count == 0 or top <= 0.0:
        return 0.0
    if count == 1:
        return UNOPPOSED_SEPARATION
    gap = (top - second) / top
    return round(max(0.0, min(1.0, gap)), _PRECISION)


def separation_statistic(scores: Iterable[Any] | None) -> float:
    """How separated the top score is from the runner-up, in ``[0, 1]``.

    Pure and order-independent. Anything that is not a finite real number is
    not a candidate and is skipped. See the module docstring for the cases.
    """
    return _statistic(*_top_two(scores))


# --- intended use ------------------------------------------------------------


def validate_intended_use(value: Any) -> str | None:
    """The canonical intended-use name, ``None`` when none was given.

    Raises :class:`ValueError` (naming the valid uses) for anything else, so a
    typo at an API boundary fails loudly instead of silently selecting the
    default band set.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if cleaned in INTENDED_USES:
            return cleaned
    # Echo a bounded excerpt: the value is caller-supplied and the message is
    # returned to the caller and logged.
    shown = value if isinstance(value, str) else type(value).__name__
    raise ValueError(f"intended_use must be one of {list(INTENDED_USES)}, got {shown[:40]!r}")


# --- policy ------------------------------------------------------------------


@dataclass(frozen=True)
class AbstentionPolicy:
    """The resolved ``recall.abstention`` configuration. Immutable."""

    enabled: bool
    default_use: str
    thresholds: Mapping[str, tuple[float, float]]
    warnings: tuple[str, ...] = field(default=())


def _is_real_fraction(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _merged_thresholds(bands: Mapping[Any, Any], warnings: list[str]) -> dict[str, tuple[float, float]] | None:
    """Defaults overlaid with *bands*, or ``None`` when anything is invalid."""
    merged: dict[str, tuple[float, float]] = dict(DEFAULT_THRESHOLDS)
    ok = True
    for use, edges in bands.items():
        if use not in INTENDED_USES:
            warnings.append(f"abstention.bands: unknown intended use {use!r} ignored (valid: {list(INTENDED_USES)})")
            continue
        if not isinstance(edges, Mapping):
            warnings.append(f"abstention.bands.{use} must be an object with 'act' and/or 'caution'")
            ok = False
            continue
        act_min, caution_min = merged[use]
        for name in ("act", "caution"):
            if name not in edges:
                continue
            if not _is_real_fraction(edges[name]):
                warnings.append(f"abstention.bands.{use}.{name} must be a number in [0, 1], got {edges[name]!r}")
                ok = False
            elif name == "act":
                act_min = float(edges[name])
            else:
                caution_min = float(edges[name])
        merged[use] = (act_min, caution_min)
    if not ok:
        return None
    previous = (0.0, 0.0)
    for use in INTENDED_USES:
        act_min, caution_min = merged[use]
        if caution_min > act_min:
            warnings.append(f"abstention.bands.{use}: caution ({caution_min}) must not exceed act ({act_min})")
            return None
        if act_min < previous[0] or caution_min < previous[1]:
            warnings.append(
                f"abstention.bands.{use}: a stricter use may not be looser than a less strict one "
                f"(order: {list(INTENDED_USES)}); a governed write must never be easier to act on than a summary"
            )
            return None
        previous = (act_min, caution_min)
    return merged


def _section_enabled(section: Any) -> bool:
    """The one definition of "opted in": a mapping carrying a literal ``enabled: true``."""
    return isinstance(section, Mapping) and section.get("enabled") is True


def resolve_policy(recall_cfg: Mapping[str, Any] | None) -> AbstentionPolicy:
    """Resolve ``recall.abstention`` from the (already loaded) recall config.

    Total and silent: a malformed section yields the shipped defaults plus
    ``warnings`` the caller may surface -- it never raises and never logs, so
    a bad config cannot fail a recall. Any invalid threshold discards the whole
    threshold table (a half-applied policy is the one no operator reasoned
    about).
    """
    warnings: list[str] = []
    section: Any = recall_cfg.get("abstention") if isinstance(recall_cfg, Mapping) else None
    if section is None:
        return AbstentionPolicy(False, DEFAULT_USE, DEFAULT_THRESHOLDS, ())
    if not isinstance(section, Mapping):
        warnings.append("recall.abstention must be an object; using defaults")
        return AbstentionPolicy(False, DEFAULT_USE, DEFAULT_THRESHOLDS, tuple(warnings))

    default_use = DEFAULT_USE
    raw_default = section.get("default_use")
    if raw_default is not None:
        try:
            default_use = validate_intended_use(raw_default) or DEFAULT_USE
        except ValueError as exc:
            warnings.append(f"abstention.default_use: {exc}")

    thresholds: Mapping[str, tuple[float, float]] = DEFAULT_THRESHOLDS
    bands = section.get("bands")
    if bands is not None:
        if not isinstance(bands, Mapping):
            warnings.append("abstention.bands must be an object keyed by intended use; using defaults")
        else:
            merged = _merged_thresholds(bands, warnings)
            if merged is not None:
                thresholds = MappingProxyType(merged)
            else:
                warnings.append("abstention.bands rejected; using the shipped defaults for every use")
    return AbstentionPolicy(_section_enabled(section), default_use, thresholds, tuple(warnings))


#: The policy for a caller that supplies none. Resolved once at import.
_DEFAULT_POLICY = resolve_policy(None)


def abstention_requested(recall_cfg: Any, intended_use: Any) -> bool:
    """Should this recall carry an abstention verdict?

    ``True`` when the caller named an intended use, or when the workspace
    opted in with a literal ``recall.abstention.enabled: true``. Only a literal
    ``true`` enables -- ``"true"``, ``1`` and ``"yes"`` are not opt-ins.

    This is the OFF-path probe, so it is deliberately inert: dictionary
    lookups on a mapping the caller already holds. No logging, no I/O, no
    clock, and it never raises on a malformed config (that reads as off).
    A non-string, non-``None`` intended use counts as a request so that
    :func:`validate_intended_use` can refuse it, rather than the typo being
    swallowed as "not asked".
    """
    if intended_use is not None and not (isinstance(intended_use, str) and not intended_use.strip()):
        return True
    if not isinstance(recall_cfg, Mapping):
        return False
    return _section_enabled(recall_cfg.get("abstention"))


# --- the verdict -------------------------------------------------------------


def _evidence_scores(results: Iterable[Any] | None) -> tuple[list[float], bool]:
    """Scores of the evidence hits, and whether all of them are RRF-fused.

    Force-surfaced guardrails are constraints, not evidence: they are
    prepended whatever their score, so they say nothing about whether the
    question was answered.
    """
    scores: list[float] = []
    all_rrf = True
    for hit in results or ():
        if not isinstance(hit, Mapping) or hit.get("guardrail"):
            continue
        score = _score_of(hit.get("score"))
        if score is None:
            continue
        scores.append(score)
        if hit.get("fusion") != _BASIS_RRF:
            all_rrf = False
    return scores, bool(scores) and all_rrf


def _band_for(separation: float, usable: bool, edges: tuple[float, float]) -> str:
    # ``usable`` is an explicit guard, not an accident of the thresholds: a
    # policy may legally set caution to 0.0, and a set with no positive
    # evidence must still never read as anything but "do not act".
    if not usable:
        return BAND_DO_NOT_ACT
    act_min, caution_min = edges
    if separation >= act_min:
        return BAND_ACT
    if separation >= caution_min:
        return BAND_CAUTION
    return BAND_DO_NOT_ACT


def _reason(count: int, top: float, band: str) -> str:
    if count == 0:
        return "no_candidates"
    if top <= 0.0:
        return "no_positive_score"
    if count == 1:
        return "single_candidate"
    if band == BAND_ACT:
        return "top_candidate_clearly_separated"
    if band == BAND_CAUTION:
        return "top_candidate_weakly_separated"
    return "no_clear_winner"


def assess(
    results: Iterable[Any] | None,
    *,
    intended_use: str | None = None,
    policy: AbstentionPolicy | None = None,
    basis: str | None = None,
) -> dict[str, Any]:
    """The abstention verdict for a served set. Pure; never mutates *results*.

    Returns the first-class response field::

        {"abstain": bool,            # True exactly when band == "do_not_act"
         "band": "act" | "proceed_with_caution" | "do_not_act",
         "separation": 0.8395,       # the set-level statistic
         "candidates": 5,            # evidence hits that carried a usable score
         "intended_use": "answer",
         "thresholds": {"act": 0.55, "caution": 0.2},
         "reason": "...",
         "basis": "sqlite"}          # where the scores came from, for calibration

    ``basis`` is ``"rrf"`` whenever every evidence hit is RRF-fused (the score
    is then rank-derived, whatever backend label the caller passes), else the
    caller's *basis*, else ``"unspecified"``.

    Raises :class:`ValueError` for an unknown *intended_use*; callers validate
    at their boundary first so this is never reached with a typo.
    """
    effective = policy if policy is not None else _DEFAULT_POLICY
    use = validate_intended_use(intended_use) or effective.default_use
    scores, all_rrf = _evidence_scores(results)
    count, top, second = _top_two(scores)
    separation = _statistic(count, top, second)
    edges = effective.thresholds[use]
    band = _band_for(separation, count > 0 and top > 0.0, edges)
    return {
        "abstain": band == BAND_DO_NOT_ACT,
        "band": band,
        "separation": separation,
        "candidates": count,
        "intended_use": use,
        "thresholds": {"act": edges[0], "caution": edges[1]},
        "reason": _reason(count, top, band),
        "basis": _BASIS_RRF if all_rrf else (basis or _BASIS_UNSPECIFIED),
    }

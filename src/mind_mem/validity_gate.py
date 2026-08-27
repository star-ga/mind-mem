"""Phase-2 recall validity gate — flag-gated, deterministic demotion.

Every hit reaching Stage 2.65 of the recall pipeline is scored against four
independent criteria — corroboration, status, contradiction cleanliness, and
lineage freshness — each in ``[0.0, 1.0]``. Governing rule: *absence of a
signal is neutral (1.0); only affirmative evidence of invalidity debits.*

The composite ``V`` is always attached to the hit as a diagnostics
annotation (``hit["validity"]``), mirroring the ``fusion_sources``
provenance pattern. When ``V`` falls below the configured threshold the hit
is **demoted, never dropped**: its score is scaled down and it is flagged
``_validity_demoted`` so downstream stages (knee cutoff, re-sort) can act on
it while it stays visible with full provenance.

Deterministic by construction: both DB reads this module performs
(contradiction log, staleness scores) are unwindowed stored state — no
``datetime.now()``, no clock, no randomness anywhere on the scored path.

Flag-gated via ``mind-mem.json`` -> ``recall.validity_gate.enabled``
(default ``False``). When disabled, :func:`apply_validity_gate` is a
complete no-op: no annotation, no DB reads, byte-identical output to the
pre-gate pipeline.

A fifth signal — **outcome attribution** (did acting on this block
actually work?) — is available behind its own default-OFF sub-flag,
``recall.validity_gate.outcome_attribution.enabled``. With that sub-flag
off the four-criteria output above is byte-identical and no outcome DB
read happens. With it on, a block repeatedly implicated in failed
outcomes is demoted and a block corroborated by successful outcomes has
its corroboration component confirmed. See
:mod:`mind_mem.outcome_attribution`.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ._recall_constants import (
    VALIDITY_DEMOTION,
    VALIDITY_GATE_THRESHOLD,
    VALIDITY_STATUS_DEAD,
    VALIDITY_STATUS_WIP,
)
from .block_parser import parse_file
from .lineage_staleness import list_staleness_scores
from .observability import get_logger
from .outcome_attribution import OutcomeSignal

__all__ = ["apply_validity_gate", "validity_components"]

_log = get_logger("validity_gate")

# Party-id extraction from CONTRADICTIONS.md entries — same shape as the
# block-id vocabulary used elsewhere (conflict_resolver, _BLOCK_ID_RE):
# one-or-more uppercase letters, an 8-digit date, a 3-digit sequence.
_PARTY_ID_RE = re.compile(r"[A-Z]+-\d{8}-\d{3}")


def apply_validity_gate(hits: list[dict[str, Any]], workspace: str, cfg: dict[str, Any]) -> None:
    """Annotate every hit with a ``validity`` diagnostic; demote low scorers.

    Mutates ``hits`` in place, matching the Stage 2.6 hard-negative idiom.
    A no-op (no annotation, no DB reads) unless
    ``cfg["validity_gate"]["enabled"]`` is truthy.

    Args:
        hits: Ranked hit dicts from the recall pipeline (mutated in place).
        workspace: Workspace root path.
        cfg: The ``recall`` config section of ``mind-mem.json`` (i.e.
            ``config.get("recall", {})``), not the full config file.
    """
    vg_cfg = cfg.get("validity_gate")
    if not isinstance(vg_cfg, dict) or not vg_cfg.get("enabled", False):
        return
    if not hits:
        return

    threshold = _unit_fraction(vg_cfg.get("threshold"), VALIDITY_GATE_THRESHOLD)
    demotion = _unit_fraction(vg_cfg.get("demotion"), VALIDITY_DEMOTION)

    contradicted_ids = _load_contradicted_party_ids(workspace)
    block_ids = [hit.get("_id", "") for hit in hits if hit.get("_id")]
    staleness = list_staleness_scores(workspace, block_ids)
    outcomes = _load_outcome_signals(vg_cfg, workspace, block_ids)

    for hit in hits:
        components = validity_components(hit, contradicted_ids, staleness, outcomes)
        hit["validity"] = components
        if components["score"] < threshold:
            hit["score"] = round(hit["score"] * demotion, 4)
            hit["_validity_demoted"] = True

    _log.debug(
        "validity_gate_applied",
        hits=len(hits),
        demoted=sum(1 for h in hits if h.get("_validity_demoted")),
    )


def validity_components(
    hit: dict[str, Any],
    contradicted_ids: set[str],
    staleness: dict[str, float],
    outcome_signals: dict[str, OutcomeSignal] | None = None,
) -> dict[str, float]:
    """Pure four-criteria validity math — the ONE source of truth shared by
    :func:`apply_validity_gate` (Stage 2.65) and
    :func:`mind_mem.retrieval_graph.feedback_quality_credit` (Stage 3.1).

    No I/O, no clock, no randomness — reads only the hit and the
    pre-fetched stored-state maps.

    ``outcome_signals`` is the opt-in fifth signal (utility, from
    :mod:`mind_mem.outcome_attribution`). ``None`` — the default, and what
    every pre-outcome caller passes — returns the exact four-key dict this
    function has always returned. When supplied:

    * a block corroborated by successful outcomes has its corroboration
      component lifted to ``1.0`` (real-world confirmation counts as a
      second source);
    * a block repeatedly implicated in failed outcomes multiplies the
      four-criteria mean by its utility factor (``[0.5, 1.0]``), which is
      what pushes it under the demotion threshold.
    """
    block_id = hit.get("_id", "")
    c1 = _corroboration_component(hit)
    c2 = _status_component(hit)
    c3 = 0.0 if block_id in contradicted_ids else 1.0
    c4 = round(1.0 - min(1.0, staleness.get(block_id, 0.0)), 4)

    if outcome_signals is None:
        return {
            "corroboration": c1,
            "status": c2,
            "contradiction": c3,
            "staleness": c4,
            "score": round(0.25 * (c1 + c2 + c3 + c4), 4),
        }

    signal = outcome_signals.get(block_id)
    if signal is None:
        factor = 1.0
    else:
        factor = signal.factor
        if signal.corroborated:
            c1 = 1.0
    return {
        "corroboration": c1,
        "status": c2,
        "contradiction": c3,
        "staleness": c4,
        "outcome": factor,
        "score": round(0.25 * (c1 + c2 + c3 + c4) * factor, 4),
    }


def _load_outcome_signals(
    vg_cfg: dict[str, Any],
    workspace: str,
    block_ids: list[str],
) -> dict[str, OutcomeSignal] | None:
    """Fetch utility evidence iff ``validity_gate.outcome_attribution`` is on.

    Returns ``None`` — the byte-identical four-criteria path — whenever the
    sub-flag is absent or false, so no DB read happens either.
    """
    # deferred: Stage 3.1 (`retrieval_graph.feedback_quality_credit`) still
    # calls validity_components/3, so its `valid` credit ignores utility —
    # deliberate, it keeps that stage byte-identical. Upgrade path: thread the
    # same pre-fetched signal map through feedback_quality_credit behind this
    # same sub-flag.
    oa_cfg = vg_cfg.get("outcome_attribution")
    if not isinstance(oa_cfg, dict) or not oa_cfg.get("enabled", False):
        return None

    from .outcome_attribution import load_outcome_signals

    return load_outcome_signals(workspace, block_ids)


def _unit_fraction(value: Any, default: float) -> float:
    """Return ``value`` if it is a real number in the half-open ``(0, 1]``
    range, else ``default``. Never raises — invalid overrides are ignored."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if 0 < value <= 1:
        return float(value)
    return default


def _corroboration_component(hit: dict[str, Any]) -> float:
    """c1 — corroboration across fusion arms (Phase-1 ``fusion_sources``).

    Absent (non-hybrid recall path) -> neutral 1.0. >=2 distinct arms ->
    1.0. Exactly one arm -> 0.5 (a single-source hit is unconfirmed).
    """
    sources = hit.get("fusion_sources")
    if not sources:
        return 1.0
    return 1.0 if len(sources) >= 2 else 0.5


def _status_component(hit: dict[str, Any]) -> float:
    """c2 — status validity.

    Missing/unknown status is neutral (1.0): a *gate* must not punish
    absent metadata, unlike ``block_maturity._status_component``.
    """
    status = str(hit.get("status") or hit.get("Status") or "").strip().lower()
    if status in VALIDITY_STATUS_WIP:
        return 0.5
    if status in VALIDITY_STATUS_DEAD:
        return 0.0
    return 1.0


def _load_contradicted_party_ids(workspace: str) -> set[str]:
    """c3 support — one batch read of the contradiction log per recall call.

    Returns the union of every block id that appears as a party to any
    entry in ``intelligence/CONTRADICTIONS.md``. File absent (or unreadable)
    -> empty set -> c3 is neutral (1.0) for every hit.

    Deliberately lighter than ``conflict_resolver.resolve_contradictions``:
    this only extracts party ids, it does not run the heavy
    ``analyze_contradiction`` resolution pass.
    """
    path = os.path.join(workspace, "intelligence", "CONTRADICTIONS.md")
    if not os.path.isfile(path):
        return set()

    try:
        blocks = parse_file(path)
    except OSError as exc:
        _log.warning("validity_gate_contradiction_log_unreadable", error=str(exc))
        return set()

    party_ids: set[str] = set()
    for block in blocks:
        for key, value in block.items():
            if key.startswith("_") or not isinstance(value, str):
                continue
            party_ids.update(_PARTY_ID_RE.findall(value))
    return party_ids

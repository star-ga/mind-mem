"""Phase-2 recall validity gate — flag-gated, deterministic demotion.

Every hit reaching Stage 2.65 of the recall pipeline is scored against
independent criteria — corroboration, status, contradiction cleanliness,
lineage freshness, and (opt-in) provenance class — each in ``[0.0, 1.0]``,
with an (opt-in) outcome-attribution factor applied on top of the mean.
Governing rule: *absence of a signal is neutral (1.0); only affirmative
evidence of invalidity debits.*

The fifth criterion, **provenance class**, is the one composite home for
"how much do we trust where this came from": ``operator`` >
``agent-verified`` > ``agent-inferred`` > ``external-ingest``
(:mod:`provenance_class`). It is a per-block, table-driven classification of
existing provenance fields — deliberately **not** a per-actor learned or
anomaly score, which would make the same corpus rank differently on two
machines. Any standalone trust surface delegates here; there is one
composite path, not two.

The composite ``V`` is always attached to the hit as a diagnostics
annotation (``hit["validity"]``), mirroring the ``fusion_sources``
provenance pattern. When ``V`` falls below the configured threshold the hit
is **demoted, never dropped**: its score is scaled down and it is flagged
``_validity_demoted`` so downstream stages (knee cutoff, re-sort) can act on
it while it stays visible with full provenance.

Deterministic by construction: every DB read this module performs
(contradiction log, staleness scores, outcome counts) is unwindowed stored
state — no ``datetime.now()``, no clock, no randomness anywhere on the
scored path.

Flag-gated via ``mind-mem.json`` -> ``recall.validity_gate.enabled``
(default ``False``). When disabled, :func:`apply_validity_gate` is a
complete no-op: no annotation, no DB reads, byte-identical output to the
pre-gate pipeline.

Two opt-in extensions hang off the gate. They are **independent**: each
carries its own default-``False`` sub-flag, either may be enabled alone,
and enabling both composes without either changing the other's meaning.

* **provenance class** — ``recall.validity_gate.provenance_class.enabled``
  (a bare ``true`` is also accepted). Folds the fifth *criterion* into the
  composite, which becomes a five-way mean.
* **outcome attribution** — did acting on this block actually work?
  ``recall.validity_gate.outcome_attribution.enabled``. Applies a *factor*
  to whichever mean is in force: a block repeatedly implicated in failed
  outcomes is demoted, and a block corroborated by successful outcomes has
  its corroboration component confirmed. See
  :mod:`mind_mem.outcome_attribution`.

With both sub-flags off the composite is the original four-criteria mean —
the returned key set and every float byte-identical to the pre-extension
gate — and neither extension's DB read happens::

    {"recall": {"validity_gate": {"enabled": true,
                                  "provenance_class": {"enabled": true},
                                  "outcome_attribution": {"enabled": true}}}}
"""

from __future__ import annotations

import os
import re
from datetime import date
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
from .provenance_class import classify_provenance, confirmed_block_ids, provenance_component

__all__ = [
    "apply_validity_gate",
    "confirmed_block_ids",
    "provenance_component",
    "validity_components",
]

_log = get_logger("validity_gate")

# Party-id extraction from CONTRADICTIONS.md entries — same shape as the
# block-id vocabulary used elsewhere (conflict_resolver, _BLOCK_ID_RE):
# one-or-more uppercase letters, an 8-digit date, a 3-digit sequence.
_PARTY_ID_RE = re.compile(r"[A-Z]+-\d{8}-\d{3}")


def apply_validity_gate(
    hits: list[dict[str, Any]],
    workspace: str,
    cfg: dict[str, Any],
    *,
    scoring_instant: date | None = None,
) -> None:
    """Annotate every hit with a ``validity`` diagnostic; demote low scorers.

    Mutates ``hits`` in place, matching the Stage 2.6 hard-negative idiom.
    A no-op (no annotation, no DB reads) unless
    ``cfg["validity_gate"]["enabled"]`` is truthy.

    Args:
        hits: Ranked hit dicts from the recall pipeline (mutated in place).
        workspace: Workspace root path.
        cfg: The ``recall`` config section of ``mind-mem.json`` (i.e.
            ``config.get("recall", {})``), not the full config file.
        scoring_instant: UTC date the one clock-sensitive input here — the
            rolling calibration window behind the ``provenance_class``
            component — is evaluated at. Everything else the gate reads
            (contradictions, staleness, unwindowed outcomes) is clock-free,
            so with this injected the whole gate is. ``None`` resolves to
            today in UTC.
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

    provenance_enabled = _provenance_enabled(vg_cfg)
    confirmed_ids = _load_confirmed_ids(workspace, block_ids, scoring_instant=scoring_instant) if provenance_enabled else frozenset()

    for hit in hits:
        components = validity_components(
            hit,
            contradicted_ids,
            staleness,
            outcomes,
            provenance_enabled=provenance_enabled,
            confirmed_ids=confirmed_ids,
        )
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
    *,
    provenance_enabled: bool = False,
    confirmed_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Pure validity math — the ONE source of truth shared by
    :func:`apply_validity_gate` (Stage 2.65),
    :func:`mind_mem.retrieval_graph.feedback_quality_credit` (Stage 3.1) and
    the standalone ``trust_scores`` surface.

    No I/O, no clock, no randomness — reads only the hit and the
    pre-fetched stored-state maps.

    Two independent opt-in extensions layer onto the four base criteria, and
    neither is aware of the other:

    * *provenance class* adds a fifth **criterion** — the mean widens from
      four terms to five.
    * *outcome attribution* applies a **factor** to whichever mean is in
      force, and may confirm the corroboration component.

    Args:
        hit: The recall hit / block dict being scored.
        contradicted_ids: Party ids of open contradictions (c3).
        staleness: ``block_id -> staleness`` in ``[0, 1]`` (c4).
        outcome_signals: Pre-fetched ``block_id -> OutcomeSignal`` utility
            evidence (:mod:`mind_mem.outcome_attribution`). ``None`` — the
            default, and what every pre-outcome caller passes — leaves the
            composite untouched: no ``outcome`` key, no factor. When
            supplied, a block corroborated by successful outcomes has its
            corroboration component lifted to ``1.0`` (real-world
            confirmation counts as a second source), and a block repeatedly
            implicated in failed outcomes multiplies the mean by its utility
            factor (``[0.5, 1.0]``), which is what pushes it under the
            demotion threshold.
        provenance_enabled: Fold the fifth criterion (provenance class) into
            the composite. Default ``False``.
        confirmed_ids: Human-confirmed block ids, used only when
            *provenance_enabled* (see :func:`confirmed_block_ids`).

    Returns:
        ``{corroboration, status, contradiction, staleness, score}``, plus
        ``{provenance, provenance_class}`` when *provenance_enabled*, plus
        ``{outcome}`` when *outcome_signals* is not ``None``. With both
        extensions off the key set and every float are byte-identical to the
        original four-criteria gate.
    """
    block_id = hit.get("_id", "")
    c1 = _corroboration_component(hit)
    c2 = _status_component(hit)
    c3 = 0.0 if block_id in contradicted_ids else 1.0
    c4 = round(1.0 - min(1.0, staleness.get(block_id, 0.0)), 4)

    factor = 1.0
    if outcome_signals is not None:
        signal = outcome_signals.get(block_id)
        if signal is not None:
            factor = signal.factor
            if signal.corroborated:
                c1 = 1.0

    components: dict[str, Any] = {
        "corroboration": c1,
        "status": c2,
        "contradiction": c3,
        "staleness": c4,
    }
    if provenance_enabled:
        c5 = provenance_component(hit, confirmed_ids)
        components["provenance"] = c5
        components["provenance_class"] = classify_provenance(hit, confirmed_ids=confirmed_ids)
        composite = 0.2 * (c1 + c2 + c3 + c4 + c5)
    else:
        composite = 0.25 * (c1 + c2 + c3 + c4)
    if outcome_signals is not None:
        components["outcome"] = factor
        composite = composite * factor
    components["score"] = round(composite, 4)
    return components


def _load_outcome_signals(
    vg_cfg: dict[str, Any],
    workspace: str,
    block_ids: list[str],
) -> dict[str, OutcomeSignal] | None:
    """Fetch utility evidence iff ``validity_gate.outcome_attribution`` is on.

    Returns ``None`` — the composite-untouched path — whenever the sub-flag
    is absent or false, so no DB read happens either. Independent of the
    ``provenance_class`` sub-flag.
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


def _provenance_enabled(vg_cfg: dict[str, Any]) -> bool:
    """Read ``validity_gate.provenance_class`` defensively (default ``False``).

    Accepts either the nested ``{"enabled": true}`` form or a bare ``true``;
    anything else is off.
    """
    section = vg_cfg.get("provenance_class")
    if isinstance(section, dict):
        return bool(section.get("enabled", False))
    return section is True


def _load_confirmed_ids(
    workspace: str,
    block_ids: list[str],
    *,
    scoring_instant: date | None = None,
) -> frozenset[str]:
    """One batch read of recorded human calibration weights per recall call.

    Non-creating and failure-tolerant (see :mod:`trust_signals`): no index ->
    empty set -> no block is promoted, which only ever costs a promotion,
    never causes a demotion.
    """
    from .trust_signals import load_calibration_weights

    return confirmed_block_ids(load_calibration_weights(workspace, block_ids, scoring_instant=scoring_instant))


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

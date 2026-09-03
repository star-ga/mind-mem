"""Workspace signal loaders for the validity gate's provenance component.

The scoring math in :mod:`provenance_class` is pure. This module is the only
place that touches disk on its behalf: it reads the calibration weights
already recorded by :mod:`calibration` and the withdrawal history already
recorded by :mod:`evidence_objects`, and hands them back as plain maps.

Only :func:`load_calibration_weights` feeds a score — it is a *per-block*,
human-sourced confirmation signal
(:func:`provenance_class.confirmed_block_ids`).
:func:`load_rollback_history` is aggregated *per actor*, which is exactly the
learned-reputation shape the determinism wedge forbids on a scoring path, so
it is kept as a read-only diagnostic and is wired into nothing.

Both loaders are **read-only and non-creating**: if the SQLite index or
the evidence chain file does not exist, they return empty maps rather
than initialising a store — recall must never write on a read path. Both
readers they delegate to create their own directory on construction, so
the existence probe has to come first; that ordering is the whole
mechanism and is covered by a test that asserts the directory is still
absent afterwards. Any other failure degrades to "no signal" (which
scores as neutral) and logs.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import os
from datetime import date

from .observability import get_logger
from .scoring_instant import as_utc_datetime, resolve_scoring_instant

_log = get_logger("trust_signals")

#: Field-level sidecar written by ``audit_chain.AuditChain``. Retained as
#: a published path constant; it is no longer where rollbacks are read
#: from -- see :func:`load_rollback_history`.
AUDIT_CHAIN_REL = os.path.join(".mind-mem-audit", "chain.jsonl")
#: The evidence chain: the ledger that actually records a withdrawal.
EVIDENCE_CHAIN_REL = os.path.join("memory", "evidence_chain.jsonl")
#: SQLite index that also holds the ``calibration_feedback`` table.
INDEX_DB_REL = os.path.join(".mind-mem-index", "recall.db")

#: Audit-sidecar operation this module used to count. No door has ever
#: written it there, so the count it produced was zero by construction,
#: not by absence of rollbacks. Kept as a published constant (it is in
#: ``__all__``) and as the name of the bug: see
#: :data:`audit_chain.RETIRED_OPERATIONS`.
ROLLBACK_OPERATION = "rollback"

#: Evidence action a withdrawal is recorded under. A governed *delete*
#: and a proposal *rollback* share it deliberately --
#: ``governance_gate.DELETE_VERB`` maps to ``EvidenceAction.ROLLBACK``
#: so that governing deletes added no enum member and no older reader
#: broke -- which is why one read here covers both verbs.
ROLLBACK_ACTION = "ROLLBACK"

#: ``metadata`` key distinguishing the two records a delete scope mints.
DELETE_PHASE_KEY = "delete_phase"
#: The delete phase that means content was actually removed. A scope
#: writes ``admitted`` on open and ``removed`` on close, so counting both
#: would charge every delete twice.
DELETE_PHASE_COUNTED = "removed"


def load_calibration_weights(
    workspace: str,
    block_ids: list[str],
    *,
    scoring_instant: date | None = None,
) -> dict[str, float]:
    """Return ``block_id -> calibration weight``, or ``{}`` when unavailable.

    Args:
        workspace: Workspace root. Falsy input yields ``{}``.
        block_ids: Blocks to look up. Empty input yields ``{}``.
        scoring_instant: UTC date opening the rolling calibration window.
            These weights are read by the *validity gate*, so a bare clock
            here would put a wall-clock read on the path the wedge calls
            deterministic. ``None`` resolves to today in UTC.

    Returns:
        The weights :meth:`calibration.CalibrationManager.get_block_weights`
        reports, or ``{}`` when the index database does not exist yet. A
        missing weight only ever costs a class promotion, never causes a
        demotion.
    """
    if not workspace or not block_ids:
        return {}
    if not os.path.isfile(os.path.join(os.path.abspath(workspace), INDEX_DB_REL)):
        return {}
    try:
        from .calibration import CalibrationManager

        return CalibrationManager(workspace).get_block_weights(
            block_ids,
            now=as_utc_datetime(resolve_scoring_instant(scoring_instant)),
        )
    except Exception as exc:  # pragma: no cover — defensive, degrade to neutral
        _log.warning("trust_calibration_unavailable", error=str(exc))
        return {}


def load_rollback_history(workspace: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return ``(rollbacks, total_actions)`` per actor from the evidence chain.

    **Read from the evidence chain, not the field-audit sidecar.** Until
    5.0.2 this counted sidecar rows whose ``operation`` was
    :data:`ROLLBACK_OPERATION` — a verb no door has ever written there,
    so the first element was ``{}`` on every workspace in existence and
    the emptiness read as "nobody has been rolled back" rather than as
    "this function is looking in the wrong ledger". Withdrawals are
    recorded by the gate under :data:`ROLLBACK_ACTION`, which covers a
    proposal rollback (``apply_engine.rollback``) and a governed delete
    alike.

    A delete scope mints two records — ``admitted`` when it opens and
    ``removed`` when it closes — so only :data:`DELETE_PHASE_COUNTED` is
    charged; a record with no ``delete_phase`` at all is a rollback
    proper and is counted. Charging both phases would report every
    delete twice.

    **Diagnostic only — never scored.** Per-actor history is a learned,
    corpus-slice-dependent quantity; folding it into recall ranking would
    make the same hit rank differently on two machines. Exposed for
    operators reading the ledger, not for the gate.

    Reads only: the chain is probed with :func:`os.path.isfile` before
    :class:`~mind_mem.evidence_objects.EvidenceChain` is constructed,
    because that constructor creates the directory it is pointed at — so
    asking "were there rollbacks?" must not be what creates ``memory/``.

    Entries with a blank ``actor`` are skipped — an unattributed action
    cannot be charged to anyone.

    Args:
        workspace: Workspace root. Falsy input yields ``({}, {})``.

    Returns:
        Two maps keyed by the evidence chain's ``actor`` field:
        withdrawals, and total recorded actions. Empty when no evidence
        chain exists.
    """
    if not workspace:
        return {}, {}
    path = os.path.join(os.path.abspath(workspace), EVIDENCE_CHAIN_REL)
    if not os.path.isfile(path):
        return {}, {}
    rollbacks: dict[str, int] = {}
    writes: dict[str, int] = {}
    try:
        from .evidence_objects import EvidenceChain

        chain = EvidenceChain(store_path=path)
        for entry in chain.get_latest(len(chain)):
            actor = str(getattr(entry, "actor", "") or "").strip()
            if not actor:
                continue
            writes[actor] = writes.get(actor, 0) + 1
            action = getattr(getattr(entry, "action", None), "value", "")
            if action != ROLLBACK_ACTION:
                continue
            metadata = getattr(entry, "metadata", None) or {}
            phase = metadata.get(DELETE_PHASE_KEY) if isinstance(metadata, dict) else None
            if phase is not None and phase != DELETE_PHASE_COUNTED:
                continue
            rollbacks[actor] = rollbacks.get(actor, 0) + 1
    except Exception as exc:  # pragma: no cover — defensive, degrade to neutral
        _log.warning("trust_rollback_history_unavailable", error=str(exc))
        return {}, {}
    return rollbacks, writes


# deferred: the audit chain records a file-level ``target``, so a rollback
# cannot be attributed to the block it reverted — that is why rollback history
# is per actor (and therefore unscorable) rather than per block. Upgrade path:
# record the block id on rollback entries, then feed a per-block rolled-back
# set into ``provenance_class.classify_provenance`` as negative evidence,
# which stays deterministic because it is per block, not per actor.

__all__ = [
    "AUDIT_CHAIN_REL",
    "DELETE_PHASE_COUNTED",
    "DELETE_PHASE_KEY",
    "EVIDENCE_CHAIN_REL",
    "INDEX_DB_REL",
    "ROLLBACK_ACTION",
    "ROLLBACK_OPERATION",
    "load_calibration_weights",
    "load_rollback_history",
]

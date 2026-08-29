"""Workspace signal loaders for the validity gate's provenance component.

The scoring math in :mod:`provenance_class` is pure. This module is the only
place that touches disk on its behalf: it reads the calibration weights
already recorded by :mod:`calibration` and the rollback history already
recorded by :mod:`audit_chain`, and hands them back as plain maps.

Only :func:`load_calibration_weights` feeds a score — it is a *per-block*,
human-sourced confirmation signal
(:func:`provenance_class.confirmed_block_ids`).
:func:`load_rollback_history` is aggregated *per actor*, which is exactly the
learned-reputation shape the determinism wedge forbids on a scoring path, so
it is kept as a read-only diagnostic and is wired into nothing.

Both loaders are **read-only and non-creating**: if the SQLite index or
the audit chain file does not exist, they return empty maps rather than
initialising a store — recall must never write on a read path. Any other
failure degrades to "no signal" (which scores as neutral) and logs.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import os
from datetime import date

from .observability import get_logger
from .scoring_instant import as_utc_datetime, resolve_scoring_instant

_log = get_logger("trust_signals")

#: Audit chain written by ``audit_chain.AuditChain``.
AUDIT_CHAIN_REL = os.path.join(".mind-mem-audit", "chain.jsonl")
#: SQLite index that also holds the ``calibration_feedback`` table.
INDEX_DB_REL = os.path.join(".mind-mem-index", "recall.db")

#: Audit operation that marks an actor's write as reverted.
ROLLBACK_OPERATION = "rollback"


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
    """Return ``(rollbacks, total_writes)`` per actor from the audit chain.

    **Diagnostic only — never scored.** Per-actor history is a learned,
    corpus-slice-dependent quantity; folding it into recall ranking would
    make the same hit rank differently on two machines. Exposed for
    operators reading the audit chain, not for the gate.

    Entries with a blank ``agent`` are skipped — an unattributed write
    cannot be charged to anyone.

    Args:
        workspace: Workspace root. Falsy input yields ``({}, {})``.

    Returns:
        Two maps keyed by the audit chain's ``agent`` field: rollback
        operations, and total operations. Empty when no chain exists.
    """
    if not workspace:
        return {}, {}
    if not os.path.isfile(os.path.join(os.path.abspath(workspace), AUDIT_CHAIN_REL)):
        return {}, {}
    rollbacks: dict[str, int] = {}
    writes: dict[str, int] = {}
    try:
        from .audit_chain import AuditChain

        for entry in AuditChain(workspace).entries():
            agent = str(getattr(entry, "agent", "") or "").strip()
            if not agent:
                continue
            writes[agent] = writes.get(agent, 0) + 1
            if getattr(entry, "operation", "") == ROLLBACK_OPERATION:
                rollbacks[agent] = rollbacks.get(agent, 0) + 1
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
    "INDEX_DB_REL",
    "ROLLBACK_OPERATION",
    "load_calibration_weights",
    "load_rollback_history",
]

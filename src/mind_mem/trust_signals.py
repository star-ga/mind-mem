"""Workspace signal loaders for per-actor trust scores.

The scoring math in :mod:`trust_scores` is pure. This module is the only
place that touches disk on its behalf: it reads the calibration weights
already recorded by :mod:`calibration` and the rollback history already
recorded by :mod:`audit_chain`, and hands them back as plain maps.

Both loaders are **read-only and non-creating**: if the SQLite index or
the audit chain file does not exist, they return empty maps rather than
initialising a store — recall must never write on a read path. Any other
failure degrades to "no signal" (which scores as neutral) and logs.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import os

from .observability import get_logger

_log = get_logger("trust_signals")

#: Audit chain written by ``audit_chain.AuditChain``.
AUDIT_CHAIN_REL = os.path.join(".mind-mem-audit", "chain.jsonl")
#: SQLite index that also holds the ``calibration_feedback`` table.
INDEX_DB_REL = os.path.join(".mind-mem-index", "recall.db")

#: Audit operation that marks an actor's write as reverted.
ROLLBACK_OPERATION = "rollback"


def load_calibration_weights(workspace: str, block_ids: list[str]) -> dict[str, float]:
    """Return ``block_id -> calibration weight``, or ``{}`` when unavailable.

    Args:
        workspace: Workspace root. Falsy input yields ``{}``.
        block_ids: Blocks to look up. Empty input yields ``{}``.

    Returns:
        The weights :meth:`calibration.CalibrationManager.get_block_weights`
        reports, or ``{}`` when the index database does not exist yet.
    """
    if not workspace or not block_ids:
        return {}
    if not os.path.isfile(os.path.join(os.path.abspath(workspace), INDEX_DB_REL)):
        return {}
    try:
        from .calibration import CalibrationManager

        return CalibrationManager(workspace).get_block_weights(block_ids)
    except Exception as exc:  # pragma: no cover — defensive, degrade to neutral
        _log.warning("trust_calibration_unavailable", error=str(exc))
        return {}


def load_rollback_history(workspace: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return ``(rollbacks, total_writes)`` per actor from the audit chain.

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


# deferred: contradiction counts still come from block Status (or an
# explicitly supplied id set) rather than from the governance contradiction
# graph — upgrade path: add a loader here that reads the edges
# ``contradiction_detector`` records and pass them as ``contradicted_ids``.

__all__ = [
    "AUDIT_CHAIN_REL",
    "INDEX_DB_REL",
    "ROLLBACK_OPERATION",
    "load_calibration_weights",
    "load_rollback_history",
]

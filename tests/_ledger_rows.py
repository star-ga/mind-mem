# Copyright 2026 STARGA, Inc.
"""One definition of "the row this write minted", shared by the ledger tests.

Through 5.0.1 a governed scope left exactly one row per ledger, so
``rows_after - rows_before == 1`` was the same statement as "the gate
authorised this write".  Since 5.0.2 it leaves **two**, and they say
different things:

*the authorisation* — minted on the way in, before the store is touched.
This is the row that says the write was allowed.

*the close record* — minted on the way out, on both the normal and the
raising exit.  This is the only row that can say whether the authorised
write actually happened, and it exists because without it a scope that
raised before ``write_block`` left a row byte-indistinguishable from one
whose block landed.  See :meth:`GovernanceGate._record_scope_close`.

The delete side has the same two halves under
:data:`~mind_mem.governance_gate.PHASE_ADMITTED` and
:data:`~mind_mem.governance_gate.PHASE_REMOVED`.

So "one governed write" is now one *authorisation* row and one close
row, and roughly two dozen assertions across this suite meant the
former while counting rows.  Rewriting each of them to say ``+ 2``
would encode the arithmetic of today's implementation in two dozen
places and go stale together the next time a scope learns a third
phase.  This module holds the convention once: ask for
:func:`authorisation_rows` wherever "the row this write minted" is
meant, and the count follows the ledger rather than the other way
round.

The phase names come from ``governance_gate`` rather than being retyped
here, so a rename in the product breaks this file loudly instead of
silently making every filter match nothing.

**Known limit, deliberately not hidden.** A *delete* scope's two halves
both carry ``DELETE`` in the hash chain — the chain has no metadata
column, and the verb was shared before the write side's close record
existed (see :data:`~mind_mem.governance_gate.CLOSE_VERB` for why the
write side chose a distinct verb instead).  So
:func:`authorisation_rows` separates the halves of a *write* in either
ledger, and the halves of a *delete* only in the evidence chain.  For a
delete, count evidence rows.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterable, Mapping

from mind_mem.governance_gate import CLOSE_VERB, PHASE_CLOSED, PHASE_REMOVED

__all__ = [
    "authorisation_rows",
    "chain_rows",
    "count_chain_authorisations",
    "count_evidence_authorisations",
    "evidence_rows",
    "is_close_row",
]


def _metadata(row: Any) -> Mapping[str, Any]:
    """The row's metadata mapping, whatever shape the row arrived in.

    Evidence rows reach tests as parsed JSONL dicts, and occasionally as
    :class:`~mind_mem.evidence_objects.EvidenceObject` instances; a hash
    chain row is a dict with no metadata at all.  All three answer here.
    """
    if isinstance(row, Mapping):
        meta = row.get("metadata")
    else:
        meta = getattr(row, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def _action(row: Any) -> str:
    """The row's verb: ``action`` on a chain row, ``action_verb`` on evidence."""
    if isinstance(row, Mapping):
        for key in ("action", "action_verb"):
            value = row.get(key)
            if isinstance(value, str):
                return value
    else:
        value = getattr(row, "action", None)
        if isinstance(value, str):
            return value
    verb = _metadata(row).get("action_verb")
    return verb if isinstance(verb, str) else ""


def is_close_row(row: Any) -> bool:
    """True when *row* is a scope's closing half rather than its authorisation.

    Three independent markers, because the two ledgers carry different
    ones and a caller should not have to know which it is holding:
    ``metadata["write_phase"] == "closed"`` (evidence, write side),
    ``metadata["delete_phase"] == "removed"`` (evidence, delete side),
    and the ``CLOSE`` verb (hash chain, write side).
    """
    meta = _metadata(row)
    if meta.get("write_phase") == PHASE_CLOSED or meta.get("delete_phase") == PHASE_REMOVED:
        return True
    return _action(row) == CLOSE_VERB


def authorisation_rows(rows: Iterable[Any]) -> list[Any]:
    """Only the rows that authorised something — one per governed scope.

    The list a pre-5.0.2 assertion meant when it counted ledger rows.
    """
    return [row for row in rows if not is_close_row(row)]


def evidence_rows(workspace: str) -> list[dict[str, Any]]:
    """Every evidence-chain row for *workspace*, in append order."""
    path = os.path.join(workspace, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chain_rows(workspace: str, db_path: str | None = None) -> list[dict[str, Any]]:
    """Every ``hash_chain`` row for *workspace*, in id order.

    Opened read-only so counting the ledger cannot alter it — a test that
    perturbs the thing it measures is not measuring it.
    """
    path = db_path or os.path.join(workspace, "memory", "hash_chain_v2.db")
    if not os.path.isfile(path):
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM hash_chain ORDER BY rowid")]
    finally:
        conn.close()


def count_evidence_authorisations(workspace: str) -> int:
    """How many governed scopes have authorised something in *workspace*."""
    return len(authorisation_rows(evidence_rows(workspace)))


def count_chain_authorisations(workspace: str, db_path: str | None = None) -> int:
    """Same count from the hash chain. See the delete caveat in the module docstring."""
    return len(authorisation_rows(chain_rows(workspace, db_path)))

# Copyright 2026 STARGA, Inc.
"""Reconcile the three ledgers against each other (5.0.2).

THE DEFECT, measured on a fresh workspace. Two governed writes, one
served recall, then the two tail rows of ``memory/hash_chain_v2.db``
deleted::

    AFTER   chain=2 evidence=4
    [ok] hash_chain: 2 entries verified
    [ok] evidence_chain: 4 entries verified
    [ok] served_ledger: 1 rows verified
    verify ok: True  exit: 0

Each ledger was walked, each was internally perfect, and nothing asked
whether they described the same history. They plainly did not: four
admissions were recorded and two chain entries survived to prove them.
Emptying the chain entirely produced ``hash_chain: 0 entries verified``
— still green.

Three legs, each using a key that ALREADY exists rather than a new field:

1. **Every close record's admission resolves.** A write scope's close
   record carries ``metadata["admission_entry_id"]``, the
   :class:`~mind_mem.hash_chain_v2.HashEntry` its admission appended. An
   id that no longer resolves is an entry that was removed.
2. **The chain is never shorter than the admissions it recorded.**
   ``GovernanceGate._write_records`` writes exactly one evidence row and
   then one chain entry, so a workspace's chain is at least as long as
   the evidence rows the gate minted. See :func:`_admission_rows` for the
   one shortfall this tolerates and why.
3. **Every served row's anchor resolves.** ``index_anchor`` is
   ``sha256(preimage(INDEX_ANCHOR_TAG, head))`` over a chain entry's hash
   at serve time, so a row anchored to an entry the chain no longer holds
   names a history that was rewritten under it.

Leg 1 and leg 3 are exact. Leg 2 is a count, which is the weakest of the
three and the only one that catches a truncation of rows nothing else
points at — which is why all three are here rather than whichever one
seemed sufficient.

Read-only, reads no clock, and creates nothing: every artifact is probed
with :func:`os.path.isfile` before a reader is built.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Optional

from .hash_chain_v2 import HashChainV2

__all__ = ["LedgerReconciliation", "reconcile"]

#: Shortfall of chain entries against admission rows that is NOT convicted.
#:
#: ``_write_records`` persists the evidence row and *then* appends the
#: chain entry, logging ``chain_append_failed_after_evidence`` if the
#: second half raises. A process killed between the two therefore leaves a
#: durable ``chain == admissions - 1``, permanently — every later
#: admission adds one to each side and the gap never closes. Convicting it
#: would make one I/O error a one-way door: the workspace could never
#: verify clean again, with no path back.
#:
#: It costs the count leg a one-entry truncation, and that is the exact
#: case :func:`mind_mem.hash_chain_v2.verify_head` convicts — removing the
#: tail leaves the seal naming an entry that is gone. The two legs are
#: complementary by construction, not by coincidence, and the residual is
#: a one-entry truncation of a chain that was never sealed (written before
#: 5.0.2 and not admitted to since), which ``--strict`` fails on for the
#: absent seal.
TOLERATED_SHORTFALL: int = 1


@dataclass(frozen=True)
class LedgerReconciliation:
    """What the three ledgers say about each other.

    ``ok`` is the verdict; the counts are what it was computed from, and
    they travel typed so a consumer never reads a number out of a
    sentence. ``checked`` is separate from ``ok`` on purpose: a workspace
    with no ledgers at all reconciles trivially, and "nothing disagreed"
    must stay distinguishable from "nothing was compared".
    """

    ok: bool
    checked: bool
    chain_entries: int
    admission_rows: int
    shortfall: int
    served_rows: int
    unresolved_admissions: tuple[str, ...]
    unresolved_anchors: tuple[int, ...]
    reasons: tuple[str, ...]

    @property
    def tolerated(self) -> bool:
        """True when the only finding is the documented crash-window gap."""
        return self.ok and 0 < self.shortfall <= TOLERATED_SHORTFALL


def reconcile(workspace: str) -> LedgerReconciliation:
    """Cross-check the hash chain, the evidence chain and the served ledger."""
    chain_path = os.path.join(workspace, "memory", "hash_chain_v2.db")
    evidence_path = os.path.join(workspace, "memory", "evidence_chain.jsonl")

    chain, readable = _open_chain(chain_path)
    if not readable:
        return LedgerReconciliation(
            ok=False,
            checked=True,
            chain_entries=0,
            admission_rows=0,
            shortfall=0,
            served_rows=0,
            unresolved_admissions=(),
            unresolved_anchors=(),
            reasons=("the hash chain cannot be read, so nothing can be reconciled against it",),
        )

    admissions, linked = _admission_rows(evidence_path)
    served = _served_anchors(workspace)
    entries = _chain_entries(chain)
    checked = bool(entries or admissions or served)

    reasons: list[str] = []

    unresolved_admissions = tuple(eid for eid in linked if eid not in entries)
    if unresolved_admissions:
        reasons.append(
            f"{len(unresolved_admissions)} close record(s) name an admission entry the chain no "
            f"longer holds: {list(unresolved_admissions[:3])}"
        )

    shortfall = max(0, admissions - len(entries))
    if shortfall > TOLERATED_SHORTFALL:
        reasons.append(f"the chain holds {len(entries)} entries against {admissions} admission rows — {shortfall} are missing")

    # Derive the anchor set only when there is a served row to check it
    # against. It is one SHA-256 per chain entry, so on a large chain in a
    # workspace whose served ledger is empty or opted out it would be the
    # most expensive thing this function does and would answer nothing —
    # and it is what pulls the attestation module into the verifier's
    # import closure, which `_anchor_for` promises not to do until a
    # workspace actually has rows to check.
    unresolved_anchors: tuple[int, ...] = ()
    if served:
        anchors = {_anchor_for(entry_hash) for entry_hash in entries.values()}
        anchors.add(_genesis_anchor())
        unresolved_anchors = tuple(seq for seq, anchor in served if anchor not in anchors)
    if unresolved_anchors:
        reasons.append(f"{len(unresolved_anchors)} served row(s) anchor to a chain entry that is gone: seq {list(unresolved_anchors[:3])}")

    return LedgerReconciliation(
        ok=not reasons,
        checked=checked,
        chain_entries=len(entries),
        admission_rows=admissions,
        shortfall=shortfall,
        served_rows=len(served),
        unresolved_admissions=unresolved_admissions,
        unresolved_anchors=unresolved_anchors,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# The three readers. Each one probes before it constructs.
# ---------------------------------------------------------------------------


def _open_chain(chain_path: str) -> tuple[Optional[HashChainV2], bool]:
    """``(chain, readable)``. An absent chain is readable and empty.

    "Absent" and "unreadable" are two facts and stay two: a workspace that
    has never admitted anything reconciles trivially, while one whose
    database will not open cannot be reconciled at all and must say so
    rather than compare against nothing.
    """
    if not os.path.isfile(chain_path):
        return None, True
    try:
        chain = HashChainV2.open_readonly(chain_path)
        # The probe. Opening is lazy, so a corrupt database raises on the
        # first query and not on the connect — asking for the length here
        # is what turns "unreadable" into an answer instead of an
        # exception thrown from the middle of a leg.
        _ = chain.length
    except (sqlite3.DatabaseError, OSError):
        return None, False
    return chain, True


def _chain_entries(chain: Optional[HashChainV2]) -> dict[str, str]:
    """``entry_id -> entry_hash`` for the whole chain.

    Both halves are needed and neither is derivable from the other: leg 1
    joins on ``entry_id`` and leg 3 joins on ``entry_hash``. Read once,
    through the same walk :meth:`HashChainV2.verify_chain` already makes,
    rather than once per join.
    """
    if chain is None:
        return {}
    return {entry.entry_id: entry.entry_hash for entry in chain.get_latest(n=chain.length)}


def _admission_rows(evidence_path: str) -> tuple[int, tuple[str, ...]]:
    """``(admission rows, admission entry ids named by close records)``.

    An *admission row* is one :meth:`GovernanceGate._write_records` minted,
    identified by ``metadata["action_verb"]`` — a key that method sets on
    every record it writes, unconditionally, and that nothing else writes.
    Counting by that rather than by row is what keeps the count leg exact:
    a row created directly through
    :meth:`~mind_mem.evidence_objects.EvidenceChain.create` has no chain
    twin to be missing, and counting it would convict a workspace of a
    truncation that never happened.

    It under-counts rows written before ``action_verb`` existed, which is
    the safe direction: an attacker truncating the chain does not touch
    the evidence rows, so the gate-minted rows are still counted and the
    shortfall still shows.

    Parsed straight from the JSONL rather than through
    :class:`~mind_mem.evidence_objects.EvidenceChain`, because this leg
    needs a COUNT of rows and the chain object's own verification is
    :func:`mind_mem.verify_cli.check_evidence_chain`'s job — running it
    twice would report one broken ledger as two findings.
    """
    if not os.path.isfile(evidence_path):
        return 0, ()

    from .governance_gate import OP_WRITE, PHASE_CLOSED

    admissions = 0
    linked: list[str] = []
    try:
        with open(evidence_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A row this module cannot parse is one check_evidence_chain
                    # convicts by name. Skipping it here keeps one defect to one
                    # finding instead of reporting the same damage twice.
                    continue
                meta = row.get("metadata")
                if not isinstance(meta, Mapping):
                    continue
                if meta.get("action_verb"):
                    admissions += 1
                if meta.get("write_phase") == PHASE_CLOSED and meta.get("operation") == OP_WRITE:
                    entry_id = meta.get("admission_entry_id")
                    if isinstance(entry_id, str) and entry_id:
                        linked.append(entry_id)
    except (OSError, UnicodeDecodeError):
        return 0, ()
    return admissions, tuple(linked)


def _served_anchors(workspace: str) -> tuple[tuple[int, str], ...]:
    """``(seq, index_anchor)`` for every served row, or empty when absent."""
    from .served_ledger import ledger_path

    if not os.path.isfile(ledger_path(workspace)):
        return ()

    from .served_ledger import read_served_runs

    try:
        rows = read_served_runs(workspace)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        # Same reasoning as the evidence reader: a broken served ledger is
        # check_served_ledger's finding, not a second copy of it here.
        return ()
    return tuple((row.seq, row.index_anchor) for row in rows)


# ---------------------------------------------------------------------------
# The anchor derivation. One definition, imported — never re-spelled.
# ---------------------------------------------------------------------------


def _anchor_for(entry_hash: str) -> str:
    """The ``index_anchor`` a run observing *entry_hash* would record.

    The tag and the construction come from
    :mod:`~mind_mem.recall_attestation`, which owns them, imported here
    rather than restated: a second spelling of ``MM_INDEX_ANCHOR_v1`` is
    two constants that have to be kept in step, and the leg would go
    silently vacuous the day they drifted. The import is function-local so
    :mod:`~mind_mem.verify_cli` keeps an import closure free of the
    serving layer until a workspace actually has served rows to check.
    """
    from .preimage import preimage
    from .recall_attestation import INDEX_ANCHOR_TAG

    return hashlib.sha256(preimage(INDEX_ANCHOR_TAG, entry_hash)).hexdigest()


def _genesis_anchor() -> str:
    """The anchor a run observing an empty chain records. Always admissible."""
    from .recall_attestation import GENESIS_ANCHOR

    return GENESIS_ANCHOR

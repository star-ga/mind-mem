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

Recovery does not reset this accounting. A recovery anchor binds the retained
hash chain's entry count and prefix-tail hash. Those historical entries form
the baseline, and admission rows written after the anchor are added to it.
Without that binding, replacing 606 evidence rows with one anchor would make
the admission count zero and turn leg 2 into a vacuous pass over the retained
hash history.

Read-only, reads no clock, and creates nothing: every artifact is probed
with :func:`os.path.isfile` before a reader is built.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .boundary_witness import (
    TS_CONTRADICTS,
    WITNESS_CONTENT_KEY,
    governed_witness_pin,
    timestamp_consistency,
    verify_witness,
)
from .evidence_recovery import (
    ATTESTS_ANCHOR_HASH_KEY,
    ATTESTS_ANCHOR_ID_KEY,
    BASELINE_VERB,
    COMPANION_ENTRIES_KEY,
    COMPANION_PRESENT_KEY,
    COMPANION_TAIL_KEY,
    DENIES_PREDECESSOR_KEY,
    RECOVERY_VERB,
    RECOVERY_VERB_KEY,
    TRUST_RESTORED_KEY,
)
from .hash_chain_v2 import GENESIS_HASH, HashChainV2

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
    recovery_baseline_present: Optional[bool] = None
    recovery_baseline_entries: Optional[int] = None
    recovery_baseline_tail: Optional[str] = None

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
            recovery_baseline_present=None,
            recovery_baseline_entries=None,
            recovery_baseline_tail=None,
            unresolved_admissions=(),
            unresolved_anchors=(),
            reasons=("the hash chain cannot be read, so nothing can be reconciled against it",),
        )

    admissions, linked, baseline, baseline_errors = _admission_rows(evidence_path)
    served = _served_anchors(workspace)
    entries = _chain_entries(chain)
    checked = bool(entries or admissions or served or baseline or baseline_errors)

    reasons: list[str] = list(baseline_errors)

    baseline_present: Optional[bool] = None
    baseline_entries: Optional[int] = None
    baseline_tail: Optional[str] = None
    if baseline is not None:
        baseline_present, baseline_entries, baseline_tail = baseline
        if baseline_present:
            hashes = list(entries.values())
            if len(hashes) < baseline_entries:
                reasons.append(f"the recovery anchor binds {baseline_entries} retained hash-chain entries, but only {len(hashes)} remain")
            elif baseline_entries and hashes[baseline_entries - 1] != baseline_tail:
                reasons.append("the retained hash-chain prefix no longer ends at the tail bound by the recovery anchor")

    unresolved_admissions = tuple(eid for eid in linked if eid not in entries)
    if unresolved_admissions:
        reasons.append(
            f"{len(unresolved_admissions)} close record(s) name an admission entry the chain no "
            f"longer holds: {list(unresolved_admissions[:3])}"
        )

    expected_entries = admissions + (baseline_entries if baseline_present and baseline_entries is not None else 0)
    shortfall = max(0, expected_entries - len(entries))
    if shortfall > TOLERATED_SHORTFALL:
        if baseline_present:
            reasons.append(
                f"the chain holds {len(entries)} entries against a retained baseline of "
                f"{baseline_entries} plus {admissions} post-recovery admission rows — {shortfall} are missing"
            )
        else:
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
        recovery_baseline_present=baseline_present,
        recovery_baseline_entries=baseline_entries,
        recovery_baseline_tail=baseline_tail,
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


def _admission_rows(
    evidence_path: str,
) -> tuple[int, tuple[str, ...], Optional[tuple[bool, int, str]], tuple[str, ...]]:
    """Admissions, links, and the recovery anchor's companion baseline.

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
        return 0, (), None, ()

    from .governance_gate import OP_WRITE, PHASE_CLOSED

    admissions = 0
    linked: list[str] = []
    baseline: Optional[tuple[bool, int, str]] = None
    baseline_errors: list[str] = []
    anchors: list[dict] = []
    attestations: list[dict] = []
    try:
        with open(evidence_path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
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
                if meta.get(RECOVERY_VERB_KEY) == RECOVERY_VERB:
                    # Collected, not convicted here. An anchor with no baseline
                    # of its own may still be bound by a LATER attestation, and
                    # that cannot be known until the whole file has been read.
                    anchors.append(
                        {
                            "line": line_no,
                            "id": row.get("evidence_id"),
                            "hash": row.get("evidence_hash"),
                            "archive": {k: meta.get(k) for k in ("archived_chain", "archived_sha256", "archived_bytes")},
                            "timestamp": row.get("timestamp"),
                            "baseline": meta,
                        }
                    )
                elif meta.get(RECOVERY_VERB_KEY) == BASELINE_VERB:
                    att = dict(meta)
                    att["line"] = line_no
                    attestations.append(att)
                if meta.get("write_phase") == PHASE_CLOSED and meta.get("operation") == OP_WRITE:
                    entry_id = meta.get("admission_entry_id")
                    if isinstance(entry_id, str) and entry_id:
                        linked.append(entry_id)
    except (OSError, UnicodeDecodeError):
        return 0, (), None, ()

    # Now resolve each anchor: its own baseline first, a later attestation
    # second, and fail-closed if neither. An anchor minted before the baseline
    # keys existed is not malformed -- the fields were never written -- but it
    # is unbound, and unbound is exactly as unusable until something binds it.
    for anc in anchors:
        meta = anc["baseline"]
        present = meta.get(COMPANION_PRESENT_KEY)
        count = meta.get(COMPANION_ENTRIES_KEY)
        tail = meta.get(COMPANION_TAIL_KEY)
        current: Optional[tuple[bool, int, str]] = None
        if well_formed_baseline(present, count, tail):
            current = (bool(present), int(count), str(tail))  # type: ignore[arg-type]
        else:
            supplemental, why = resolve_supplemental_baseline(
                evidence_path,
                str(anc["id"] or ""),
                str(anc["hash"] or ""),
                int(anc["line"]),
                anc["archive"],
                attestations,
                str(anc.get("timestamp") or ""),
                governed_witness_pin(evidence_path, str(anc["id"] or "")),
            )
            if supplemental is not None:
                current = supplemental
            else:
                baseline_errors.append(why or "a recovery anchor carries a missing or malformed companion hash-chain baseline")
                continue
        if baseline is not None and baseline != current:
            baseline_errors.append("multiple recovery anchors carry conflicting companion hash-chain baselines")
        baseline = current

    return admissions, tuple(linked), baseline, tuple(baseline_errors)


def verify_companion_prefix(workspace: str, entries: int, tail: str) -> tuple[bool, str]:
    """Does the live companion chain's PREFIX of *entries* rows end in *tail*?

    Not "is the current tail equal to tail" -- the companion chain keeps
    growing, so that test would start failing the moment anything appended
    after the attestation was written. This reads back to the attested row and
    compares there, so the claim stays checkable forever.
    """
    from .hash_chain_v2 import GENESIS_HASH as _GEN
    from .hash_chain_v2 import HashChainV2

    db_path = os.path.join(os.path.dirname(os.path.abspath(workspace)), "hash_chain_v2.db")
    if not os.path.isfile(db_path):
        return False, "the companion hash chain is absent"
    try:
        chain = HashChainV2.open_readonly(db_path)
        ok, broken_at = chain.verify_chain()
        if not ok:
            return False, f"the companion hash chain breaks at entry {broken_at}"
        length = chain.length
        if entries > length:
            return False, f"the attestation claims {entries} companion entries but the chain holds {length}"
        if entries == 0:
            return (tail == _GEN), ("" if tail == _GEN else "an empty companion prefix must end at genesis")
        rows = chain.get_latest(length - entries + 1)
        if not rows:
            return False, "the companion chain returned no rows for the attested prefix"
        actual = rows[0].entry_hash
    except Exception as exc:  # noqa: BLE001 -- any read failure is a refusal, never a pass
        return False, f"the companion hash chain could not be read: {exc}"
    if actual != tail:
        return False, "the companion prefix does not end in the attested tail"
    return True, ""


def verify_recovery_boundary(workspace: str, entries: int, anchor_timestamp: str) -> tuple[bool, str]:
    """Is *entries* the prefix that existed AT THE SEAL -- not merely A valid one?

    A cryptographically valid prefix is not the recovery boundary. Every prefix
    of a hash chain verifies, so an attestation offering a LARGER prefix -- the
    606 pre-recovery rows plus admissions appended afterwards, with their
    correctly matching tail -- passes a validity check while silently absorbing
    post-recovery admission obligations into the baseline. A SMALLER prefix
    understates it. Both are valid; neither is the boundary.

    The boundary is bracketed against the anchor's own timestamp, which is
    hash-bound in the evidence record, using companion timestamps, which are
    hash-bound in the chain preimage:

        row[entries]      must be at or before the seal
        row[entries + 1]  if it exists, must be after the seal

    TRUST ASSUMPTION, stated because it is not eliminable here. Timestamps are
    tamper-EVIDENT: hashing them means they cannot be altered after the fact
    without breaking the chain. They are not tamper-PROOF: a writer supplies its
    own timestamp, so a row backdated at creation would satisfy this bracket.
    This detects an honest-but-wrong boundary and a later edit; it does not
    defeat a lying writer at write time. Where that matters, the boundary needs
    an operator-reviewed attestation carrying its own governance authority
    rather than automatic inference from metadata.

    No count offsets and no tolerance: anything unreadable, unparseable or
    out of order fails closed.
    """
    from datetime import datetime

    from .hash_chain_v2 import HashChainV2

    def _parse(value: object) -> "Optional[datetime]":
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    sealed_at = _parse(anchor_timestamp)
    if sealed_at is None:
        return False, "the anchor carries no parseable timestamp, so no boundary can be established"

    db_path = os.path.join(os.path.dirname(os.path.abspath(workspace)), "hash_chain_v2.db")
    if not os.path.isfile(db_path):
        return False, "the companion hash chain is absent"
    try:
        chain = HashChainV2.open_readonly(db_path)
        length = chain.length
        if entries > length:
            return False, f"the attestation claims {entries} companion entries but the chain holds {length}"
        # Read from the attested row to the end: row[entries] is first.
        window = chain.get_latest(length - entries + 1) if entries >= 1 else chain.get_latest(length)
    except Exception as exc:  # noqa: BLE001 -- any read failure is a refusal
        return False, f"the companion hash chain could not be read: {exc}"

    if entries >= 1:
        if not window:
            return False, "the companion chain returned no rows at the attested boundary"
        at_boundary = _parse(getattr(window[0], "timestamp", None))
        if at_boundary is None:
            return False, "the companion row at the attested boundary carries no parseable timestamp"
        if at_boundary > sealed_at:
            return False, (
                "the attested prefix is TOO LARGE: it includes a companion row written after the seal, "
                "which would absorb a post-recovery admission into the baseline"
            )
        following = window[1] if len(window) > 1 else None
    else:
        following = window[0] if window else None

    if following is not None:
        after = _parse(getattr(following, "timestamp", None))
        if after is None:
            return False, "the companion row after the attested boundary carries no parseable timestamp"
        if after <= sealed_at:
            return False, (
                "the attested prefix is TOO SMALL: a companion row written at or before the seal "
                "falls outside it, so the baseline understates the sealed history"
            )
    return True, ""


def resolve_supplemental_baseline(
    workspace: str,
    anchor_id: str,
    anchor_hash: str,
    anchor_line: int,
    anchor_archive: Mapping[str, object],
    attestations: "Sequence[Mapping[str, object]]",
    anchor_timestamp: str = "",
    governed_witness_digest: str = "",
    notes: "Optional[list[str]]" = None,
) -> tuple[Optional[tuple[bool, int, str]], str]:
    """A baseline for one legacy anchor, or a reason it stays fail-closed.

    Every clause below is checked against an artifact, never against the
    attestation's own assertion: the archive is re-hashed off disk, and the
    companion prefix is re-derived from the live chain. The record supplies a
    binding to check; it is not itself evidence.
    """
    if notes is None:
        notes = []
    matches: list[tuple[bool, int, str]] = []
    for att in attestations:
        att_line = att.get("line", 0)
        if isinstance(att_line, int) and att_line <= anchor_line:
            continue  # forward-only: an attestation cannot precede its anchor
        if att.get(ATTESTS_ANCHOR_ID_KEY) != anchor_id:
            continue
        if att.get(ATTESTS_ANCHOR_HASH_KEY) != anchor_hash:
            return None, "an attestation names this anchor with the wrong evidence hash"
        # The denial is a REQUIREMENT, not decoration. An attestation binds an
        # anchor to an archive; it must not be readable as rehabilitating the
        # sealed history. Enforced as exact identities -- a truthy string or a
        # missing key is malformed, and malformed fails closed.
        if att.get(DENIES_PREDECESSOR_KEY) is not True:
            return None, "a baseline attestation does not deny predecessor continuity"
        if att.get(TRUST_RESTORED_KEY) is not False:
            return None, "a baseline attestation does not deny predecessor trust restoration"
        present = att.get(COMPANION_PRESENT_KEY)
        count = att.get(COMPANION_ENTRIES_KEY)
        tail = att.get(COMPANION_TAIL_KEY)
        if not well_formed_baseline(present, count, tail):
            return None, "a baseline attestation carries a malformed companion hash-chain baseline"
        for key in ("archived_chain", "archived_sha256", "archived_bytes"):
            if att.get(key) != anchor_archive.get(key):
                return None, "a baseline attestation names a different archive than its anchor"
        ok, why = _archive_matches(workspace, att)
        if not ok:
            return None, why
        assert isinstance(count, int) and isinstance(tail, str)  # well_formed_baseline
        ok, why = verify_companion_prefix(workspace, count, tail)
        if not ok:
            return None, why
        # A valid prefix is necessary and NOT sufficient: it must be the prefix
        # that existed at the seal. THE AUTHORITY FOR THAT IS A GOVERNED WITNESS,
        # never the wall clock. A timestamp bracket does not just miss a
        # post-recovery admission -- one row appended after the seal and stamped
        # before it makes the bracket refuse the true boundary and ACCEPT N+1,
        # certifying the overclaim. Reproduced; see boundary_witness.
        witness, why = verify_witness(
            workspace,
            anchor_id,
            anchor_hash,
            att.get(WITNESS_CONTENT_KEY),
            governed_witness_digest,
        )
        if witness is None:
            return None, why
        if witness.prefix_count != count or witness.prefix_tail != tail:
            return None, "the attested baseline does not match the witnessed boundary"

        # The clock is EVIDENCE, not authority. It may corroborate or contradict;
        # it can never move a witnessed boundary, and it must never raise.
        consistency = timestamp_consistency(workspace, count, anchor_timestamp)
        if consistency == TS_CONTRADICTS:
            notes.append("timestamp evidence contradicts the witnessed boundary")

        matches.append((bool(present), count, tail))

    if not matches:
        return None, ""
    if len(set(matches)) > 1:
        return None, "competing baseline attestations disagree for one anchor"
    # Byte-identical duplicates agree by definition and are collapsed rather
    # than convicted: replaying the same claim adds nothing and removes nothing.
    return matches[0], ""


def _archive_matches(workspace: str, claim: Mapping[str, object]) -> tuple[bool, str]:
    """Re-hash the named archive off disk and compare against the claim."""
    import hashlib

    name = str(claim.get("archived_chain") or "")
    want_sha = str(claim.get("archived_sha256") or "")
    want_bytes = claim.get("archived_bytes")
    if not name or not want_sha:
        return False, "a baseline attestation names no archive"
    # REJECT a traversal-bearing claim; do not normalise it away. Passing the
    # name through basename() would silently accept "../../etc/shadow" as
    # "shadow" -- a claim that named something it had no business naming would
    # be quietly rewritten into one that verifies. A name that is not already
    # its own basename is malformed, and malformed fails closed.
    if name != os.path.basename(name) or name in (os.curdir, os.pardir):
        return False, "a baseline attestation names a non-basename archive path"
    # A malformed byte length must REJECT, not skip the constraint. Guarding
    # the comparison with isinstance meant a string, None or bool length
    # bypassed the check entirely while looking stricter than it was.
    if not isinstance(want_bytes, int) or isinstance(want_bytes, bool) or want_bytes < 0:
        return False, "a baseline attestation carries a malformed archive byte length"
    path = os.path.join(os.path.dirname(os.path.abspath(workspace)), name)
    if not os.path.isfile(path):
        return False, "the archive a baseline attestation binds is missing"
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        # exc carries the absolute path; a public error must not leak it.
        return False, f"the bound archive could not be read ({exc.__class__.__name__})"
    if digest.hexdigest() != want_sha:
        return False, "the bound archive does not match its attested digest"
    if size != want_bytes:
        return False, "the bound archive does not match its attested byte length"
    return True, ""


def well_formed_baseline(present: object, count: object, tail: object) -> bool:
    """The ONE test for a companion hash-chain baseline.

    Both the direct path (a baseline in the anchor's own metadata) and the
    supplemental path (a baseline supplied by a later attestation) call this.
    A second, weaker copy of this rule would be a way in, so there is exactly
    one.
    """
    return bool(
        isinstance(present, bool)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and isinstance(tail, str)
        and len(tail) == 128
        and all(ch in "0123456789abcdef" for ch in tail)
        and (count > 0 or tail == GENESIS_HASH)
        and (present or (count == 0 and tail == GENESIS_HASH))
    )


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

# Copyright 2026 STARGA, Inc.
"""Admission receipts — proof that a block mutation passed the governance gate.

The product claim is that every write goes through
``propose_update`` → ``approve_apply`` and lands in an append-only hash
chain. Until this module that was a convention: ``GovernanceGate.admit()``
had two callers while ``BlockStore.write_block`` had thirteen, so the
drop-folder ingest and the agent-message channel wrote ``Status: active``
blocks with no chain entry naming where they came from.

A *receipt* makes the bypass impossible rather than discouraged.
:class:`~mind_mem.governance_gate.GovernanceGate` mints one when it admits
a write and publishes it on a context variable for the duration of the
admitted operation; every ``write_block`` implementation calls
:func:`require_admission` first and raises :class:`UngatedWriteError` when
no receipt is open. Nothing is threaded through unrelated signatures, and
no ``BlockStore`` implementation has to be trusted to remember the rule —
the structural test in ``tests/test_governed_write_paths.py`` enumerates
every raw caller against an explicit allowlist and fails the build on a
new one.

Two properties worth stating plainly rather than papering over:

**Receipts do not cross a new thread.** ``contextvars`` are per-context,
so a background writer must open its own scope inside its own thread.
That is fail-closed and deliberate: the alternative — a process-global
flag — is exactly the ambient authority this module removes.

**Binding is by block id, not by content.** The receipt carries the
admitted content's digest, but ``write_block`` checks only that the
receipt covers the id. Content-binding would require every backend to
reproduce the gate's serialisation byte-for-byte (markdown render vs.
Postgres row), which couples them harder than it protects. The upgrade
path is a per-block ``WRITE`` chain entry emitted by ``write_block``
itself and linked to ``receipt.entry_id``.

**A receipt names its ingest tier.** ``tier`` is required with no
default, so a new ingest source that has not been given an
:class:`~mind_mem.enums.IngestTier` cannot obtain a receipt at all and
``write_block`` refuses it. The tier's row in
:data:`~mind_mem.enums.INITIAL_STATUS` is the only place an initial
status is decided; :func:`require_admission` refuses a write whose
``Status`` is servable under a tier that cannot mint one, so a
quarantine-tier door cannot carry an ``active`` block in.

**A receipt names the operation it authorises.** ``operation`` is
:data:`OP_WRITE` or :data:`OP_DELETE`, and :func:`require_admission`
refuses a receipt whose operation is not the one the caller is
performing. A write receipt therefore cannot be reused to delete, which
matters because the two scopes are opened by different doors for
different reasons: a WRITE scope is opened by an ingest path that has
content to land, a DELETE scope by an operator or an HTTP door that has
a rationale for removing some. Before this field, ``delete_block`` had no
admission check at all in any of the five stores — an ungated delete
returned ``True`` and the block was gone, with no receipt and no chain
record. See :func:`require_delete_admission`.

**A restore is admitted at the seam too.** ``restore`` is the third
mutation on a ``BlockStore`` and was the last one held up by
convention: the RESTORE scope lived in the callers
(``apply_engine.restore_snapshot``), so a direct ``store.restore(snap)``
withdrew governed blocks with both ledgers unmoved while ``write_block``
and ``delete_block`` at the same seam refused. Every ``restore``
implementation now calls :func:`require_restore_admission` first and
raises :class:`UngatedRestoreError` when the open receipt was not minted
for a restore.

**A DELETE scope reports back what it removed.** The store calls
:meth:`AdmissionReceipt.record_removal` with the content it actually
took out; the gate reads that ledger when the scope closes and writes
ONE chain record covering it. The alternative — every door remembering
to write its own record — is the shape that produced the ungated delete
in the first place.

**A WRITE scope reports back what it consumed.** The delete side's
ledger had no write-side twin, so the chain recorded that a write was
*authorised* and never that it happened: a scope that raised after the
gate minted its entry left an ``APPLY`` row byte-indistinguishable from
one whose block landed. Measured on a fresh workspace — a scope that
raises before ``write_block`` moves the chain from 1 row to 2, the block
is absent, and the last row carries no outcome marker of any kind.
:func:`require_admission` now records every id it authorises into
:class:`LandingLedger`, and the gate writes one close record naming the
outcome and the consumed ids on **both** exits of every write scope. See
:meth:`~mind_mem.governance_gate.GovernanceGate._run_write_scope` for
what that record can and cannot claim.

**Reads have an admission too.** :func:`admit_read` and
:func:`admit_read_one` expose the decision the recall legs already make
(``admissibility.admit_leg``) so a tool handler applies the *same*
predicate rather than its own status check, and can report how many
items were withheld. A read surface that routes through them cannot
serve quarantined content by forgetting a filter, because the filter is
the only path to the rows.

This module imports nothing from ``mind_mem`` at module scope except the
leaf ``enums`` module — no I/O, no config, no other package import — so
the write surface can depend on it without dragging the whole governance
stack, and its private state cannot be reached from anywhere else. The
two functions that need the egress predicate (:func:`admit_read` and
:func:`_require_write_within_tier`) import ``admissibility`` inside the
function body, deliberately, to keep that true.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final, Iterator, Mapping, Optional, Sequence

from .enums import INITIAL_STATUS, TIER_ID_PREFIXES, IngestTier, Status, is_servable

__all__ = [
    "AdmissionReceipt",
    "BATCH",
    "BLOCK",
    "GovernanceBypassError",
    "OP_DELETE",
    "OP_WRITE",
    "PROPOSAL",
    "LandingLedger",
    "ReadAdmission",
    "RemovalLedger",
    "RESTORE_TIER",
    "UngatedDeleteError",
    "UngatedRestoreError",
    "UngatedWriteError",
    "admit_read",
    "admit_read_one",
    "current_admission",
    "require_admission",
    "require_delete_admission",
    "require_restore_admission",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GovernanceBypassError(Exception):
    """Raised when a write is blocked because governance refused it.

    Defined here rather than in ``governance_gate`` so the write surface
    can catch it without importing the gate. ``governance_gate`` re-exports
    it, so ``from .governance_gate import GovernanceBypassError`` — the
    historical spelling — keeps resolving to this exact class.
    """


class UngatedWriteError(GovernanceBypassError):
    """A block mutation was attempted with no governance admission open.

    A subclass of :class:`GovernanceBypassError` because that is what it
    is: the apply engine already aborts an apply when that class escapes,
    and an ungated write is a bypass by any other name.

    Named for writes because writes are all it could mean when it was
    introduced. It is now the base for the delete side as well
    (:class:`UngatedDeleteError`), so a handler that already catches this
    class keeps catching every ungated mutation without being edited —
    which is the forward-compatible direction, and the reason the
    inheritance runs this way round rather than through a new common
    parent that existing ``except`` clauses would miss.
    """


class UngatedDeleteError(UngatedWriteError):
    """A block delete was attempted with no DELETE admission open.

    Raised when nothing is open, when the open receipt authorises writes
    rather than deletes, when its chain entry was never confirmed, or
    when it does not cover the id being removed. Deliberately raised
    *before* the store resolves the target, so a probe for "does this id
    exist" and a probe for "is this door gated" fail identically.
    """


class UngatedRestoreError(UngatedWriteError):
    """A store restore was attempted with no RESTORE admission open.

    Raised when nothing is open, when the open receipt was minted for a
    delete, for a single block, or for a whole proposal, when its chain
    entry was never confirmed, or when it was not minted for a re-stamp
    of already-governed content. See :func:`require_restore_admission`
    for what each of those refusals is protecting.

    A subclass of :class:`UngatedWriteError` for the same
    forward-compatibility reason :class:`UngatedDeleteError` is one: a
    restore re-writes content, the apply engine already aborts on that
    class, and every existing ``except UngatedWriteError`` keeps catching
    every ungated mutation without being edited.
    """


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

#: Admission covering exactly one block.
BLOCK: Final = "block"

#: Admission covering a named set of blocks written in one operation.
BATCH: Final = "batch"

#: Admission covering every block written while applying one approved
#: proposal. Ambient *inside the sanctioned apply path only*: the gate
#: admits once per proposal (one chain entry keyed on the proposal id),
#: but the apply engine writes blocks from five separate ops. Encoding
#: that as "covers whatever this proposal touches" is the honest shape;
#: claiming a per-block content-bound receipt there would be a false one.
PROPOSAL: Final = "proposal"


# ---------------------------------------------------------------------------
# Operations. `kind` says how much a receipt covers; `operation` says what it
# lets the holder DO with that coverage. They are orthogonal and both are
# checked: a BATCH/WRITE receipt authorises writes to a fixed id set, a
# BATCH/DELETE receipt authorises deletes of one.
# ---------------------------------------------------------------------------

#: Receipt authorises ``BlockStore.write_block`` for the ids it covers.
OP_WRITE: Final = "write"

#: Receipt authorises ``BlockStore.delete_block`` for the ids it covers.
OP_DELETE: Final = "delete"

#: The closed set. A receipt naming anything else is refused at construction:
#: an operation nobody has classified authorises nothing, which is the
#: fail-closed direction.
OPERATIONS: Final[frozenset[str]] = frozenset({OP_WRITE, OP_DELETE})


#: The ingest tier a restore has to be admitted under.
#:
#: A restore reinstates blocks that were already admitted once and mints
#: no status of its own, which is exactly what
#: :data:`~mind_mem.enums.INITIAL_STATUS` calls a *carrying* tier — its row
#: is ``None``. Naming it here rather than open-coding ``IngestTier.RESTAMP``
#: at the seam gives the rule one definition and one place to change, and
#: makes "the receipt was minted for a re-stamp, not for an ingest" a
#: property :func:`require_restore_admission` can state rather than imply.
RESTORE_TIER: Final[IngestTier] = IngestTier.RESTAMP


class RemovalLedger:
    """What a DELETE scope actually removed, reported back by the store.

    The gate authorises a delete *before* the store resolves the target,
    so at admission time it does not yet know what content it is about to
    lose. The store reports it here — one call per removed block — and
    the gate reads the ledger when the scope closes, writing exactly one
    chain record over the whole set. A ``/clear`` that removes ten
    thousand blocks therefore produces one record, not ten thousand
    unlinked ones.

    Mutable by design, and the only mutable thing a receipt carries. It
    is excluded from equality and ``repr`` so a receipt still compares by
    its identity fields.

    **Memory is bounded.** The raw content of the *first* removal is kept
    (a single-block delete records the removed content's own hash as its
    payload); from the second removal on only ``(block_id, sha256)``
    leaves are kept and the raw copy is dropped, so clearing a corpus
    costs one hash per block rather than the corpus in RAM.
    """

    __slots__ = ("_leaves", "_sole_content")

    def __init__(self) -> None:
        self._leaves: list[tuple[str, str]] = []
        self._sole_content: Optional[str] = None

    def record(self, block_id: str, content: str) -> None:
        """Record that *block_id* was removed, carrying *content*."""
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self._leaves.append((str(block_id), digest))
        self._sole_content = content if len(self._leaves) == 1 else None

    @property
    def leaves(self) -> list[tuple[str, str]]:
        """``(block_id, sha256(content))`` per removal, in removal order.

        The shape :class:`~mind_mem.merkle_tree.MerkleTree` builds from,
        so the chain record can carry a root the removed set verifies
        against.
        """
        return list(self._leaves)

    @property
    def sole_content(self) -> Optional[str]:
        """The removed content when exactly one block was removed."""
        return self._sole_content

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Ids removed, in removal order."""
        return tuple(bid for bid, _ in self._leaves)

    def __len__(self) -> int:
        return len(self._leaves)

    def __bool__(self) -> bool:
        return bool(self._leaves)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"RemovalLedger(removed={len(self._leaves)})"


class LandingLedger:
    """Which ids a WRITE scope authorised a store to write, in order.

    The write-side twin of :class:`RemovalLedger`, and the reason a close
    record can say something rather than nothing. Every
    :func:`require_admission` that returns a WRITE receipt records its id
    here, so when the scope closes the gate knows which of the ids the
    receipt *covered* were actually consumed — the distinction the chain
    could not make before, because ``covers`` is a statement of intent
    minted before any store was touched.

    **What a recorded id means, exactly.** ``require_admission`` is the
    first statement of every ``write_block``, so an id lands here when a
    store *began* an authorised write of it, not when the bytes are
    durable. That gap is bounded rather than open: a ``write_block`` that
    raises propagates out through the scope, and the scope's close record
    then reads ``scope_outcome="error"``, so the ambiguous combination is
    "outcome=ok with an id that a store accepted and then discarded
    without raising". Closing that last gap needs the store to report
    back after the durable write the way ``record_removal`` does —
    deferred: it is a five-backend change to ``write_block``, and the
    upgrade path is a ``receipt.record_landed(id)`` call at the end of
    each implementation with this ledger recording *attempts* under a
    separate key.

    Mutable by design and excluded from receipt equality, exactly as
    :class:`RemovalLedger` is. Ids are de-duplicated, so re-writing one
    block twice inside a scope records it once; memory is one id per
    distinct block, which is the same order as the ``covers`` set the
    receipt already holds.
    """

    __slots__ = ("_ids", "_seen")

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._seen: set[str] = set()

    def record(self, block_id: str) -> None:
        """Record that *block_id* was authorised for writing. Idempotent."""
        bid = str(block_id)
        if bid in self._seen:
            return
        self._seen.add(bid)
        self._ids.append(bid)

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Ids consumed, in the order they were first authorised."""
        return tuple(self._ids)

    def __contains__(self, block_id: object) -> bool:
        return str(block_id) in self._seen

    def __len__(self) -> int:
        return len(self._ids)

    def __bool__(self) -> bool:
        return bool(self._ids)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"LandingLedger(landed={len(self._ids)})"


@dataclass(frozen=True)
class AdmissionReceipt:
    """Immutable proof that :meth:`GovernanceGate.admit` accepted a mutation.

    A write or a delete — ``operation`` says which, and the two are not
    interchangeable.

    Attributes:
        entry_id: ``HashEntry.entry_id`` of the chain entry the gate
            appended. Unique in ``hash_chain.entry_id``.
        content_hash: SHA3-512 of the content the gate admitted.
        kind: :data:`BLOCK`, :data:`BATCH` or :data:`PROPOSAL`.
        covers: Block ids this receipt authorises. Empty for
            :data:`PROPOSAL`, which authorises the whole apply.
        chain_verified: True once the gate has read the entry back out of
            the durable chain. :func:`require_admission` refuses a receipt
            without it, so "the receipt resolves" is a checked property
            rather than an assumption — checked once per admission rather
            than once per block, so a bulk import pays one extra read, not
            one per row.
        actor: Resolved identity the gate recorded for the admission.
        tier: Which ingest source this write came from. Required with no
            default: a door with no :class:`~mind_mem.enums.IngestTier`
            cannot get a receipt, and without a receipt it cannot write.
            Its :data:`~mind_mem.enums.INITIAL_STATUS` row decides the
            status the write may carry. ``None`` on — and only on — a
            :data:`OP_DELETE` receipt: a deletion has no ingest source,
            and inventing a tier for it would put a false claim about
            provenance in the audit record.
        operation: :data:`OP_WRITE` or :data:`OP_DELETE`. Checked by
            :func:`require_admission`, so a write receipt cannot
            authorise a delete and a delete receipt cannot authorise a
            write. Defaults to :data:`OP_WRITE`, which is what every
            receipt minted before this field existed was.
        evidence_id: ``EvidenceObject.evidence_id`` of the evidence row
            the gate wrote for this admission. The pairing key a close
            record points back with: the evidence row for an *open*
            admission cannot carry the chain ``entry_id`` (the evidence
            row is written first, before the chain entry exists), so
            without this an auditor has no field linking the two halves
            of one scope. Defaults to ``""`` for a receipt minted by
            something other than the gate.
        removals: The :class:`RemovalLedger` a DELETE scope's store
            reports into. Empty for a write.
        landings: The :class:`LandingLedger` :func:`require_admission`
            records every authorised WRITE id into. Empty for a delete.
    """

    entry_id: str
    content_hash: str
    kind: str
    tier: Optional[IngestTier]
    covers: frozenset[str] = field(default_factory=frozenset)
    chain_verified: bool = False
    actor: str = ""
    operation: str = OP_WRITE
    evidence_id: str = ""
    removals: RemovalLedger = field(default_factory=RemovalLedger, compare=False, repr=False)
    landings: LandingLedger = field(default_factory=LandingLedger, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Refuse a receipt whose operation and tier disagree.

        Structural rather than defensive: the combinations refused here
        are the ones that would let a receipt claim something untrue
        about itself, and there is no code path that legitimately builds
        one. Raising ``ValueError`` at construction means the gate cannot
        mint a malformed receipt even by mistake.
        """
        if self.operation not in OPERATIONS:
            raise ValueError(f"admission receipt names operation {self.operation!r}, which is not one of {sorted(OPERATIONS)}")
        if self.operation == OP_WRITE:
            if self.tier is None:
                raise ValueError(
                    "a WRITE receipt must name the ingest tier its status is minted from; None is only valid for a DELETE receipt"
                )
            return
        if self.tier is not None:
            raise ValueError(
                f"a DELETE receipt names no ingest tier (got {self.tier!r}): removing content is not "
                "an ingest, and the audit record must not claim it was"
            )
        if self.kind == PROPOSAL:
            raise ValueError(
                "a DELETE receipt may not be proposal-scoped: a PROPOSAL receipt authorises every "
                "id it is asked about, which is ambient authority to delete anything"
            )

    def authorizes(self, block_id: str) -> bool:
        """True when this receipt permits its operation on *block_id*."""
        if self.kind == PROPOSAL:
            return True
        return block_id in self.covers

    def record_removal(self, block_id: str, content: str) -> None:
        """Report that *block_id* was removed, carrying *content*.

        Called by ``BlockStore.delete_block`` immediately after the block
        is gone, inside the scope that authorised it. The gate turns the
        ledger into one chain record when the scope closes.

        Raises:
            UngatedDeleteError: On a write receipt (a write removes
                nothing), or for an id this receipt does not cover — a
                removal outside the admitted set must not be recorded
                under it, because the record would then name a scope that
                never authorised the removal.
        """
        if self.operation != OP_DELETE:
            raise UngatedDeleteError(
                f"admission {self.entry_id} authorises a {self.operation}, not a delete; it cannot record the removal of {block_id!r}"
            )
        bid = str(block_id)
        if not self.authorizes(bid):
            covered = ", ".join(sorted(self.covers)[:8]) or "(none)"
            raise UngatedDeleteError(
                f"admission {self.entry_id} does not cover block {bid!r} (covers: {covered}); refusing to record its removal"
            )
        self.removals.record(bid, content)

    def record_landing(self, block_id: str) -> None:
        """Report that *block_id* was authorised for writing under this receipt.

        Called by :func:`require_admission` — the first statement of every
        ``write_block`` — once the id has passed every admission check, so
        the ledger holds exactly the ids this scope let through and never
        one it refused.

        Raises:
            UngatedWriteError: On a delete receipt (a delete lands
                nothing; it reports through :meth:`record_removal`), or
                for an id this receipt does not cover. Both are
                unreachable from ``require_admission``, which checks the
                same two things first; they are here so a receipt cannot
                be made to carry a false landing by any other caller.
        """
        if self.operation != OP_WRITE:
            raise UngatedWriteError(
                f"admission {self.entry_id} authorises a {self.operation}, not a write; it cannot record a landing for {block_id!r}"
            )
        bid = str(block_id)
        if not self.authorizes(bid):
            covered = ", ".join(sorted(self.covers)[:8]) or "(none)"
            raise UngatedWriteError(
                f"admission {self.entry_id} does not cover block {bid!r} (covers: {covered}); refusing to record it as landed"
            )
        self.landings.record(bid)


# ---------------------------------------------------------------------------
# The context variable. Private by design: publishing a receipt is the
# gate's privilege, and `tests/test_governed_write_paths.py` fails the
# build if either of these names is referenced outside this module and
# `governance_gate.py`.
# ---------------------------------------------------------------------------

_active_admission: ContextVar[Optional[AdmissionReceipt]] = ContextVar("mind_mem_active_admission", default=None)


@contextmanager
def _open_admission(receipt: AdmissionReceipt) -> Iterator[AdmissionReceipt]:
    """Publish *receipt* for the duration of the block. Gate-internal."""
    token = _active_admission.set(receipt)
    try:
        yield receipt
    finally:
        _active_admission.reset(token)


def current_admission() -> Optional[AdmissionReceipt]:
    """The receipt open in this context, or ``None``. Read-only."""
    return _active_admission.get()


def _ungated(operation: str, message: str) -> UngatedWriteError:
    """The refusal class for *operation* — delete gets its own subclass."""
    if operation == OP_DELETE:
        return UngatedDeleteError(message)
    return UngatedWriteError(message)


def require_delete_admission(block_id: str) -> AdmissionReceipt:
    """Return the open receipt authorising a delete of *block_id*.

    The delete-side twin of :func:`require_admission`, and the first line
    of every ``BlockStore.delete_block`` implementation. Call it *before*
    resolving the target: a delete of a non-existent id inside a scope
    that covers it returns ``False`` from the store, while a delete of
    any id with no scope open raises — so the two are told apart by
    whether the caller was authorised, never by whether the block was
    there.

    A named function rather than a keyword because it is what the delete
    surface greps for, the same way ``require_admission`` is what the
    write surface greps for; both structural tests read the call, not the
    argument.

    Raises:
        UngatedDeleteError: No admission is open, the open one authorises
            writes rather than deletes, its chain entry was never
            verified, or it does not cover *block_id*.
    """
    return require_admission(block_id, operation=OP_DELETE)


def require_restore_admission(snap_dir: str) -> AdmissionReceipt:
    """Return the open receipt authorising a restore from *snap_dir*.

    The **first statement of every** ``BlockStore.restore`` implementation,
    for the reason :func:`require_delete_admission` is the first statement
    of every ``delete_block``: until this existed the restore seam was the
    one mutation door held up by convention rather than by construction.
    Measured on 5.0.2 with the delete and write gates already closed, a
    governed write followed by ``store.restore(snap)`` called directly with
    no scope::

        restore returned normally
        block D-002 readable before/after: True False
        (evidence, hash_chain) before/after: (4, 4) (4, 4)

    — a governed block died and neither ledger moved. The positive control
    in the same run at the same seam: ``ungated delete_block -> raised
    UngatedDeleteError``, ``ungated write_block -> raised
    UngatedWriteError``. The RESTORE scope existed
    (``apply_engine.restore_snapshot``, ``backup_restore.restore_workspace``)
    and every sanctioned caller opened it; nothing made a caller that did
    not fail.

    **What a restore receipt has to be.** Four properties, each refusing a
    receipt that would otherwise be silently transferable into the most
    destructive operation the product has:

    ``operation`` is :data:`OP_WRITE`
        A restore re-writes content, so it is admitted on the write side.
        A DELETE receipt is not transferable to it, exactly as it is not
        transferable to a write.

    ``kind`` is :data:`BATCH`
        A restore reinstates a *set* and withdraws another. A
        :data:`BLOCK` receipt covers one id and cannot honestly authorise
        overwriting a workspace; a :data:`PROPOSAL` receipt authorises
        every id it is asked about, which is ambient authority to
        overwrite anything — the same reason
        :meth:`AdmissionReceipt.__post_init__` refuses a proposal-scoped
        delete. This is the load-bearing one: ``apply_engine`` rolls back
        from *inside* an open ``admit_proposal``, so without it the
        proposal's ambient receipt would authorise a bare
        ``store.restore()`` on that path.

    ``tier`` is :data:`RESTORE_TIER`
        The receipt was minted for a re-stamp of already-governed
        content. A receipt minted for an ingest is a licence to land what
        that door brought in, never a licence to reinstate a snapshot
        over the corpus.

    ``chain_verified``
        The gate read the admission back out of the durable chain. An
        unconfirmed receipt authorises nothing here for the same reason
        it authorises no write.

    Args:
        snap_dir: The snapshot being restored. Not resolved, not read and
            not required to exist — it names the subject in the refusal,
            and this function is called *before* the store touches the
            snapshot so that an ungated caller and a caller naming a
            missing snapshot fail by authorisation rather than by
            existence, exactly as :func:`require_delete_admission` does
            for a missing block.

    Returns:
        The open receipt, so the store can name ``receipt.entry_id`` in
        its own log record and an operator can join the store's log to
        the chain entry that authorised it.

    Raises:
        UngatedRestoreError: no admission is open, or the open one fails
            any of the four properties above.

    Note:
        The receipt carries no field naming the snapshot, so this cannot
        yet check that the open scope recorded *this* manifest digest.
        The gap is bounded by ``tests/test_governed_restore_seam.py``,
        which pins ``apply_engine.restore_snapshot`` as the only opener of
        a ``.restore(`` call in ``src/`` — and that function passes the
        same ``snap_dir`` to the store that it hashed into its record, so
        a scope recording one snapshot while restoring another is not
        reachable. Closing it by construction needs the gate to mint a
        receipt that covers ``sha256(MANIFEST.json)``; see this lane's
        report for the exact change.
    """
    receipt = _active_admission.get()
    if receipt is None:
        raise UngatedRestoreError(
            f"ungated restore from {snap_dir!r}: no governance admission is open. "
            "A restore withdraws every block written since the snapshot and "
            "reinstates the versions under it; it must run inside "
            "apply_engine.restore_snapshot, which opens the RESTORE scope that "
            "records what was reinstated and what was withdrawn. See "
            "docs/GOVERNED_WRITES.md."
        )
    if receipt.operation != OP_WRITE:
        raise UngatedRestoreError(
            f"admission {receipt.entry_id} authorises a {receipt.operation}, not a restore, "
            f"so it does not cover the snapshot at {snap_dir!r}. A receipt is not transferable "
            "between operations; open a RESTORE scope."
        )
    if receipt.kind != BATCH:
        raise UngatedRestoreError(
            f"admission {receipt.entry_id} is {receipt.kind}-scoped and cannot authorise the "
            f"restore of {snap_dir!r}: a restore reinstates a set of blocks and withdraws "
            f"another, so it needs a {BATCH} receipt naming both. A {BLOCK} receipt covers one "
            f"id, and a {PROPOSAL} receipt covers whatever it is asked about — which is ambient "
            "authority to overwrite the whole workspace."
        )
    if receipt.tier is not RESTORE_TIER:
        named = receipt.tier.value if receipt.tier is not None else None
        raise UngatedRestoreError(
            f"admission {receipt.entry_id} was minted under ingest tier {named!r}, not "
            f"{RESTORE_TIER.value!r}, so it does not authorise the restore of {snap_dir!r}. A "
            "restore re-stamps content the corpus already admitted; a receipt minted for an "
            "ingest lets that door land what it brought in, not reinstate a snapshot over "
            "everything else."
        )
    if not receipt.chain_verified:
        raise UngatedRestoreError(
            f"admission {receipt.entry_id} for the restore of {snap_dir!r} was never confirmed in the hash chain; refusing the restore"
        )
    return receipt


def require_admission(block_id: str, *, status: object = None, operation: str = OP_WRITE) -> AdmissionReceipt:
    """Return the open receipt authorising *operation* on *block_id*.

    Called at the top of every ``BlockStore.write_block`` implementation,
    which passes the block's ``Status`` field as *status*. The check is
    one-sided on purpose: a tier that mints a withheld status may not
    write a **servable** block, because that is the escalation the tier
    table exists to prevent. Moving between two withheld statuses is not
    an escalation, and a carrying tier (``INITIAL_STATUS`` row ``None``)
    constrains nothing — it rewrites blocks that were already admitted.

    Every ``write_block`` implementation is required to pass *status*;
    ``tests/test_governed_write_paths.py`` fails the build on one that
    does not, so the default here is a signature convenience and not an
    opt-out. *status* is meaningless for a delete and is not consulted
    there — a removal cannot escalate a status it is taking away.

    A WRITE that passes every check is recorded in the receipt's
    :class:`LandingLedger` before this returns, which is what lets the
    scope's close record name the ids it consumed rather than only the
    ids it covered. Nothing is recorded for a refusal or for a delete.

    Args:
        operation: :data:`OP_WRITE` (default) or :data:`OP_DELETE`. The
            open receipt must name the same one. Delete surfaces should
            call :func:`require_delete_admission` instead of passing this
            by hand.

    Raises:
        UngatedWriteError: no admission is open, the open admission
            authorises the other operation, it does not cover *block_id*,
            its chain entry was never verified, or the block's status
            outranks what its tier can mint. A refused *delete* raises the
            :class:`UngatedDeleteError` subclass, so a caller that catches
            the base class is unaffected.
    """
    if operation not in OPERATIONS:
        raise ValueError(f"require_admission asked for operation {operation!r}, which is not one of {sorted(OPERATIONS)}")
    receipt = _active_admission.get()
    if receipt is None:
        raise _ungated(
            operation,
            f"ungated {operation} of {block_id!r}: no governance admission is open. "
            "Every block write must run inside GovernanceGate.admit_block / "
            "admit_batch / admit_proposal, and every block delete inside "
            "GovernanceGate.admit_delete / admit_delete_batch. See "
            "docs/GOVERNED_WRITES.md.",
        )
    if receipt.operation != operation:
        raise _ungated(
            operation,
            f"admission {receipt.entry_id} authorises a {receipt.operation}, not a {operation}, "
            f"so it does not cover {block_id!r}. Open a "
            f"{'delete' if operation == OP_DELETE else 'write'} scope for this operation; a "
            "receipt is not transferable between them.",
        )
    if not receipt.chain_verified:
        raise _ungated(
            operation, f"admission {receipt.entry_id} for {block_id!r} was never confirmed in the hash chain; refusing the {operation}"
        )
    if not receipt.authorizes(block_id):
        covered = ", ".join(sorted(receipt.covers)[:8]) or "(none)"
        raise _ungated(operation, f"admission {receipt.entry_id} does not cover block {block_id!r} (covers: {covered})")
    if operation == OP_WRITE:
        _require_write_within_tier(receipt, block_id, status)
        # Last, after every refusal: the ledger holds ids this scope let
        # through, never one it turned away. The gate reads it when the
        # scope closes and names the consumed ids in the close record, so
        # an aborted scope's record says `landed: []` instead of leaving
        # an authorisation row that reads exactly like a landed write.
        receipt.record_landing(block_id)
    return receipt


def _require_write_within_tier(receipt: AdmissionReceipt, block_id: str, status: object) -> None:
    """Refuse a write the receipt's ingest tier is not entitled to make.

    Two rules, and the tier table decides both. **Confinement**
    (:data:`~mind_mem.enums.TIER_ID_PREFIXES`): a confined tier may write
    only ids whose prefix it names, and only the one status its
    ``INITIAL_STATUS`` row mints — that narrowness is what lets
    ``DETECTOR_FINDING`` mint ``open``, a status recall recognises,
    without becoming a second ``admit_proposal``. **Status**, for every
    other tier: under a tier that mints a withheld status, a block must
    arrive in a state recall will not serve.

    The status predicate is :func:`~mind_mem.admissibility.is_admissible_status`
    — the **same** one the recall allow-list applies — and using it here
    is the whole point. This check used to ask ``is_servable(status)``,
    which is a different question about the same value, and the two
    disagreed on an **unstated** status: ``is_servable(None)`` is False,
    so the write was let through, while ``is_admissible_status(None)`` is
    True, so recall served it. An external-ingest door that simply
    omitted the ``Status`` field therefore got its content served — a
    complete bypass of the quarantine, reached through the sanctioned
    gate API and without ever naming a status the gate could refuse.

    Asking the reader's question at the writer's door closes it by
    construction. Moving between two withheld
    statuses is still not an escalation, and a carrying tier
    (``INITIAL_STATUS`` row ``None``) still constrains nothing.
    """
    from .admissibility import is_admissible_status

    if receipt.tier is None:
        # Unreachable through the gate: __post_init__ refuses a WRITE
        # receipt with no tier, and only a WRITE receipt reaches here.
        # Kept because "no tier" must never read as "no constraint" —
        # the one direction this check must not fail in is open.
        raise UngatedWriteError(
            f"admission {receipt.entry_id} for {block_id!r} names no ingest tier, so no status rule applies to it; refusing the write"
        )
    confined_to = TIER_ID_PREFIXES.get(receipt.tier)
    if confined_to is not None:
        prefix = str(block_id).split("-", 1)[0]
        if prefix not in confined_to:
            raise UngatedWriteError(
                f"block {block_id!r} was admitted under ingest tier {receipt.tier.value!r}, "
                f"which may only write ids prefixed {sorted(confined_to)} — {prefix!r} is not "
                "one of them. A confined tier is confined so that the status it mints cannot "
                "reach a corpus that status does not belong to; open the scope that owns this "
                "corpus instead."
            )
    row = INITIAL_STATUS[receipt.tier]
    if row is None or is_servable(row):
        return
    if confined_to is not None:
        # A confined tier mints exactly its own row and nothing else, on
        # ids it is confined to. That is the whole of its licence: it
        # cannot escalate (the row is not servable, checked above), it
        # cannot pick a different lifecycle status, and it cannot leave
        # its corpora. Without this arm the generic rule below would
        # refuse the tier's OWN status whenever recall recognises it —
        # which is the case that made the confinement necessary.
        if _status_is(status, row):
            return
        raise UngatedWriteError(
            f"block {block_id!r} was admitted under ingest tier {receipt.tier.value!r}, "
            f"which mints exactly {row.value!r}, but the write carries {status!r}. A confined "
            f"tier has one status to give; stamp {row.value!r} or open a different scope."
        )
    if is_admissible_status(status):
        served = "a servable status" if is_servable(status) else "no status at all"
        raise UngatedWriteError(
            f"block {block_id!r} was admitted under ingest tier {receipt.tier.value!r}, "
            f"which mints {row.value!r}, but the write carries {served} "
            f"({status!r}) and recall would serve it. Stamp {row.value!r} at the "
            "door and release it through a governance proposal instead."
        )


def _status_is(status: object, row: Status) -> bool:
    """True when *status* is exactly *row*, on the corpus's spelling terms.

    Same normalisation :func:`~mind_mem.enums.is_servable` applies (case,
    surrounding space), because a live corpus holds ``Open`` beside
    ``open`` and they are one state. Anything that is not a string or a
    :class:`~mind_mem.enums.Status` is not the row — an unstated status
    included, which is the point: a confined tier cannot land a block
    with no status any more than an ingest tier can.
    """
    if isinstance(status, Status):
        return status is row
    if not isinstance(status, str):
        return False
    return status.strip().lower() == row.value


# ---------------------------------------------------------------------------
# Read admission — the egress half of the seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadAdmission:
    """The servable subset of what a read surface was about to return.

    Attributes:
        admitted: The items a caller may be shown, order preserved,
            copied so the caller cannot mutate the source rows.
        withheld: How many items were dropped. A count, never the ids —
            a list surface that named what it withheld would leak the
            existence of quarantined blocks to every caller, which is
            most of what withholding them was for. The one-block form is
            different and deliberately so: the caller already named the
            id, so ``withheld == 1`` there tells it nothing it did not
            supply, and it is what lets ``get_block`` answer "withheld"
            rather than the false "not found".
    """

    admitted: list[dict]
    withheld: int = 0

    @property
    def sole(self) -> Optional[dict]:
        """The single admitted item, or ``None`` — for one-block reads."""
        return self.admitted[0] if self.admitted else None

    def __bool__(self) -> bool:
        return bool(self.admitted)

    def __len__(self) -> int:
        return len(self.admitted)


def admit_read(
    items: Sequence[Mapping[str, Any]],
    *,
    workspace: Optional[str] = None,
    status_key: str = "Status",
    allow: frozenset[str] = frozenset(),
    surface: Optional[str] = None,
) -> ReadAdmission:
    """The subset of *items* a read surface may serve, and how many it may not.

    This is the recall pipeline's own egress decision, exposed so every
    other content-returning surface reaches the *same* verdict instead of
    writing its own status check. It is the whole decision, not the
    predicate alone, because the three parts have to run together and in
    order:

    1. **Refresh cached statuses** (when *workspace* is given). An index
       caches ``status``, and it goes stale in the fail-OPEN direction —
       a block quarantined after it was indexed still reads ``active``
       there. A surface reading rows out of an index and not refreshing
       serves what the corpus has already withdrawn.
    2. **Allow-list the status.** ``admissibility.is_admissible_status``
       is an allow-list, so a status nobody has named is withheld.
    3. **Resolve the release set** — but only after step 2 has failed for
       something, so an all-servable list touches no disk.

    Args:
        items: Rows carrying a status under *status_key*, and their block
            id under ``"_id"`` — the key ``admissibility`` reads. A row
            keyed some other way still gets the status check; it just
            cannot be matched against the release set, which withholds it
            rather than serving it, so the mismatch fails safe and stays
            visible in ``withheld``. Blocks parsed off disk carry
            ``"Status"``; index rows carry ``"status"``.
        workspace: Enables steps 1 and 3. Omit only when the caller
            parsed the blocks itself *and* no release decision can apply
            — the release set can readmit a quarantined block, so leaving
            this out is the strict direction, never the permissive one.
        allow: Statuses to readmit for this call. A per-call widening
            with an operator behind it (``recall(include_pending=True)``
            is the precedent), never a default.
        surface: Name recorded on the withheld metric, for the same
            reason ``admit_leg`` takes ``leg``.

    Returns:
        A :class:`ReadAdmission`. ``withheld`` is the number of items
        dropped, which a tool handler should report so an incomplete
        answer is visibly incomplete rather than silently short.

    Raises:
        Whatever the status refresh raises. Deliberately not swallowed: a
        surface that cannot confirm a status must fail rather than serve
        the cached copy it could not check.
    """
    from .admissibility import admit_leg, is_admissible_status, live_statuses, with_live_statuses, workspace_release_ids

    rows: list[Mapping[str, Any]] = list(items)
    if not rows:
        return ReadAdmission([], 0)
    if workspace is not None:
        rows = list(with_live_statuses([dict(r) for r in rows], live_statuses(workspace), status_key=status_key))
    if all(is_admissible_status(row.get(status_key)) for row in rows):
        return ReadAdmission([dict(row) for row in rows], 0)
    releases: frozenset[str] = frozenset()
    if workspace is not None:
        releases = workspace_release_ids(workspace)
    kept = admit_leg(rows, status_key=status_key, releases=releases, allow=allow, leg=surface or "read")
    return ReadAdmission(kept, len(rows) - len(kept))


def admit_read_one(
    block: Optional[Mapping[str, Any]],
    *,
    workspace: Optional[str] = None,
    status_key: str = "Status",
    allow: frozenset[str] = frozenset(),
    surface: Optional[str] = None,
) -> ReadAdmission:
    """:func:`admit_read` for a surface that resolved exactly one block.

    ``block is None`` (nothing there) returns an empty admission with
    ``withheld == 0``; a block that exists but is not servable returns
    one with ``withheld == 1``. Those two are different answers and a
    single-block tool needs to tell them apart — "not found" and
    "withheld" are both refusals, but only one of them is true.
    """
    if block is None:
        return ReadAdmission([], 0)
    return admit_read([block], workspace=workspace, status_key=status_key, allow=allow, surface=surface)

# Copyright 2026 STARGA, Inc.
"""Admission receipts — proof that a block write passed the governance gate.

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

This module imports nothing from ``mind_mem`` except the leaf
``enums`` module — no I/O, no config, no other package import — so the
write surface can depend on it without dragging the whole governance
stack, and its private state cannot be reached from anywhere else.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Final, Iterator, Optional

from .enums import INITIAL_STATUS, IngestTier, is_servable

__all__ = [
    "AdmissionReceipt",
    "BATCH",
    "BLOCK",
    "GovernanceBypassError",
    "PROPOSAL",
    "UngatedWriteError",
    "current_admission",
    "require_admission",
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
    """A block write was attempted with no governance admission open.

    A subclass of :class:`GovernanceBypassError` because that is what it
    is: the apply engine already aborts an apply when that class escapes,
    and an ungated write is a bypass by any other name.
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


@dataclass(frozen=True)
class AdmissionReceipt:
    """Immutable proof that :meth:`GovernanceGate.admit` accepted a write.

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
            status the write may carry.
    """

    entry_id: str
    content_hash: str
    kind: str
    tier: IngestTier
    covers: frozenset[str] = field(default_factory=frozenset)
    chain_verified: bool = False
    actor: str = ""

    def authorizes(self, block_id: str) -> bool:
        """True when this receipt permits a write to *block_id*."""
        if self.kind == PROPOSAL:
            return True
        return block_id in self.covers


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


def require_admission(block_id: str, *, status: object = None) -> AdmissionReceipt:
    """Return the open receipt authorising a write to *block_id*.

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
    opt-out.

    Raises:
        UngatedWriteError: no admission is open, the open admission does
            not cover *block_id*, its chain entry was never verified, or
            the block's status outranks what its tier can mint.
    """
    receipt = _active_admission.get()
    if receipt is None:
        raise UngatedWriteError(
            f"ungated write to {block_id!r}: no governance admission is open. "
            "Every block write must run inside GovernanceGate.admit_block / "
            "admit_batch / admit_proposal. See docs/GOVERNED_WRITES.md."
        )
    if not receipt.chain_verified:
        raise UngatedWriteError(f"admission {receipt.entry_id} for {block_id!r} was never confirmed in the hash chain; refusing the write")
    if not receipt.authorizes(block_id):
        covered = ", ".join(sorted(receipt.covers)[:8]) or "(none)"
        raise UngatedWriteError(f"admission {receipt.entry_id} does not cover block {block_id!r} (covers: {covered})")
    _require_status_within_tier(receipt, block_id, status)
    return receipt


def _require_status_within_tier(receipt: AdmissionReceipt, block_id: str, status: object) -> None:
    """Refuse a servable status under a tier that cannot mint one."""
    row = INITIAL_STATUS[receipt.tier]
    if row is None or is_servable(row):
        return
    if is_servable(status):
        raise UngatedWriteError(
            f"block {block_id!r} was admitted under ingest tier {receipt.tier.value!r}, "
            f"which mints {row.value!r}, but the write carries a servable status "
            f"({status!r}). Release it through a governance proposal instead of "
            "stamping it at the door."
        )

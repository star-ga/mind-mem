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

This module imports nothing from ``mind_mem`` so the write surface can
depend on it without dragging the whole governance stack — and its
private state cannot be reached from anywhere else.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Final, Iterator, Optional

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
    """

    entry_id: str
    content_hash: str
    kind: str
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


def require_admission(block_id: str) -> AdmissionReceipt:
    """Return the open receipt authorising a write to *block_id*.

    Called at the top of every ``BlockStore.write_block`` implementation.

    Raises:
        UngatedWriteError: no admission is open, the open admission does
            not cover *block_id*, or its chain entry was never verified.
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
    return receipt

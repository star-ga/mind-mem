# Copyright 2026 STARGA, Inc.
"""GovernanceGate — single choke-point for all block writes.

Every block write must pass through this gate.  The gate verifies
spec-hash consistency, creates an evidence object, and appends an entry
to the SHA3-512 hash chain.  If the spec-hash has drifted the gate raises
GovernanceBypassError and the write is blocked.

Callers do not use :meth:`GovernanceGate.admit` directly — they open an
*admission scope*, which admits the write and publishes an
:class:`~mind_mem.admission.AdmissionReceipt` that
:func:`~mind_mem.admission.require_admission` reads inside every
``BlockStore.write_block``.  A write with no open scope raises
:class:`~mind_mem.admission.UngatedWriteError`::

    from .governance_gate import get_gate

    gate = get_gate(workspace)
    with gate.admit_block("WRITE", block_id, content, tier=IngestTier.EXTERNAL_INGEST):
        store.write_block(block)

Three scopes, matching the three shapes a governed write actually takes:

``admit_block``     one block (an inbox drop, a single message).
``admit_batch``     a named set written in one operation (bulk ingest,
                    a backend migration, a re-stamp pass).
``admit_proposal``  every block written while applying one approved
                    proposal — the gate admits once per proposal, so
                    this is the honest encoding of an apply's scope.

All three mint their chain entry inside ``_admit_lock``, preserving the
evidence-then-chain ordering that lock exists to protect.

**Every receipt names an ingest tier, and the tier decides the status.**
``admit_block`` / ``admit_batch`` take a required ``tier`` keyword and
refuse any tier whose :data:`~mind_mem.enums.INITIAL_STATUS` row is
servable; ``admit_proposal`` takes no tier at all and always mints
:attr:`~mind_mem.enums.IngestTier.PROPOSAL_APPLY`. So "only an approved
proposal can produce a recallable block" is enforced by construction
rather than by every door remembering to stamp ``Status: quarantined``
— a caller cannot request a status, and a source with no tier cannot
obtain a receipt at all.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

from .admission import (
    BATCH,
    BLOCK,
    PROPOSAL,
    AdmissionReceipt,
    GovernanceBypassError,
    UngatedWriteError,
    _open_admission,
)
from .enums import INITIAL_STATUS, IngestTier, mints_servable
from .evidence_objects import EvidenceAction, EvidenceChain
from .hash_chain_v2 import HashChainV2, HashEntry
from .observability import get_logger
from .spec_binding import SpecBindingManager

__all__ = [
    "AdmissionReceipt",
    "GovernanceBypassError",
    "GovernanceGate",
    "IngestTier",
    "UngatedWriteError",
    "get_gate",
]

#: Tiers a non-proposal scope may open. Derived from the table rather
#: than hand-listed, so a new row is classified the moment it exists.
MINTABLE_TIERS: frozenset[IngestTier] = frozenset(t for t in IngestTier if not mints_servable(t))


# Resolve the current_agent_id contextvar lazily so the API layer is not a
# hard dependency of the governance layer (the REST API is optional).
def _current_agent() -> str:
    """Return the authenticated agent ID from the REST context, or 'system'."""
    try:
        from mind_mem.api.rest import current_agent_id  # noqa: PLC0415

        return current_agent_id.get()
    except Exception:
        return "system"


_log = get_logger("governance_gate")


# ---------------------------------------------------------------------------
# Module-level lazy singletons keyed by workspace path
# ---------------------------------------------------------------------------

_gate_lock = threading.Lock()
_gates: dict[str, "GovernanceGate"] = {}


def get_gate(workspace: str) -> "GovernanceGate":
    """Return (creating if needed) the GovernanceGate singleton for *workspace*."""
    ws = os.path.realpath(workspace)
    with _gate_lock:
        if ws not in _gates:
            _gates[ws] = GovernanceGate(ws)
        return _gates[ws]


# ---------------------------------------------------------------------------
# GovernanceGate
# ---------------------------------------------------------------------------


class GovernanceGate:
    """Single choke-point for all governance-audited block writes.

    Initialised once per workspace.  Holds references to the shared
    HashChainV2 and EvidenceChain so every write contributes to the
    same persistent audit trail.

    Args:
        workspace: Absolute path to the mind-mem workspace.
        config_path: Optional path to mind-mem.json.  Defaults to
            ``<workspace>/mind-mem.json``.
    """

    def __init__(self, workspace: str, config_path: Optional[str] = None) -> None:
        self._ws = os.path.realpath(workspace)
        self._config_path = config_path or os.path.join(self._ws, "mind-mem.json")

        memory_dir = os.path.join(self._ws, "memory")
        os.makedirs(memory_dir, exist_ok=True)

        self._chain = HashChainV2(os.path.join(memory_dir, "hash_chain_v2.db"))
        self._evidence = EvidenceChain(store_path=os.path.join(memory_dir, "evidence_chain.jsonl"))
        self._spec_mgr = SpecBindingManager(self._config_path)
        # Serialize admit() so evidence-then-chain writes cannot interleave
        # across threads: two interleaved admits could write evidence A,
        # evidence B, chain B, chain A — the evidence and chain orderings
        # would diverge, breaking audit-trail-to-chain-head correlation.
        self._admit_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def chain(self) -> HashChainV2:
        """The underlying HashChainV2 instance."""
        return self._chain

    @property
    def evidence(self) -> EvidenceChain:
        """The underlying EvidenceChain instance."""
        return self._evidence

    def admit(
        self,
        action: str,
        block_id: str,
        content: str,
        actor: str = "",
        target_file: str = "",
        metadata: Optional[dict] = None,
    ) -> HashEntry:
        """Admit a write through the governance gate.

        Prefer :meth:`admit_block` / :meth:`admit_batch` /
        :meth:`admit_proposal`: they call this and then publish the
        receipt that ``write_block`` requires.  Calling ``admit`` alone
        records the admission but authorises no write.

        Steps:
        1. Verify spec-hash is current.  Raise GovernanceBypassError if drifted.
        2. Create an evidence object for the action.
        3. Append a hash-chain entry.

        Args:
            action:  Verb describing the write (e.g. "WRITE", "APPLY", "DELETE").
            block_id: Logical block identifier.
            content: Raw content being written (hashed, not stored inline).
            actor:   Identity of the caller (default "system").
            target_file: Relative path of the target file (optional).
            metadata: Extra contextual data (optional).

        Returns:
            The appended :class:`HashEntry`.  Previously ``True``; the
            entry carries the identity a receipt is built from, and both
            historical callers ignored the return value.

        Raises:
            GovernanceBypassError: When the spec-hash has drifted and the
                write is blocked.
        """
        # Resolve actor: explicit argument wins; fall back to contextvar then "system"
        effective_actor = actor if actor else _current_agent()

        with self._admit_lock:
            # Step 1 — spec-hash check (only when a binding exists)
            spec_hash = self._current_spec_hash()
            if spec_hash is None:
                _log.debug("governance_gate.no_binding", block_id=block_id, action=action)
            else:
                valid, reason = self._spec_mgr.verify()
                if not valid:
                    _log.warning(
                        "governance_gate.spec_drifted",
                        block_id=block_id,
                        action=action,
                        reason=reason,
                    )
                    raise GovernanceBypassError(f"GovernanceGate blocked write to '{block_id}': spec-hash drifted. {reason}")

            # Step 2 — create evidence object
            ev_action = _map_action(action)
            meta = dict(metadata or {})
            if spec_hash:
                meta["spec_hash"] = spec_hash
            # Always surface the resolved agent ID in metadata so the audit
            # record carries attribution regardless of which field consumers read.
            meta.setdefault("agent_id", effective_actor)
            self._evidence.create(
                action=ev_action,
                actor=effective_actor,
                target_block_id=block_id,
                target_file=target_file,
                payload=content,
                metadata=meta,
            )

            # Step 3 — append to hash chain. If this fails after evidence
            # was persisted, log the inconsistency loudly so operators can
            # reconcile the two stores. A true two-phase commit is not
            # possible across JSONL + SQLite; best-effort atomicity via the
            # lock + ordered write is the strongest guarantee here.
            try:
                entry = self._chain.append(block_id, action, content)
            except Exception:
                _log.error(
                    "governance_gate.chain_append_failed_after_evidence",
                    block_id=block_id,
                    action=action,
                    actor=effective_actor,
                )
                raise

            _log.debug(
                "governance_gate.admitted",
                block_id=block_id,
                action=action,
                actor=effective_actor,
            )
            return entry

    # ------------------------------------------------------------------
    # Admission scopes — the only way to authorise a block write
    # ------------------------------------------------------------------

    @contextmanager
    def admit_block(
        self,
        action: str,
        block_id: str,
        content: str,
        *,
        tier: IngestTier,
        actor: str = "",
        target_file: str = "",
        metadata: Optional[dict] = None,
    ) -> Iterator[AdmissionReceipt]:
        """Admit and authorise a write to exactly *block_id*.

        A ``write_block`` for any other id inside this scope raises
        :class:`UngatedWriteError`, and so does one whose ``Status``
        outranks what *tier* can mint.

        Args:
            tier: Required. Must be in :data:`MINTABLE_TIERS` — a scope
                that is not an approved proposal cannot mint a servable
                status, whatever the caller passes.
        """
        receipt = self._mint(action, block_id, content, BLOCK, frozenset({str(block_id)}), tier, actor, target_file, metadata)
        with _open_admission(receipt) as open_receipt:
            yield open_receipt

    @contextmanager
    def admit_batch(
        self,
        action: str,
        batch_id: str,
        block_ids: Iterable[str],
        content: str,
        *,
        tier: IngestTier,
        actor: str = "",
        target_file: str = "",
        metadata: Optional[dict] = None,
    ) -> Iterator[AdmissionReceipt]:
        """Admit and authorise writes to a fixed set of block ids.

        For operations a per-block proposal would make impossible: a bulk
        external ingest, a backend migration, a re-stamp pass.  The id set
        is fixed when the scope opens, so a batch cannot grow to cover a
        block the chain entry never named.

        Args:
            tier: Required, and bound by :data:`MINTABLE_TIERS` exactly as
                :meth:`admit_block` is.
        """
        covers = frozenset(str(bid) for bid in block_ids)
        receipt = self._mint(action, batch_id, content, BATCH, covers, tier, actor, target_file, metadata)
        with _open_admission(receipt) as open_receipt:
            yield open_receipt

    @contextmanager
    def admit_proposal(
        self,
        proposal_id: str,
        content: str,
        actor: str = "",
        target_file: str = "",
        metadata: Optional[dict] = None,
    ) -> Iterator[AdmissionReceipt]:
        """Admit one approved proposal and authorise the blocks it writes.

        Broader than the other two on purpose, and only honest because it
        is narrow in *reach*: the gate records one chain entry per
        proposal (keyed on the proposal id, hashing its ops), while the
        apply engine writes blocks from several ops.  A per-block
        content-bound receipt is not derivable from that entry, so this
        scope authorises the apply rather than pretending otherwise.

        Takes no ``tier``: it always mints
        :attr:`~mind_mem.enums.IngestTier.PROPOSAL_APPLY`. That is what
        makes "only an approved proposal can produce a servable block"
        structural — the one scope that reaches ``ACTIVE`` gives its
        caller nothing to pass.
        """
        receipt = self._mint("APPLY", proposal_id, content, PROPOSAL, frozenset(), IngestTier.PROPOSAL_APPLY, actor, target_file, metadata)
        with _open_admission(receipt) as open_receipt:
            yield open_receipt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mint(
        self,
        action: str,
        block_id: str,
        content: str,
        kind: str,
        covers: frozenset[str],
        tier: IngestTier,
        actor: str,
        target_file: str,
        metadata: Optional[dict],
    ) -> AdmissionReceipt:
        """Admit the write, then read the chain entry back before trusting it.

        The read-back is what makes "a receipt that does not resolve is an
        error" a checked property rather than an assumption.  It costs one
        indexed SELECT per admission — not per block — so a 10 000-block
        import pays for it once.

        The tier is checked *before* the admission is recorded, so a
        refused tier leaves no chain entry, and is copied into the entry's
        metadata so the audit trail names the source rather than only the
        receipt in memory.
        """
        self._check_tier(kind, block_id, tier)
        entry = self.admit(
            action=action,
            block_id=block_id,
            content=content,
            actor=actor,
            target_file=target_file,
            metadata={**(metadata or {}), "ingest_tier": tier.value},
        )
        confirmed = self._chain.get_by_entry_id(entry.entry_id)
        if confirmed is None or confirmed.entry_hash != entry.entry_hash:
            raise GovernanceBypassError(
                f"admission for {block_id!r} did not resolve in the hash chain (entry {entry.entry_id}); refusing to authorise the write"
            )
        return AdmissionReceipt(
            entry_id=entry.entry_id,
            content_hash=entry.content_hash,
            kind=kind,
            tier=tier,
            covers=covers,
            chain_verified=True,
            actor=actor or _current_agent(),
        )

    @staticmethod
    def _check_tier(kind: str, block_id: str, tier: IngestTier) -> None:
        """Refuse a tier the scope is not entitled to mint.

        ``admit_proposal`` hardcodes its tier, so the ``PROPOSAL`` arm is
        defence in depth; the ``BLOCK`` / ``BATCH`` arm is the live rule.
        """
        if not isinstance(tier, IngestTier):
            raise GovernanceBypassError(f"admission for {block_id!r} named ingest tier {tier!r}, which is not an IngestTier member")
        if kind == PROPOSAL:
            if tier is not IngestTier.PROPOSAL_APPLY:
                raise GovernanceBypassError(f"a proposal admission may only use {IngestTier.PROPOSAL_APPLY.value!r}, not {tier.value!r}")
            return
        if tier not in MINTABLE_TIERS:
            row = INITIAL_STATUS[tier]
            raise GovernanceBypassError(
                f"refusing admission for {block_id!r}: ingest tier {tier.value!r} mints "
                f"{(row.value if row else None)!r}, which recall serves. Only an approved "
                f"proposal (admit_proposal) may do that; a {kind} scope must use one of "
                f"{sorted(t.value for t in MINTABLE_TIERS)}."
            )

    def current_spec_hash(self) -> Optional[str]:
        """Return the current spec_hash from the binding, or None."""
        return self._current_spec_hash()

    def _current_spec_hash(self) -> Optional[str]:
        binding = self._spec_mgr.get_binding()
        return binding.spec_hash if binding is not None else None


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

_ACTION_MAP: dict[str, EvidenceAction] = {
    "WRITE": EvidenceAction.APPLY,
    "APPLY": EvidenceAction.APPLY,
    "CREATE": EvidenceAction.APPLY,
    "DELETE": EvidenceAction.ROLLBACK,
    "ROLLBACK": EvidenceAction.ROLLBACK,
    "PROPOSE": EvidenceAction.PROPOSE,
    "VERIFY": EvidenceAction.VERIFY,
    "RESOLVE": EvidenceAction.RESOLVE,
    "DRIFT": EvidenceAction.DRIFT,
    "CONTRADICT": EvidenceAction.CONTRADICT,
}


def _map_action(action: str) -> EvidenceAction:
    """Map a free-form action string to an EvidenceAction enum value."""
    upper = action.upper()
    return _ACTION_MAP.get(upper, EvidenceAction.APPLY)

# Copyright 2026 STARGA, Inc.
"""GovernanceGate — single choke-point for all block writes and deletes.

Every block write must pass through this gate.  The gate creates an
evidence object and appends an entry to the SHA3-512 hash chain, and —
*when the workspace carries a spec binding* — verifies spec-hash
consistency first, raising GovernanceBypassError and blocking the write
if the hash has drifted.

That conditional is load-bearing and is not a detail: the spec-hash step
only runs where ``.spec_binding.json`` exists, and **no workspace is born
with one**.  ``init_workspace`` does not write one, so on a fresh
workspace the gate does **not** detect an edit to ``mind-mem.json``
(``governance_mode``, ``mcp_acl.admin_tools``, ``proposal_budget``); it
logs one ``governance_gate.unbound_config`` warning per gate and admits.
Arm it with ``mm bind`` (``mm_cli._cmd_bind`` →
:meth:`SpecBindingManager.bind`); ``mm verify`` then reports the binding
as present and current.  ``mm bind`` refuses to re-attest a config that
has already drifted unless ``--rebind`` is passed, so re-binding cannot
silently launder an unreviewed config edit.  Do not read the paragraph
above as "config tampering is caught by default" — until ``mm bind`` has
run for that workspace, it is not.

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

Five scopes. Three for the shapes a governed write takes:

``admit_block``     one block (an inbox drop, a single message).
``admit_batch``     a named set written in one operation (bulk ingest,
                    a backend migration, a re-stamp pass).
``admit_proposal``  every block written while applying one approved
                    proposal — the gate admits once per proposal, so
                    this is the honest encoding of an apply's scope.

…and two for the shapes a governed *delete* takes, which had no gate at
all before 5.0.2 (``delete_block`` checked nothing in any of the five
stores, so an ungated call removed the block and left no record):

``admit_delete``       one block, with a rationale.
``admit_delete_batch`` a frozen id set removed as one decision — what
                       ``POST /clear`` needs, so a corpus wipe is one
                       authorisation and one removal record rather than
                       N unlinked ones.

All five mint their chain entry inside ``_admit_lock``, preserving the
evidence-then-chain ordering that lock exists to protect. A receipt
names the *operation* it authorises, and
:func:`~mind_mem.admission.require_admission` refuses a mismatch, so a
write receipt cannot be spent on a delete.

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
from typing import Final, Iterable, Iterator, Optional

from .admission import (
    BATCH,
    BLOCK,
    OP_DELETE,
    OP_WRITE,
    PROPOSAL,
    AdmissionReceipt,
    GovernanceBypassError,
    UngatedDeleteError,
    UngatedWriteError,
    _open_admission,
)
from .enums import INITIAL_STATUS, IngestTier, mints_servable
from .evidence_objects import EvidenceAction, EvidenceChain
from .hash_chain_v2 import HashChainV2, HashEntry
from .merkle_tree import MerkleTree
from .observability import get_logger
from .spec_binding import SpecBindingManager

__all__ = [
    "AdmissionReceipt",
    "GovernanceBypassError",
    "GovernanceGate",
    "IngestTier",
    "UngatedDeleteError",
    "UngatedWriteError",
    "evict_gate",
    "get_gate",
]

#: Verb every delete scope records, in both its phases. Mapped by
#: :data:`_ACTION_MAP` to :attr:`EvidenceAction.ROLLBACK` — the evidence
#: vocabulary's existing word for "content withdrawn" — so governing DELETE
#: adds **no** enum member and a reader from an older release parses every
#: record a delete writes. The two phases are told apart by
#: ``metadata["delete_phase"]``, which an older reader carries through
#: untouched.
DELETE_VERB: Final = "DELETE"

#: ``metadata["delete_phase"]`` on the record minted when the scope opens:
#: this delete was authorised. Written before the store is touched.
PHASE_ADMITTED: Final = "admitted"

#: ``metadata["delete_phase"]`` on the record minted when the scope closes:
#: this is what was actually removed. One record per scope, whatever the
#: number of blocks — a ``/clear`` writes one, not one per block.
PHASE_REMOVED: Final = "removed"

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


def evict_gate(workspace: str) -> bool:
    """Drop and close the cached gate for a workspace that is being destroyed.

    Caching a gate per workspace forever is right for a real workspace
    and wrong for an ephemeral one. Every ``mm review`` preview mints a
    temporary sandbox, asks for its gate, and then deletes the
    directory — so without this the cache grows one gate per preview,
    each holding a chain and an evidence log, every one of them keyed on
    a path that no longer exists.

    Call this **only for a workspace the caller owns and is tearing
    down**. It is not a refresh. Two live gates for one workspace do not
    share ``_admit_lock``, ``HashChainV2._lock`` or
    ``EvidenceChain._lock`` — those are per-instance, and the singleton
    is the only thing that makes them process-wide. Measured: three
    appends across two evidence chains on one file leave the JSONL
    loading as *zero* entries with ``load_integrity_compromised``, i.e.
    a fork destroys the whole audit history, not merely its tail.
    :meth:`GovernanceGate.close` is what contains a mistake here — an
    evicted gate refuses to admit rather than forking.

    Returns:
        True when a gate was cached for *workspace* and has been closed.
    """
    ws = os.path.realpath(workspace)
    with _gate_lock:
        gate = _gates.pop(ws, None)
        if gate is not None:
            # Refuse new admissions ATOMICALLY with the pop. Popping alone
            # only stops callers who have not yet obtained a reference; a
            # thread already holding this gate could still open an admission
            # in the window between the pop and close(), because _closed was
            # still False. That is a write authorised against a workspace we
            # are in the middle of destroying. _begin_retire is a plain flag
            # set, so the module lock is held for no longer than before.
            gate._begin_retire()
    if gate is None:
        return False
    # The DRAIN stays outside _gate_lock: close() takes the gate's own
    # _admit_lock and can block behind an admission already in flight.
    # Holding the module lock across that would stall get_gate() for every
    # OTHER workspace in the process.
    gate.close()
    return True


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
        # Set by close(); an evicted gate must refuse, not fork. See close().
        self._closed = False
        # One warning per gate for an unbound config — see admit() step 1.
        # Per gate, not per admit: the condition is a property of the
        # workspace, and repeating it on every write would train operators
        # to filter it out.
        self._warned_unbound = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _begin_retire(self) -> None:
        """Refuse NEW admissions at once, without waiting for in-flight ones.

        Called by :func:`evict_gate` while it holds the module ``_gate_lock``,
        so the instant a gate stops being reachable through the cache it also
        stops admitting. :meth:`close` then takes ``_admit_lock`` to wait out
        any admission already in progress.

        The split is the point. Doing it all in ``close`` means blocking on
        ``_admit_lock`` while the module lock is held, which stalls every other
        workspace; doing only the pop leaves a window in which the gate is
        unreachable yet still authorising writes. A bare attribute store is
        atomic under CPython, and ``admit`` re-reads the flag under
        ``_admit_lock``, so no admission can begin after this returns.
        """
        self._closed = True

    def close(self) -> None:
        """Retire this gate: refuse every further admission. Idempotent.

        A gate is retired when the workspace under it is being destroyed
        (see :func:`evict_gate`). What makes that safe is this refusal,
        not the cache eviction: if a caller still holds a retired gate
        and a *new* gate has since been built for the same path, the two
        would each compute the next ``previous_hash`` from their own
        in-memory evidence tail and fork the log — which makes the whole
        JSONL unloadable, not just the forked tail. Raising is strictly
        better than that, and ``GovernanceBypassError`` is already this
        module's vocabulary for "this write is not authorised".

        Taken under ``_admit_lock`` so retirement can never land between
        the evidence write and the chain append — the exact window that
        lock exists to protect (see ``__init__``).

        Nothing here releases an OS handle, and it must not be described
        as though it did: :class:`HashChainV2` opens and closes a
        connection per call, and :class:`EvidenceChain` is JSONL opened
        per append. The gate holds paths, locks and loaded entries.
        ``_entries`` is deliberately left intact so a reader holding a
        retired gate (``gate.evidence``) still sees the history it
        already loaded rather than silently seeing nothing.
        """
        with self._admit_lock:
            self._closed = True

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
        1. Verify spec-hash is current, **if a binding exists**.  Raise
           GovernanceBypassError if it has drifted.  With no binding on
           disk this step is skipped (one ``unbound_config`` warning per
           gate) — see the module docstring for why that is the default.
           A binding that exists but is unparseable is *not* treated as
           absent: ``get_binding()`` raises SpecBindingCorruptedError out
           of here and the write is blocked.
        2. Create an evidence object for the action.
        3. Append a hash-chain entry.

        Args:
            action:  Verb describing the write. Must be a key of
                :data:`_ACTION_MAP` (e.g. "WRITE", "INGEST", "DELETE") — that
                table is an allowlist, and a verb it does not classify is
                refused rather than recorded under a guessed label.
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
                write is blocked; when this gate has been retired; or when
                *action* is not classifiable by :data:`_ACTION_MAP`. The
                last case is refused before either store is written, so no
                record is left claiming an action the gate could not name.
        """
        # Resolve actor: explicit argument wins; fall back to contextvar then "system"
        effective_actor = actor if actor else _current_agent()

        with self._admit_lock:
            # Step 0 — a retired gate authorises nothing. Checked inside the
            # lock so it cannot race a concurrent close() into the middle of
            # the evidence-then-chain write below.
            if self._closed:
                raise GovernanceBypassError(
                    f"GovernanceGate for {self._ws!r} was retired (its workspace is gone); refusing to admit {block_id!r}"
                )

            # Step 1 — spec-hash check (only when a binding exists).
            # No binding means this step is inert: config tampering is NOT
            # detected for this workspace. That was previously a debug log,
            # which made an unarmed gate indistinguishable from an armed one
            # in any normal deployment — and no workspace is created with a
            # binding, so unarmed is the default state, not the exception.
            # Warn once per gate instead so the gap is visible; the warning
            # names `mm bind` because that is the fix.
            spec_hash = self._current_spec_hash()
            if spec_hash is None:
                if not self._warned_unbound:
                    self._warned_unbound = True
                    _log.warning(
                        "governance_gate.unbound_config",
                        workspace=self._ws,
                        config_path=self._config_path,
                        msg=(
                            "no spec binding for this config; admitting without a "
                            "spec-hash check — edits to mind-mem.json will not be "
                            "detected until a binding is written. Run `mm bind` "
                            "for this workspace to arm it."
                        ),
                    )
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

            # Step 2 — create evidence object.
            # Classify FIRST, before either store is touched: _map_action
            # refuses a verb it cannot label, and doing that here means a
            # refusal leaves no evidence object and no chain entry to
            # reconcile. An unclassifiable verb is not recorded as an apply.
            ev_action = _map_action(action)
            meta = dict(metadata or {})
            if spec_hash:
                meta["spec_hash"] = spec_hash
            # Always surface the resolved agent ID in metadata so the audit
            # record carries attribution regardless of which field consumers read.
            meta.setdefault("agent_id", effective_actor)
            # Carry the raw verb into the evidence record. EvidenceAction is a
            # deliberately small closed vocabulary, so WRITE / INGEST /
            # MESSAGE / MIGRATE / REEXTRACT all land as APPLY; without this an
            # auditor reading evidence alone cannot tell an operator-run store
            # migration from an inbox drop. The hash chain already keeps the
            # verb (hashed, in its `action` column); this keeps the two stores
            # saying the same thing. Additive metadata, so a reader from an
            # older release just carries the extra key through untouched.
            meta.setdefault("action_verb", action)
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
    # Delete scopes — the only way to authorise a block delete
    # ------------------------------------------------------------------

    @contextmanager
    def admit_delete(
        self,
        block_id: str,
        *,
        rationale: str,
        actor: str = "",
        target_file: str = "",
        metadata: Optional[dict] = None,
    ) -> Iterator[AdmissionReceipt]:
        """Admit and authorise the deletion of exactly *block_id*.

        The delete-side twin of :meth:`admit_block`, and the only way to
        open a scope ``BlockStore.delete_block`` will accept. Until this
        existed, ``delete_block`` had no admission check in any of the
        five stores: an ungated call returned ``True``, the block was
        gone, and neither chain held a record of it.

        Two records, both under the ``DELETE`` verb, both reaching the
        evidence chain as :attr:`EvidenceAction.ROLLBACK` (no new enum
        member, so nothing an older reader cannot parse):

        ``delete_phase="admitted"``
            written before the store is touched, naming who asked, what
            id, and why. This is the one the receipt is minted from.
        ``delete_phase="removed"``
            written when the scope closes, **only if something was
            actually removed**, carrying the removed content's hash as
            its payload hash. A delete of an id that was not there
            removes nothing and records no second row.

        The second record exists because the first cannot carry what was
        lost: the gate authorises before the store resolves the target,
        so at admission time the content is still unknown. The store
        reports it with
        :meth:`~mind_mem.admission.AdmissionReceipt.record_removal`.

        Args:
            block_id: The only id this scope may delete.
            rationale: Why. Required and non-empty — an audit record that
                cannot say why content was destroyed is most of the way
                to no record. The doors impose their own stricter rules
                (``POST /clear`` wants ≥16 characters); this is the floor.
            actor: Identity to attribute the deletion to. Falls back to
                the authenticated REST agent, then ``"system"``.

        Raises:
            GovernanceBypassError: Empty rationale, retired gate, drifted
                spec binding.
        """
        covers = frozenset({str(block_id)})
        receipt = self._mint_delete(str(block_id), covers, BLOCK, rationale, actor, target_file, metadata)
        yield from self._run_delete_scope(receipt, str(block_id), target_file)

    @contextmanager
    def admit_delete_batch(
        self,
        batch_id: str,
        block_ids: Iterable[str],
        *,
        rationale: str,
        actor: str = "",
        target_file: str = "",
        metadata: Optional[dict] = None,
    ) -> Iterator[AdmissionReceipt]:
        """Admit and authorise the deletion of a fixed set of block ids.

        For the operation a per-block scope would misrepresent: ``POST
        /clear`` wipes the corpus one ``delete_block`` call at a time, and
        N separate receipts would leave N chain records with nothing
        saying they were one decision. One scope, one authorisation
        record, and one removal record carrying a Merkle root over every
        ``(block_id, content_hash)`` actually removed.

        The id set is frozen when the scope opens, exactly as
        :meth:`admit_batch` freezes it, so a clear cannot grow to cover a
        block the chain entry never named — including one written *while*
        the clear runs.

        Raises:
            GovernanceBypassError: Empty rationale or an empty id set. An
                empty set is refused rather than admitted as a no-op: a
                receipt covering nothing authorises nothing, and minting
                one would put a decision in the chain that never had a
                subject.
        """
        covers = frozenset(str(bid) for bid in block_ids)
        if not covers:
            raise GovernanceBypassError(
                f"refusing a delete batch {batch_id!r} that covers no block ids: a receipt covering nothing authorises nothing"
            )
        receipt = self._mint_delete(str(batch_id), covers, BATCH, rationale, actor, target_file, metadata)
        yield from self._run_delete_scope(receipt, str(batch_id), target_file)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_delete_scope(self, receipt: AdmissionReceipt, subject_id: str, target_file: str) -> Iterator[AdmissionReceipt]:
        """Publish *receipt*, then record whatever the store removed under it.

        The removal record is written on the way out of **both** exits.
        A scope that raised half-way through a clear still destroyed the
        blocks it got to, and a chain that only records tidy deletions is
        a chain that under-reports exactly the cases an auditor cares
        about. On the error path the record attempt can never mask the
        original exception; on the success path a failure to record
        propagates, because a deletion nobody recorded is the failure
        this whole scope exists to prevent.
        """
        try:
            with _open_admission(receipt) as open_receipt:
                yield open_receipt
        except BaseException:
            try:
                self._record_removals(receipt, subject_id, target_file, outcome="error")
            except Exception:  # pragma: no cover - never mask the original failure
                _log.error(
                    "governance_gate.delete_record_failed_after_error",
                    block_id=subject_id,
                    removed=len(receipt.removals),
                    actor=receipt.actor,
                )
            raise
        self._record_removals(receipt, subject_id, target_file, outcome="ok")

    def _record_removals(self, receipt: AdmissionReceipt, subject_id: str, target_file: str, *, outcome: str) -> None:
        """Write the one record naming what this delete scope destroyed.

        Silent when the ledger is empty, which is the honest reading: a
        delete of an id that was not in the store removed nothing, and a
        "removed" record for it would claim content died that never
        existed. The authorisation record is still there either way, so
        the attempt is not lost.

        The payload is the removed content itself for a single-block
        removal — so the record's ``payload_hash`` **is**
        ``sha256(removed content)`` — and the Merkle root over every
        ``(block_id, sha256)`` leaf when more than one block went. The
        root is in ``metadata`` in both cases, so one verification
        procedure covers both shapes.
        """
        ledger = receipt.removals
        if not ledger:
            return
        leaves = ledger.leaves
        tree = MerkleTree()
        tree.build(leaves)
        root = tree.root_hash
        sole = ledger.sole_content
        payload = sole if (len(ledger) == 1 and sole is not None) else root
        self.admit(
            action=DELETE_VERB,
            block_id=subject_id,
            content=payload,
            actor=receipt.actor,
            target_file=target_file,
            metadata={
                "delete_phase": PHASE_REMOVED,
                "operation": OP_DELETE,
                "admission_entry_id": receipt.entry_id,
                "removed_count": len(ledger),
                "merkle_root": root,
                "scope_outcome": outcome,
            },
        )

    def _mint_delete(
        self,
        subject_id: str,
        covers: frozenset[str],
        kind: str,
        rationale: str,
        actor: str,
        target_file: str,
        metadata: Optional[dict],
    ) -> AdmissionReceipt:
        """Mint the DELETE receipt for a scope over *covers*."""
        if not rationale or not rationale.strip():
            raise GovernanceBypassError(
                f"refusing to admit a delete of {subject_id!r} with no rationale: the chain record "
                "would say content was destroyed and not say why"
            )
        content = _delete_preimage(subject_id, covers, rationale)
        meta = {
            **(metadata or {}),
            "delete_phase": PHASE_ADMITTED,
            "rationale": rationale,
            "covers_count": len(covers),
        }
        return self._mint(DELETE_VERB, subject_id, content, kind, covers, None, actor, target_file, meta, operation=OP_DELETE)

    def _mint(
        self,
        action: str,
        block_id: str,
        content: str,
        kind: str,
        covers: frozenset[str],
        tier: Optional[IngestTier],
        actor: str,
        target_file: str,
        metadata: Optional[dict],
        *,
        operation: str = OP_WRITE,
    ) -> AdmissionReceipt:
        """Admit the mutation, then read the chain entry back before trusting it.

        The read-back is what makes "a receipt that does not resolve is an
        error" a checked property rather than an assumption.  It costs one
        indexed SELECT per admission — not per block — so a 10 000-block
        import pays for it once.

        The tier is checked *before* the admission is recorded, so a
        refused tier leaves no chain entry, and is copied into the entry's
        metadata so the audit trail names the source rather than only the
        receipt in memory. A DELETE admission names no tier: it ingests
        nothing, so there is no provenance to record and no status to
        constrain. That combination is checked here too, before the
        record is written — ``AdmissionReceipt.__post_init__`` would
        refuse it, but only after the chain entry had already landed.
        """
        if operation == OP_WRITE:
            self._check_tier(kind, block_id, tier)
        elif tier is not None:
            raise GovernanceBypassError(
                f"refusing a {operation} admission for {block_id!r} that names ingest tier {tier!r}: removing content is not an ingest"
            )
        entry = self.admit(
            action=action,
            block_id=block_id,
            content=content,
            actor=actor,
            target_file=target_file,
            metadata={
                **(metadata or {}),
                **({"ingest_tier": tier.value} if tier is not None else {}),
                "operation": operation,
            },
        )
        confirmed = self._chain.get_by_entry_id(entry.entry_id)
        if confirmed is None or confirmed.entry_hash != entry.entry_hash:
            raise GovernanceBypassError(
                f"admission for {block_id!r} did not resolve in the hash chain "
                f"(entry {entry.entry_id}); refusing to authorise the {operation}"
            )
        return AdmissionReceipt(
            entry_id=entry.entry_id,
            content_hash=entry.content_hash,
            kind=kind,
            tier=tier,
            covers=covers,
            chain_verified=True,
            actor=actor or _current_agent(),
            operation=operation,
        )

    @staticmethod
    def _check_tier(kind: str, block_id: str, tier: Optional[IngestTier]) -> None:
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
# Delete admission preimage
# ---------------------------------------------------------------------------


def _delete_preimage(subject_id: str, covers: frozenset[str], rationale: str) -> str:
    """Canonical text a delete admission's chain entry hashes.

    A write admission hashes the content it is about to land. A delete
    admission has no such content — the thing being destroyed is not yet
    resolved, and refusing to resolve it first is what makes the door
    probing-resistant. So it hashes the *decision* instead: the subject,
    the frozen id set, and the stated reason. Tamper with any of the
    three afterwards and the chain entry stops verifying.

    Sorted, newline-separated, and NUL-free by construction (block ids
    cannot contain a newline), so the composition is unambiguous: no
    field can be shifted into another by choosing a clever id.
    """
    ids = "\n".join(sorted(covers))
    return f"DELETE\nsubject={subject_id}\nrationale={rationale}\ncovers={len(covers)}\n{ids}\n"


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

#: Every verb a governed write may carry, and the evidence class it is
#: recorded as. This is an **allowlist, not a lookup with a default**: the
#: evidence chain is the product's audit claim, so a record may only carry a
#: label some human deliberately chose. A verb absent from this table has no
#: chosen label, and :func:`_map_action` refuses the admission rather than
#: guessing one — see the rationale on that function.
#:
#: Adding a verb is a one-line change here, and the choice of
#: :class:`EvidenceAction` for it is the audit decision being made. Reuse an
#: existing member; do **not** add an enum member for a new verb, because a
#: reader from an older release deserialises the chain with
#: ``EvidenceAction(d["action"])`` and would raise ``ValueError`` on a member
#: it has never heard of. The verb itself is not lost to the coarse class: it
#: is stored raw and in the clear in the hash chain's ``action`` column (and
#: is covered by that entry's hash), and is copied verbatim into the evidence
#: record's ``metadata["action_verb"]``. Verify against the raw verb; dispatch
#: on the enum.
_ACTION_MAP: dict[str, EvidenceAction] = {
    # --- Content landed. "APPLY" is the evidence vocabulary's word for it. ---
    "WRITE": EvidenceAction.APPLY,
    "APPLY": EvidenceAction.APPLY,
    "CREATE": EvidenceAction.APPLY,
    # INGEST / MESSAGE / MIGRATE / REEXTRACT are the verbs the shipped write
    # doors actually pass (inbox, importers, ingest-serve, agent_messaging,
    # `mm migrate-store`, pipeline_hash's re-stamp pass). Each one wrote
    # content and each one is an APPLY — but until they were listed here they
    # reached the chain via a silent default, i.e. correct by luck with
    # nothing gating it. They are classified explicitly now.
    "INGEST": EvidenceAction.APPLY,
    "MESSAGE": EvidenceAction.APPLY,
    "MIGRATE": EvidenceAction.APPLY,
    "REEXTRACT": EvidenceAction.APPLY,
    # --- Content withdrawn. ---
    "DELETE": EvidenceAction.ROLLBACK,
    "ROLLBACK": EvidenceAction.ROLLBACK,
    # --- Governance verbs that own an EvidenceAction member outright. ---
    "PROPOSE": EvidenceAction.PROPOSE,
    "VERIFY": EvidenceAction.VERIFY,
    "RESOLVE": EvidenceAction.RESOLVE,
    "DRIFT": EvidenceAction.DRIFT,
    "CONTRADICT": EvidenceAction.CONTRADICT,
}


def _map_action(action: str) -> EvidenceAction:
    """Classify a governed write's verb, or refuse to record it at all.

    This used to end in ``_ACTION_MAP.get(upper, EvidenceAction.APPLY)``, so
    any verb the table did not know was written into the evidence chain as an
    ``APPLY``. That is the one failure mode an audit chain cannot absorb: the
    record was not merely coarse, it was a *false statement* about what
    happened, indistinguishable after the fact from a genuine apply, and
    sealed under a hash that made the lie tamper-evident rather than
    detectable. A ``DELETE`` misspelled ``DELET`` recorded as an apply is
    worse than no record.

    So the mapping fails closed. There is no honest partial outcome here —
    the gate cannot write "something happened, label unknown", because
    :class:`~mind_mem.evidence_objects.EvidenceAction` has no such member and
    adding one would break older readers (see :data:`_ACTION_MAP`). Refusing
    is the only outcome that leaves the chain true. The refusal is raised
    before :meth:`GovernanceGate.admit` creates the evidence object or
    appends to the hash chain, so a rejected verb leaves no half-written
    record in either store.

    Raises:
        GovernanceBypassError: When *action* is not in :data:`_ACTION_MAP`.
    """
    upper = action.upper()
    try:
        return _ACTION_MAP[upper]
    except KeyError:
        raise GovernanceBypassError(
            f"refusing admission for action {action!r}: no evidence classification exists "
            f"for it, so the gate cannot label the chain record truthfully. Recording it as "
            f"{EvidenceAction.APPLY.value!r} by default (the pre-5.0.2 behaviour) would put a "
            f"claim in the audit chain that nothing chose. Pass one of "
            f"{sorted(_ACTION_MAP)}, or add {upper!r} to _ACTION_MAP mapped to the "
            f"EvidenceAction that is true of it."
        ) from None

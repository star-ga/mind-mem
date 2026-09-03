# Copyright 2026 STARGA, Inc.
"""GovernanceGate — single choke-point for all block writes and deletes.

Every block write must pass through this gate.  The gate creates an
evidence object and appends an entry to the SHA3-512 hash chain, and —
*when the workspace carries a spec binding* — verifies spec-hash
consistency first, raising GovernanceBypassError and blocking the write
if the hash has drifted.

That conditional is load-bearing and is not a detail: the spec-hash step
only runs where ``.spec_binding.json`` exists.  Through 5.0.1 **no
workspace was born with one** — ``init_workspace`` wrote no binding, so
on every workspace the shipped tooling could produce, step 1 was inert
and an edit to ``mind-mem.json`` (``governance_mode``,
``proposal_budget``) went undetected.  As of 5.0.2
:func:`~mind_mem.init_workspace.init` binds the config it writes, so a
new workspace is armed from birth; the ``unbound_config`` warning now
marks a workspace made before 5.0.2, or one whose binding was removed.
Arm such a workspace with ``mm bind`` (``mm_cli._cmd_bind`` →
:meth:`SpecBindingManager.bind`); ``mind-mem-verify`` then reports the
binding as present and current.  ``mm bind`` refuses to re-attest a
config that has already drifted unless ``--rebind`` is passed, and
``init`` binds only a config *it* just wrote, so neither can silently
launder an unreviewed config edit.

**Why arming at birth needed the drift response fixed first.**  Step 1
used to respond to drift by *raising*, unconditionally: it never
consulted ``governance_mode``, so a workspace in the shipped default
``detect_only`` enforced config drift as hard as one in ``enforce``.
Measured on a fresh workspace (5.0.1): ``init`` → ``bind`` → set
``auto_recall`` to ``false`` — a documented setting, and hand-editing
this file is the only configuration path the product offers, there is no
``mm config set`` — → every governed write then failed with
:class:`GovernanceBypassError`.  Binding at ``init`` time would have
turned the documented way to configure mind-mem into a total write
outage for every new workspace.

So the two landed in the only order that works: step 1 honours
``governance_mode`` first — record a ``DRIFT`` row and warn under
``detect_only``, raise otherwise — and ``init`` arms second.  Reversing
that ships the outage.  See :meth:`GovernanceGate._check_spec_drift` for
what the mode can and cannot be trusted for; in short, the response is
downgradable by the same edit it is judging, and the record is not.

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

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from typing import Final, Iterable, Iterator, Optional, Sequence

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
from .evidence_objects import EvidenceAction, EvidenceChain, EvidenceObject
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
    "config_path_for",
    "evict_gate",
    "get_gate",
    "read_governance_mode",
    "unclosed_write_scopes",
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

#: Verb the drift record carries. Mapped by :data:`_ACTION_MAP` to
#: :attr:`EvidenceAction.DRIFT`, which already existed — recording config
#: drift adds no enum member.
DRIFT_VERB: Final = "DRIFT"

#: ``block_id`` prefix for a record whose subject is the config file
#: rather than a block. Prefixed rather than bare so a config record can
#: never collide with a real block id in ``get_block_chain``.
CONFIG_SUBJECT_PREFIX: Final = "config:"

#: Verb every write scope's close record carries. Deliberately **not** the
#: scope's own verb, which is where this differs from the delete side's two
#: phases: existing consumers count rows by verb — ``restore_snapshot``
#: writes a ``RESTORE`` batch and its caller asserts exactly one ``RESTORE``
#: row — so reusing the verb would silently double every such count. A
#: distinct verb keeps every verb-based reader correct without any of them
#: being changed, which is the only version of this that a future scope
#: cannot get wrong. The scope's verb is not lost: it is in the close
#: record's ``metadata["scope_verb"]``.
CLOSE_VERB: Final = "CLOSE"

#: ``metadata["write_phase"]`` on the record every write scope appends when
#: it closes. The write-side twin of :data:`PHASE_REMOVED`, and the only
#: row that can say whether the authorised write actually happened. Absent
#: from the open record, which is byte-identical to what 5.0.1 wrote.
PHASE_CLOSED: Final = "closed"

#: ``metadata["scope_outcome"]`` values. ``ok`` means the scope's body ran
#: to completion; ``error`` means it raised and the exception left the
#: scope. Same vocabulary the delete side already records, so one
#: verification procedure reads both.
OUTCOME_OK: Final = "ok"
OUTCOME_ERROR: Final = "error"

#: The one ``governance_mode`` that refuses every apply.
#:
#: Named because two different decisions read this config and their
#: strict answers point in *opposite* directions: for spec drift the
#: strict answer is :data:`ENFORCE_MODE` (refuse the write), while for
#: :func:`mind_mem.apply_engine.apply_proposal` the strict answer is this
#: one (refuse the apply) — under ``enforce`` an apply proceeds. A single
#: "fail closed" constant would therefore be wrong for one of them, so
#: each reader names the value that is strict *for its own decision* and
#: both derive it from one parse of one file
#: (:func:`read_governance_mode`).
DETECT_ONLY_MODE: Final = "detect_only"

#: ``governance_mode`` values that downgrade the drift response from a
#: refusal to a record. Exactly one value, the shipped default: fail
#: closed means an unknown mode enforces rather than admits.
DETECT_ONLY_MODES: frozenset[str] = frozenset({DETECT_ONLY_MODE})

#: The response used when the config cannot be read at all — a missing
#: file, invalid JSON, a document that is not an object, or a
#: ``governance_mode`` that is not a string. The mode is then unknowable,
#: and unknowable takes the strict response.
ENFORCE_MODE: Final = "enforce"

#: The mode a *readable* config with no ``governance_mode`` key gets. Not
#: :data:`ENFORCE_MODE`: an absent key is not an unreadable config, and
#: this product already has one answer for it —
#: :data:`~mind_mem.init_workspace.DEFAULT_CONFIG` ships ``detect_only``,
#: ``apply_engine._governance_mode`` defaults to ``detect_only``, and
#: ``intel_scan`` defaults to ``detect_only``. A fourth reader inventing
#: a stricter default would not be fail-closed, it would be this module
#: disagreeing with the rest of the package about what the shipped
#: default is. Measured: doing that refused 69 governed writes across 24
#: test files whose workspaces carry a config with no such key.
DEFAULT_MODE: Final = DETECT_ONLY_MODE


def config_path_for(ws: str) -> str:
    """The config file a gate for workspace *ws* binds, attests and reads.

    One definition, because it is the file the spec binding hashes: any
    second reader that spells the path itself is a reader the attestation
    does not cover, and the whole point of binding a config is that the
    file which changes behaviour is the file whose changes are recorded.
    :meth:`GovernanceGate.__init__` and
    :func:`mind_mem.apply_engine._get_mode` both come through here.
    """
    return os.path.join(ws, "mind-mem.json")


def read_governance_mode(config_path: str) -> str | None:
    """``governance_mode`` from *config_path*, or ``None`` when unknowable.

    The single parse behind every governance-mode decision in this
    package. It reports three outcomes, and the caller — not this
    function — chooses the strict answer for its own decision, because
    the strict answers differ (see :data:`DETECT_ONLY_MODE`):

    * ``None`` — **unknowable**: the file is missing, is not JSON, is not
      an object, or carries a ``governance_mode`` that is not a string.
    * :data:`DEFAULT_MODE` — **readable, key absent**. An absent key is
      not an unreadable config, and this package already ships one answer
      for it.
    * the configured string, verbatim, otherwise.

    Returning ``None`` rather than a mode string is what keeps the two
    cases apart: a config that literally says ``"enforce"`` and a config
    that cannot be read must not arrive at a caller as the same value,
    or every caller silently inherits the drift gate's notion of strict.

    Read-only and side-effect free — no log line, no clock, no write —
    so a caller may probe the mode on a path that must stay inert.
    """
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    mode = config.get("governance_mode", DEFAULT_MODE)
    return mode if isinstance(mode, str) else None


#: Cap on how many ids a close record lists inline. Beyond it the record
#: carries the first :data:`_MAX_LANDED_LISTED` ids, ``landed_truncated``,
#: and — always, at any size — the exact ``landed_count`` and a Merkle
#: root over the whole set, so a truncated list never costs the record its
#: ability to verify what a scope consumed.
_MAX_LANDED_LISTED: Final = 256

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
        self._config_path = config_path or config_path_for(self._ws)

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
        # The drift reason last written as a DRIFT row, so a bound
        # workspace whose config was edited records the observation once
        # rather than on every admission for the life of the process. A
        # *different* drift is a new observation and records again.
        self._last_drift_reason: Optional[str] = None

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
        with self._admit_lock:
            # Step 0 — a retired gate authorises nothing. Checked inside the
            # lock so it cannot race a concurrent close() into the middle of
            # the evidence-then-chain write below.
            if self._closed:
                raise GovernanceBypassError(
                    f"GovernanceGate for {self._ws!r} was retired (its workspace is gone); refusing to admit {block_id!r}"
                )

            # Step 1 — spec-hash check (only when a binding exists).
            spec_hash = self._check_spec_drift(block_id, action)

            return self._write_records(action, block_id, content, actor, target_file, metadata, spec_hash)[1]

    def _write_records(
        self,
        action: str,
        block_id: str,
        content: str,
        actor: str,
        target_file: str,
        metadata: Optional[dict],
        spec_hash: Optional[str],
    ) -> tuple[EvidenceObject, HashEntry]:
        """Steps 2 and 3 alone: create the evidence object, append the chain entry.

        Split out of :meth:`admit` so a record *about* step 1 — the
        ``DRIFT`` row :meth:`_record_drift` writes — can be appended
        without re-running step 1 on itself. Recursion is not the only
        reason: a drift record that re-verified the binding would warn
        twice for one observation and, on any future response that is not
        purely advisory, would be judged by the very drift it is
        reporting.

        Returns the evidence row as well as the chain entry, because a
        write scope needs the evidence ``evidence_id`` to link its close
        record back to the admission — the evidence row is written before
        the chain entry exists, so it cannot carry the chain's id.

        Called with ``_admit_lock`` held. The lock is re-entrant, so a
        caller already inside :meth:`admit` re-takes it for free while a
        direct caller still gets the evidence-then-chain ordering the lock
        exists to protect.
        """
        # Resolve actor: explicit argument wins; fall back to contextvar then "system"
        effective_actor = actor if actor else _current_agent()

        with self._admit_lock:
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
            evidence = self._evidence.create(
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
            return evidence, entry

    def _check_spec_drift(self, block_id: str, action: str) -> Optional[str]:
        """Step 1 — verify the config against its binding. Returns the spec hash.

        Three outcomes, and which one applies is decided by
        ``governance_mode`` in the bound config:

        **No binding.** The step is inert: config tampering is NOT
        detected for this workspace. That was previously a debug log,
        which made an unarmed gate indistinguishable from an armed one in
        any normal deployment. Warn once per gate instead so the gap is
        visible; the warning names ``mm bind`` because that is the fix.
        :func:`~mind_mem.init_workspace.init` now writes a binding for
        every workspace it creates, so this is the state of a workspace
        made before 5.0.2 or one whose binding was removed.

        **Drift, any mode.** Record one ``DRIFT`` row naming the drift and
        the mode about to be applied, and warn. Recorded *before* the mode
        is consulted and regardless of what it says, which is the whole
        reason the record is worth anything (see below). Recorded once per
        distinct drift observation, not once per admission — see
        :meth:`_record_drift`.

        **Then, under** ``detect_only`` (the shipped default): admit.
        "Detect only" now means what it says, and that is the change that
        makes arming at birth safe rather than a write outage —
        hand-editing ``mind-mem.json`` is the only configuration path this
        product offers, and under the previous unconditional raise, a
        bound workspace that used it lost every governed write.
        **Under any other mode:** raise, blocking the write, which is what
        this step has always done.

        **What the mode can and cannot be trusted for.** The mode is read
        from the config under audit, so an edit that also sets
        ``governance_mode`` to ``detect_only`` chooses its own response.
        That is exactly why the DRIFT row is appended first and carries
        the mode that was then applied: the *response* is downgradable by
        the edit it is judging, the *record* is not, and a reader can see
        both the drift and the fact that enforcement was relaxed for it.
        Fail closed on anything unreadable — a missing or unparseable
        config, or a mode that is not a string — and on any value that is
        not exactly ``detect_only``. An *absent* key is not unreadable and
        takes the shipped default, :data:`DEFAULT_MODE`; see
        :meth:`_governance_mode`. Making
        the response itself tamper-proof needs the binding to attest the
        mode it was bound with (a ``governance_mode`` field on
        :class:`~mind_mem.spec_binding.SpecBinding`); until then, do not
        read ``enforce`` as "an attacker cannot turn this off".
        """
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
            return None

        valid, reason = self._spec_mgr.verify()
        if valid:
            return spec_hash

        mode = self._governance_mode()
        _log.warning(
            "governance_gate.spec_drifted",
            block_id=block_id,
            action=action,
            reason=reason,
            governance_mode=mode,
        )
        # Record BEFORE responding, so the loudest response is not also the
        # one that leaves the least evidence: an enforce-mode refusal used
        # to raise and write nothing at all, leaving the tamper it caught
        # in a log line and nowhere in the ledger.
        self._record_drift(reason, mode)
        if mode not in DETECT_ONLY_MODES:
            raise GovernanceBypassError(
                f"GovernanceGate blocked write to '{block_id}': spec-hash drifted. {reason} "
                f"governance_mode={mode!r}. Review the config change and re-attest it with "
                f"`mm bind --rebind` for this workspace; a workspace whose config is edited "
                f"after `init` armed it stays blocked until someone says the edit was intended."
            )
        return spec_hash

    def _record_drift(self, reason: str, mode: str) -> None:
        """Append one ``DRIFT`` row for a newly observed drift. Idempotent per drift.

        Recorded once per distinct drift *observation* rather than once
        per admission: a bound workspace whose config was edited would
        otherwise write a DRIFT row on every write for the life of the
        process, which floods the ledger the auditor has to read. A
        different drift (a second edit, so a different reason string)
        is a new observation and records again.

        Writes through :meth:`_write_records` rather than :meth:`admit`,
        because :meth:`admit` runs step 1 — which is what called this.
        This is the *only* caller that goes around :meth:`admit`, and it
        is a record about the check rather than an admission through it.
        It passes ``spec_hash=None``: the bound hash is stale and the
        current one is what drifted, so stamping either on this record
        would put a "verified under this spec" claim on the record that
        says verification failed. Both hashes are named in ``reason``.
        """
        if reason == self._last_drift_reason:
            return
        self._last_drift_reason = reason
        try:
            self._write_records(
                DRIFT_VERB,
                f"{CONFIG_SUBJECT_PREFIX}{os.path.basename(self._config_path)}",
                reason,
                "governance_gate",
                self._config_path,
                {"governance_mode": mode, "drift_reason": reason},
                None,
            )
        except Exception:  # pragma: no cover - a ledger that cannot record must not swallow the write
            _log.error("governance_gate.drift_record_failed", workspace=self._ws, reason=reason)
            raise

    def _governance_mode(self) -> str:
        """The drift gate's reading of ``governance_mode``. Fail closed.

        Two different situations, and collapsing them is a mistake this
        function made once — :func:`read_governance_mode` now keeps them
        apart and this method only picks the strict answer:

        **Unreadable** (``None``) — a missing file, invalid JSON, a
        document that is not an object, or a ``governance_mode`` that is
        not a string. The mode cannot be determined, so
        :data:`ENFORCE_MODE` applies.

        **Readable, key absent** — :data:`DEFAULT_MODE`. The key has a
        shipped default (``detect_only``) that three other readers in
        this package already apply; treating its absence as ``enforce``
        is not caution, it is a fourth reader disagreeing with the
        package about what the default is.

        ``ENFORCE_MODE`` is strict *here* and permissive in the apply
        engine, which is why the two readers must not share one answer;
        see :data:`DETECT_ONLY_MODE`.
        """
        mode = read_governance_mode(self._config_path)
        return ENFORCE_MODE if mode is None else mode

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
        yield from self._run_write_scope(receipt, action, str(block_id), target_file)

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
        yield from self._run_write_scope(receipt, action, str(batch_id), target_file)

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
        yield from self._run_write_scope(receipt, "APPLY", str(proposal_id), target_file)

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

    def _run_write_scope(self, receipt: AdmissionReceipt, action: str, subject_id: str, target_file: str) -> Iterator[AdmissionReceipt]:
        """Publish *receipt*, then record how the scope ended and what it consumed.

        The write-side twin of :meth:`_run_delete_scope`, and the reason
        the chain can distinguish an authorised write from a landed one.
        Before it, ``_mint`` appended the ``APPLY`` row on the way *in*
        and nothing was written on the way out, so a scope that raised
        before ``write_block`` left a row byte-indistinguishable from one
        whose block landed — the chain over-reported what it stored, and
        "prove what it stored" was not a property the ledger held.
        Measured on a fresh workspace: a scope raising before the write
        moved the chain 1 → 2 rows with the block absent, and the last row
        carried no outcome marker of any kind.

        The close record is written on **both** exits, unconditionally —
        including the one that consumed nothing. This is where it differs
        from :meth:`_record_removals`, which is silent on an empty ledger:
        a delete that removed nothing has nothing to claim, while a write
        scope that landed nothing is exactly the case an auditor is
        looking for, and silence there would make it indistinguishable
        from a process that died mid-scope. Absence of a close record
        therefore means one thing only — the scope never closed.

        On the error path a failure to record can never mask the original
        exception; on the success path it propagates, because a write
        nobody accounted for is the failure this scope exists to prevent.

        **What the record claims.** ``scope_outcome`` describes how the
        scope's *body* ended, and ``landed`` names the ids
        ``require_admission`` authorised inside it (see
        :class:`~mind_mem.admission.LandingLedger` for the precise
        meaning). It does not claim durability, and — this is the one
        limit every caller has to design around — it cannot see a body
        that rolls its own work back and then returns normally: a
        ``return`` is a normal exit and reads ``ok``. The obligation that
        follows belongs to the caller and is written down once, here, so
        it is not rediscovered per site: **a body that withdraws its own
        work must leave this scope by raising.**

        ``apply_engine`` is the worked example. Both of its rollback
        branches — an op failure, and a post-check failure after every op
        landed — raise ``ApplyAborted`` from inside the scope and catch
        it immediately outside the ``with``, so its callers still get the
        ``(False, message)`` tuple they have always got while the chain
        gets the truth. Measured before that was so: a post-check
        rollback closed the proposal's scope ``outcome=ok landed=1``,
        byte-identical to a successful apply's close record, while the
        receipt and the proposal status both said ``rolled_back``. The
        chain was not silent — the restore writes its own ``RESTORE`` row
        naming what it withdrew — but the one row that says how the scope
        ended said the wrong word, and the two had to be read together to
        find that out.

        ``metadata["scope_error_type"]`` names the exception class on the
        error exit, and is absent on the ok exit. Additive detail, never
        a second outcome value: a deliberate, fully-handled withdrawal
        (``ApplyAborted``) and a crash (``OSError``) are both ``error``,
        so a reader that knows only ``ok`` and ``error`` is complete
        without it, and one that wants to separate the two does not have
        to parse a message to do it.
        """
        try:
            with _open_admission(receipt) as open_receipt:
                yield open_receipt
        except BaseException as exc:
            try:
                self._record_scope_close(receipt, action, subject_id, target_file, outcome=OUTCOME_ERROR, error_type=type(exc).__name__)
            except Exception:  # pragma: no cover - never mask the original failure
                _log.error(
                    "governance_gate.write_close_failed_after_error",
                    block_id=subject_id,
                    landed=len(receipt.landings),
                    actor=receipt.actor,
                )
            raise
        self._record_scope_close(receipt, action, subject_id, target_file, outcome=OUTCOME_OK)

    def _record_scope_close(
        self,
        receipt: AdmissionReceipt,
        action: str,
        subject_id: str,
        target_file: str,
        *,
        outcome: str,
        error_type: Optional[str] = None,
    ) -> None:
        """Write the one record naming how a write scope ended.

        Carries two back-references, because the two ledgers are keyed
        differently and an auditor may hold either: ``admission_entry_id``
        names the hash-chain entry the admission appended (the same field
        a delete's removal record carries), and ``admission_evidence_id``
        names the evidence row. The second is what makes "an opened scope
        that never closed" *computable* from the evidence chain alone —
        an open admission's evidence row cannot carry the chain entry_id,
        because it is written before the chain entry exists.

        Recorded under :data:`CLOSE_VERB`, not the scope's own verb, so a
        consumer that counts rows by verb does not see one scope twice —
        the delete side could share ``DELETE`` across its two phases only
        because nothing counted ``DELETE`` rows before it existed. The
        scope's verb is kept in ``metadata["scope_verb"]``, so both halves
        are still one grep, and ``CLOSE`` maps to the existing
        :attr:`EvidenceAction.VERIFY` member rather than adding one.

        Size is bounded: ``landed_count`` and ``landed_root`` are exact at
        any size, and only the inline id list is capped
        (:data:`_MAX_LANDED_LISTED`), so a 10 000-block batch writes one
        bounded record rather than a 10 000-element metadata blob.

        Forward compatibility. Nothing here adds an
        :class:`~mind_mem.evidence_objects.EvidenceAction` member:
        :data:`CLOSE_VERB` maps through :data:`_ACTION_MAP` onto the
        existing ``VERIFY``, and every field this method contributes is a
        ``metadata`` key. A 5.0.1 reader round-trips ``metadata`` as an
        opaque dict (``EvidenceObject.from_dict``) and hashes it as
        sorted JSON, so it verifies a record carrying ``scope_outcome``
        or ``scope_error_type`` without knowing either word, and reports
        the row as the ``VERIFY`` it is. That is also why a *new
        ``scope_outcome`` value* would be the wrong shape for any future
        distinction: readers dispatch on that field, and a third word
        would be a word they cannot dispatch. Additive keys, always.
        """
        landed = receipt.landings.block_ids
        tree = MerkleTree()
        tree.build([(bid, _sha256_text(bid)) for bid in landed])
        metadata: dict = {
            "write_phase": PHASE_CLOSED,
            "operation": OP_WRITE,
            "admission_entry_id": receipt.entry_id,
            "admission_evidence_id": receipt.evidence_id,
            "scope_outcome": outcome,
            "landed_count": len(landed),
            "landed_root": tree.root_hash,
            "landed": list(landed[:_MAX_LANDED_LISTED]),
            "scope_verb": action,
        }
        if len(landed) > _MAX_LANDED_LISTED:
            metadata["landed_truncated"] = True
        if error_type is not None:
            # Set only on the error exit, so the ok record stays exactly the
            # record this release already writes — a key that appeared on
            # every row would make the successful close look changed to
            # anyone diffing two chains, for no information.
            metadata["scope_error_type"] = error_type
        self.admit(
            action=CLOSE_VERB,
            block_id=subject_id,
            content=_close_preimage(subject_id, outcome, landed),
            actor=receipt.actor,
            target_file=target_file,
            metadata=metadata,
        )

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
        # Through :meth:`admit`, not around it. A scope needs the evidence
        # row's id as well as the chain entry, and the obvious refactor —
        # a private method returning both, called here — quietly moved the
        # choke point: `admit` stopped being on the mint path at all, and
        # a gate made to refuse by replacing `admit` then admitted, opened
        # a scope, and let a summary reach disk. Reading the evidence tail
        # back instead keeps every admission going through the one public
        # method, and the read is inside `_admit_lock` (re-entrant, and
        # `admit` takes it too) so the pair cannot be split by a
        # concurrent admission.
        with self._admit_lock:
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
            tail = self._evidence.get_latest(1)
            evidence_id = tail[0].evidence_id if tail else ""
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
            evidence_id=evidence_id,
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


def _sha256_text(text: str) -> str:
    """SHA-256 of *text*. The leaf hash a landed-id Merkle root is built on."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _close_preimage(subject_id: str, outcome: str, landed: Sequence[str]) -> str:
    """Canonical text a write scope's close record hashes.

    The open record hashes the content the scope was about to land. The
    close record has a different subject — what the scope *did* — so it
    hashes that: the subject, how the scope ended, and the ids it
    consumed, in consumption order. Tamper with any of the three
    afterwards and the record stops verifying.

    Newline-separated and unambiguous by composition (block ids cannot
    contain a newline), and order-preserving rather than sorted, because
    the order ids were consumed in is part of what happened.
    """
    ids = "\n".join(landed)
    return f"CLOSE\nsubject={subject_id}\noutcome={outcome}\nlanded={len(landed)}\n{ids}\n"


def unclosed_write_scopes(evidence: EvidenceChain) -> list[EvidenceObject]:
    """Evidence rows for write scopes that opened and never closed.

    "Opened, not landed": the gate minted an admission, and no close
    record for it was ever written. Every write scope writes one on both
    exits, so a row left unpaired means the process died inside the
    scope, the ledger was truncated, or the close record was suppressed —
    each of which is a reason to distrust what the open row appears to
    claim.

    Pairing is by ``metadata["admission_evidence_id"]`` on the close row,
    which names the open row's own ``evidence_id``. Rows written before
    5.0.2 carry no ``operation`` key and are not counted as open scopes:
    they predate the close record and would every one of them be reported
    as unclosed, which is a statement about the release, not about the
    workspace.

    Read-only. Takes the chain rather than a workspace path so a caller
    that already holds one does not open a second reader over the same
    JSONL — two live :class:`EvidenceChain` objects over one file is the
    fork :meth:`GovernanceGate.close` exists to prevent.
    """
    rows = evidence.get_latest(n=len(evidence))
    closed: set[str] = set()
    opened: list[EvidenceObject] = []
    for row in rows:
        metadata = row.metadata or {}
        if metadata.get("operation") != OP_WRITE:
            continue
        if metadata.get("write_phase") == PHASE_CLOSED:
            linked = metadata.get("admission_evidence_id")
            if isinstance(linked, str) and linked:
                closed.add(linked)
            continue
        opened.append(row)
    return [row for row in opened if row.evidence_id not in closed]


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
    # A whole-corpus restore withdraws every block written since the
    # snapshot and resurrects the versions under it. That is the same
    # coarse class as a rollback and carries no new enum member — a 5.0.1
    # reader parses the record and sees ``ROLLBACK`` — while the raw verb
    # in the chain's ``action`` column and in ``metadata["action_verb"]``
    # keeps a restore distinguishable from the rollback of one proposal.
    "RESTORE": EvidenceAction.ROLLBACK,
    # --- Attestation of what an admitted scope actually did. ---
    # A close record neither lands content nor withdraws it: it states the
    # outcome of a scope that was already admitted, and names the ids that
    # scope consumed. VERIFY is the vocabulary's existing word for an
    # attestation record, so closing a write scope adds no enum member and
    # a 5.0.1 reader parses every row a 5.0.2 write scope writes.
    "CLOSE": EvidenceAction.VERIFY,
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

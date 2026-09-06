# Copyright 2026 STARGA, Inc.
"""RA.3 — lifecycle deaths leave a receipt, like every other governed act.

This product records content arriving (``admit`` / ``admit_batch``),
content being withdrawn (``admit_delete`` / ``admit_delete_batch``), and
content being argued about (contradiction, drift, resolution). It did not
record content quietly ceasing to be findable, and three shipped paths do
exactly that:

======================================  =====================================
transition                              recorded before this module
======================================  =====================================
``TierManager.demote``                  an in-process ``emit_event`` and a
                                        log line. No ledger row anywhere.
``TierManager._evict``                  nothing at all.
``compaction.archive_completed_blocks``  one BATCH ``MIGRATE`` receipt naming
                                        the run. No row said which block
                                        left the served surface.
======================================  =====================================

A tombstone table was the obvious answer and is the wrong one: a mutable
side table contradicts tamper-evidence, and a second place to keep the
truth is a second place for it to drift. So a lifecycle loss is written
where every other governed act is written — the append-only evidence
chain, plus the field-audit sidecar — under three verbs that say what
actually happened:

``DEMOTE``   the block lost standing on the tier ladder.
``ARCHIVE``  the block left the *served* surface. It is still governed
             and ``BlockStore.get_by_id`` still resolves it; only the
             recall corpus stops seeing it.
``FORGET``   the record named by ``subject`` was destroyed outright and
             cannot be recovered from the store that held it.

**The record names its subject, so ``FORGET`` cannot overstate.** A tier
eviction destroys a ``block_tiers`` row, not the block — the block's text
is untouched and still readable. Every row therefore carries
``metadata["subject"]``: :data:`SUBJECT_TIER_ASSIGNMENT` for a loss of
ladder standing, :data:`SUBJECT_BLOCK` for a loss the corpus itself felt.
Without that field a ``FORGET`` on an eviction would read as "the block is
gone", which is false, and a false record under a hash is worse than no
record.

**The delete doors are deliberately absent.** ``block_store.delete_block``,
``POST /clear``, ``memory_ops.delete_memory_item`` and
``compaction.compact_signals`` all already open a governed delete scope
that writes an authorisation row, a removal row carrying a Merkle root,
and a ``deleted_blocks.jsonl`` recovery journal. Adding a ``FORGET``
beside those would double-count every death, and an auditor counting
deaths would get the wrong number. This module records the acts that had
NO receipt, not the acts whose receipt could be spelled differently.

**Post-hoc, and never fatal.** A receipt is written after the transition
has landed, in the same order ``compact_signals`` reports its removals:
the demotion has already happened, so raising here would not un-demote
anything — it would only abort the rest of a decay sweep, leaving the
blocks after it unrecorded as well. A failed write is therefore logged,
counted (``lifecycle_evidence_write_failed``) and reported through
:meth:`LifecycleRecorder.record`'s return value, never raised. That is
the same ruling ``event_fanout.emit_event`` makes, for the same reason.

Configuration — **opt-in, and this is a compatibility decision, not a
safety one**::

    {"lifecycle_evidence": {"enabled": true}}

:class:`~mind_mem.evidence_objects.EvidenceAction` gained ``DEMOTE`` /
``ARCHIVE`` / ``FORGET`` in the same release that taught the reader to
tolerate a verb it does not model (``UnknownAction`` /
``EvidenceAction.parse``). A release *older* than that one parses
strictly: ``EvidenceAction(raw)`` raises, ``_load_from_file`` reads the
raise as "unreadable record", and the whole chain freezes for that
process. So a member and the writer that emits it are separate landings,
reader first — and until an operator confirms every process sharing a
workspace can read the new verbs, this writer stays inert. Flipping the
flag is that confirmation. The ``served_ledger`` shipped the same way and
defaults on now that the fleet has caught up.

Inertness is a performance claim as well as a behavioural one, so the OFF
path costs nothing per block: :meth:`LifecycleRecorder.for_workspace`
resolves the flag ONCE, at the outermost sensible point (a
``TierManager`` construction, an archive sweep), and answers ``None``.
The per-block cost of a disabled workspace is a ``None`` check. A
workspace that never opened the flag also never gets the ledger files
created for it — the chains are constructed lazily, on the first row.

Zero new dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from .evidence_objects import EvidenceAction
from .observability import get_logger, metrics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .audit_chain import AuditChain
    from .evidence_objects import EvidenceChain

_log = get_logger("lifecycle_evidence")

#: Config section in ``mind-mem.json``.
CONFIG_SECTION = "lifecycle_evidence"

#: Sidecar operation for a loss of tier standing.
DEMOTE_OPERATION = "demote_block"

#: Sidecar operation for a block leaving the served surface.
ARCHIVE_OPERATION = "archive_block"

#: Sidecar operation for a record destroyed outright.
FORGET_OPERATION = "forget_block"

#: The lifecycle verbs, as strings rather than enum members.
#:
#: RA.3's text says to add DEMOTE/ARCHIVE/FORGET to ``EvidenceAction``. Doing
#: that breaks forward compatibility and the tree already locks against it:
#: ``EvidenceObject.from_dict`` performs a strict ``EvidenceAction(value)``
#: lookup, so a member this release invents makes a 5.0.1 reader fail to load a
#: chain 5.0.2 wrote. Governing DELETE hit the same wall and answered it by
#: reusing ``ROLLBACK`` and moving the distinction into additive metadata
#: (``governance_gate.DELETE_VERB``, ``metadata["delete_phase"]``); the same
#: shape is used here, because a lifecycle loss IS content withdrawn and the
#: vocabulary already has the word for that.
DEMOTE_VERB = "DEMOTE"
ARCHIVE_VERB = "ARCHIVE"
FORGET_VERB = "FORGET"

#: Every lifecycle verb writes ``ROLLBACK`` into the chain. An older reader
#: parses the record and simply does not see the finer verb, which is exactly
#: the degradation forward compatibility asks for: it loses detail, never the
#: ability to read.
EVIDENCE_ACTION_FOR_VERB: dict[str, EvidenceAction] = {
    DEMOTE_VERB: EvidenceAction.ROLLBACK,
    ARCHIVE_VERB: EvidenceAction.ROLLBACK,
    FORGET_VERB: EvidenceAction.ROLLBACK,
}

#: ``metadata`` key carrying the verb the chain cannot express in its action.
LIFECYCLE_VERB_KEY = "lifecycle_verb"

#: The verb -> sidecar operation mapping, complete in both directions. Both
#: ledgers carry every lifecycle loss, so a row in one with no twin in the
#: other is a defect rather than a design.
OPERATION_FOR_VERB: dict[str, str] = {
    DEMOTE_VERB: DEMOTE_OPERATION,
    ARCHIVE_VERB: ARCHIVE_OPERATION,
    FORGET_VERB: FORGET_OPERATION,
}

#: The sidecar verbs this module writes. ``audit_chain.VALID_OPERATIONS``
#: is a superset; ``tests/test_ledger_hierarchy.py`` re-derives that set
#: from the source, so a verb here with no ``append`` behind it fails the
#: build.
LIFECYCLE_OPERATIONS: frozenset[str] = frozenset(OPERATION_FOR_VERB.values())

#: The thing that was lost was the block itself — its place on the served
#: surface, or its content.
SUBJECT_BLOCK = "block"

#: The thing that was lost was the block's standing on the tier ladder.
#: The block's text is untouched and still readable.
SUBJECT_TIER_ASSIGNMENT = "tier_assignment"

#: Values a receipt may carry beside its ids. Ids, enum names, counts and
#: flags — never block text. Same rule as ``event_fanout.scrub_payload``,
#: for the same reason: these rows are read by tooling well outside the
#: admission gate, so a receipt that quoted a block would be a content
#: egress the quarantine cannot see.
DetailValue = Union[str, int, float, bool]


def lifecycle_evidence_enabled(workspace: Union[str, Path]) -> bool:
    """True only when *workspace* has opted in explicitly.

    Exactly one value opts in — ``{"lifecycle_evidence": {"enabled":
    true}}``, the literal boolean. A missing file, a missing section, a
    missing key, a non-dict section, the string ``"true"`` and the
    integer ``1`` all mean OFF, so a workspace cannot start emitting a
    verb an older reader chokes on by typo.

    That is the reverse of :func:`mind_mem.served_ledger.ledger_enabled`,
    which is on unless a literal ``false`` opts out, and the asymmetry is
    the point: the served ledger's rows are new *files* that no other
    release parses, while these rows go into a chain older releases read
    strictly. See this module's header.

    Reads the config file directly rather than through the MCP config
    helper, following ``served_ledger``: this module must stay importable
    without pulling the server layer in behind it.
    """
    path = os.path.join(str(workspace), "mind-mem.json")
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(config, dict):
        return False
    section = config.get(CONFIG_SECTION)
    if not isinstance(section, dict):
        return False
    return section.get("enabled") is True


class LifecycleRecorder:
    """Writes one lifecycle receipt into both ledgers, or nothing at all.

    Obtained from :meth:`for_workspace`, which answers ``None`` for a
    workspace that has not opted in — so a caller holds either a live
    recorder or nothing, and there is no "enabled" branch to forget on
    the per-block path.
    """

    __slots__ = ("_audit", "_evidence", "_workspace")

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace
        # Built on the first row, not here: constructing either chain
        # creates its directory, and a sweep that finds nothing to demote
        # must leave a workspace exactly as it found it.
        self._evidence: Optional["EvidenceChain"] = None
        self._audit: Optional["AuditChain"] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def for_workspace(cls, workspace: Optional[str]) -> Optional["LifecycleRecorder"]:
        """A recorder for *workspace*, or ``None`` when nothing should be written.

        Resolve this ONCE per sweep and hold the result; the flag read is
        a file read, and doing it per block would put a config parse on
        the decay path. ``None`` for an absent workspace **without
        touching the disk** — a caller addressed only by database path
        cannot name a workspace, and guessing one would be worse than
        recording nothing.
        """
        if not workspace:
            return None
        if not lifecycle_evidence_enabled(workspace):
            return None
        return cls(workspace)

    # ------------------------------------------------------------------
    # The door
    # ------------------------------------------------------------------

    def record(
        self,
        verb: str,
        block_id: str,
        *,
        subject: str,
        door: str,
        actor: str = "system",
        target_file: str = "",
        reason: str = "",
        detail: Optional[Mapping[str, DetailValue]] = None,
    ) -> bool:
        """Record one lifecycle loss. Returns True when both rows landed.

        Args:
            verb: One of :data:`OPERATION_FOR_VERB`'s keys.
            block_id: The block whose standing, surface or record was lost.
            subject: :data:`SUBJECT_BLOCK` or
                :data:`SUBJECT_TIER_ASSIGNMENT` — what the verb is true
                *of*. A ``FORGET`` without it would overstate.
            door: ``module.function`` of the caller, so an auditor can
                get from a row back to the code that minted it.
            actor: Who asked. ``"system"`` for a scheduled sweep, which
                has no human identity to claim.
            target_file: Relative path the block lives in *after* the
                transition, when there is one.
            reason: Free-text justification recorded in the sidecar row.
            detail: Ids, enum names, counts and flags only.

        Never raises: see this module's header for why a post-hoc receipt
        must not be able to abort the sweep that earned it.
        """
        operation = OPERATION_FOR_VERB.get(verb)
        if operation is None:
            # Not a lifecycle verb. Refused rather than recorded under a
            # borrowed operation, for the reason ``_map_action`` refuses
            # an unclassified gate verb: a row nobody chose the label for
            # is a claim nobody made.
            _log.warning("lifecycle_verb_unknown", verb=str(verb), block_id=block_id)
            metrics.inc("lifecycle_evidence_write_failed")
            return False

        payload: dict[str, Any] = {
            "action": EVIDENCE_ACTION_FOR_VERB[verb].value,
            "block_id": block_id,
            "subject": subject,
            "door": door,
        }
        payload.update(dict(detail or {}))

        metadata: dict[str, Any] = {"subject": subject, "door": door, LIFECYCLE_VERB_KEY: verb}
        metadata.update(dict(detail or {}))
        # The link back to the chain of record, read from the ambient
        # scope exactly as ``AuditChain.append`` reads it: an archive
        # receipt is minted inside the batch scope that authorised the
        # move, and a tier receipt is minted outside any scope and simply
        # has no link to name. A ContextVar read — no syscall, no parse.
        admission_entry_id = _current_admission_entry_id()
        if admission_entry_id is not None:
            metadata["admission_entry_id"] = admission_entry_id

        try:
            self._evidence_chain().create(
                action=EVIDENCE_ACTION_FOR_VERB[verb],
                actor=actor,
                target_block_id=block_id,
                target_file=target_file,
                payload=payload,
                metadata=metadata,
            )
            self._append_sidecar(verb, block_id, agent=actor, reason=reason, payload=payload)
        except Exception as exc:
            # The transition already landed. Reporting the miss is the
            # only honest outcome left; hiding it would recreate the
            # silence this module exists to end.
            _log.warning(
                "lifecycle_evidence_write_failed",
                action=EVIDENCE_ACTION_FOR_VERB[verb].value,
                block_id=block_id,
                door=door,
                error=str(exc),
            )
            metrics.inc("lifecycle_evidence_write_failed")
            return False

        metrics.inc("lifecycle_evidence_recorded")
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evidence_chain(self) -> "EvidenceChain":
        if self._evidence is None:
            from .evidence_objects import EvidenceChain

            self._evidence = EvidenceChain(store_path=os.path.join(self._workspace, "memory", "evidence_chain.jsonl"))
        return self._evidence

    def _append_sidecar(
        self,
        verb: str,
        block_id: str,
        *,
        agent: str,
        reason: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Append the sidecar row for *action*, one branch per verb.

        Two shapes here are load-bearing rather than stylistic, and both
        are dictated by the gate in ``tests/test_ledger_hierarchy.py``,
        which re-derives ``audit_chain.VALID_OPERATIONS`` from this source
        by AST so the sidecar cannot advertise a verb no door writes:

        * the chain is appended to through ``self._audit``, the attribute
          the scanner saw bound to ``AuditChain(...)``. A local rebound
          from a helper's return value is not a receipt it can follow, so
          the writes would vanish from the derived set;
        * the verbs are named at the call sites instead of looked up in
          :data:`OPERATION_FOR_VERB`. A verb arriving through a dict is
          unresolvable, and the scanner fails the build rather than
          under-report — correctly, since a verb it cannot read is a verb
          the contract cannot be checked against.
        """
        if self._audit is None:
            from .audit_chain import AuditChain

            self._audit = AuditChain(self._workspace)
        # Dispatched explicitly, one literal per branch, rather than through
        # OPERATION_FOR_VERB. tests/test_ledger_hierarchy.py AST-scans this
        # source to re-derive VALID_OPERATIONS, so a verb reached only through
        # a dict lookup is invisible to it and the sidecar vocabulary silently
        # stops being checkable. Collapsing these three lines is a real
        # simplification and it costs that gate; the gate is worth more.
        if verb == DEMOTE_VERB:
            self._audit.append(DEMOTE_OPERATION, block_id, agent=agent, reason=reason, payload=dict(payload))
        elif verb == ARCHIVE_VERB:
            self._audit.append(ARCHIVE_OPERATION, block_id, agent=agent, reason=reason, payload=dict(payload))
        elif verb == FORGET_VERB:
            self._audit.append(FORGET_OPERATION, block_id, agent=agent, reason=reason, payload=dict(payload))
        else:  # pragma: no cover - unreachable: record() filters first
            raise ValueError(f"not a lifecycle verb: {verb!r}")


def _current_admission_entry_id() -> Optional[str]:
    """The open admission scope's chain entry id, or ``None``."""
    from .admission import current_admission

    receipt = current_admission()
    if receipt is None:
        return None
    entry_id = getattr(receipt, "entry_id", None)
    return entry_id if isinstance(entry_id, str) and entry_id else None


__all__ = [
    "ARCHIVE_OPERATION",
    "CONFIG_SECTION",
    "DEMOTE_OPERATION",
    "FORGET_OPERATION",
    "LIFECYCLE_OPERATIONS",
    "OPERATION_FOR_VERB",
    "EVIDENCE_ACTION_FOR_VERB",
    "LIFECYCLE_VERB_KEY",
    "DEMOTE_VERB",
    "ARCHIVE_VERB",
    "FORGET_VERB",
    "SUBJECT_BLOCK",
    "SUBJECT_TIER_ASSIGNMENT",
    "LifecycleRecorder",
    "lifecycle_evidence_enabled",
]

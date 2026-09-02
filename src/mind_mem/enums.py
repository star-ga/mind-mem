"""Centralised enum definitions for mind-mem.

Single source of truth for status strings and other closed-set values
duplicated across the codebase. Per the 2026-04-18 audit, the task
status literals (`todo`, `doing`, `blocked`, `done`, `canceled`) are
hardcoded in eight files — `sqlite_index.py`, `validate_py.py`,
`_recall_core.py`, `intel_scan.py`, `recall_vector.py`,
`_recall_constants.py`, `capture.py`, and the `validate.sh` bash
mirror. This module is the landing pad for those constants; the
call-site migration ships as a coordinated patch in v3.2.0.

All enums here inherit from ``str`` so they remain serialisation-
compatible with the on-disk Markdown schema. An existing string
``"todo"`` literal can be replaced with ``TaskStatus.TODO`` without
changing any file-format output or JSON payload.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional


class TaskStatus(str, Enum):
    """State machine for a Task block (``[T-YYYYMMDD-###]``).

    Lifecycle:

        TODO -> DOING -> DONE
           \\      ^       ^
            \\     |       |
             -> BLOCKED ---+
              \\
               -> CANCELED  (terminal)

    Transitions are enforced at the apply-engine level; the enum
    itself captures the universe of legal states.
    """

    TODO = "todo"
    DOING = "doing"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELED = "canceled"

    @classmethod
    def open(cls) -> frozenset["TaskStatus"]:
        """Statuses that still count as open loops."""
        return frozenset({cls.TODO, cls.DOING, cls.BLOCKED})

    @classmethod
    def closed(cls) -> frozenset["TaskStatus"]:
        """Terminal statuses."""
        return frozenset({cls.DONE, cls.CANCELED})

    def is_open(self) -> bool:
        return self in self.open()

    def is_closed(self) -> bool:
        return self in self.closed()


# ---------------------------------------------------------------------------
# Servability — the allow-list on block Status
# ---------------------------------------------------------------------------


class Status(str, Enum):
    """Block ``Status`` values the governed write paths can *mint*.

    Deliberately not a catalogue of every status string a live corpus
    holds. Two dozen distinct values exist there, several of them
    :class:`TaskStatus` members written into a Task block's ``Status``
    field, and folding them in here would settle that collision by
    accident. This enum covers exactly the statuses :data:`INITIAL_STATUS`
    decides, so the admission table is closed over a closed set.

    Servability does not wait on that migration: :func:`is_servable` is
    total over arbitrary strings, and everything it has not been told
    about is withheld.

    deferred: reconcile ``_recall_constants.VALIDITY_STATUS_DEAD`` /
    ``apply_engine.VALID_STATUSES`` / ``contradiction_detector.
    COMMITTED_STATUSES`` into this enum — upgrade path: fold them in once
    the ``TaskStatus``-vs-block-``Status`` overlap on ``todo`` / ``doing``
    / ``done`` has an owner.
    """

    #: Servable. Reachable only by applying an approved proposal.
    ACTIVE = "active"
    #: Withheld until a governance release moves it to :attr:`ACTIVE`.
    QUARANTINED = "quarantined"
    #: Withheld: an auto-captured signal nobody has reviewed yet.
    PENDING = "pending"
    #: An unresolved detector finding (a contradiction, a drift signal).
    #:
    #: NOT a quarantine state and NOT servable in the :data:`SERVABLE`
    #: sense: ``is_servable(OPEN)`` is False, so no tier minting it can
    #: reach ``admit_proposal``'s privilege. It is nonetheless a status
    #: recall RECOGNISES (``admissibility.RECOGNISED_STATUSES``), because
    #: a contradiction nobody can recall is a finding the product never
    #: made. The tier that mints it is confined by
    #: :data:`TIER_ID_PREFIXES` to the two corpora those findings live
    #: in, so "recall-visible" is bought with a prefix allow-list rather
    #: than with trust.
    OPEN = "open"


#: The allow-list. A block is servable only when its status is named here.
#:
#: This is an allow-list rather than a deny-list on "quarantined", and the
#: inversion is the point: the old rule served every status it had never
#: heard of, so a new ingest door minting ``Status: staged`` was served by
#: default. Anything nobody has named is now withheld by default.
SERVABLE: frozenset[Status] = frozenset({Status.ACTIVE})


def is_servable(status: object) -> bool:
    """True when *status* names a servable block state.

    Total by construction and fail-closed. ``None``, a non-string, a
    status a future door invents, an empty field — all withheld. Spelling
    is normalised (case, surrounding space) because a corpus really does
    hold ``Active`` alongside ``active``, and they are one state.
    """
    if isinstance(status, Status):
        return status in SERVABLE
    if not isinstance(status, str):
        return False
    normalised = status.strip().lower()
    return any(normalised == member.value for member in SERVABLE)


# ---------------------------------------------------------------------------
# Retrieval legs — the closed set of paths a candidate can enter recall by
# ---------------------------------------------------------------------------


class Leg(str, Enum):
    """A retrieval path that can put a candidate in front of the caller.

    Closed on purpose. The admissibility gate is only as complete as this
    vocabulary: the quarantine acceptance test parametrises over
    ``list(Leg)``, so a member added here without a fixture fails the
    suite rather than quietly leaving a path untested.

    ``hybrid`` is deliberately absent. It is a *fusion mode* — the label
    the attestation uses when the lexical and vector legs both ran — not
    a way for a candidate to arrive, and putting it here would generate a
    parametrisation row that corresponds to no retrieval path.

    :attr:`GRAPH`, :attr:`KG` and :attr:`ENTITY_PREFETCH` are the three
    legs that splice raw parsed blocks into the result list. They are
    named here because the previous vocabulary knew only about ``bm25``
    and ``vector``, and a gate that cannot name a leg cannot test it.
    """

    #: Lexical retrieval — FTS5 index or the in-memory BM25 scan.
    BM25 = "bm25"
    #: Dense retrieval over the embedding index.
    VECTOR = "vector"
    #: Cross-reference walk from the ranked seeds.
    GRAPH = "graph"
    #: Typed knowledge-graph edge walk from the query's entities.
    KG = "kg"
    #: Entity-tier prefetch plus its one-hop neighbourhood.
    ENTITY_PREFETCH = "entity_prefetch"


# ---------------------------------------------------------------------------
# Ingest tiers — the closed set of sources a governed write arrives from
# ---------------------------------------------------------------------------


class IngestTier(str, Enum):
    """Where an admitted write came from. Closed by construction.

    ``AdmissionReceipt.tier`` is required with no default, so a new ingest
    source that has not been given a tier here cannot obtain a receipt,
    and ``BlockStore.write_block`` refuses it. That is the door this enum
    exists to close: not "remember to quarantine", but "unrepresentable".

    Values match what is already written to disk where an equivalent
    string exists (``external-ingest`` is ``provenance_class.
    EXTERNAL_INGEST``, the value the ``IngestTier:`` block field already
    carries), so nothing in a live corpus has to be re-stamped.
    """

    #: Bulk importer and the inbox drop folder — untrusted external input.
    EXTERNAL_INGEST = "external-ingest"
    #: A peer agent's message. Untrusted: see :data:`INITIAL_STATUS`.
    AGENT_MESSAGE = "agent-message"
    #: Auto-captured signals mined out of the daily log.
    AUTO_CAPTURE = "auto-capture"
    #: Re-stamping blocks already in the store. Carries; mints nothing.
    RESTAMP = "restamp"
    #: Operator copy of an already-governed corpus between backends.
    STORE_MIGRATION = "store-migration"
    #: Applying an approved proposal. The only tier that reaches ACTIVE.
    PROPOSAL_APPLY = "proposal-apply"
    #: The integrity scanner's own findings — contradictions and drift.
    #:
    #: Derived content: every input is a block already admitted to the
    #: corpus, so the quarantine axis (which exists for untrusted INPUT)
    #: has nothing to say about it. What the gate buys here is the
    #: receipt and the chain row, not withholding. Confined by
    #: :data:`TIER_ID_PREFIXES` so it can write nothing but a finding.
    DETECTOR_FINDING = "detector-finding"


#: The **only** place an initial status is decided.
#:
#: ``None`` names a *carrying* tier: it rewrites blocks that were already
#: admitted once and mints no status of its own, so there is no honest
#: value to put here. A carrying tier is still barred from minting
#: ``ACTIVE`` in the sense that matters — it cannot raise a block's status,
#: only preserve it.
#:
#: OPERATOR DECISION (this change): ``AGENT_MESSAGE`` arrives
#: :attr:`Status.QUARANTINED`, reversing what shipped. A peer agent is the
#: standard prompt-injection carrier, and a single-operator model does not
#: make its *inputs* trusted. The consequence is deliberate: an agent
#: message is no longer recallable until a governance release admits it.
INITIAL_STATUS: Mapping[IngestTier, Optional[Status]] = MappingProxyType(
    {
        IngestTier.EXTERNAL_INGEST: Status.QUARANTINED,
        IngestTier.AGENT_MESSAGE: Status.QUARANTINED,
        IngestTier.AUTO_CAPTURE: Status.PENDING,
        IngestTier.RESTAMP: None,
        IngestTier.STORE_MIGRATION: None,
        IngestTier.PROPOSAL_APPLY: Status.ACTIVE,
        IngestTier.DETECTOR_FINDING: Status.OPEN,
    }
)


#: Block-id prefixes a tier may mint for. A tier listed here is
#: **confined**: :func:`~mind_mem.admission.require_admission` refuses its
#: receipt for any id outside the set, and refuses any status but the one
#: its :data:`INITIAL_STATUS` row names.
#:
#: This is what makes :attr:`IngestTier.DETECTOR_FINDING` narrow enough to
#: mint a status recall recognises. Unconfined tiers are absent from this
#: table and keep the general rule (a withheld-minting tier may not carry
#: in anything recall would serve); a confined tier trades that blanket
#: rule for a much smaller reach — two corpora, one status, no choice.
#:
#: Keep in step with ``block_store._BLOCK_PREFIX_MAP``: a prefix here that
#: the store cannot route is a tier that can write nothing at all
#: (pinned by ``tests/test_governed_detector_writes.py``).
TIER_ID_PREFIXES: Mapping[IngestTier, frozenset[str]] = MappingProxyType(
    {
        IngestTier.DETECTOR_FINDING: frozenset({"C", "DREF"}),
    }
)


def is_confined(tier: IngestTier) -> bool:
    """True when *tier* may only write ids with a named prefix.

    Derived from :data:`TIER_ID_PREFIXES` rather than hand-listed, so a
    new confined tier is classified the moment it has a row.
    """
    return tier in TIER_ID_PREFIXES


def mints_quarantine(tier: IngestTier) -> bool:
    """True when *tier*'s row means "withheld until governance admits it".

    The distinction :data:`INITIAL_STATUS` alone cannot draw. A row is a
    quarantine marker when the tier is an *input* door: an unconfined
    tier whose status is not servable. A **confined** tier's row is the
    lifecycle state of one corpus (an ``open`` contradiction has passed
    the gate; it is unresolved, not unadmitted), so it is not evidence
    that a block skipped governance and must not withhold every other
    block that happens to share the spelling.

    ``admissibility.UNADMITTED`` derives from this, so a new *input* tier
    still withholds its content with no edit anywhere else.
    """
    row = INITIAL_STATUS[tier]
    return row is not None and not is_servable(row) and not is_confined(tier)


def mints_servable(tier: IngestTier) -> bool:
    """True when *tier* mints a status that recall will serve.

    Derived from :data:`INITIAL_STATUS`, never hand-listed, so a new row
    is classified the moment it exists. ``GovernanceGate`` uses this to
    refuse a servable tier from ``admit_block`` / ``admit_batch``.
    """
    return is_servable(INITIAL_STATUS[tier])


__all__ = [
    "INITIAL_STATUS",
    "SERVABLE",
    "TIER_ID_PREFIXES",
    "IngestTier",
    "Leg",
    "Status",
    "TaskStatus",
    "is_confined",
    "is_servable",
    "mints_quarantine",
    "mints_servable",
]

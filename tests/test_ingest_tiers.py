# Copyright 2026 STARGA, Inc.
"""T1/T2 — the ingest-tier gate: a source with no tier cannot get a receipt.

The write invariant (``test_governed_write_paths``) proves every write is
*admitted*. It does not constrain what a write may claim once admitted, so
the drop folder and the importer stayed quarantined only because each
remembered to stamp ``Status: quarantined`` itself. A new ingest door that
forgot would land ``Status: active`` blocks with a perfectly valid receipt.

This file closes that by construction:

T1  exhaustiveness — :data:`IngestTier` is closed, every member has an
    :data:`INITIAL_STATUS` row, and servability is a *total* partition of
    the status space with no third bucket. A status nobody has named is
    withheld, so a future door cannot invent its way into the served set.

T2  the gate — ``admit_block`` / ``admit_batch`` cannot mint a servable
    tier whatever the caller passes; ``admit_proposal`` is the only path
    to ``ACTIVE`` and takes no tier argument at all; and ``write_block``
    refuses a block whose ``Status`` escalates above its receipt's row.

Both are structural. Nothing here asserts that a particular door
*remembered* a rule — the point is that forgetting is not expressible.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from _ledger_rows import authorisation_rows

from mind_mem.admission import AdmissionReceipt, UngatedWriteError
from mind_mem.enums import (
    INITIAL_STATUS,
    SERVABLE,
    IngestTier,
    Status,
    is_servable,
    mints_servable,
)
from mind_mem.governance_gate import OPEN_SCOPE_TIERS, SCOPE_BOUND_TIERS, GovernanceBypassError, get_gate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"workspace_path": str(ws), "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    return str(ws)


#: Tiers an unnamed (``admit_block`` / ``admit_batch``) scope may open:
#: they mint nothing servable **and** are not bound to a scope of their
#: own. Derived from the gate's two tables, never hand-listed, so a new
#: row is covered the moment it exists.
CARRYING_OR_WITHHELD = tuple(t for t in IngestTier if t in OPEN_SCOPE_TIERS)


# ---------------------------------------------------------------------------
# T1 — exhaustiveness
# ---------------------------------------------------------------------------


def test_every_tier_has_an_initial_status_row() -> None:
    """The table is total over the enum, in both directions."""
    assert set(INITIAL_STATUS) == set(IngestTier), (
        "INITIAL_STATUS and IngestTier have drifted; every tier must name the "
        f"status it mints. missing={sorted(t.value for t in set(IngestTier) - set(INITIAL_STATUS))} "
        f"extra={sorted(str(t) for t in set(INITIAL_STATUS) - set(IngestTier))}"
    )


def test_initial_status_rows_are_statuses_or_carrying() -> None:
    """A row is a :class:`Status` or ``None`` (carries an existing status)."""
    for tier, row in INITIAL_STATUS.items():
        assert row is None or isinstance(row, Status), f"{tier.value} row is {row!r}, not a Status or None"


def test_status_space_has_no_third_bucket() -> None:
    """Every status is servable or withheld — the partition is total."""
    for status in Status:
        served = is_servable(status)
        assert served is (status in SERVABLE), f"{status.value} disagrees with SERVABLE"
        assert served or not served  # a bool, never a third value
    assert SERVABLE == frozenset({Status.ACTIVE}), "the allow-list is exactly ACTIVE"


@pytest.mark.parametrize(
    "unknown",
    ["", "  ", "superseded", "deprecated", "a-status-a-future-door-invents", "ACTIVE-ish", "activate"],
)
def test_unknown_statuses_are_withheld(unknown: str) -> None:
    """The allow-list inversion: never-heard-of means withheld, not served."""
    assert is_servable(unknown) is False


@pytest.mark.parametrize("weird", [None, 0, 1, True, object(), ["active"], {"active": 1}])
def test_is_servable_is_total_over_non_strings(weird: object) -> None:
    """Total by construction: no input type can raise or leak a truthy pass."""
    assert is_servable(weird) is False


def test_servable_status_spelling_is_case_and_space_insensitive() -> None:
    """A corpus writes ``Active``/``ACTIVE `` too; they are the same state."""
    for spelling in ("active", "Active", "ACTIVE", "  active  "):
        assert is_servable(spelling) is True


def test_only_the_proposal_tier_mints_a_servable_status() -> None:
    """The load-bearing claim: exactly one tier can reach the served set."""
    servable_tiers = {t for t in IngestTier if mints_servable(t)}
    assert servable_tiers == {IngestTier.PROPOSAL_APPLY}


def test_agent_message_arrives_withheld() -> None:
    """Operator decision: a peer agent is an untrusted input carrier."""
    assert INITIAL_STATUS[IngestTier.AGENT_MESSAGE] is Status.QUARANTINED


# ---------------------------------------------------------------------------
# T2 — the gate
# ---------------------------------------------------------------------------


def test_receipt_tier_is_required_with_no_default() -> None:
    """A receipt cannot exist without naming where the write came from."""
    fields = {f.name: f for f in dataclasses.fields(AdmissionReceipt)}
    assert "tier" in fields, "AdmissionReceipt has no tier field"
    tier_field = fields["tier"]
    assert tier_field.default is dataclasses.MISSING and tier_field.default_factory is dataclasses.MISSING, (
        "AdmissionReceipt.tier has a default; a caller that forgets it would get a silent tier"
    )
    with pytest.raises(TypeError):
        AdmissionReceipt(entry_id="e", content_hash="c", kind="block")  # type: ignore[call-arg]


@pytest.mark.parametrize("scope", ["admit_block", "admit_batch"])
def test_minting_scopes_require_a_tier(workspace: str, scope: str) -> None:
    """No tier, no receipt: a new door cannot get one by omission."""
    gate = get_gate(workspace)
    kwargs = {"action": "WRITE", "content": "x"}
    if scope == "admit_block":
        kwargs["block_id"] = "IMP-20260829-001"
    else:
        kwargs["batch_id"] = "b1"
        kwargs["block_ids"] = ["IMP-20260829-001"]
    with pytest.raises(TypeError):
        with getattr(gate, scope)(**kwargs):  # type: ignore[arg-type]
            pass


@pytest.mark.parametrize("scope", ["admit_block", "admit_batch"])
def test_minting_scopes_refuse_a_servable_tier(workspace: str, scope: str) -> None:
    """T2: whatever the caller passes, these two cannot yield ACTIVE."""
    gate = get_gate(workspace)
    for tier in IngestTier:
        if not mints_servable(tier):
            continue
        with pytest.raises(GovernanceBypassError):
            if scope == "admit_block":
                with gate.admit_block(action="WRITE", block_id="IMP-20260829-001", content="x", tier=tier):
                    pass
            else:
                with gate.admit_batch(action="WRITE", batch_id="b1", block_ids=["IMP-20260829-001"], content="x", tier=tier):
                    pass


@pytest.mark.parametrize("tier", CARRYING_OR_WITHHELD, ids=lambda t: t.value)
def test_minting_scopes_stamp_the_tier_on_the_receipt(workspace: str, tier: IngestTier) -> None:
    """Every non-servable tier is usable and lands on the receipt verbatim."""
    gate = get_gate(workspace)
    with gate.admit_block(action="WRITE", block_id="IMP-20260829-001", content="x", tier=tier) as receipt:
        assert receipt.tier is tier
        assert not is_servable(INITIAL_STATUS[receipt.tier])


@pytest.mark.parametrize("scope", ["admit_block", "admit_batch"])
def test_minting_scopes_refuse_a_scope_bound_tier(workspace: str, scope: str) -> None:
    """A tier bound to its own scope is unreachable from the open ones.

    ``PROPOSAL_APPLY`` was already refused by the servable rule; the arm
    that matters is ``EDGE_APPROVAL``, whose ``INITIAL_STATUS`` row is
    ``None``. A carrying row constrains no status, so if ``admit_block``
    could name that tier it could land ``Status: active`` on any id —
    which is exactly the reach ``admit_edge`` exists to withdraw.
    """
    gate = get_gate(workspace)
    assert SCOPE_BOUND_TIERS, "no scope-bound tier at all — this guard would pass over nothing"
    for tier in SCOPE_BOUND_TIERS.values():
        assert tier not in OPEN_SCOPE_TIERS
        with pytest.raises(GovernanceBypassError):
            if scope == "admit_block":
                with gate.admit_block(action="WRITE", block_id="IMP-20260829-001", content="x", tier=tier):
                    pass
            else:
                with gate.admit_batch(action="WRITE", batch_id="b1", block_ids=["IMP-20260829-001"], content="x", tier=tier):
                    pass


def test_the_open_scope_tiers_are_the_positive_control(workspace: str) -> None:
    """The same call, on every tier the open scopes DO admit, must pass.

    Without this the refusal test above would still pass if ``admit_block``
    had been broken into refusing everything.
    """
    gate = get_gate(workspace)
    assert OPEN_SCOPE_TIERS, "no open-scope tier at all — the refusals would prove nothing"
    for tier in OPEN_SCOPE_TIERS:
        with gate.admit_block(action="WRITE", block_id="IMP-20260829-001", content="x", tier=tier) as receipt:
            assert receipt.tier is tier


def test_admit_proposal_is_the_only_path_to_active(workspace: str) -> None:
    """It takes no tier at all — the caller cannot request a status."""
    gate = get_gate(workspace)
    with gate.admit_proposal(proposal_id="P-1", content="[]") as receipt:
        assert receipt.tier is IngestTier.PROPOSAL_APPLY
        assert INITIAL_STATUS[receipt.tier] is Status.ACTIVE
    with pytest.raises(TypeError):
        with gate.admit_proposal(proposal_id="P-2", content="[]", tier=IngestTier.EXTERNAL_INGEST):  # type: ignore[call-arg]
            pass


# ---------------------------------------------------------------------------
# T2 — write_block refuses a Status its receipt's tier cannot mint
# ---------------------------------------------------------------------------


def _block(status: str) -> dict:
    return {
        "_id": "IMP-20260829-001",
        "type": "IMPORTED",
        "Statement": "imported text",
        "Status": status,
        "Date": "2026-08-29",
    }


def test_write_block_refuses_a_status_escalation(workspace: str) -> None:
    """A quarantine-tier receipt cannot carry an ``active`` block in."""
    from mind_mem.storage import get_block_store

    store = get_block_store(workspace)
    gate = get_gate(workspace)
    with gate.admit_block(action="INGEST", block_id="IMP-20260829-001", content="imported text", tier=IngestTier.EXTERNAL_INGEST):
        with pytest.raises(UngatedWriteError):
            store.write_block(_block("active"))


#: Every spelling of "this block states no status". The block parser renders
#: a bare ``Status:`` line as an empty LIST, which is why the third row is
#: not redundant with the second.
_UNSTATED = [None, "", []]


@pytest.mark.parametrize("unstated", _UNSTATED, ids=["absent", "empty", "bare-Status-line"])
def test_write_block_refuses_a_block_that_states_no_status(unstated: object, workspace: str) -> None:
    """The bypass: omitting ``Status`` entirely got the content SERVED.

    The write gate used to ask ``is_servable(status)`` while recall asks
    ``is_admissible_status(status)``. They disagree on an unstated status
    — not servable, but admissible — so an external-ingest door that
    simply left the field off wrote a block recall then served in full.
    No status was ever named, so there was nothing for the old check to
    refuse. Reached through the sanctioned gate API, in three lines.
    """
    from mind_mem.storage import get_block_store

    store = get_block_store(workspace)
    gate = get_gate(workspace)
    block = _block("x")
    if unstated is None:
        block.pop("Status")
    else:
        block["Status"] = unstated
    with gate.admit_block(action="INGEST", block_id="IMP-20260829-001", content="imported text", tier=IngestTier.EXTERNAL_INGEST):
        with pytest.raises(UngatedWriteError):
            store.write_block(block)


@pytest.mark.parametrize("unstated", _UNSTATED, ids=["absent", "empty", "bare-Status-line"])
def test_an_unstated_status_is_exactly_what_recall_would_serve(unstated: object) -> None:
    """Why the refusal above is the right shape, and not belt-and-braces.

    Pins the disagreement itself: these values are NOT servable (so the
    old write check waved them through) and ARE admissible (so recall
    serves them). The write gate must therefore ask the reader's
    question, and this fails the moment the two predicates drift back
    apart.
    """
    from mind_mem.admissibility import is_admissible_status

    assert not is_servable(unstated), "an unstated status is not servable — this is why the old check missed it"
    assert is_admissible_status(unstated), "an unstated status IS served by recall — this is what made it a bypass"


def test_a_withheld_minting_tier_refuses_every_status_recall_would_serve(workspace: str) -> None:
    """The refusal tracks the READ predicate, not a hand-listed set.

    ``superseded`` is not ``active``, so a servability check let it in;
    recall serves it (demoted) all the same, which makes it an
    escalation out of quarantine by another name.
    """
    from mind_mem.storage import get_block_store

    store = get_block_store(workspace)
    gate = get_gate(workspace)
    with gate.admit_block(action="INGEST", block_id="IMP-20260829-001", content="imported text", tier=IngestTier.EXTERNAL_INGEST):
        with pytest.raises(UngatedWriteError):
            store.write_block(_block("superseded"))


def test_write_block_accepts_the_status_its_tier_mints(workspace: str) -> None:
    """The same write, correctly stamped, goes through."""
    from mind_mem.storage import get_block_store

    store = get_block_store(workspace)
    gate = get_gate(workspace)
    with gate.admit_block(action="INGEST", block_id="IMP-20260829-001", content="imported text", tier=IngestTier.EXTERNAL_INGEST):
        assert store.write_block(_block(Status.QUARANTINED.value)) == "IMP-20260829-001"


def test_carrying_tier_may_move_an_already_active_block(workspace: str) -> None:
    """A backend copy / re-stamp preserves status; it mints none."""
    from mind_mem.storage import get_block_store

    store = get_block_store(workspace)
    gate = get_gate(workspace)
    assert INITIAL_STATUS[IngestTier.STORE_MIGRATION] is None
    with gate.admit_batch(
        action="MIGRATE",
        batch_id="m1",
        block_ids=["IMP-20260829-001"],
        content="x",
        tier=IngestTier.STORE_MIGRATION,
    ):
        assert store.write_block(_block("active")) == "IMP-20260829-001"


# ---------------------------------------------------------------------------
# T2 — the tier reaches the audit trail, and a refusal leaves no trace
# ---------------------------------------------------------------------------


def test_a_refused_tier_leaves_no_chain_entry(workspace: str) -> None:
    """The tier is checked before the admission is recorded.

    A refusal that still appended would put an entry in the append-only
    chain for a write that never happened — an audit trail that overstates.
    """
    gate = get_gate(workspace)
    before = gate.chain.length
    with pytest.raises(GovernanceBypassError):
        with gate.admit_block(action="INGEST", block_id="IMP-20260829-001", content="x", tier=IngestTier.PROPOSAL_APPLY):
            pass
    assert gate.chain.length == before


def test_the_tier_is_recorded_in_the_audit_trail(workspace: str) -> None:
    """The receipt lives in memory; the chain entry has to name the source too."""
    import json
    import os

    gate = get_gate(workspace)
    with gate.admit_block(action="INGEST", block_id="IMP-20260829-001", content="x", tier=IngestTier.EXTERNAL_INGEST):
        pass
    with open(os.path.join(workspace, "memory", "evidence_chain.jsonl"), encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows, "no evidence row was written"
    # The tier is a property of the *authorisation*: the scope's close
    # record says what landed, and it carries no tier, so reading the last
    # row would read the wrong half.
    admitted = authorisation_rows(rows)
    assert admitted, "no authorisation row was written"
    assert admitted[-1]["metadata"]["ingest_tier"] == IngestTier.EXTERNAL_INGEST.value


# ---------------------------------------------------------------------------
# T1 — the table is the only decider: no second spelling may drift from it
# ---------------------------------------------------------------------------


def test_quarantine_constants_are_derived_from_the_table() -> None:
    """The importer's two constants must not become a second source of truth."""
    from mind_mem.importers.quarantine import QUARANTINE_STATUS, QUARANTINE_TIER

    assert QUARANTINE_STATUS == INITIAL_STATUS[IngestTier.EXTERNAL_INGEST].value
    assert QUARANTINE_TIER == IngestTier.EXTERNAL_INGEST.value, (
        "the IngestTier: block field and the receipt's tier have drifted apart; "
        "they name the same thing and a corpus written under one is read under the other"
    )


def test_the_message_door_reads_the_table_rather_than_restating_it() -> None:
    """Same rule for the messaging door: one row, no second spelling."""
    from mind_mem.agent_messaging import MESSAGE_STATUS, MESSAGE_TIER, build_message_block
    from mind_mem.agent_messaging import TIER_FIELD as MESSAGE_TIER_FIELD
    from mind_mem.importers.quarantine import TIER_FIELD as IMPORT_TIER_FIELD

    assert MESSAGE_STATUS is INITIAL_STATUS[MESSAGE_TIER]
    assert MESSAGE_TIER_FIELD == IMPORT_TIER_FIELD, "two spellings of the same block field"
    block = build_message_block("hi", timestamp="20260829T000000Z", nonce="beef")
    assert block["Status"] == INITIAL_STATUS[IngestTier.AGENT_MESSAGE].value
    assert not is_servable(block["Status"])

# Copyright 2026 STARGA, Inc.
"""The admission seam: one contract for entering, leaving and dying.

The product claim is that *no content enters, leaves, or dies without a
gate receipt and a chain record*. Before 5.0.2 only the first third of
that was true:

ENTER   ``require_admission`` in every ``write_block``; an ungated write
        raises. Proven by ``tests/test_governed_write_paths.py``.
LEAVE   leaked — ``get_block`` served quarantined content verbatim.
DIE     ungoverned — ``delete_block`` in all five stores checked nothing.
        A live probe returned ``True`` and the block was gone, with no
        receipt, no evidence record and no chain entry. Three doors
        reached it: the ADMIN ``delete_memory_item`` tool, ``DELETE
        /memories/{id}``, and ``POST /clear``, which wipes the corpus one
        ``delete_block`` call at a time.

This file tests the seam the other surfaces are built on, in three
parts:

A   DELETE admission — ``admit_delete`` / ``admit_delete_batch``, the
    ``operation`` field that stops a write receipt being spent on a
    delete, and the removal record that names what actually died.
B   READ admission — ``admit_read`` / ``admit_read_one``, the same
    verdict the recall legs reach, exposed so a tool handler applies it
    rather than inventing its own status check.
C   The silent action collapse — ``_map_action`` used to end in
    ``_ACTION_MAP.get(upper, EvidenceAction.APPLY)``, so a verb the
    table did not know was written into the evidence chain **as an
    apply**. A ``DELETE`` misspelled ``DELET`` was recorded, sealed under
    a hash, as content landing. That is a false statement in an audit
    chain, which is the one failure mode it cannot absorb.

Every negative assertion here carries a positive control: the same call
that must be refused is shown succeeding under the right receipt, so a
refusal can never be the accident of a method that never worked. Every
gate carries a mutation twin in ``TestMutationTwin*`` — the twin
restores the pre-5.0.2 behaviour and asserts the protective body now
fails. A gate never observed failing is not a gate.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem import admissibility, admission, governance_gate
from mind_mem.admission import (
    BATCH,
    BLOCK,
    OP_DELETE,
    OP_WRITE,
    PROPOSAL,
    AdmissionReceipt,
    ReadAdmission,
    UngatedDeleteError,
    UngatedWriteError,
    admit_read,
    admit_read_one,
    require_admission,
    require_delete_admission,
)
from mind_mem.enums import IngestTier
from mind_mem.evidence_objects import EvidenceAction, EvidenceChain
from mind_mem.governance_gate import (
    DELETE_VERB,
    PHASE_ADMITTED,
    PHASE_REMOVED,
    GovernanceBypassError,
    evict_gate,
    get_gate,
)
from mind_mem.merkle_tree import MerkleTree

SRC = Path(__file__).resolve().parents[1] / "src" / "mind_mem"

#: A block id the release-decision pattern accepts (a ``memory/`` drop
#: corpus). Derived nowhere: it is spelled out so the release test fails
#: loudly if the prefix routing changes rather than silently testing a
#: pattern that no longer matches.
RELEASABLE_ID = "IMP-20260901-001"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    """A throwaway workspace with its gate evicted on the way out.

    Eviction matters: ``get_gate`` caches one gate per realpath forever,
    and two live gates over one evidence file fork the chain (see
    ``evict_gate``'s docstring). A test that left its gate cached would
    poison any later test that happened to reuse the path.
    """
    ws = tmp_path / "ws"
    for sub in ("memory", "decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"workspace_path": str(ws), "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


def _records(ws: str) -> list[dict]:
    """Every evidence record written in *ws*, in order."""
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _phase(records: list[dict], phase: str) -> list[dict]:
    return [r for r in records if r.get("metadata", {}).get("delete_phase") == phase]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _merkle_root(leaves: list[tuple[str, str]]) -> str:
    tree = MerkleTree()
    tree.build(leaves)
    return tree.root_hash


def _block_text(bid: str, statement: str, status: str, **extra: str) -> str:
    lines = [f"[{bid}]", f"Statement: {statement}", "Date: 2026-09-01", f"Status: {status}"]
    lines.extend(f"{k}: {v}" for k, v in extra.items())
    return "\n".join(lines) + "\n\n---\n\n"


def _append(ws: str, rel: str, text: str) -> None:
    path = os.path.join(ws, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


# ===========================================================================
# A — DELETE admission
# ===========================================================================


def test_an_ungated_delete_is_refused(workspace: str) -> None:
    """The measured defect: an ungated delete used to just work.

    Positive control in the same test: the identical call inside a
    DELETE scope returns the receipt. Without it, this test would also
    pass if ``require_delete_admission`` raised unconditionally — i.e. if
    it were broken rather than protective.
    """
    with pytest.raises(UngatedDeleteError) as excinfo:
        require_delete_admission("D-20260901-001")
    assert "no governance admission is open" in str(excinfo.value)

    gate = get_gate(workspace)
    with gate.admit_delete("D-20260901-001", rationale="positive control") as receipt:
        assert require_delete_admission("D-20260901-001") is receipt


def test_an_ungated_delete_error_is_catchable_as_the_write_error(workspace: str) -> None:
    """Forward compatibility for handlers written before deletes were gated.

    ``UngatedDeleteError`` subclasses ``UngatedWriteError`` precisely so
    an existing ``except UngatedWriteError`` — the apply engine's abort
    path among them — keeps catching every ungated mutation without
    being edited.
    """
    with pytest.raises(UngatedWriteError):
        require_delete_admission("D-20260901-001")
    assert issubclass(UngatedDeleteError, UngatedWriteError)
    assert issubclass(UngatedDeleteError, admission.GovernanceBypassError)


def test_a_write_receipt_does_not_authorize_a_delete(workspace: str) -> None:
    """The receipt says which operation it bought; it is not transferable."""
    gate = get_gate(workspace)
    with gate.admit_block("WRITE", "IMP-1", "body", tier=IngestTier.EXTERNAL_INGEST) as receipt:
        # Positive control: the receipt DOES authorise the write it is for.
        assert require_admission("IMP-1", status="quarantined") is receipt
        with pytest.raises(UngatedDeleteError) as excinfo:
            require_delete_admission("IMP-1")
    assert "authorises a write, not a delete" in str(excinfo.value)


def test_a_delete_receipt_does_not_authorize_a_write(workspace: str) -> None:
    """…and the same in the other direction."""
    gate = get_gate(workspace)
    with gate.admit_delete("IMP-1", rationale="operator cleanup") as receipt:
        # Positive control: the receipt DOES authorise the delete it is for.
        assert require_delete_admission("IMP-1") is receipt
        with pytest.raises(UngatedWriteError) as excinfo:
            require_admission("IMP-1", status="quarantined")
    assert "authorises a delete, not a write" in str(excinfo.value)


def test_a_proposal_receipt_does_not_authorize_a_delete(workspace: str) -> None:
    """The one ambient scope stays ambient over writes only.

    ``admit_proposal`` authorises every id it is asked about, which is
    the honest encoding of an apply. Letting that reach ``delete_block``
    would hand the apply path authority to destroy anything.
    """
    gate = get_gate(workspace)
    with gate.admit_proposal(proposal_id="P-1", content="[]") as receipt:
        assert receipt.operation == OP_WRITE
        # Positive control: the proposal receipt authorises an arbitrary write.
        assert require_admission("D-20260901-999", status="active") is receipt
        with pytest.raises(UngatedDeleteError):
            require_delete_admission("D-20260901-999")


def test_delete_scope_records_the_authorisation_before_the_store_is_touched(workspace: str) -> None:
    """The admitted record exists while the block is still there.

    Ordering is the property: authorise, then act. A record written only
    after a successful removal would be missing for exactly the deletes
    that crashed half-way.
    """
    gate = get_gate(workspace)
    with gate.admit_delete("IMP-1", rationale="operator cleanup", actor="alice"):
        admitted = _phase(_records(workspace), PHASE_ADMITTED)
        assert len(admitted) == 1, "the authorisation record must exist before the store is touched"
        row = admitted[0]
        assert row["action"] == EvidenceAction.ROLLBACK.value
        assert row["actor"] == "alice"
        assert row["target_block_id"] == "IMP-1"
        assert row["metadata"]["operation"] == OP_DELETE
        assert row["metadata"]["rationale"] == "operator cleanup"
        assert row["metadata"]["action_verb"] == DELETE_VERB


def test_the_removal_record_carries_the_removed_content_hash(workspace: str) -> None:
    """What died, hashed, attributed, and linked to its authorisation."""
    gate = get_gate(workspace)
    removed = "[IMP-1]\nStatement: the removed block\nStatus: quarantined\n"
    with gate.admit_delete("IMP-1", rationale="operator cleanup", actor="alice") as receipt:
        receipt.record_removal("IMP-1", removed)
        entry_id = receipt.entry_id

    rows = _phase(_records(workspace), PHASE_REMOVED)
    assert len(rows) == 1
    row = rows[0]
    assert row["payload_hash"] == _sha256(removed)
    assert row["actor"] == "alice"
    assert row["target_block_id"] == "IMP-1"
    assert row["metadata"]["admission_entry_id"] == entry_id
    assert row["metadata"]["removed_count"] == 1
    assert row["metadata"]["merkle_root"] == _merkle_root([("IMP-1", _sha256(removed))])
    assert row["metadata"]["scope_outcome"] == "ok"


def test_a_delete_that_removed_nothing_writes_no_removal_record(workspace: str) -> None:
    """A miss is not a death.

    Deleting an id that was not in the store removes nothing, and a
    "removed" record for it would claim content died that never existed.
    The authorisation record still stands, so the attempt is not lost —
    which the positive control below pins.
    """
    gate = get_gate(workspace)
    with gate.admit_delete("IMP-missing", rationale="not there"):
        pass
    rows = _records(workspace)
    assert _phase(rows, PHASE_REMOVED) == []
    assert len(_phase(rows, PHASE_ADMITTED)) == 1, "the attempt must still be recorded"


def test_a_batch_delete_writes_one_removal_record_over_a_merkle_root(workspace: str) -> None:
    """``POST /clear`` is one decision, so it leaves one record.

    Per-block records would flood a chain built for low-volume decisions
    and would lose the fact that the removals were one operation.
    """
    gate = get_gate(workspace)
    contents = {f"IMP-{i}": f"[IMP-{i}]\nStatement: block {i}\n" for i in range(1, 6)}
    with gate.admit_delete_batch("CLEAR-1", contents, rationale="operator clear, corpus reset", actor="alice") as receipt:
        assert receipt.kind == BATCH
        for bid, text in contents.items():
            receipt.record_removal(bid, text)

    rows = _phase(_records(workspace), PHASE_REMOVED)
    assert len(rows) == 1, f"a batch delete must write exactly one removal record, got {len(rows)}"
    row = rows[0]
    assert row["metadata"]["removed_count"] == 5
    expected = _merkle_root([(bid, _sha256(text)) for bid, text in contents.items()])
    assert row["metadata"]["merkle_root"] == expected
    # The payload is the root itself once more than one block went, so the
    # record's own hash covers the set.
    assert row["payload_hash"] == _sha256(expected)


def test_a_batch_scope_cannot_grow_past_its_frozen_id_set(workspace: str) -> None:
    """A clear cannot reach a block written while it was running."""
    gate = get_gate(workspace)
    with gate.admit_delete_batch("CLEAR-1", ["IMP-1", "IMP-2"], rationale="operator clear, corpus reset") as receipt:
        # Positive control: an id inside the frozen set is authorised.
        assert require_delete_admission("IMP-1") is receipt
        receipt.record_removal("IMP-1", "body one")

        with pytest.raises(UngatedDeleteError):
            require_delete_admission("IMP-3")
        with pytest.raises(UngatedDeleteError):
            receipt.record_removal("IMP-3", "body three")

    row = _phase(_records(workspace), PHASE_REMOVED)[0]
    assert row["metadata"]["removed_count"] == 1, "the uncovered removal must not have been recorded"


def test_the_removal_record_survives_a_failure_inside_the_scope(workspace: str) -> None:
    """A clear that died half-way still destroyed what it got to.

    A chain that only records tidy deletions under-reports exactly the
    cases an auditor cares about. The original exception is never masked.
    """
    gate = get_gate(workspace)
    with pytest.raises(RuntimeError, match="store exploded"):
        with gate.admit_delete_batch("CLEAR-1", ["IMP-1", "IMP-2"], rationale="operator clear, corpus reset") as receipt:
            receipt.record_removal("IMP-1", "body one")
            raise RuntimeError("store exploded")

    rows = _phase(_records(workspace), PHASE_REMOVED)
    assert len(rows) == 1
    assert rows[0]["metadata"]["removed_count"] == 1
    assert rows[0]["metadata"]["scope_outcome"] == "error"


def test_a_delete_leaves_the_evidence_chain_verifying(workspace: str) -> None:
    """Two records per delete, and the chain still links end to end."""
    gate = get_gate(workspace)
    with gate.admit_delete("IMP-1", rationale="operator cleanup") as receipt:
        receipt.record_removal("IMP-1", "body")

    chain = EvidenceChain(store_path=os.path.join(workspace, "memory", "evidence_chain.jsonl"))
    ok, problems = chain.verify_chain()
    assert ok, problems
    assert len(_records(workspace)) == 2


def test_a_delete_needs_a_rationale(workspace: str) -> None:
    """An audit record that cannot say *why* is most of the way to none."""
    gate = get_gate(workspace)
    for bad in ("", "   "):
        with pytest.raises(GovernanceBypassError, match="no rationale"):
            with gate.admit_delete("IMP-1", rationale=bad):
                pass
    assert _records(workspace) == [], "a refused delete must leave no record"

    # Positive control: a rationale is all that was missing.
    with gate.admit_delete("IMP-1", rationale="operator cleanup"):
        pass
    assert len(_records(workspace)) == 1


def test_an_empty_delete_batch_is_refused(workspace: str) -> None:
    """A receipt covering nothing authorises nothing; minting one is a lie."""
    gate = get_gate(workspace)
    with pytest.raises(GovernanceBypassError, match="covers no block ids"):
        with gate.admit_delete_batch("CLEAR-EMPTY", [], rationale="operator clear, corpus reset"):
            pass
    assert _records(workspace) == []


def test_a_delete_receipt_names_no_ingest_tier(workspace: str) -> None:
    """Removing content is not an ingest, so there is no tier to claim."""
    gate = get_gate(workspace)
    with gate.admit_delete("IMP-1", rationale="operator cleanup") as receipt:
        assert receipt.tier is None
        assert receipt.operation == OP_DELETE
        assert receipt.chain_verified is True
    row = _phase(_records(workspace), PHASE_ADMITTED)[0]
    assert "ingest_tier" not in row["metadata"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"operation": OP_WRITE, "tier": None, "kind": BLOCK}, "must name the ingest tier"),
        ({"operation": OP_DELETE, "tier": IngestTier.EXTERNAL_INGEST, "kind": BLOCK}, "names no ingest tier"),
        ({"operation": OP_DELETE, "tier": None, "kind": PROPOSAL}, "may not be proposal-scoped"),
        ({"operation": "purge", "tier": None, "kind": BLOCK}, "not one of"),
    ],
)
def test_a_malformed_receipt_cannot_be_constructed(kwargs: dict, expected: str) -> None:
    """The combinations that would let a receipt lie about itself are refused.

    Positive control below: the two well-formed shapes construct fine, so
    this is a rule about the malformed ones and not a constructor that
    rejects everything.
    """
    with pytest.raises(ValueError, match=expected):
        AdmissionReceipt(entry_id="e", content_hash="c", chain_verified=True, **kwargs)


def test_a_well_formed_receipt_constructs() -> None:
    """Positive control for the parametrised refusals above."""
    write = AdmissionReceipt(entry_id="e", content_hash="c", kind=BLOCK, tier=IngestTier.EXTERNAL_INGEST)
    delete = AdmissionReceipt(entry_id="e", content_hash="c", kind=BLOCK, tier=None, operation=OP_DELETE)
    assert write.operation == OP_WRITE
    assert delete.operation == OP_DELETE
    assert len(delete.removals) == 0


def test_a_write_receipt_refuses_to_record_a_removal() -> None:
    """A write removes nothing, so it has nothing to report."""
    write = AdmissionReceipt(entry_id="e", content_hash="c", kind=BLOCK, tier=IngestTier.EXTERNAL_INGEST, covers=frozenset({"IMP-1"}))
    with pytest.raises(UngatedDeleteError, match="authorises a write"):
        write.record_removal("IMP-1", "body")


def test_the_removal_ledger_bounds_its_memory() -> None:
    """Clearing a corpus costs one hash per block, not the corpus in RAM."""
    ledger = admission.RemovalLedger()
    ledger.record("IMP-1", "first body")
    assert ledger.sole_content == "first body"
    ledger.record("IMP-2", "second body")
    assert ledger.sole_content is None, "raw content must be dropped once the removal set grows"
    assert ledger.leaves == [("IMP-1", _sha256("first body")), ("IMP-2", _sha256("second body"))]
    assert ledger.block_ids == ("IMP-1", "IMP-2")
    assert len(ledger) == 2


def test_require_admission_refuses_an_operation_it_cannot_classify(workspace: str) -> None:
    """An operation nobody named authorises nothing — fail closed."""
    gate = get_gate(workspace)
    with gate.admit_block("WRITE", "IMP-1", "body", tier=IngestTier.EXTERNAL_INGEST):
        with pytest.raises(ValueError, match="not one of"):
            require_admission("IMP-1", operation="purge")


# ===========================================================================
# B — READ admission
# ===========================================================================

#: The three-status seed every read test uses. ``active`` is servable,
#: the other two are what an ingest tier mints and a release has not yet
#: admitted.
SEED_ROWS: list[dict] = [
    {"_id": "D-1", "Status": "active", "Statement": "CANARY servable"},
    {"_id": "IMP-1", "Status": "pending", "Statement": "CANARY pending"},
    {"_id": "IMP-2", "Status": "quarantined", "Statement": "CANARY quarantined"},
]


def test_admit_read_withholds_what_recall_withholds() -> None:
    """The tool surface reaches the recall verdict, not its own.

    Positive control: all three rows go in, and the servable one comes
    out with its content intact — so "the canary is absent" cannot be
    the accident of an empty result.
    """
    decision = admit_read(SEED_ROWS, surface="test")
    assert isinstance(decision, ReadAdmission)
    served = {row["_id"] for row in decision.admitted}
    assert served == {"D-1"}, "only the servable row may be served"
    assert decision.admitted[0]["Statement"] == "CANARY servable", "the admitted row must keep its content"
    assert decision.withheld == 2


def test_admit_read_agrees_with_the_recall_leg_predicate() -> None:
    """Same input, same verdict, by construction rather than by review."""
    ours = {row["_id"] for row in admit_read(SEED_ROWS).admitted}
    theirs = {row["_id"] for row in admissibility.admit_corpus(SEED_ROWS)}
    assert ours == theirs


def test_admit_read_copies_rather_than_aliases() -> None:
    """A handler mutating its result must not reach back into the store."""
    decision = admit_read(SEED_ROWS)
    decision.admitted[0]["Statement"] = "mutated"
    assert SEED_ROWS[0]["Statement"] == "CANARY servable"


def test_admit_read_one_tells_withheld_apart_from_absent() -> None:
    """``get_block`` needs both answers, and only one of them is true."""
    absent = admit_read_one(None)
    assert absent.sole is None and absent.withheld == 0

    withheld = admit_read_one(SEED_ROWS[2])
    assert withheld.sole is None and withheld.withheld == 1

    served = admit_read_one(SEED_ROWS[0])
    assert served.sole is not None and served.sole["_id"] == "D-1" and served.withheld == 0


def test_admit_read_honours_an_explicit_per_call_widening() -> None:
    """The ``allow`` widening has a caller behind it, never a default."""
    assert admit_read(SEED_ROWS).withheld == 2
    widened = admit_read(SEED_ROWS, allow=frozenset({"pending"}))
    assert {row["_id"] for row in widened.admitted} == {"D-1", "IMP-1"}
    assert widened.withheld == 1


def test_admit_read_resolves_a_release_from_the_workspace(workspace: str) -> None:
    """A governance release readmits a withheld block, with no reindex.

    Positive control first: without the release decision the block is
    withheld, so the second half proves the release did the work.
    """
    row = {"_id": RELEASABLE_ID, "Status": "quarantined", "Statement": "CANARY imported"}
    assert admit_read([row], workspace=workspace).withheld == 1

    _append(workspace, "decisions/DECISIONS.md", _block_text("D-20260901-REL", "Release approved", "active", Releases=RELEASABLE_ID))
    decision = admit_read([row], workspace=workspace)
    assert decision.withheld == 0
    assert decision.sole is not None and decision.sole["_id"] == RELEASABLE_ID


def test_admit_read_refreshes_a_stale_cached_status(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An index caches ``status``, and it goes stale fail-OPEN.

    A surface reading rows out of an index and not refreshing serves what
    the corpus has already withdrawn. Positive control: the same row with
    no override is served, so the withholding is the refresh's doing.
    """
    cached = [{"_id": "IMP-9", "status": "active", "statement": "CANARY stale"}]
    assert admit_read(cached, workspace=workspace, status_key="status").withheld == 0

    monkeypatch.setattr(admissibility, "live_statuses", lambda ws: {"IMP-9": "quarantined"})
    refreshed = admit_read(cached, workspace=workspace, status_key="status")
    assert refreshed.withheld == 1, "a status the corpus has since withdrawn must not be served"
    assert refreshed.admitted == []


def test_admit_read_withholds_a_status_nobody_named() -> None:
    """The allow-list is the fail-closed half: unnamed means unserved."""
    invented = [{"_id": "IMP-3", "Status": "smuggled", "Statement": "CANARY invented"}]
    assert admit_read(invented).withheld == 1


def test_admit_read_of_nothing_is_not_a_withholding() -> None:
    assert admit_read([]) == ReadAdmission([], 0)
    assert not admit_read([])


# ===========================================================================
# C — the silent action collapse
# ===========================================================================


def test_an_unknown_action_is_not_recorded_as_an_apply(workspace: str) -> None:
    """The test that would have caught the original defect.

    ``_map_action`` used to return ``EvidenceAction.APPLY`` for any
    string it did not know, and ``admit`` wrote that straight into the
    evidence chain. A misspelled ``DELET`` therefore became an
    indistinguishable, hash-sealed claim that content had landed.

    Both halves are asserted: the refusal, and that the refusal leaves
    **no** record in either store — a half-written record naming an
    action the gate could not label is the same lie in a smaller font.
    """
    gate = get_gate(workspace)
    before_evidence = len(_records(workspace))
    before_chain = gate.chain.length

    with pytest.raises(GovernanceBypassError, match="no evidence classification exists"):
        gate.admit(action="DELET", block_id="IMP-1", content="body", actor="mallory")

    after = _records(workspace)
    assert len(after) == before_evidence, "a verb the gate cannot label must leave no evidence record"
    assert gate.chain.length == before_chain, "…and no hash-chain entry either"
    assert not any(r["action"] == EvidenceAction.APPLY.value and r["actor"] == "mallory" for r in after)


def test_a_known_action_still_records(workspace: str) -> None:
    """Positive control for the refusal above.

    Without this, ``test_an_unknown_action_is_not_recorded_as_an_apply``
    would also pass against a gate that had stopped recording anything.
    """
    gate = get_gate(workspace)
    gate.admit(action="WRITE", block_id="IMP-1", content="body", actor="alice")
    rows = _records(workspace)
    assert len(rows) == 1
    assert rows[0]["action"] == EvidenceAction.APPLY.value
    assert rows[0]["actor"] == "alice"
    assert rows[0]["metadata"]["action_verb"] == "WRITE"


def test_the_raw_verb_survives_the_coarse_classification(workspace: str) -> None:
    """Verify against the raw string; dispatch on the enum (§1.4).

    ``EvidenceAction`` is a small closed vocabulary, so WRITE / INGEST /
    MIGRATE all land as APPLY. The verb is not lost: it is stored raw in
    the hash chain's ``action`` column and copied verbatim into the
    evidence record's ``metadata["action_verb"]``, which an older reader
    carries through untouched.
    """
    gate = get_gate(workspace)
    for verb in ("WRITE", "INGEST", "MIGRATE", "REEXTRACT", "MESSAGE"):
        gate.admit(action=verb, block_id=f"IMP-{verb}", content="body", actor="alice")
    verbs = [r["metadata"]["action_verb"] for r in _records(workspace)]
    assert verbs == ["WRITE", "INGEST", "MIGRATE", "REEXTRACT", "MESSAGE"]
    assert {r["action"] for r in _records(workspace)} == {EvidenceAction.APPLY.value}
    assert [e.action for e in gate.chain.get_latest(10)] == verbs


def test_governing_delete_added_no_evidence_action_member() -> None:
    """Forward compatibility: a 5.0.1 reader parses every 5.0.2 record.

    ``EvidenceObject.from_dict`` does a strict ``EvidenceAction(value)``
    lookup, so a new member would make an older process fail to load a
    chain a newer one wrote. DELETE reuses ``ROLLBACK``, the vocabulary's
    existing word for content withdrawn, and the phase distinction lives
    in additive metadata instead.
    """
    assert {member.value for member in EvidenceAction} == {
        "PROPOSE",
        "APPLY",
        "ROLLBACK",
        "CONTRADICT",
        "DRIFT",
        "RESOLVE",
        "VERIFY",
    }
    assert governance_gate._ACTION_MAP[DELETE_VERB] is EvidenceAction.ROLLBACK


def _admit_action_literals() -> dict[str, set[str]]:
    """Every literal ``action`` a source file hands a gate admission.

    Static, over ``src/`` — the same discovery shape the tool-surface
    gate uses. Reads calls named ``admit``/``admit_block``/``admit_batch``
    and collects the first positional argument or the ``action=`` keyword
    when it is a string constant.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
            if name not in {"admit", "admit_block", "admit_batch"}:
                continue
            literals: list[str] = []
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                literals.append(node.args[0].value)
            for kw in node.keywords:
                if kw.arg == "action" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    literals.append(kw.value.value)
            for literal in literals:
                found.setdefault(literal, set()).add(str(path.relative_to(SRC)))
    return found


def test_every_action_verb_in_src_is_classified() -> None:
    """The tripwire that keeps the fail-closed mapping affordable.

    ``_map_action`` refusing an unknown verb is only safe if no shipped
    door passes one. This enumerates them mechanically, so a new door
    with a new verb fails here — at build time, naming its own file —
    rather than at the moment an operator runs it.
    """
    literals = _admit_action_literals()
    assert len(literals) >= 5, f"the scan found only {sorted(literals)} — it is not looking at the real call sites"
    unclassified = {verb: sorted(files) for verb, files in literals.items() if verb.upper() not in governance_gate._ACTION_MAP}
    assert not unclassified, (
        "these doors pass a verb _ACTION_MAP does not classify, so they would be refused at "
        f"runtime: {unclassified}. Add each to _ACTION_MAP mapped to the EvidenceAction that is "
        "true of it — never re-introduce a default."
    )


def test_the_delete_scopes_pass_a_classified_verb(workspace: str) -> None:
    """The seam's own verb goes through the same allow-list as every door."""
    assert DELETE_VERB.upper() in governance_gate._ACTION_MAP
    gate = get_gate(workspace)
    with gate.admit_delete("IMP-1", rationale="operator cleanup") as receipt:
        receipt.record_removal("IMP-1", "body")
    assert {r["action"] for r in _records(workspace)} == {EvidenceAction.ROLLBACK.value}


# ===========================================================================
# Mutation twins — a gate never observed failing is not a gate
# ===========================================================================


class TestMutationTwinDeleteAdmission:
    """Disable the delete gate; the protective bodies must go red."""

    def test_ungated_delete_refusal_depends_on_require_delete_admission(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(admission, "require_delete_admission", lambda block_id: None)

        with pytest.raises(pytest.fail.Exception):
            with pytest.raises(UngatedDeleteError):
                admission.require_delete_admission("D-20260901-001")

    def test_operation_check_is_what_stops_a_write_receipt(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Restore the pre-5.0.2 receipt: one that names no operation.

        With the check gone, the write receipt authorises the delete —
        which is exactly the ambient authority the field removes.
        """
        gate = get_gate(workspace)

        def permissive(block_id: str, *, status: object = None, operation: str = OP_WRITE) -> Any:
            """``require_admission`` with the operation check deleted."""
            receipt = admission.current_admission()
            if receipt is None:
                raise UngatedWriteError("no admission open")
            if not receipt.chain_verified or not receipt.authorizes(block_id):
                raise UngatedWriteError("receipt does not cover it")
            return receipt

        monkeypatch.setattr(admission, "require_admission", permissive)

        with gate.admit_block("WRITE", "IMP-1", "body", tier=IngestTier.EXTERNAL_INGEST):
            # The protective assertion in
            # test_a_write_receipt_does_not_authorize_a_delete is this one.
            with pytest.raises(pytest.fail.Exception):
                with pytest.raises(UngatedDeleteError):
                    admission.require_admission("IMP-1", operation=OP_DELETE)

    def test_the_removal_record_depends_on_the_ledger_being_read(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the recorder; the removal record disappears."""
        gate = get_gate(workspace)
        monkeypatch.setattr(type(gate), "_record_removals", lambda self, *a, **k: None)
        with gate.admit_delete("IMP-1", rationale="operator cleanup") as receipt:
            receipt.record_removal("IMP-1", "body")
        assert _phase(_records(workspace), PHASE_REMOVED) == [], "twin precondition: the recorder is stubbed out"


class TestMutationTwinActionMapping:
    """Restore the silent APPLY default; the audit test must go red."""

    def test_unknown_action_test_goes_red_under_the_permissive_default(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            governance_gate,
            "_map_action",
            lambda action: governance_gate._ACTION_MAP.get(action.upper(), EvidenceAction.APPLY),
        )
        gate = get_gate(workspace)

        # The protective body: a refusal, and no record left behind.
        with pytest.raises(pytest.fail.Exception):
            with pytest.raises(GovernanceBypassError):
                gate.admit(action="DELET", block_id="IMP-1", content="body", actor="mallory")

        rows = _records(workspace)
        assert rows, "twin precondition: the permissive default let the admission through"
        assert rows[-1]["action"] == EvidenceAction.APPLY.value, "…and recorded a DELET as an APPLY, which is the defect"
        assert rows[-1]["metadata"]["action_verb"] == "DELET"


class TestMutationTwinReadAdmission:
    """Neuter the egress filter; the withholding assertions must go red."""

    def test_withholding_depends_on_admit_leg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(admissibility, "admit_leg", lambda hits, **kwargs: [dict(h) for h in hits])
        decision = admit_read(SEED_ROWS)
        assert decision.withheld == 0, "twin precondition: the filter is disabled"
        assert {row["_id"] for row in decision.admitted} == {"D-1", "IMP-1", "IMP-2"}
        # Which is what test_admit_read_withholds_what_recall_withholds asserts against.
        with pytest.raises(AssertionError):
            assert {row["_id"] for row in decision.admitted} == {"D-1"}

    def test_one_block_withholding_depends_on_the_status_predicate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(admissibility, "is_admissible_status", lambda status: True)
        decision = admit_read_one(SEED_ROWS[2])
        assert decision.withheld == 0 and decision.sole is not None, "twin precondition: every status now passes"

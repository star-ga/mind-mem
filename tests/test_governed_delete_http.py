# Copyright 2026 STARGA, Inc.
"""The two HTTP doors that kill content, and the records they now leave.

``DELETE /memories/{id}`` and ``POST /clear`` both reached
``store.delete_block`` directly. With the store seam gated they could no
longer work at all — which is the fail-closed direction, but not the
product: an operator deleting a block is a legitimate governed act, so
the doors open a scope rather than being denied one.

What each door owes:

``DELETE /memories/{id}``
    one authorisation record and, if something was removed, one removal
    record, naming the actor and the target. The route carries no body,
    so it records :data:`~mind_mem.http_transport.DEFAULT_DELETE_RATIONALE`
    — naming the door rather than inventing a reason nobody gave.

``POST /clear``
    **one** bulk record, not N unlinked ones, over an id set frozen when
    the scope opens. Per-block records would flood a chain built for
    low-volume decisions and would lose the fact that the removals were
    one decision.

Every refusal here is paired with a positive control, and the two
``TestMutation*`` classes restore the pre-5.0.2 shape and show the
protective test going red — a gate never observed failing is not a gate.

A note on ``POST /clear`` and the corpus it enumerates: the loop used to
iterate ``store.list_blocks()``, which every store in the tree implements
as the list of *artifacts* (``.md`` paths, ``file_path`` values) rather
than block ids, so the endpoint removed nothing at all. That defect is
closed — the loop now walks
:func:`~mind_mem.http_transport._corpus_block_ids`, and
``tests/test_governed_delete_clear_enumeration.py`` proves it against a
real Markdown corpus. The property tested *here* is the one the scope
guarantees whatever the enumeration returns: **the receipt covers exactly
the set the loop iterates**, so the scope can never drift from the
deletions it authorises.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem import http_transport
from mind_mem.admission import require_delete_admission
from mind_mem.governance_gate import PHASE_ADMITTED, PHASE_REMOVED, evict_gate, get_gate
from mind_mem.http_transport import (
    DEFAULT_DELETE_RATIONALE,
    _handle_clear,
    _handle_delete_memory,
)
from mind_mem.merkle_tree import MerkleTree

SEED_ID = "D-20260901-001"
CLEAR_BODY = {"rationale": "operator reset for the release rehearsal", "confirm": "yes-i-really-want-to-clear"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _records(ws: str) -> list[dict]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _phase(ws: str, phase: str) -> list[dict]:
    return [r for r in _records(ws) if r.get("metadata", {}).get("delete_phase") == phase]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _merkle_root(leaves: list[tuple[str, str]]) -> str:
    tree = MerkleTree()
    tree.build(leaves)
    return tree.root_hash


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
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


def _seed(ws: str, bid: str, statement: str) -> None:
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{bid}]\nStatement: {statement}\nDate: 2026-09-01\nStatus: active\n\n---\n\n")


def _present(ws: str, bid: str) -> bool:
    from mind_mem.storage import get_block_store

    return get_block_store(ws).get_by_id(bid) is not None


class _CorpusStore:
    """A store the ``/clear`` loop can enumerate, and whose deletes work.

    ``get_all`` is what the door reads (block dicts carrying ``_id``, the
    protocol shape every real store returns); ``list_blocks`` stays the
    artifact list the protocol documents, so the double cannot make the
    door look right by handing it ids through the wrong method. It obeys
    the delete contract — check first, report the removal on success — so
    the bulk-record assertions below measure the door, not a permissive
    double.
    """

    def __init__(self, rows: dict[str, str]) -> None:
        self.rows = dict(rows)
        self.attempts: list[str] = []

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        return [{"_id": bid, "Statement": self.rows[bid], "Status": "active"} for bid in sorted(self.rows)]

    def block_ids(self) -> list[str]:
        """The ids the door will enumerate — the test's read of the same set."""
        return sorted(self.rows)

    def list_blocks(self) -> list[str]:
        return [f"decisions/{bid}.md" for bid in sorted(self.rows)]

    def delete_block(self, block_id: str) -> bool:
        self.attempts.append(block_id)
        receipt = require_delete_admission(str(block_id))
        removed = self.rows.pop(str(block_id), None)
        if removed is None:
            return False
        receipt.record_removal(str(block_id), removed)
        return True

    def get_by_id(self, block_id: str) -> dict[str, Any] | None:
        return {"_id": block_id, "Statement": self.rows[block_id]} if block_id in self.rows else None


@pytest.fixture
def corpus(monkeypatch: pytest.MonkeyPatch) -> _CorpusStore:
    """Route the door's ``get_block_store`` at a store that holds block ids."""
    store = _CorpusStore({f"D-2026090{i}-001": f"block {i}" for i in range(1, 6)})
    monkeypatch.setattr("mind_mem.storage.get_block_store", lambda ws, config=None: store)
    return store


# ===========================================================================
# DELETE /memories/{id}
# ===========================================================================


def test_the_delete_route_records_the_death_it_caused(workspace: str) -> None:
    """The happy path: one authorisation, one removal, both attributed."""
    _seed(workspace, SEED_ID, "the block the operator asked to remove")
    assert _present(workspace, SEED_ID), "fixture never seeded the block; every assertion below would be vacuous"

    status, body = _handle_delete_memory(workspace, SEED_ID, actor="alice")

    assert status == 200
    assert body["ok"] is True
    assert not _present(workspace, SEED_ID)

    admitted = _phase(workspace, PHASE_ADMITTED)
    assert len(admitted) == 1
    assert admitted[0]["actor"] == "alice"
    assert admitted[0]["target_block_id"] == SEED_ID
    assert admitted[0]["metadata"]["operation"] == "delete"
    assert admitted[0]["metadata"]["rationale"] == DEFAULT_DELETE_RATIONALE

    removed = _phase(workspace, PHASE_REMOVED)
    assert len(removed) == 1
    assert removed[0]["metadata"]["removed_count"] == 1
    # The response hands the caller the receipt id, so a client can verify
    # its own deletion against the chain rather than trust the 200. The
    # removal record carries the same id, which is what links the two.
    assert body["admission"] == removed[0]["metadata"]["admission_entry_id"]


def test_a_caller_supplied_rationale_reaches_the_record(workspace: str) -> None:
    """The default names the door; a caller that has a reason overrides it."""
    _seed(workspace, SEED_ID, "superseded by the 5.0.2 decision")
    status, _body = _handle_delete_memory(workspace, SEED_ID, rationale="superseded by D-20260901-002")
    assert status == 200
    assert _phase(workspace, PHASE_ADMITTED)[0]["metadata"]["rationale"] == "superseded by D-20260901-002"


def test_a_delete_the_gate_refuses_is_reported_as_a_refusal(workspace: str) -> None:
    """403, and the block is still there.

    An empty rationale is refused by the gate — a chain record that
    cannot say why content was destroyed is most of the way to no
    record. The door must not translate that into a 200: telling a
    caller their content is gone when it is not is the one answer a
    memory product must never give.
    """
    _seed(workspace, SEED_ID, "the block that must survive a refused delete")
    assert _present(workspace, SEED_ID)

    status, body = _handle_delete_memory(workspace, SEED_ID, rationale="   ")

    assert status == 403
    assert body["error"] == "delete refused by governance"
    assert _present(workspace, SEED_ID), "a refused delete removed the block anyway"
    assert _records(workspace) == [], "a refused authorisation must leave no record"


def test_deleting_a_block_that_is_not_there_stays_a_404(workspace: str) -> None:
    """The external contract is unchanged, and nothing is recorded as dead."""
    _seed(workspace, SEED_ID, "a different block")
    status, body = _handle_delete_memory(workspace, "D-20260901-999")
    assert status == 404
    assert body["error"] == "block not found"
    assert _phase(workspace, PHASE_REMOVED) == []


def test_an_invalid_block_id_is_refused_before_any_scope_opens(workspace: str) -> None:
    status, _body = _handle_delete_memory(workspace, "../../etc/passwd")
    assert status == 400
    assert _records(workspace) == []


# ===========================================================================
# POST /clear
# ===========================================================================


def test_clear_writes_one_bulk_record_over_everything_it_removed(workspace: str, corpus: _CorpusStore) -> None:
    """Five removals, one decision, one record — not five unlinked ones."""
    expected = dict(corpus.rows)
    assert len(expected) == 5, "fixture never seeded the corpus"

    status, body = _handle_clear(workspace, CLEAR_BODY, actor="alice")

    assert status == 200
    assert body["deleted"] == 5
    assert corpus.rows == {}, "the positive control: the clear really removed them"

    admitted = _phase(workspace, PHASE_ADMITTED)
    assert len(admitted) == 1, f"a clear is one decision, so it authorises once; got {len(admitted)}"
    assert admitted[0]["metadata"]["covers_count"] == 5
    assert admitted[0]["metadata"]["rationale"] == CLEAR_BODY["rationale"]
    assert admitted[0]["actor"] == "alice"

    removed = _phase(workspace, PHASE_REMOVED)
    assert len(removed) == 1, f"a clear must leave ONE removal record, not one per block; got {len(removed)}"
    assert removed[0]["metadata"]["removed_count"] == 5
    assert removed[0]["metadata"]["merkle_root"] == _merkle_root([(bid, _sha256(text)) for bid, text in sorted(expected.items())])
    assert body["admission"] == removed[0]["metadata"]["admission_entry_id"]


def test_the_clear_scope_covers_exactly_what_the_loop_iterates(workspace: str, corpus: _CorpusStore) -> None:
    """The receipt and the loop can never disagree about the id set.

    Whatever the corpus enumeration returns, the scope is opened over
    that exact sequence and the loop walks the same one — so the
    authorisation always names the deletions it authorised.
    """
    enumerated = corpus.block_ids()
    status, _body = _handle_clear(workspace, CLEAR_BODY)
    assert status == 200
    assert corpus.attempts == enumerated
    assert _phase(workspace, PHASE_ADMITTED)[0]["metadata"]["covers_count"] == len(enumerated)


def test_a_block_written_during_the_clear_is_outside_the_receipt(workspace: str, corpus: _CorpusStore) -> None:
    """The frozen set is the point: a clear cannot grow to reach a new block.

    The intruder is added while the scope is open. The loop never sees
    it (it walks the frozen list), and even a store that tried to delete
    it would be refused, because the receipt does not cover it.
    """
    from mind_mem.admission import UngatedDeleteError

    original = corpus.block_ids()
    real_delete = corpus.delete_block
    intruder = "D-20260901-777"
    #: What the open receipt said about the intruder, checked from inside
    #: the live scope — after the scope closes every id raises, which
    #: would prove nothing about coverage.
    verdict: list[str] = []

    def delete_and_intrude(block_id: str) -> bool:
        if intruder not in corpus.rows and block_id == original[0]:
            corpus.rows[intruder] = "written while the clear was running"
            # Positive control, inside the scope: the receipt authorises
            # the id it froze...
            assert require_delete_admission(block_id) is not None
            # ...and refuses the one that arrived after it.
            try:
                require_delete_admission(intruder)
            except UngatedDeleteError as exc:
                verdict.append(str(exc))
            else:  # pragma: no cover - the failure this test exists for
                verdict.append("AUTHORISED")
        return real_delete(block_id)

    corpus.delete_block = delete_and_intrude  # type: ignore[method-assign]

    status, body = _handle_clear(workspace, CLEAR_BODY)

    assert status == 200
    assert body["deleted"] == len(original)
    assert intruder in corpus.rows, "the clear took a block its receipt never covered"
    assert intruder not in corpus.attempts, "the loop walked past its own frozen list"
    assert verdict and verdict[0] != "AUTHORISED", "the open receipt covered a block written after it was minted"
    assert "does not cover" in verdict[0]


def test_clearing_an_empty_corpus_authorises_nothing(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A receipt covering nothing authorises nothing, so none is minted."""
    monkeypatch.setattr("mind_mem.storage.get_block_store", lambda ws, config=None: _CorpusStore({}))
    status, body = _handle_clear(workspace, CLEAR_BODY)
    assert status == 200
    assert body["deleted"] == 0
    assert body["admission"] is None
    assert _records(workspace) == [], "nothing died, so nothing may be recorded as a death"


def test_clear_still_requires_its_rationale_and_confirmation(workspace: str, corpus: _CorpusStore) -> None:
    """The pre-existing guards are untouched, and refuse before any scope."""
    status, _body = _handle_clear(workspace, {"rationale": "too short", "confirm": "yes-i-really-want-to-clear"})
    assert status == 400
    status, _body = _handle_clear(workspace, {"rationale": CLEAR_BODY["rationale"], "confirm": "no"})
    assert status == 400
    assert corpus.rows, "a refused clear deleted something"
    assert _records(workspace) == []


def test_a_store_refusing_a_covered_block_aborts_the_clear(workspace: str, corpus: _CorpusStore) -> None:
    """A governance refusal inside the loop is never swallowed.

    The loop skips a block that merely errors — one bad block must not
    abort a wipe. A ``GovernanceBypassError`` is different in kind: it
    means the receipt and the loop disagree about what was authorised,
    which is exactly the failure the scope exists to surface.
    """
    from mind_mem.admission import UngatedDeleteError

    real_delete = corpus.delete_block
    target = corpus.block_ids()[2]

    def refuse_one(block_id: str) -> bool:
        if block_id == target:
            raise UngatedDeleteError("simulated disagreement between the receipt and the loop")
        return real_delete(block_id)

    corpus.delete_block = refuse_one  # type: ignore[method-assign]

    status, body = _handle_clear(workspace, CLEAR_BODY)

    assert status == 403
    assert body["error"] == "clear refused by governance"
    # The blocks removed before the refusal are still recorded: a chain
    # that only records tidy deletions under-reports exactly the cases an
    # auditor cares about.
    removed = _phase(workspace, PHASE_REMOVED)
    assert len(removed) == 1
    assert removed[0]["metadata"]["removed_count"] == 2
    assert removed[0]["metadata"]["scope_outcome"] == "error"


def test_a_block_that_merely_errors_does_not_abort_the_clear(workspace: str, corpus: _CorpusStore) -> None:
    """Positive control for the test above: an ordinary failure is skipped."""
    real_delete = corpus.delete_block
    target = corpus.block_ids()[2]

    def blow_up_on_one(block_id: str) -> bool:
        if block_id == target:
            raise OSError("disk hiccup")
        return real_delete(block_id)

    corpus.delete_block = blow_up_on_one  # type: ignore[method-assign]

    status, body = _handle_clear(workspace, CLEAR_BODY)

    assert status == 200
    assert body["deleted"] == 4
    assert _phase(workspace, PHASE_REMOVED)[0]["metadata"]["removed_count"] == 4


# ===========================================================================
# Mutation twins — restore the 5.0.1 shape and watch the gate go red
# ===========================================================================


class TestMutationTwins:
    """Each twin reproduces the pre-5.0.2 door and shows the test failing.

    These do not patch production code; they run the *old* body against
    the new store seam, which is the sharper demonstration: the ungated
    door cannot work at all any more, so the defect is unreachable rather
    than merely tested-against.
    """

    def test_the_old_ungated_delete_door_now_raises(self, workspace: str) -> None:
        """5.0.1's ``_handle_delete_memory`` body: no scope, straight to the store."""
        from mind_mem.admission import UngatedDeleteError
        from mind_mem.storage import get_block_store

        _seed(workspace, SEED_ID, "the block the old door would have taken")
        store = get_block_store(workspace)
        assert store.get_by_id(SEED_ID) is not None

        with pytest.raises(UngatedDeleteError):
            store.delete_block(SEED_ID)  # the 5.0.1 door, verbatim

        assert store.get_by_id(SEED_ID) is not None
        assert _records(workspace) == []

    def test_the_old_per_block_clear_would_have_written_n_records(self, workspace: str, corpus: _CorpusStore) -> None:
        """Why the batch scope exists: per-block scopes flood the chain.

        Five blocks through five ``admit_delete`` scopes produce ten
        records with nothing linking them; the batch produces two. The
        assertion that ``len(removed) == 1`` in
        :func:`test_clear_writes_one_bulk_record_over_everything_it_removed`
        is therefore load-bearing and observably failable.
        """
        gate = get_gate(workspace)
        for bid in corpus.block_ids():
            with gate.admit_delete(bid, rationale="the pre-batch shape, one scope per block"):
                corpus.delete_block(bid)

        assert len(_phase(workspace, PHASE_ADMITTED)) == 5, "the pre-batch shape authorises once per block"
        removed = _phase(workspace, PHASE_REMOVED)
        assert len(removed) == 5, "…and records once per block, which is what the batch scope replaces"
        # Ten records, and not one of them says the removals were one
        # decision: each covers exactly its own block.
        assert {r["metadata"]["removed_count"] for r in removed} == {1}


def test_the_module_exports_the_rationale_default() -> None:
    """The audit record and the tests read one string, not two."""
    assert http_transport.DEFAULT_DELETE_RATIONALE == "http-delete"

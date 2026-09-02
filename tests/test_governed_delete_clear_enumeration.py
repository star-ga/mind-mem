# Copyright 2026 STARGA, Inc.
"""``POST /clear`` must clear the corpus, not a list of filenames.

The defect this file closes was measured on a real ``MarkdownBlockStore``
before it existed. Three blocks were written through the governance gate;
``POST /clear`` was called with a valid rationale and the destructive
confirmation string; it answered::

    200 {"ok": true, "deleted": 0, "admission": "d1b5cb6e-…"}

and all three blocks were still there. The cause is a type confusion the
``BlockStore`` protocol documents in plain sight: ``list_blocks()``
returns block-*containing artifacts* — ``.md`` paths on the Markdown and
encrypted backends, distinct ``file_path`` values on Postgres — and the
door handed those to ``delete_block``, which resolves a path to no block
at all. It predates 5.0.2 (the same loop is in the v5.0.1 tag), so the
endpoint has never wiped anything.

Two things make it worse than an inert endpoint, and both are governance
failures rather than cosmetic ones:

* an operator who clears a workspace to purge content is told the purge
  succeeded when nothing was purged — a **false erasure**, which for a
  memory product is the one answer worse than an error; and
* once the delete scope landed, the door minted a DELETE authorisation
  over that set of filenames, so the chain carried a receipt for a death
  that never happened — the exact mirror of the ungated delete 5.0.2
  closed.

The fix is :func:`~mind_mem.http_transport._corpus_block_ids`, which
reads ids off ``get_all(active_only=False)`` — on the protocol, and
implemented by all five stores. Everything below is asserted against a
**real** Markdown corpus rather than a double, because a double is what
let the original defect hide: every double in the sibling file returned
ids from ``list_blocks``, so the door looked correct when driven by one.

:class:`TestMutationTwin` runs the pre-fix enumeration through the same
governed loop and shows it removing nothing — the defect is reproducible
on demand, so the assertions above are observably failable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem.governance_gate import PHASE_ADMITTED, PHASE_REMOVED, evict_gate, get_gate
from mind_mem.http_transport import _corpus_block_ids, _handle_clear
from mind_mem.storage import get_block_store

CLEAR_BODY = {
    "rationale": "operator purge rehearsal for the release",
    "confirm": "yes-i-really-want-to-clear",
}

ACTIVE_IDS = ("D-20260901-001", "D-20260901-002", "D-20260901-003")
QUARANTINED_ID = "D-20260901-009"


# ---------------------------------------------------------------------------
# Fixtures — a real corpus on disk, not a double
# ---------------------------------------------------------------------------


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


def _seed(ws: str, bid: str, statement: str, status: str = "active") -> None:
    """Append one block to the corpus file the ``D-`` prefix routes to."""
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{bid}]\nStatement: {statement}\nDate: 2026-09-01\nStatus: {status}\n\n---\n\n")


def _records(ws: str) -> list[dict[str, Any]]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _phase(ws: str, phase: str) -> list[dict[str, Any]]:
    return [r for r in _records(ws) if r.get("metadata", {}).get("delete_phase") == phase]


def _live_ids(ws: str) -> set[str]:
    """Ids actually parsed off disk right now, read through a fresh store."""
    return {str(b["_id"]) for b in get_block_store(ws).get_all(active_only=False) if b.get("_id")}


@pytest.fixture
def seeded(workspace: str) -> str:
    for i, bid in enumerate(ACTIVE_IDS, start=1):
        _seed(workspace, bid, f"the block a wipe must take, number {i}")
    # POSITIVE CONTROL for every assertion in this file: the corpus is
    # non-empty and readable before anything is cleared. Without it a
    # "nothing is left" assertion would pass against a corpus that never
    # held anything.
    assert _live_ids(workspace) == set(ACTIVE_IDS), "the fixture never wrote the corpus it is about to clear"
    return workspace


# ---------------------------------------------------------------------------
# The enumeration itself
# ---------------------------------------------------------------------------


def test_the_enumeration_reads_block_ids_not_artifact_paths(seeded: str) -> None:
    """The two lists are different things, and the door needs the id one."""
    store = get_block_store(seeded)

    ids, unreachable = _corpus_block_ids(store)
    artifacts = store.list_blocks()

    assert sorted(ids) == sorted(ACTIVE_IDS)
    assert unreachable == 0
    # The mechanism, stated as an assertion: list_blocks is a file list,
    # so no element of it is a block id and none could ever be deleted.
    assert artifacts, "the corpus has no artifacts, so this comparison proves nothing"
    assert set(artifacts).isdisjoint(set(ids))
    assert all(a.endswith(".md") for a in artifacts)


def test_the_enumeration_deduplicates_and_keeps_first_seen_order() -> None:
    """A repeated id must be authorised once, not twice."""

    class _Duplicating:
        def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
            return [{"_id": "D-1"}, {"_id": "D-2"}, {"_id": "D-1"}, {"_id": "D-3"}]

    ids, unreachable = _corpus_block_ids(_Duplicating())
    assert ids == ["D-1", "D-2", "D-3"]
    assert unreachable == 0


def test_a_block_with_no_id_is_counted_rather_than_silently_dropped() -> None:
    """An unreachable block makes the wipe partial; that has to be visible."""

    class _Malformed:
        def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
            return [{"_id": "D-1"}, {"Statement": "no id at all"}, {"_id": ""}]

    ids, unreachable = _corpus_block_ids(_Malformed())
    assert ids == ["D-1"]
    assert unreachable == 2


def test_the_enumeration_asks_for_withheld_blocks_too() -> None:
    """``active_only=False`` is load-bearing, so the call is asserted."""

    class _Recording:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
            self.calls.append(active_only)
            return [{"_id": "D-1"}]

    store = _Recording()
    _corpus_block_ids(store)
    assert store.calls == [False], "a wipe that skipped withheld blocks would be a partial purge reported as a whole one"


# ---------------------------------------------------------------------------
# The door, against a real corpus
# ---------------------------------------------------------------------------


def test_clear_actually_empties_a_real_markdown_corpus(seeded: str) -> None:
    """The defect, inverted: the wipe removes what it says it removed."""
    status, body = _handle_clear(seeded, CLEAR_BODY, actor="alice")

    assert status == 200
    assert body["deleted"] == len(ACTIVE_IDS)
    assert "unreachable" not in body
    assert _live_ids(seeded) == set(), "the clear reported success and left the corpus behind"


def test_the_wipe_leaves_exactly_one_bulk_record(seeded: str) -> None:
    """One decision, one authorisation, one removal record over the set."""
    _status, body = _handle_clear(seeded, CLEAR_BODY, actor="alice")

    admitted = _phase(seeded, PHASE_ADMITTED)
    assert len(admitted) == 1, f"a clear is one decision; got {len(admitted)} authorisations"
    assert admitted[0]["metadata"]["covers_count"] == len(ACTIVE_IDS)
    assert admitted[0]["metadata"]["operation"] == "delete"
    assert admitted[0]["actor"] == "alice"

    removed = _phase(seeded, PHASE_REMOVED)
    assert len(removed) == 1, f"a clear must leave ONE removal record, not one per block; got {len(removed)}"
    assert removed[0]["metadata"]["removed_count"] == len(ACTIVE_IDS)
    assert removed[0]["metadata"]["merkle_root"], "the record names no root for the content it destroyed"
    assert body["admission"] == removed[0]["metadata"]["admission_entry_id"]


def test_the_authorisation_names_the_ids_it_will_take(seeded: str) -> None:
    """The receipt covers the corpus, so no removal falls outside it.

    Read from inside the live scope: after it closes every id is refused,
    which would prove nothing about coverage.
    """
    from mind_mem.admission import current_admission

    seen: list[frozenset[str]] = []
    real = get_block_store(seeded).delete_block

    def peek(block_id: str) -> bool:
        receipt = current_admission()
        assert receipt is not None
        seen.append(receipt.covers)
        return real(block_id)

    store = get_block_store(seeded)
    store.delete_block = peek  # type: ignore[method-assign]
    import mind_mem.storage as storage_mod

    original_factory = storage_mod.get_block_store
    storage_mod.get_block_store = lambda ws, config=None: store  # type: ignore[assignment]
    try:
        status, _body = _handle_clear(seeded, CLEAR_BODY)
    finally:
        storage_mod.get_block_store = original_factory  # type: ignore[assignment]

    assert status == 200
    assert seen, "the loop never ran, so nothing about coverage was observed"
    assert seen[0] == frozenset(ACTIVE_IDS)


def test_a_wipe_takes_the_withheld_blocks_as_well(workspace: str) -> None:
    """A purge that left quarantined content behind would be a false one."""
    for bid in ACTIVE_IDS:
        _seed(workspace, bid, "servable content")
    _seed(workspace, QUARANTINED_ID, "content the corpus is withholding", status="quarantined")

    store = get_block_store(workspace)
    # Positive control on both sides: the block is really there, and it
    # really is withheld — so `active_only=False` is what reaches it.
    assert QUARANTINED_ID in _live_ids(workspace)
    assert QUARANTINED_ID not in {str(b["_id"]) for b in store.get_all(active_only=True) if b.get("_id")}

    status, body = _handle_clear(workspace, CLEAR_BODY)

    assert status == 200
    assert body["deleted"] == len(ACTIVE_IDS) + 1
    assert _live_ids(workspace) == set()


def test_an_unreachable_block_is_reported_on_the_response(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial wipe must be visibly partial."""

    class _PartlyMalformed:
        def __init__(self) -> None:
            self.rows = {"D-1": "content"}

        def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
            return [{"_id": "D-1", "Statement": "content"}, {"Statement": "no id"}]

        def delete_block(self, block_id: str) -> bool:
            from mind_mem.admission import require_delete_admission

            receipt = require_delete_admission(block_id)
            removed = self.rows.pop(block_id, None)
            if removed is None:
                return False
            receipt.record_removal(block_id, removed)
            return True

    monkeypatch.setattr("mind_mem.storage.get_block_store", lambda ws, config=None: _PartlyMalformed())
    status, body = _handle_clear(workspace, CLEAR_BODY)

    assert status == 200
    assert body["deleted"] == 1
    assert body["unreachable"] == 1, "a block the wipe could not reach was not reported"


def test_a_store_that_cannot_enumerate_fails_the_clear(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never ``ok: true, deleted: 0`` again — that is the defect's shape."""

    class _Broken:
        def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
            raise RuntimeError("backend unavailable")

        def delete_block(self, block_id: str) -> bool:  # pragma: no cover - must never run
            raise AssertionError("the clear reached the store despite a failed enumeration")

    monkeypatch.setattr("mind_mem.storage.get_block_store", lambda ws, config=None: _Broken())
    status, body = _handle_clear(workspace, CLEAR_BODY)

    assert status == 500
    assert body == {"error": "internal block store error"}
    assert _records(workspace) == [], "a failed enumeration must not mint an authorisation"


def test_an_empty_corpus_still_authorises_nothing(workspace: str) -> None:
    """Unchanged by the fix: no subject, no receipt, no record."""
    status, body = _handle_clear(workspace, CLEAR_BODY)

    assert status == 200
    assert body["deleted"] == 0
    assert body["admission"] is None
    assert _records(workspace) == []


# ---------------------------------------------------------------------------
# Mutation twin
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """Reproduce the pre-fix enumeration and watch the wipe do nothing."""

    def test_the_old_artifact_enumeration_removes_nothing(self, seeded: str) -> None:
        """The v5.0.1 loop body, verbatim, against the same real corpus.

        This is the failure the tests above are protecting against, run
        on demand rather than asserted about: the door's own governed
        loop, fed ``list_blocks()`` instead of block ids, deletes zero
        blocks and leaves the corpus whole — while still minting an
        authorisation, which is the receipt-without-a-death half of the
        defect.
        """
        store = get_block_store(seeded)
        artifact_paths = list(store.list_blocks())
        assert artifact_paths, "no artifacts to feed the old enumeration"

        gate = get_gate(seeded)
        deleted = 0
        with gate.admit_delete_batch(
            "clear-pre-fix-shape",
            artifact_paths,
            rationale=str(CLEAR_BODY["rationale"]),
        ):
            for path in artifact_paths:
                if store.delete_block(path):
                    deleted += 1

        assert deleted == 0, "the pre-fix enumeration is supposed to be incapable of deleting anything"
        assert _live_ids(seeded) == set(ACTIVE_IDS), "the corpus changed, so this is no longer the defect being reproduced"
        # The authorisation was minted anyway — a receipt for a death
        # that never happened.
        assert len(_phase(seeded, PHASE_ADMITTED)) == 1
        assert _phase(seeded, PHASE_REMOVED) == []

    def test_the_current_door_on_the_same_corpus_removes_everything(self, seeded: str) -> None:
        """The other half of the twin: same corpus, current door, real wipe."""
        status, body = _handle_clear(seeded, CLEAR_BODY)

        assert status == 200
        assert body["deleted"] == len(ACTIVE_IDS)
        assert _live_ids(seeded) == set()

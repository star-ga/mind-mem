# Copyright 2026 STARGA, Inc.
"""``supersede_decision`` writes the successor pointer, both directions.

``_op_supersede_decision`` is the one place in the product where the
retired block's id and its successor's id are both in hand. It flipped
``Status`` and stopped, so ``SupersededBy`` — read by ``sqlite_index``'s
xref extraction, ``_recall_scoring``'s cross-reference graph,
``evidence_bundle``'s relation extraction, the ADR schema template and
the conflict resolver's report — was a field with five readers and no
writer.

These tests cover both write paths (legacy filesystem and BlockStore) and
finish by proving the pointer is *readable by a consumer*: writing a field
nothing can read is not the fix.
"""

from __future__ import annotations

import os

from mind_mem.apply_engine import _op_supersede_decision
from mind_mem.block_parser import parse_blocks
from mind_mem.evidence_bundle import build_bundle

OLD_ID = "D-20260213-001"
NEW_ID = "D-20260213-002"

_OLD_BLOCK = f"[{OLD_ID}]\nStatus: active\nStatement: Old decision\n"
_NEW_BLOCK = f"[{NEW_ID}]\nStatus: active\nStatement: New decision\n"


class _FakeStore:
    """Minimal BlockStore stand-in: get_by_id + write_block, nothing else."""

    def __init__(self, blocks: list[dict]) -> None:
        self._blocks = {str(b["_id"]): dict(b) for b in blocks}
        self.written: list[dict] = []

    def get_by_id(self, block_id: str):
        found = self._blocks.get(block_id)
        return dict(found) if found is not None else None

    def write_block(self, block: dict) -> str:
        self._blocks[str(block["_id"])] = dict(block)
        self.written.append(dict(block))
        return str(block["_id"])


def _write_corpus(tmp_path) -> str:
    path = os.path.join(str(tmp_path), "DECISIONS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_OLD_BLOCK)
    return path


def _blocks_by_id(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        blocks = parse_blocks(fh.read())
    return {str(b["_id"]): b for b in blocks}


# ---------------------------------------------------------------------------
# Legacy filesystem path
# ---------------------------------------------------------------------------


def test_file_path_writes_both_directions(tmp_path):
    path = _write_corpus(tmp_path)
    ok, msg = _op_supersede_decision(path, {"target": OLD_ID, "new_block": _NEW_BLOCK})
    assert ok, msg

    by_id = _blocks_by_id(path)
    # Positive control: both blocks are really in the file, so the
    # assertions below are about content and not about an empty parse.
    assert set(by_id) == {OLD_ID, NEW_ID}
    assert by_id[OLD_ID]["Status"] == "superseded"
    assert by_id[OLD_ID]["SupersededBy"] == NEW_ID
    assert by_id[NEW_ID]["Supersedes"] == OLD_ID


def test_file_path_updates_an_existing_pointer_in_place(tmp_path):
    """A stale pointer is corrected, not duplicated."""
    path = os.path.join(str(tmp_path), "DECISIONS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"[{OLD_ID}]\nStatus: active\nSupersededBy: D-19990101-001\nStatement: Old\n")

    ok, msg = _op_supersede_decision(
        path,
        {"target": OLD_ID, "new_block": f"[{NEW_ID}]\nStatus: active\nSupersedes: D-19990101-999\nStatement: New\n"},
    )
    assert ok, msg

    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert content.count("SupersededBy:") == 1
    assert content.count("Supersedes:") == 1

    by_id = _blocks_by_id(path)
    assert by_id[OLD_ID]["SupersededBy"] == NEW_ID
    assert by_id[NEW_ID]["Supersedes"] == OLD_ID


def test_file_path_still_refuses_a_missing_status_field(tmp_path):
    """The pre-existing refusal is unchanged — no pointer, no write."""
    path = os.path.join(str(tmp_path), "DECISIONS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"[{OLD_ID}]\nStatement: no status here\n")
    before = open(path, encoding="utf-8").read()

    ok, msg = _op_supersede_decision(path, {"target": OLD_ID, "new_block": _NEW_BLOCK})
    assert not ok
    assert "Status" in msg
    assert open(path, encoding="utf-8").read() == before


# ---------------------------------------------------------------------------
# BlockStore path
# ---------------------------------------------------------------------------


def test_store_path_writes_both_directions():
    store = _FakeStore([{"_id": OLD_ID, "Status": "active", "Statement": "Old decision"}])
    ok, msg = _op_supersede_decision(None, {"target": OLD_ID, "new_block": _NEW_BLOCK}, store=store)
    assert ok, msg

    written = {str(b["_id"]): b for b in store.written}
    assert set(written) == {OLD_ID, NEW_ID}
    assert written[OLD_ID]["Status"] == "superseded"
    assert written[OLD_ID]["SupersededBy"] == NEW_ID
    assert written[NEW_ID]["Supersedes"] == OLD_ID


def test_store_path_reports_the_successor_it_wrote():
    store = _FakeStore([{"_id": OLD_ID, "Status": "active"}])
    ok, msg = _op_supersede_decision(None, {"target": OLD_ID, "new_block": _NEW_BLOCK}, store=store)
    assert ok
    assert NEW_ID in msg


# ---------------------------------------------------------------------------
# The pointer has to be READABLE by something downstream
# ---------------------------------------------------------------------------


def test_pointer_is_read_by_evidence_bundle(tmp_path):
    """A consumer turns the written field into the relation it exists for.

    ``evidence_bundle`` maps ``SupersededBy`` → ``superseded_by`` and
    ``Supersedes`` → ``supersedes``. If the supersession did not write
    the fields, this bundle carries neither relation — which is exactly
    what it carried before the fix.
    """
    path = _write_corpus(tmp_path)
    ok, msg = _op_supersede_decision(path, {"target": OLD_ID, "new_block": _NEW_BLOCK})
    assert ok, msg

    with open(path, encoding="utf-8") as fh:
        blocks = parse_blocks(fh.read())
    assert len(blocks) == 2

    bundle = build_bundle("what replaced the old decision", blocks)
    relations = {(r.subject, r.predicate, r.object) for r in bundle.relations}
    assert (OLD_ID, "superseded_by", NEW_ID) in relations
    assert (NEW_ID, "supersedes", OLD_ID) in relations


def test_pointer_is_read_by_the_recall_xref_graph(tmp_path):
    """The second consumer: the cross-reference adjacency used by recall."""
    from mind_mem._recall_scoring import build_xref_graph

    path = _write_corpus(tmp_path)
    ok, msg = _op_supersede_decision(path, {"target": OLD_ID, "new_block": _NEW_BLOCK})
    assert ok, msg

    with open(path, encoding="utf-8") as fh:
        blocks = parse_blocks(fh.read())
    graph = build_xref_graph(blocks)
    assert NEW_ID in graph[OLD_ID]
    assert OLD_ID in graph[NEW_ID]

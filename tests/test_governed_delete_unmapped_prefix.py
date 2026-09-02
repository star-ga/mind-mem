# Copyright 2026 STARGA, Inc.
"""A block this store can read must be a block it can remove.

``MarkdownBlockStore.delete_block`` resolved its target through
``_resolve_block_file``, which answers only for the prefixes in
``_BLOCK_PREFIX_MAP``. That function's own docstring says what a caller
owes when it does not: *"Returns None for unrecognised prefixes. Callers
must fall back to full-corpus scan when the mapping is absent (e.g.,
signals, one-off entity types not in the prefix map)."* ``delete_block``
did not fall back. It returned ``False``.

Measured before the fix, on a corpus holding one ``D-`` block and one
unmapped-prefix block in ``entities/signals.md`` (the probe used a
``SIG-`` id, which 5.0.2 later routed through the prefix map when GAP-2
made signals writable through the store; the fixture below carries a
``TRAJ-`` id instead, so it keeps measuring the fallback rather than the
canonical-file path):

* ``get_by_id`` and ``get_all`` both returned the unmapped block, so
  recall served it;
* ``DELETE /memories/<unmapped id>`` answered ``404 block not found``
  while the block sat on disk; and
* ``POST /clear`` answered ``200 {"ok": true, "deleted": 1}`` and left it
  behind — a partial purge reported as a whole one.

An id readable through every door and destroyable through none is an
undeletable record, which is the one thing a governed memory must not
hold: the DIE half of the thesis is *"nothing dies without a receipt"*,
not *"some things cannot die"*.

The fix is at the store, not at the doors, so every caller inherits it —
the two HTTP doors, the admin tool's store path, and anything added
later. The canonical file is still tried first, so a mapped prefix costs
one open exactly as before; the corpus walk happens only where the old
code returned the wrong answer.

:class:`TestMutationTwin` restores the prefix-map-only resolution and
shows these tests going red.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem.admission import UngatedDeleteError
from mind_mem.block_store import MarkdownBlockStore, _resolve_block_file
from mind_mem.governance_gate import PHASE_ADMITTED, PHASE_REMOVED, evict_gate, get_gate
from mind_mem.http_transport import _handle_clear, _handle_delete_memory

#: In ``_BLOCK_PREFIX_MAP`` — resolves to ``decisions/DECISIONS.md``.
MAPPED_ID = "D-20260901-001"

#: Not in the map. ``TRAJ`` blocks are real (``trajectory.py`` writes
#: them, one file per capture) and no row of ``corpus_registry.CORPUS_TABLE``
#: gives the prefix a canonical file, so ``_resolve_block_file`` returns
#: ``None`` for them — which is the whole premise of this file. It used to
#: be ``SIG``; 5.0.2 mapped that prefix (GAP-2), and
#: :func:`test_the_unmapped_block_is_real_and_the_prefix_map_does_not_know_it`
#: is what said so.
UNMAPPED_ID = "TRAJ-20260901-007"
UNMAPPED_ID_2 = "TRAJ-20260901-008"

#: A mapped prefix filed somewhere other than its canonical file.
DISPLACED_ID = "D-20260901-042"

CLEAR_BODY = {
    "rationale": "operator purge rehearsal for the release",
    "confirm": "yes-i-really-want-to-clear",
}


def _block(bid: str, statement: str) -> str:
    return f"[{bid}]\nStatement: {statement}\nDate: 2026-09-01\nStatus: active\n\n---\n\n"


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    ws = tmp_path / "ws"
    for sub in ("memory", "decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"workspace_path": str(ws), "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    (ws / "decisions" / "DECISIONS.md").write_text(_block(MAPPED_ID, "a block the prefix map knows"), encoding="utf-8")
    (ws / "entities" / "signals.md").write_text(
        _block(UNMAPPED_ID, "an observed signal") + _block(UNMAPPED_ID_2, "a second observed signal"),
        encoding="utf-8",
    )
    (ws / "decisions" / "ARCHIVE.md").write_text(_block(DISPLACED_ID, "a mapped prefix in a non-canonical file"), encoding="utf-8")
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


def _ids(ws: str) -> set[str]:
    return {str(b["_id"]) for b in MarkdownBlockStore(ws).get_all(active_only=False) if b.get("_id")}


def _records(ws: str) -> list[dict[str, Any]]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _phase(ws: str, phase: str) -> list[dict[str, Any]]:
    return [r for r in _records(ws) if r.get("metadata", {}).get("delete_phase") == phase]


# ---------------------------------------------------------------------------
# The premise, established before anything is asserted about it
# ---------------------------------------------------------------------------


def test_the_unmapped_block_is_real_and_the_prefix_map_does_not_know_it(workspace: str) -> None:
    """Positive control for every test below.

    If the prefix map ever grows a ``TRAJ`` row, this test says so — and
    the rest of the file would be measuring the mapped path while
    claiming to measure the fallback.
    """
    assert _resolve_block_file(workspace, UNMAPPED_ID) is None, "TRAJ is mapped now; pick a prefix that is not"
    assert _resolve_block_file(workspace, MAPPED_ID) is not None
    store = MarkdownBlockStore(workspace)
    assert store.get_by_id(UNMAPPED_ID) is not None, "the store cannot read the block, so nothing here is about deleting it"
    assert _ids(workspace) == {MAPPED_ID, UNMAPPED_ID, UNMAPPED_ID_2, DISPLACED_ID}


def test_the_canonical_file_is_still_tried_first(workspace: str) -> None:
    """The fast path is unchanged: a mapped prefix costs one open."""
    store = MarkdownBlockStore(workspace)
    candidates = store._delete_candidates(MAPPED_ID)
    assert candidates[0] == os.path.join(workspace, "decisions", "DECISIONS.md")
    assert len(candidates) > 1, "the fallback list is empty, so no fallback could ever happen"


# ---------------------------------------------------------------------------
# The doors
# ---------------------------------------------------------------------------


def test_the_delete_route_removes_a_block_outside_the_prefix_map(workspace: str) -> None:
    """404-while-present was the defect; 200-and-gone is the fix."""
    status, body = _handle_delete_memory(workspace, UNMAPPED_ID, actor="alice")

    assert status == 200, f"the door still cannot reach the block: {body}"
    assert body["ok"] is True
    assert UNMAPPED_ID not in _ids(workspace)
    # Untouched neighbours: the fallback scan must splice one block, not a file.
    assert {MAPPED_ID, UNMAPPED_ID_2, DISPLACED_ID} <= _ids(workspace)


def test_the_removal_record_names_the_unmapped_block(workspace: str) -> None:
    """A death on the fallback path is recorded like any other."""
    _handle_delete_memory(workspace, UNMAPPED_ID, actor="alice")

    admitted = _phase(workspace, PHASE_ADMITTED)
    removed = _phase(workspace, PHASE_REMOVED)
    assert len(admitted) == 1 and len(removed) == 1
    assert admitted[0]["target_block_id"] == UNMAPPED_ID
    assert admitted[0]["actor"] == "alice"
    assert removed[0]["metadata"]["operation"] == "delete"
    assert removed[0]["metadata"]["removed_count"] == 1
    assert removed[0]["metadata"]["merkle_root"]


def test_a_mapped_prefix_in_a_non_canonical_file_is_reachable(workspace: str) -> None:
    """The map says where a ``D-`` block *should* be, not where it is."""
    status, _body = _handle_delete_memory(workspace, DISPLACED_ID, actor="alice")

    assert status == 200
    assert DISPLACED_ID not in _ids(workspace)


def test_the_clear_takes_every_block_including_the_unmapped_ones(workspace: str) -> None:
    """A purge that skipped them would be partial and say it was whole."""
    status, body = _handle_clear(workspace, CLEAR_BODY, actor="alice")

    assert status == 200
    assert body["deleted"] == 4, f"the wipe left blocks behind: {sorted(_ids(workspace))}"
    assert _ids(workspace) == set()
    removed = _phase(workspace, PHASE_REMOVED)
    assert len(removed) == 1, "still one bulk record, not one per block"
    assert removed[0]["metadata"]["removed_count"] == 4


# ---------------------------------------------------------------------------
# The fallback does not weaken anything
# ---------------------------------------------------------------------------


def test_an_ungated_delete_of_an_unmapped_block_still_raises(workspace: str) -> None:
    """The new resolution path is behind the same admission check."""
    store = MarkdownBlockStore(workspace)
    assert store.get_by_id(UNMAPPED_ID) is not None, "positive control: the block is here to be refused over"

    with pytest.raises(UngatedDeleteError):
        store.delete_block(UNMAPPED_ID)

    assert store.get_by_id(UNMAPPED_ID) is not None, "the refused delete removed it anyway"
    assert _records(workspace) == []


def test_an_absent_id_still_returns_false_after_the_whole_corpus_is_walked(workspace: str) -> None:
    """The scan must not invent a hit — and must not raise on a miss."""
    store = MarkdownBlockStore(workspace)
    gate = get_gate(workspace)
    before = _ids(workspace)

    with gate.admit_delete("TRAJ-20260901-999", rationale="deleting an id that is not here"):
        assert store.delete_block("TRAJ-20260901-999") is False

    assert _ids(workspace) == before, "a miss changed the corpus"
    assert _phase(workspace, PHASE_REMOVED) == [], "a miss recorded a removal"


def test_a_covered_scope_still_refuses_an_id_it_does_not_cover(workspace: str) -> None:
    """Coverage is checked before resolution, so the scan never runs uncovered."""
    store = MarkdownBlockStore(workspace)
    gate = get_gate(workspace)

    with gate.admit_delete(UNMAPPED_ID, rationale="a scope for one block only"):
        with pytest.raises(UngatedDeleteError, match="does not cover"):
            store.delete_block(UNMAPPED_ID_2)

    assert UNMAPPED_ID_2 in _ids(workspace)


# ---------------------------------------------------------------------------
# Mutation twin
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """Restore prefix-map-only resolution and watch the fallback vanish."""

    def test_prefix_map_only_resolution_cannot_reach_the_block(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-fix candidate list, run against the same corpus.

        ``_delete_candidates`` is narrowed to what ``_resolve_block_file``
        alone could offer — the 5.0.1 shape — and the governed delete of
        a readable block goes back to reporting ``False``.
        """

        def prefix_map_only(self: MarkdownBlockStore, block_id: str) -> list[str]:
            mapped = _resolve_block_file(self._workspace, block_id)
            return [mapped] if mapped is not None and os.path.isfile(mapped) else []

        monkeypatch.setattr(MarkdownBlockStore, "_delete_candidates", prefix_map_only)

        store = MarkdownBlockStore(workspace)
        assert store.get_by_id(UNMAPPED_ID) is not None, "positive control: readable, and about to be undeletable"

        gate = get_gate(workspace)
        with gate.admit_delete(UNMAPPED_ID, rationale="the pre-fix resolution, reproduced"):
            assert store.delete_block(UNMAPPED_ID) is False

        assert UNMAPPED_ID in _ids(workspace), "the mutation did not reproduce the defect"

        status, body = _handle_clear(workspace, CLEAR_BODY)
        assert status == 200
        assert body["deleted"] == 1, "only the canonically-filed block was reachable"
        assert _ids(workspace) == {UNMAPPED_ID, UNMAPPED_ID_2, DISPLACED_ID}, "the partial purge, reported as a whole one"

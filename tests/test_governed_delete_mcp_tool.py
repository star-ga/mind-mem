# Copyright 2026 STARGA, Inc.
"""The third door that killed content without a record: the MCP admin tool.

``tests/test_admission_seam.py`` names three doors that reached
``delete_block`` before 5.0.2 — the ADMIN ``delete_memory_item`` tool,
``DELETE /memories/{id}`` and ``POST /clear``. The two HTTP doors were
wired to ``admit_delete`` in that release. This one was not, and its two
legs failed in opposite directions. Both measured by live probe against a
workspace built by ``mind-mem-init``:

* **Markdown leg** (backend ``markdown``, the zero-config default) — it
  never called ``delete_block`` at all. It resolved the corpus file and
  spliced the lines out itself, so the store-side gate could not see it.
  The tool answered ``{"status": "deleted"}``, the block left
  ``DECISIONS.md``, and ``memory/evidence_chain.jsonl`` gained **zero**
  rows. Content died with no receipt and no record, on the most-used
  delete door in the product.
* **Store leg** (any non-Markdown backend) — it *did* call
  ``delete_block``, which now refuses an ungated caller. The
  ``UngatedDeleteError`` was caught by the leg's own ``except Exception``
  and returned as ``{"error": "Delete failed on the postgres block
  store: ungated delete of ..."}``; the block survived. Fail-closed,
  which is the right direction, but not a working delete surface.

One ``admit_delete`` scope now covers both legs, opened at a single call
site in the tool. The store leg inherits ``record_removal`` from the
store; the Markdown leg reports its own removal, because it is the code
that does the removing.

Every negative assertion here carries a positive control — the block is
shown present before the call — because ``assert gone`` passes just as
well against a fixture that never seeded anything.
:class:`TestMutationTwin` removes the scope and the removal call and
shows these tests going red: a gate never observed failing is not a gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem.admission import UngatedDeleteError, current_admission
from mind_mem.governance_gate import PHASE_ADMITTED, PHASE_REMOVED, evict_gate
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import memory_ops
from mind_mem.mcp.tools.memory_ops import DEFAULT_DELETE_RATIONALE, delete_memory_item

SEED_ID = "D-20260901-001"
KEEP_ID = "D-20260901-002"
CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _admin_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """``delete_memory_item`` is admin-scoped; opt the test process in."""
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


def _make_ws(tmp_path: Path, name: str, config: dict) -> str:
    ws = tmp_path / name
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n\n", encoding="utf-8")
    return str(ws)


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    """The zero-config default: blocks of record on the Markdown corpus."""
    ws = _make_ws(tmp_path, "mdws", {"block_store": {"backend": "markdown"}})
    try:
        yield ws
    finally:
        evict_gate(ws)


def _seed(ws: str, bid: str, statement: str) -> None:
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{bid}]\nStatement: {statement}\nDate: 2026-09-01\nStatus: active\n\n---\n\n")


def _corpus(ws: str) -> str:
    return Path(ws, "decisions", "DECISIONS.md").read_text(encoding="utf-8")


def _present(ws: str, bid: str) -> bool:
    """The positive control: is the block actually in the corpus of record?"""
    return f"[{bid}]" in _corpus(ws)


def _records(ws: str) -> list[dict]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _phase(ws: str, phase: str) -> list[dict]:
    return [r for r in _records(ws) if (r.get("metadata") or {}).get("delete_phase") == phase]


def _call(ws: str, bid: str, **kwargs: Any) -> dict:
    with use_workspace(ws):
        return json.loads(delete_memory_item(bid, **kwargs))


# ---------------------------------------------------------------------------
# A — the Markdown leg, which reached no store at all
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_markdown_delete_records_the_death_it_caused(workspace: str) -> None:
    """The measured hole: this leg used to leave the chain empty."""
    _seed(workspace, SEED_ID, "the block the operator asked to remove")
    _seed(workspace, KEEP_ID, "the block that must survive")
    assert _present(workspace, SEED_ID), "fixture never seeded the block; every assertion below would be vacuous"

    payload = _call(workspace, SEED_ID)

    assert payload["status"] == "deleted"
    assert payload["block_id"] == SEED_ID
    assert not _present(workspace, SEED_ID)
    assert _present(workspace, KEEP_ID), "the splice took a block it was not asked for"

    admitted = _phase(workspace, PHASE_ADMITTED)
    assert len(admitted) == 1, f"expected exactly one authorisation record, got {len(admitted)}"
    assert admitted[0]["target_block_id"] == SEED_ID
    assert admitted[0]["metadata"]["operation"] == "delete"
    assert admitted[0]["metadata"]["rationale"] == DEFAULT_DELETE_RATIONALE
    assert admitted[0]["metadata"]["door"] == "mcp.delete_memory_item"
    assert admitted[0]["metadata"]["backend"] == "markdown"
    assert admitted[0]["actor"], "the record must name an actor, even an unauthenticated one"

    removed = _phase(workspace, PHASE_REMOVED)
    assert len(removed) == 1, "the block is gone and no record says so"
    assert removed[0]["target_block_id"] == SEED_ID
    assert removed[0]["metadata"]["removed_count"] == 1
    # The response hands the caller the receipt id so a client can verify
    # its own deletion against the chain rather than trust the payload.
    assert payload["admission"] == removed[0]["metadata"]["admission_entry_id"]


@pytest.mark.unit
def test_the_removal_record_carries_the_hash_of_what_died(workspace: str) -> None:
    """A record that does not identify the content is most of the way to none."""
    import hashlib

    _seed(workspace, SEED_ID, "content whose hash must reach the chain")
    assert _present(workspace, SEED_ID)
    lines = _corpus(workspace).split("\n")
    start = lines.index(f"[{SEED_ID}]")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("---")) + 1
    expected = hashlib.sha256("\n".join(lines[start:end]).encode("utf-8")).hexdigest()

    _call(workspace, SEED_ID)

    removed = _phase(workspace, PHASE_REMOVED)
    assert len(removed) == 1
    assert removed[0]["payload_hash"] == expected, "the removal record does not hash the content that was removed"


@pytest.mark.unit
def test_a_caller_supplied_rationale_reaches_the_record(workspace: str) -> None:
    """The default names the door; a caller with a reason overrides it."""
    _seed(workspace, SEED_ID, "superseded by the 5.0.2 decision")
    assert _present(workspace, SEED_ID)

    _call(workspace, SEED_ID, rationale="superseded by D-20260901-002")

    assert _phase(workspace, PHASE_ADMITTED)[0]["metadata"]["rationale"] == "superseded by D-20260901-002"


@pytest.mark.unit
def test_a_whitespace_rationale_is_not_a_rationale(workspace: str) -> None:
    """Blank input falls back to the door name rather than refusing.

    The gate refuses an empty rationale, so passing ``"   "`` straight
    through would turn a blank optional field into a failed delete. The
    door name is what the tool honestly has; the fallback is applied
    before the gate sees it.
    """
    _seed(workspace, SEED_ID, "the block a blank rationale must still be able to remove")
    payload = _call(workspace, SEED_ID, rationale="   ")
    assert payload["status"] == "deleted"
    assert _phase(workspace, PHASE_ADMITTED)[0]["metadata"]["rationale"] == DEFAULT_DELETE_RATIONALE


@pytest.mark.unit
def test_a_refused_scope_is_reported_as_a_refusal_not_a_deletion(workspace: str) -> None:
    """403-shaped: the block is still there, and nothing claims otherwise."""
    from mind_mem.governance_gate import GovernanceBypassError, get_gate

    _seed(workspace, SEED_ID, "the block that must survive a refused delete")
    assert _present(workspace, SEED_ID)

    gate = get_gate(workspace)

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise GovernanceBypassError("spec binding drifted")

    original = type(gate).admit_delete
    try:
        type(gate).admit_delete = _refuse  # type: ignore[method-assign]
        payload = _call(workspace, SEED_ID)
    finally:
        type(gate).admit_delete = original  # type: ignore[method-assign]

    assert "error" in payload
    assert payload.get("status") != "deleted"
    assert _present(workspace, SEED_ID), "a refused delete removed the block anyway"
    assert _records(workspace) == [], "a refused authorisation must leave no record"


@pytest.mark.unit
def test_the_markdown_leg_runs_inside_an_open_delete_scope(workspace: str) -> None:
    """Read the seam directly, not just its side effects.

    ``record_removal`` on the wrong receipt would raise, so the previous
    tests already prove *a* delete receipt was open. This one names the
    property: at the moment the corpus is spliced, the ambient admission
    is a DELETE covering exactly this id.
    """
    _seed(workspace, SEED_ID, "the block whose splice must be covered")
    seen: dict[str, Any] = {}
    original = memory_ops._delete_from_corpus

    def _spy(ws: str, block_id: str, filepath: str, receipt: Any) -> str:
        active = current_admission()
        seen["receipt_is_ambient"] = active is receipt
        seen["operation"] = None if active is None else active.operation
        seen["covers"] = set() if active is None else set(active.covers)
        return original(ws, block_id, filepath, receipt)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(memory_ops, "_delete_from_corpus", _spy)
        payload = _call(workspace, SEED_ID)
    finally:
        monkey.undo()

    assert payload["status"] == "deleted"
    assert seen["receipt_is_ambient"] is True
    assert seen["operation"] == "delete"
    assert seen["covers"] == {SEED_ID}


# ---------------------------------------------------------------------------
# B — the store leg, which fail-closed into being unusable
# ---------------------------------------------------------------------------


class _GovernedStore:
    """A store double that obeys the whole delete contract.

    Check first, then ``record_removal`` on success — the same order the
    five real stores use. A permissive double would make this suite
    measure the double instead of the tool.
    """

    def __init__(self, rows: dict[str, str]) -> None:
        self.rows = dict(rows)

    def ping(self) -> dict[str, Any]:
        return {"ok": True}

    def list_blocks(self) -> list[str]:
        return ["decisions/DECISIONS.md"]

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        return [{"_id": bid, "Statement": text, "Status": "active"} for bid, text in sorted(self.rows.items())]

    def get_by_id(self, block_id: str) -> dict[str, Any] | None:
        return {"_id": block_id, "Statement": self.rows[block_id]} if block_id in self.rows else None

    def delete_block(self, block_id: str) -> bool:
        from mind_mem.admission import require_delete_admission

        receipt = require_delete_admission(str(block_id))
        removed = self.rows.pop(str(block_id), None)
        if removed is None:
            return False
        receipt.record_removal(str(block_id), removed)
        return True


@pytest.fixture
def store_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, _GovernedStore]]:
    """A workspace whose blocks of record live in a non-Markdown store."""
    ws = _make_ws(tmp_path, "pgws", {"block_store": {"backend": "postgres", "dsn": "postgresql:///unused"}})
    store = _GovernedStore({SEED_ID: "the store-resident block", KEEP_ID: "the survivor"})
    factory = lambda workspace, config=None: store  # noqa: E731 - test double
    monkeypatch.setattr("mind_mem.storage.get_block_store", factory)
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.get_block_store", factory)
    try:
        yield ws, store
    finally:
        evict_gate(ws)


@pytest.mark.unit
def test_the_store_leg_works_again_and_records_the_death(store_workspace: tuple[str, _GovernedStore]) -> None:
    """Before the scope existed this call raised ``UngatedDeleteError``."""
    ws, store = store_workspace
    assert store.get_by_id(SEED_ID) is not None, "fixture never seeded the block"

    payload = _call(ws, SEED_ID)

    assert payload["status"] == "deleted"
    assert payload["backend"] == "postgres"
    assert store.get_by_id(SEED_ID) is None
    assert store.get_by_id(KEEP_ID) is not None

    admitted = _phase(ws, PHASE_ADMITTED)
    assert len(admitted) == 1
    assert admitted[0]["metadata"]["backend"] == "postgres"
    removed = _phase(ws, PHASE_REMOVED)
    assert len(removed) == 1
    assert removed[0]["metadata"]["removed_count"] == 1
    assert payload["admission"] == removed[0]["metadata"]["admission_entry_id"]


@pytest.mark.unit
def test_an_absent_store_id_is_authorised_but_records_no_death(store_workspace: tuple[str, _GovernedStore]) -> None:
    """Probing resistance: the scope opens before the target is resolved."""
    ws, _store = store_workspace
    payload = _call(ws, "D-20260901-999")
    assert "block store" in payload["error"]
    assert payload.get("status") != "deleted"
    assert len(_phase(ws, PHASE_ADMITTED)) == 1, "the attempt must still be recorded"
    assert _phase(ws, PHASE_REMOVED) == [], "a delete that removed nothing must claim no death"


# ---------------------------------------------------------------------------
# C — the refusals that must survive the change
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_ungated_store_delete_still_raises(store_workspace: tuple[str, _GovernedStore]) -> None:
    """The tool got a scope; the seam did not get a bypass.

    Positive control in the same test: the block is present before the
    ungated call and still present after it, so the refusal cannot be an
    artefact of a store that never held it.
    """
    _ws, store = store_workspace
    assert store.get_by_id(SEED_ID) is not None
    with pytest.raises(UngatedDeleteError):
        store.delete_block(SEED_ID)
    assert store.get_by_id(SEED_ID) is not None, "an ungated delete removed the block anyway"


@pytest.mark.unit
def test_an_unroutable_prefix_is_refused_before_any_scope_opens(workspace: str) -> None:
    """A prefix this door cannot route mints no authorisation record."""
    _seed(workspace, SEED_ID, "an unrelated block")
    payload = _call(workspace, "UNKNOWN-123")
    assert "Unrecognized" in payload["error"]
    assert _records(workspace) == [], "an id the door never routed left a record anyway"
    assert _present(workspace, SEED_ID)


@pytest.mark.unit
def test_an_invalid_block_id_is_refused_before_any_scope_opens(workspace: str) -> None:
    payload = _call(workspace, "invalid!!!")
    assert "Invalid block ID" in payload["error"]
    assert _records(workspace) == []


@pytest.mark.unit
def test_a_markdown_id_that_is_not_there_records_no_death(workspace: str) -> None:
    """Authorised, attempted, removed nothing — and says exactly that."""
    _seed(workspace, KEEP_ID, "the only block in the corpus")
    payload = _call(workspace, SEED_ID)
    assert "not found" in payload["error"].lower()
    assert payload.get("status") != "deleted"
    assert len(_phase(workspace, PHASE_ADMITTED)) == 1
    assert _phase(workspace, PHASE_REMOVED) == []
    assert _present(workspace, KEEP_ID)


# ---------------------------------------------------------------------------
# D — the mutation twin. A gate never observed failing is not a gate.
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """Restore the pre-fix shape and watch the protective tests go red."""

    @pytest.mark.unit
    def test_without_the_scope_the_markdown_delete_leaves_no_record(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 5.0.1 behaviour, reproduced: deleted, and nothing recorded.

        The twin bypasses the scope exactly as the old code did — call the
        splice with no admission open — and asserts the measured defect.
        If this ever fails, the splice acquired a second, hidden gate and
        the test above is measuring something other than the scope.
        """
        from contextlib import nullcontext

        _seed(workspace, SEED_ID, "the block the unscoped splice takes")
        assert _present(workspace, SEED_ID)

        class _NoReceipt:
            entry_id = "no-scope"

            def record_removal(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        def _unscoped(*_args: Any, **_kwargs: Any) -> Any:
            return nullcontext(_NoReceipt())

        monkeypatch.setattr("mind_mem.governance_gate.GovernanceGate.admit_delete", _unscoped)
        payload = _call(workspace, SEED_ID)

        assert payload["status"] == "deleted", "the twin must reproduce a working delete, not a broken one"
        assert not _present(workspace, SEED_ID)
        assert _records(workspace) == [], "the twin did not actually bypass the gate"

    @pytest.mark.unit
    def test_dropping_record_removal_loses_the_removal_row(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The half-fix: a scope that authorises and never reports.

        Wrapping the splice without reporting the removal would leave an
        authorisation for a death the chain never recorded — the mirror
        of the ``/clear`` receipt-for-a-death-that-never-happened defect.
        This twin proves the ``record_removal`` call in
        ``_delete_from_corpus`` is what produces the second row.
        """
        _seed(workspace, SEED_ID, "the block whose removal goes unreported")
        assert _present(workspace, SEED_ID)

        monkeypatch.setattr(
            "mind_mem.admission.AdmissionReceipt.record_removal",
            lambda self, block_id, content: None,
        )
        payload = _call(workspace, SEED_ID)

        assert payload["status"] == "deleted"
        assert not _present(workspace, SEED_ID)
        assert len(_phase(workspace, PHASE_ADMITTED)) == 1
        assert _phase(workspace, PHASE_REMOVED) == [], "the twin did not actually drop the removal report"

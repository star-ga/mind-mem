"""``delete_memory_item`` and ``memory_health`` on a non-Markdown backend.

Two defects, both about a tool asking a different question than its own
precondition did:

* ``delete_memory_item`` passed the backend-aware workspace gate (which
  pings the configured store) and then did pure Markdown work — resolving
  a corpus file, splicing lines out of ``DECISIONS.md``. On a store-backed
  workspace those files are empty init templates, so the delete reported
  the block id as unknown and removed nothing while the block stayed in
  the store.
* ``memory_health`` produced its single ``total_active`` field from two
  non-equivalent predicates: ``block_parser.get_active`` over files that
  exclude ``*_ARCHIVE.md`` on the Markdown branch, versus an open-coded
  ``Status`` lookup that treated a MISSING status as active, ignored the
  store's own activity flag, and scanned archived rows, on the store
  branch. The store branch now delegates the Status half to the very
  function the Markdown branch uses, so the two cannot drift again.

The store here is a stand-in that implements the slice of the block-store
protocol these tools use, so the coverage does not need a live Postgres.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import delete_memory_item, memory_health

CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory")


@pytest.fixture(autouse=True)
def _admin_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """``delete_memory_item`` is admin-scoped; opt the test process in."""
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


class _FakeStore:
    """The read/delete slice of the block-store protocol, in memory."""

    def __init__(self, blocks: list[dict[str, Any]]) -> None:
        self._blocks = {str(b["_id"]): dict(b) for b in blocks}

    # health probe used by the workspace gate
    def ping(self) -> dict[str, Any]:
        return {"ok": True}

    def list_blocks(self) -> list[str]:
        return sorted({str(b.get("_source_file", "")) for b in self._blocks.values()})

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        return [dict(b) for b in self._blocks.values()]

    def get_by_id(self, block_id: str) -> dict[str, Any] | None:
        block = self._blocks.get(block_id)
        return dict(block) if block else None

    def delete_block(self, block_id: str) -> bool:
        return self._blocks.pop(block_id, None) is not None

    def holds(self, block_id: str) -> bool:
        return block_id in self._blocks


def _store_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocks: list[dict]) -> tuple[str, _FakeStore]:
    """A workspace whose configured backend is not the Markdown corpus."""
    ws = tmp_path / "ws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    # What init_workspace leaves on disk for a store-backed workspace: the
    # corpus files exist but hold no blocks.
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n", encoding="utf-8")
    (ws / "mind-mem.json").write_text(
        json.dumps({"block_store": {"backend": "postgres", "dsn": "postgresql:///unused"}}),
        encoding="utf-8",
    )
    store = _FakeStore(blocks)
    factory = lambda workspace, config=None: store  # noqa: E731 - test double
    monkeypatch.setattr("mind_mem.storage.get_block_store", factory)
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.get_block_store", factory)
    return str(ws), store


def _markdown_workspace(tmp_path: Path, body: str) -> str:
    ws = tmp_path / "mdws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(json.dumps({"recall": {"backend": "scan"}}), encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text(body, encoding="utf-8")
    (ws / "decisions" / "DECISIONS_ARCHIVE.md").write_text(
        "[D-20260613-404]\nStatement: archived\nStatus: active\nDate: 2026-06-13\n\n---\n",
        encoding="utf-8",
    )
    return str(ws)


# ─── delete_memory_item ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_delete_removes_the_store_resident_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws, store = _store_workspace(
        tmp_path,
        monkeypatch,
        [{"_id": "D-20260501-001", "_source_file": "decisions/DECISIONS.md", "Statement": "delete me", "Status": "active"}],
    )
    with use_workspace(ws):
        payload = json.loads(delete_memory_item("D-20260501-001"))
    assert payload["status"] == "deleted"
    assert payload["block_id"] == "D-20260501-001"
    assert not store.holds("D-20260501-001")


@pytest.mark.unit
def test_delete_of_an_absent_store_block_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws, _store = _store_workspace(tmp_path, monkeypatch, [])
    with use_workspace(ws):
        payload = json.loads(delete_memory_item("D-20260501-999"))
    assert "block store" in payload["error"]
    assert "status" not in payload


@pytest.mark.unit
def test_delete_on_the_markdown_default_still_splices_the_file(tmp_path: Path) -> None:
    """The zero-config path must be untouched by the backend branch."""
    ws = _markdown_workspace(
        tmp_path,
        "[D-20260613-501]\nStatement: keep\nStatus: active\nDate: 2026-06-13\n\n---\n"
        "[D-20260613-502]\nStatement: drop\nStatus: active\nDate: 2026-06-13\n\n---\n",
    )
    with use_workspace(ws):
        payload = json.loads(delete_memory_item("D-20260613-502"))
    assert payload["status"] == "deleted"
    assert payload["file"] == "DECISIONS.md"
    remaining = (Path(ws) / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")
    assert "D-20260613-501" in remaining
    assert "D-20260613-502" not in remaining


# ─── memory_health activity predicate ─────────────────────────────────────────

_STATUS_CASES = [
    # (block_id, Status field or None, extra fields)
    ("D-20260613-601", "active", {}),
    ("D-20260613-602", "Active", {}),  # a spelling of the same state
    ("D-20260613-603", None, {"_active": False}),  # missing Status read as active
    ("D-20260613-604", "superseded", {}),
]


@pytest.mark.unit
def test_store_activity_matches_the_markdown_predicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks: list[dict] = []
    for bid, status, extra in _STATUS_CASES:
        block: dict[str, Any] = {"_id": bid, "_source_file": "decisions/DECISIONS.md", "Statement": bid, **extra}
        if status is not None:
            block["Status"] = status
        blocks.append(block)
    # An archived block: the Markdown branch never opens *_ARCHIVE.md, so
    # the store branch must not count it either.
    blocks.append(
        {
            "_id": "D-20260613-605",
            "_source_file": "decisions/DECISIONS_ARCHIVE.md",
            "Statement": "archived",
            "Status": "active",
        }
    )
    ws, _store = _store_workspace(tmp_path, monkeypatch, blocks)
    with use_workspace(ws):
        health = json.loads(memory_health())

    assert health["total_blocks"] == 4  # the archived block is out of scope
    # active/Active are one state; a MISSING Status is not active, and the
    # store's own deactivation outranks whatever the metadata says.
    assert health["total_active"] == 2
    assert health["corpus"]["decisions"] == {"total": 4, "active": 2}

    # The same four blocks as Markdown must give the same two numbers —
    # that comparability is the whole point of the shared field name.
    md_body = "".join(
        f"[{bid}]\nStatement: {bid}\n" + (f"Status: {status}\n" if status else "") + "Date: 2026-06-13\n\n---\n"
        for bid, status, _extra in _STATUS_CASES
    )
    md_ws = _markdown_workspace(tmp_path, md_body)
    with use_workspace(md_ws):
        md_health = json.loads(memory_health())
    assert (md_health["total_blocks"], md_health["total_active"]) == (health["total_blocks"], health["total_active"])


@pytest.mark.unit
def test_store_deactivated_row_is_not_active_despite_its_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The store's own activity column outranks a stale metadata Status."""
    ws, _store = _store_workspace(
        tmp_path,
        monkeypatch,
        [
            {
                "_id": "D-20260613-701",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "deactivated in the store",
                "Status": "active",
                "_active": False,
            }
        ],
    )
    with use_workspace(ws):
        health = json.loads(memory_health())
    assert health["total_blocks"] == 1
    assert health["total_active"] == 0

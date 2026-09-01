"""Regression: ``v4.block_kinds`` / ``v4.kind_summaries`` must CLOSE connections.

Both modules used ``with sqlite3.connect(...) as conn:`` at every call site.
That context manager commits (or rolls back) and then leaves the handle
**open** — closing is not part of what it does. Nothing else reclaimed it
either: a :class:`sqlite3.Connection` owns a prepared-statement cache that
refers back to the connection, so each one sits in a reference cycle that
refcounting cannot break; only the cyclic collector could, at an arbitrary
later time. The result was one leaked descriptor set per call.

**Every observation below is taken while the cyclic collector is off**, and
only then compared. That is load-bearing, not incidental: an earlier draft of
these tests re-enabled gc before asserting, and two of them passed against the
unfixed code because a gen-0 collection happened to run in between and closed
the leaked connections first. A test that lets the collector clean up before
it looks cannot see the defect it exists to catch.

The workspace is seeded in WAL journal mode — persistent in the DB header, and
how the rest of the v4 surface leaves this store — so an unclosed handle also
pins the ``-wal`` / ``-shm`` sidecars. That gives a portable observation:
SQLite removes the sidecars when the *last* connection closes, so their
survival is direct evidence of a live handle on every platform, including the
Windows case where such a handle makes ``unlink`` / ``rmdir`` fail.
"""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from mind_mem.v4.block_kinds import (
    BlockKind,
    ensure_block_kind_column,
    ensure_block_kind_tags_table,
    get_block_kind,
    get_block_kind_tags,
    list_blocks_by_kind,
    set_block_kinds,
)
from mind_mem.v4.kind_summaries import (
    ensure_kind_summary_schema,
    get_summary,
    list_summaries,
    refresh_summary,
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Workspace with both flags ON and a WAL-mode ``index.db`` already seeded."""
    cfg = {"v4": {"block_kinds": {"enabled": True}, "kind_summaries": {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))

    seed = sqlite3.connect(tmp_path / "index.db")
    try:
        seed.execute("PRAGMA journal_mode=WAL")
        seed.execute("CREATE TABLE IF NOT EXISTS blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT NOT NULL DEFAULT 'unspecified')")
        seed.execute("INSERT OR REPLACE INTO blocks (id, content, kind) VALUES ('b1', 'head\nrest', 'entity')")
        seed.commit()
    finally:
        seed.close()
    return tmp_path


@contextmanager
def _no_gc() -> Iterator[None]:
    """Run *and observe* with the cyclic collector off. See module docstring."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _exercise(ws: Path) -> None:
    """Call every public entry point of both modules once."""
    ensure_block_kind_column(ws)
    get_block_kind(ws, "b1")
    list_blocks_by_kind(ws, BlockKind.ENTITY)
    ensure_block_kind_tags_table(ws)
    set_block_kinds(ws, "b1", [BlockKind.ENTITY, BlockKind.CODE])
    get_block_kind_tags(ws, "b1")
    ensure_kind_summary_schema(ws)
    refresh_summary(ws, "entity")
    get_summary(ws, "entity")
    list_summaries(ws)


def _sidecars(ws: Path) -> list[str]:
    return sorted(p.name for p in ws.iterdir() if p.name.startswith("index.db-"))


def _open_index_db_fds() -> list[str]:
    """Descriptors this process holds on ``index.db`` and its sidecars (Linux)."""
    held: list[str] = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:  # fd closed between listdir and readlink
            continue
        if "index.db" in target:
            held.append(target)
    return held


@pytest.mark.unit
def test_wal_sidecars_do_not_survive_the_call(workspace: Path) -> None:
    """No ``-wal`` / ``-shm`` left behind — SQLite drops them on the last close.

    Portable stand-in for a handle count: on every platform the sidecars
    outlive the call only while some connection is still open.
    """
    with _no_gc():
        for _ in range(5):
            _exercise(workspace)
        surviving = _sidecars(workspace)

    assert surviving == [], f"sidecars survived, so a connection is still open: {surviving}"


@pytest.mark.unit
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/fd is Linux-only")
def test_no_descriptors_left_open_on_the_v4_store(workspace: Path) -> None:
    """The definitive count: zero descriptors on ``index.db*`` after the calls."""
    with _no_gc():
        for _ in range(10):
            _exercise(workspace)
        held = _open_index_db_fds()

    assert held == [], f"{len(held)} descriptor(s) still open on the v4 store: {sorted(set(held))}"


@pytest.mark.unit
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/fd is Linux-only")
def test_descriptor_count_is_flat_across_repeated_calls(workspace: Path) -> None:
    """Descriptor use must not grow with call count — the leak was unbounded."""
    with _no_gc():
        for _ in range(2):  # warm up: imports, schema creation
            _exercise(workspace)
        before = len(os.listdir("/proc/self/fd"))
        for _ in range(20):
            _exercise(workspace)
        after = len(os.listdir("/proc/self/fd"))

    assert after == before, f"descriptor count grew {before} -> {after} over 20 rounds"


@pytest.mark.unit
def test_writes_still_commit_before_the_close(workspace: Path) -> None:
    """Closing must not swallow the write.

    ``close()`` never commits, so the transaction context has to exit
    *before* it. If the two were ordered the other way, the tag set and the
    summary row below would be rolled back and this read — made from a
    brand-new connection, after the writer is gone — would come up empty.
    """
    _exercise(workspace)

    probe = sqlite3.connect(workspace / "index.db")
    try:
        tags = {r[0] for r in probe.execute("SELECT kind FROM block_kind_tags WHERE block_id = 'b1'")}
        summaries = probe.execute("SELECT kind, block_count FROM kind_summaries").fetchall()
    finally:
        probe.close()

    assert tags == {"entity", "code"}
    assert summaries == [("entity", 1)]

    # And through the module's own readers.
    assert get_block_kind_tags(workspace, "b1") == {BlockKind.ENTITY, BlockKind.CODE}
    stored = get_summary(workspace, "entity")
    assert stored is not None and stored.block_count == 1


@pytest.mark.unit
def test_failed_write_rolls_back_and_still_closes(workspace: Path) -> None:
    """An exception inside the block rolls back *and* releases the handle.

    Guards the other half of the ordering: the transaction context must keep
    its rollback behaviour when a close is layered on top of it.
    """
    ensure_block_kind_tags_table(workspace)
    set_block_kinds(workspace, "b1", [BlockKind.ENTITY])

    with _no_gc():
        with pytest.raises(ValueError):
            # 'not-a-kind' fails BlockKind() validation, aborting the call.
            set_block_kinds(workspace, "b1", [BlockKind.CODE, "not-a-kind"])
        tags = get_block_kind_tags(workspace, "b1")
        surviving = _sidecars(workspace)

    assert tags == {BlockKind.ENTITY}, "the failed call must not have changed the tag set"
    assert surviving == [], f"sidecars survived a failed call: {surviving}"

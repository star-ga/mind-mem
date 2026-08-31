"""Regression: ``v4.block_metadata`` / ``v4.pq`` must CLOSE their connections.

Both modules opened every connection as ``with sqlite3.connect(...) as conn:``.
That context manager commits (or, on an exception, rolls back) and then leaves
the handle **open** — closing is not part of what it does. Nothing else
reclaimed it either: a :class:`sqlite3.Connection` owns a prepared-statement
cache that refers back to the connection, so each one sits in a reference
cycle that refcounting cannot break; only the cyclic collector could, at some
arbitrary later time. The result was a leaked descriptor set per call, growing
without bound.

**Every observation below is taken while the cyclic collector is off**, and
only then compared. That is load-bearing, not incidental: with gc enabled a
generation-0 pass can run mid-loop, break the cycles and close the leaked
connections, so a test that looks afterwards sees a clean process and passes
against the unfixed code — it cannot see the defect it exists to catch.

The workspace is seeded in WAL journal mode — persistent in the DB header, and
how the rest of mind-mem leaves this store — so an unclosed handle also pins
the ``-wal`` / ``-shm`` sidecars. That gives a portable observation: SQLite
removes the sidecars when the *last* connection closes, so their survival is
direct evidence of a live handle on every platform, including the Windows case
where such a handle makes ``unlink`` / ``rmdir`` on the workspace fail.
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

from mind_mem.v4.block_metadata import (
    delete_block_metadata,
    ensure_metadata_schema,
    get_block_metadata,
    list_blocks_by_tag,
    set_block_metadata,
)
from mind_mem.v4.pq import (
    Codebook,
    PQConfig,
    ensure_pq_schema,
    load_code,
    load_codebook,
    store_code,
    store_codebook,
)

_CFG = PQConfig(subvectors=4, centroids=4)


def _codebook() -> Codebook:
    """A small, fully-specified 4x4x2 codebook (D = 8). No training needed."""
    return Codebook(
        cfg=_CFG,
        centroids=tuple(tuple((float(m + k), float(m - k)) for k in range(4)) for m in range(4)),
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Workspace with both flags ON and a WAL-mode ``index.db`` already seeded."""
    cfg = {
        "v4": {
            "block_metadata": {"enabled": True},
            "pq": {"enabled": True, "subvectors": 4, "centroids": 4},
        }
    }
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))

    seed = sqlite3.connect(tmp_path / "index.db")
    try:
        seed.execute("PRAGMA journal_mode=WAL")
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
    """Call every connection-opening entry point of both modules once."""
    ensure_metadata_schema(ws)
    set_block_metadata(ws, "b1", {"proj": "mind", "env": "dev"}, ttl_seconds=60)
    get_block_metadata(ws, "b1")
    list_blocks_by_tag(ws, "proj", "mind", limit=10)
    delete_block_metadata(ws, "b1")

    ensure_pq_schema(ws)
    store_codebook(ws, "cb", _codebook())
    load_codebook(ws, "cb")
    store_code(ws, "b1", "cb", b"\x00\x01\x02\x03")
    load_code(ws, "b1", "cb")


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
def test_every_connection_handed_out_is_closed(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the mechanism directly: each connection opened is later closed.

    A zero descriptor count is the symptom; this is the cause. Every
    connection the two modules open is recorded, and afterwards each one is
    asked to run a trivial statement — a closed connection raises
    ``ProgrammingError``, an open one answers. No vocabulary flag is set, so
    nothing else in the call path opens a database.
    """
    handed_out: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def recording_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        handed_out.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", recording_connect)
    with _no_gc():
        _exercise(workspace)
    monkeypatch.undo()

    # Ten public entry points, three of which re-enter an ensure_*_schema
    # helper that opens its own connection: 13 today. Asserted as a floor
    # rather than an equality so collapsing a redundant open stays a legal
    # change, while "we observed nothing at all" still fails.
    opened = len(handed_out)
    assert opened >= 10, f"expected at least one connection per entry point, saw {opened}"

    still_open = 0
    for conn in handed_out:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            continue  # closed, as required
        still_open += 1
        conn.close()  # don't leak out of the test itself
    handed_out.clear()

    assert still_open == 0, f"{still_open} of {opened} connections were left open"


@pytest.mark.unit
def test_writes_still_commit_before_the_close(workspace: Path) -> None:
    """Closing must not swallow the write.

    ``close()`` never commits, so the transaction context has to exit
    *before* it. If the two were ordered the other way, the metadata row and
    the PQ code below would be discarded and these reads — made from a
    brand-new connection, after the writer is gone — would come up empty.
    """
    set_block_metadata(workspace, "keep-1", {"proj": "mind"}, ttl_seconds=90)
    store_codebook(workspace, "cb", _codebook())
    store_code(workspace, "keep-1", "cb", b"\x01\x02\x03\x04")

    probe = sqlite3.connect(workspace / "index.db")
    try:
        meta = probe.execute("SELECT tags, ttl_seconds FROM block_metadata WHERE block_id = 'keep-1'").fetchone()
        code = probe.execute("SELECT code FROM pq_codes WHERE block_id = 'keep-1' AND codebook = 'cb'").fetchone()
        book = probe.execute("SELECT subvectors, centroids, sub_dim FROM pq_codebook WHERE name = 'cb'").fetchone()
    finally:
        probe.close()

    assert meta is not None and json.loads(meta[0]) == {"proj": "mind"} and meta[1] == 90
    assert code is not None and bytes(code[0]) == b"\x01\x02\x03\x04"
    assert book == (4, 4, 2)

    # And through the modules' own readers.
    stored = get_block_metadata(workspace, "keep-1")
    assert stored is not None and stored.tags == {"proj": "mind"} and stored.ttl_seconds == 90
    assert load_code(workspace, "keep-1", "cb") == b"\x01\x02\x03\x04"
    reloaded = load_codebook(workspace, "cb")
    assert reloaded is not None and reloaded.centroids == _codebook().centroids


@pytest.mark.unit
def test_delete_still_reports_rowcount_after_the_close(workspace: Path) -> None:
    """``delete_block_metadata`` reads its cursor's rowcount, then closes.

    The return value has to survive the close being layered underneath it:
    True when a row went away, False when there was nothing to delete.
    """
    set_block_metadata(workspace, "gone", {"proj": "mind"})
    with _no_gc():
        first = delete_block_metadata(workspace, "gone")
        second = delete_block_metadata(workspace, "gone")
        surviving = _sidecars(workspace)

    assert first is True, "deleting an existing row must report True"
    assert second is False, "deleting an absent row must report False"
    assert get_block_metadata(workspace, "gone") is None
    assert surviving == [], f"sidecars survived the deletes: {surviving}"


@pytest.mark.unit
def test_failed_write_releases_the_handle_and_writes_nothing(workspace: Path) -> None:
    """An exception inside the block must still release the handle.

    Guards the other half of the ordering: layering a close under the
    transaction context must not cost the exception path. ``_serialise_codebook``
    raises from inside the connection block here (the codebook declares four
    subvector tables but carries one), and ``ttl_seconds`` below is a type
    SQLite cannot bind, so the INSERT itself raises.

    Note what this does *not* claim: neither call has a committed statement
    ahead of the failure, because no public entry point in these two modules
    issues two writes in one transaction. What is asserted is that the failed
    call leaves no row and no open handle.
    """
    ensure_pq_schema(workspace)
    ensure_metadata_schema(workspace)
    malformed = Codebook(cfg=_CFG, centroids=(((0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)),))

    with _no_gc():
        with pytest.raises(IndexError):
            store_codebook(workspace, "bad", malformed)
        with pytest.raises((sqlite3.InterfaceError, sqlite3.ProgrammingError)):
            set_block_metadata(workspace, "bad", {"proj": "mind"}, ttl_seconds=object())  # type: ignore[arg-type]
        surviving = _sidecars(workspace)

    assert surviving == [], f"sidecars survived a failed call: {surviving}"
    assert load_codebook(workspace, "bad") is None, "the failed store must not have written a codebook row"
    assert get_block_metadata(workspace, "bad") is None, "the failed upsert must not have written a metadata row"

"""Every SQLite connection ``tool_output/store.py`` and ``v4/hnsw_kind_index.py``
open must be CLOSED before the call that opened it returns.

Both modules used the ``with sqlite3.connect(...) as conn`` idiom. That context
manager commits (or, on an exception, rolls back) and then leaves the handle
**open** — its ``__exit__`` documents exactly that and nothing more. Nothing
reclaims it afterwards either: a ``sqlite3.Connection`` and its
prepared-statement cache reference each other, so the object is unreachable only
to the *cyclic* collector, never to refcounting.

Consequences, all of them observable and all of them tested here:

* descriptors accumulate — one per call, on modules called once per stored tool
  run and once per registered embedding;
* the ``-wal`` / ``-shm`` sidecars cannot be checkpointed away while a
  connection still holds them, so they outlive the work;
* on Windows an open handle makes ``os.unlink`` / ``rmdir`` fail, so a workspace
  containing either database cannot be deleted.

The tests below come in three flavours deliberately:

* ``_closes_every_sqlite_connection`` probes the connection objects themselves
  — the mechanism, i.e. was ``close()`` actually reached on each one;
* ``_leaves_no_wal_sidecars`` asserts only on files on disk, with no
  instrumentation at all: SQLite removes the sidecars when the last connection
  to the database closes, so their survival is the leak made visible;
* ``_survive_the_close`` / ``_rolls_back_before_closing`` pin the transaction
  half. ``close()`` never commits, and closing over an open transaction
  discards it, so adding a close in the wrong place would trade a descriptor
  leak for silently lost writes. These fail if commit-then-close order breaks.

A fix that closed connections but dropped the commit — or committed but left
handles open — fails one of the groups.

The cyclic collector is DISABLED throughout: with it running, a well-timed
collection can close the leaked connections and make the leak look fixed.
"""

from __future__ import annotations

import gc
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from mind_mem.tool_output import ToolOutputStore
from mind_mem.v4.hnsw_kind_index import FLAG as HNSW_FLAG
from mind_mem.v4.hnsw_kind_index import (
    backend_status,
    ensure_hnsw_schema,
    knn_by_kind,
    register_block_embedding,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def no_gc() -> Iterator[None]:
    """Run the body with the cyclic collector off.

    Without this the test measures the collector's timing rather than the
    module's behaviour: the leaked connections sit in a reference cycle, so a
    collection that happens to fire mid-test closes them and the leak vanishes.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@pytest.fixture
def opened_connections(monkeypatch: pytest.MonkeyPatch) -> list[sqlite3.Connection]:
    """Record every ``sqlite3.Connection`` handed out while the test runs.

    Note this list holds a strong reference to each connection on purpose — it
    is what lets the assertion ask a *specific* connection whether it is closed.
    That is a testing device and must not be copied into library code, where
    exactly this shape (a registry that outlives the borrower) is itself a
    descriptor leak. ``monkeypatch`` restores ``sqlite3.connect`` and the list
    is dropped when the test ends.
    """
    real_connect = sqlite3.connect
    seen: list[sqlite3.Connection] = []

    def spy(*args: object, **kwargs: object) -> sqlite3.Connection:
        con = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        seen.append(con)
        return con

    monkeypatch.setattr(sqlite3, "connect", spy)
    return seen


def _still_open(connections: list[sqlite3.Connection]) -> int:
    """How many of *connections* are still usable (i.e. were never closed)."""
    count = 0
    for con in connections:
        try:
            con.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            continue  # "Cannot operate on a closed database."
        count += 1
    return count


def _seed_wal(db: Path) -> bool:
    """Persist ``journal_mode=WAL`` in *db*'s header, then fully close.

    WAL is a property of the file, so every later connection inherits it and
    creates the ``-wal`` / ``-shm`` sidecars. Returns False when the filesystem
    refuses WAL, so the sidecar tests can skip rather than pass vacuously.
    """
    con = sqlite3.connect(db)
    try:
        mode = con.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        con.commit()
    finally:
        con.close()
    return str(mode).lower() == "wal"


def _sidecars(db: Path) -> list[str]:
    return [suffix for suffix in ("-wal", "-shm") if db.with_name(db.name + suffix).exists()]


@pytest.fixture
def hnsw_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {HNSW_FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


def _exercise_hnsw(workspace: Path) -> None:
    """One pass over every public entry point that opens a connection."""
    backend_status(workspace)
    ensure_hnsw_schema(workspace)
    register_block_embedding(workspace, "B-1", "entity", [1.0, 0.0, 0.5])
    knn_by_kind(workspace, "entity", [1.0, 0.0, 0.5], k=3)


def _exercise_store(store: ToolOutputStore) -> None:
    """One pass over every public entry point that opens a connection."""
    handle = store.store_and_summarize("alpha\nbeta\n", source="cmd", ts="").handle
    assert store.recall_output(handle) == "alpha\nbeta\n"
    assert store.meta(handle) is not None
    store.gc()


# ---------------------------------------------------------------------------
# tool_output/store.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tool_output_store_closes_every_sqlite_connection(
    tmp_path: Path,
    no_gc: None,
    opened_connections: list[sqlite3.Connection],
) -> None:
    store = ToolOutputStore(sqlite_path=str(tmp_path / "t.db"), max_rows=5)
    for _ in range(3):
        _exercise_store(store)

    # __init__ + 3 × (store + recall + meta + gc) = 13 connections.
    assert len(opened_connections) >= 13, "spy never saw the module's connections — test wired wrong"
    assert _still_open(opened_connections) == 0


@pytest.mark.unit
def test_tool_output_store_leaves_no_wal_sidecars(tmp_path: Path, no_gc: None) -> None:
    db = tmp_path / "t.db"
    if not _seed_wal(db):
        pytest.skip("filesystem does not support WAL journal mode")

    store = ToolOutputStore(sqlite_path=str(db), max_rows=5)
    _exercise_store(store)

    assert _sidecars(db) == []


# ---------------------------------------------------------------------------
# v4/hnsw_kind_index.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hnsw_kind_index_closes_every_sqlite_connection(
    hnsw_workspace: Path,
    no_gc: None,
    opened_connections: list[sqlite3.Connection],
) -> None:
    for _ in range(3):
        _exercise_hnsw(hnsw_workspace)

    # 3 × (backend_status + ensure + register [which re-ensures] + knn) = 15.
    assert len(opened_connections) >= 15, "spy never saw the module's connections — test wired wrong"
    assert _still_open(opened_connections) == 0


@pytest.mark.unit
def test_hnsw_kind_index_leaves_no_wal_sidecars(hnsw_workspace: Path, no_gc: None) -> None:
    ensure_hnsw_schema(hnsw_workspace)  # creates index.db
    db = hnsw_workspace / "index.db"
    if not _seed_wal(db):
        pytest.skip("filesystem does not support WAL journal mode")

    _exercise_hnsw(hnsw_workspace)

    assert _sidecars(db) == []


@pytest.mark.unit
def test_hnsw_workspace_directory_is_removable_after_use(hnsw_workspace: Path, no_gc: None) -> None:
    """The Windows symptom, asserted where it actually bites.

    Honest scope: this test discriminates on **Windows only**. POSIX unlinks a
    file that still has open descriptors, so it passed against the leaking code
    too; it is Windows where an open handle makes ``unlink``/``rmdir`` raise
    ``PermissionError``. It is kept as the direct guard for that platform (CI
    runs a Windows matrix row) — the leak itself is caught everywhere by the
    ``_closed`` and ``_sidecar`` tests above.
    """
    _exercise_hnsw(hnsw_workspace)

    db = hnsw_workspace / "index.db"
    assert db.is_file()
    for path in sorted(hnsw_workspace.iterdir()):
        path.unlink()
    hnsw_workspace.rmdir()
    assert not hnsw_workspace.exists()


# ---------------------------------------------------------------------------
# Transaction ordering — the half of the fix that is easy to get wrong
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stored_rows_survive_the_close(tmp_path: Path, no_gc: None) -> None:
    """Closing must not cost us the write.

    ``close()`` on its own never commits, and closing a connection with an open
    transaction discards it — so a "fix" that only added ``close()`` would turn
    every store and every eviction into a silent no-op. Read the rows back
    through an independent connection, which can only see committed data.
    """
    db = tmp_path / "t.db"
    store = ToolOutputStore(sqlite_path=str(db), max_rows=3)
    for i in range(8):
        store.store_and_summarize(f"row-{i}\n", source=f"cmd{i}", ts="")

    independent = sqlite3.connect(db)
    try:
        rows = independent.execute("SELECT COUNT(*) FROM tool_outputs").fetchone()[0]
    finally:
        independent.close()
    # 8 stored, retention keeps the newest 3 — both the INSERTs and the
    # eviction DELETE had to commit for this number to be right.
    assert rows == 3


@pytest.mark.unit
def test_hnsw_embeddings_survive_the_close(hnsw_workspace: Path, no_gc: None) -> None:
    """Same guard for the other module: registration must still commit."""
    for i in range(5):
        register_block_embedding(hnsw_workspace, f"B-{i}", "entity", [1.0, 0.0, float(i)])

    independent = sqlite3.connect(hnsw_workspace / "index.db")
    try:
        rows = independent.execute("SELECT COUNT(*) FROM block_kind_embeddings").fetchone()[0]
    finally:
        independent.close()
    assert rows == 5


@pytest.mark.unit
def test_sqlite_session_rolls_back_before_closing(tmp_path: Path, no_gc: None) -> None:
    """An exception inside the session must roll the write back, not commit it.

    This pins the *ordering* of the two steps. The inner ``with con`` has to
    exit — rolling back — before ``close()`` runs. Were the close to happen
    first, or the ``with con`` to be dropped in favour of a bare close, the
    half-written row would either be discarded silently by a different path or
    committed. Only the rollback-then-close order gives this result.
    """
    db = tmp_path / "t.db"
    store = ToolOutputStore(sqlite_path=str(db))

    with pytest.raises(RuntimeError, match="deliberate"):
        with store._sqlite() as con:
            con.execute(
                "INSERT OR REPLACE INTO tool_outputs "
                "(handle, source, exit_code, ts, full_text, summary, line_count, byte_count) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("H-rollback", "cmd", 0, "", "text", "summary", 1, 4),
            )
            # Visible within the transaction, so the INSERT really ran.
            assert con.execute("SELECT COUNT(*) FROM tool_outputs WHERE handle='H-rollback'").fetchone()[0] == 1
            raise RuntimeError("deliberate")

    independent = sqlite3.connect(db)
    try:
        surviving = independent.execute("SELECT COUNT(*) FROM tool_outputs WHERE handle='H-rollback'").fetchone()[0]
    finally:
        independent.close()
    assert surviving == 0

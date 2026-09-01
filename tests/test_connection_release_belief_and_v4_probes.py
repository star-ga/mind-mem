"""Every SQLite connection these modules open is CLOSED, not just committed.

``with sqlite3.connect(...) as conn`` commits (or, on an exception, rolls back)
and then leaves the handle **open** — that is all its ``__exit__`` does. Nothing
reclaims it afterwards either: a ``sqlite3.Connection`` owns its
prepared-statement cache and that cache references the connection back, so the
object sits in a reference cycle that plain refcounting cannot free. It survives
until the cyclic collector happens to run.

So every one of these tests disables the collector first. With ``gc`` on, a leak
and a fix look the same a moment later; with ``gc`` off, only an actual
``close()`` releases the descriptor. That is the mechanism under test — not
"the numbers went down".

Three consequences are asserted, in increasing order of platform independence:

* descriptors on the database accumulate (Linux, via ``/proc/self/fd``);
* the ``-wal`` / ``-shm`` sidecars survive, because SQLite only checkpoints
  them away when the LAST connection closes (any platform);
* the directory holding the database cannot be removed on Windows, where an
  open handle makes ``unlink``/``rmdir`` fail (asserted here via ``rmtree``).

Modules covered: :mod:`mind_mem.kalman_belief` (writes) and
:mod:`mind_mem.v4.block_versioning` (reads).
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from mind_mem.kalman_belief import BeliefStore
from mind_mem.v4 import block_versioning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_fds(prefix: str) -> list[str] | None:
    """Descriptors held under *prefix*, or None where that is unknowable.

    Returns None — NOT [] — when ``/proc/self/fd`` is absent. An empty list
    means "checked, nothing open"; None means "could not check". Collapsing the
    two would make ``assert _open_fds(...) == []`` pass for free on macOS and
    Windows, which is an assertion about an absence that holds trivially.
    """
    fd_dir = "/proc/self/fd"
    if not os.path.isdir(fd_dir):  # pragma: no cover - non-Linux
        return None
    held = []
    for name in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue
        if target.startswith(prefix):
            held.append(os.path.basename(target))
    return sorted(held)


def _assert_no_open_fds(prefix: str, what: str) -> None:
    """Assert nothing under *prefix* is open, where that is checkable.

    Off Linux this is a deliberate no-op; the sidecar and rmtree assertions
    carry the proof there.
    """
    held = _open_fds(prefix)
    if held is None:  # pragma: no cover - non-Linux
        return
    assert held == [], f"{what} left open descriptors: {held}"


def _sidecars(db: str) -> list[str]:
    return [suffix for suffix in ("-wal", "-shm") if os.path.exists(db + suffix)]


def _assert_released(db: str, what: str) -> None:
    # SQLite checkpoints and deletes -wal/-shm when the LAST connection on the
    # database closes, so a surviving sidecar means one is still open. This is
    # the platform-neutral half of the proof.
    assert _sidecars(db) == [], f"{what} left WAL sidecars behind: a connection is still open"
    _assert_no_open_fds(db, what)


def _wal(db: Path) -> str:
    """Create *db* in WAL mode so its sidecars become observable."""
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    finally:
        conn.close()
    return str(db)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace with a WAL-mode ``index.db`` and the self-editing flag on."""
    flags = {"self_editing": {"enabled": True}}
    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": flags}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))

    db = _wal(tmp_path / "index.db")
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT)")
        conn.execute("INSERT INTO blocks VALUES ('B-1', 'hello world', 'note')")
        conn.execute("CREATE TABLE block_tier_vclock (block_id TEXT)")
        conn.execute(
            "CREATE TABLE block_edits (edit_id INTEGER PRIMARY KEY, block_id TEXT, "
            "old_content TEXT, new_content TEXT, reason TEXT, status TEXT, "
            "proposed_at TEXT, approved_at TEXT, approver TEXT)"
        )
        conn.execute(
            "INSERT INTO block_edits VALUES (1, 'B-1', 'first', 'second', 'why', "
            "'applied', '2026-01-01T00:00:00+00:00', '2026-01-02T00:00:00+00:00', 'ops')"
        )
        conn.commit()
    finally:
        conn.close()
    assert _sidecars(db) == [], "fixture itself left a connection open"
    return tmp_path


# ---------------------------------------------------------------------------
# kalman_belief.BeliefStore — the write path
# ---------------------------------------------------------------------------


def test_belief_store_writes_close_their_connections(tmp_path: Path) -> None:
    db = _wal(tmp_path / "beliefs.db")
    gc.disable()
    try:
        store = BeliefStore(db_path=db)  # _init_db + _load_from_db
        for i in range(5):
            store.update_belief(f"D-{i}", observation=1.0, source="approve_apply")
        # The store is still alive here: this asserts the connections were
        # released per call, not that the object was garbage.
        _assert_released(db, "BeliefStore")
    finally:
        gc.enable()


def test_belief_store_decay_pass_does_not_leak_per_block(tmp_path: Path) -> None:
    """A decay pass persists every belief — once one connection per block leaked."""
    db = _wal(tmp_path / "beliefs.db")
    store = BeliefStore(db_path=db)
    for i in range(20):
        store.update_belief(f"D-{i}", observation=0.9, source="s")
    gc.disable()
    try:
        store.decay_all(hours_elapsed=48.0)
        _assert_released(db, "BeliefStore.decay_all")
    finally:
        gc.enable()


def test_belief_store_still_commits_before_closing(tmp_path: Path) -> None:
    """Closing must not cost durability — the mechanism check for the ordering.

    ``close()`` never commits, so if the close ran instead of (or before) the
    transaction exit, the write would silently vanish. Reading it back through
    a *second* store proves the commit happened first.
    """
    db = str(tmp_path / "beliefs.db")
    store = BeliefStore(db_path=db)
    store.update_belief("D-1", observation=1.0, source="approve_apply")
    written = store.get_belief("D-1")

    reopened = BeliefStore(db_path=db)
    restored = reopened.get_belief("D-1")
    assert restored.observation_count == written.observation_count
    assert restored.estimate == pytest.approx(written.estimate)


def test_belief_store_session_still_rolls_back_on_error(tmp_path: Path) -> None:
    """The other half of the ordering: close must not turn a rollback into a commit."""
    db = str(tmp_path / "beliefs.db")
    store = BeliefStore(db_path=db)

    with pytest.raises(RuntimeError):
        with store._session(db) as conn:
            conn.execute("INSERT INTO beliefs VALUES ('ROLLED-BACK', 0.5, 0.1, '2026-01-01T00:00:00+00:00', 1, '{}')")
            raise RuntimeError("boom")

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT block_id FROM beliefs WHERE block_id = 'ROLLED-BACK'").fetchall()
    finally:
        conn.close()
    assert rows == [], "the failed session committed instead of rolling back"


def test_belief_store_session_actually_closes_the_connection(tmp_path: Path) -> None:
    """Name the mechanism: the connection object is closed, not merely released."""
    db = str(tmp_path / "beliefs.db")
    store = BeliefStore(db_path=db)
    with store._session(db) as conn:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# v4 read surfaces
# ---------------------------------------------------------------------------


def test_block_versioning_reads_close_their_connections(workspace: Path) -> None:
    db = str(workspace / "index.db")
    gc.disable()
    try:
        for _ in range(5):
            assert block_versioning.block_history(workspace, "B-1")
            assert block_versioning.versioned_block_ids(workspace) == ["B-1"]
            assert block_versioning.content_as_of(workspace, "B-1", "2026-06-01T00:00:00+00:00") == "second"
        _assert_released(db, "block_versioning")
    finally:
        gc.enable()


def test_block_versioning_closes_even_when_the_table_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The early ``return []`` path opens a connection too."""
    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": {"self_editing": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    db = _wal(tmp_path / "index.db")
    gc.disable()
    try:
        for _ in range(5):
            assert block_versioning.block_history(tmp_path, "B-1") == []
            assert block_versioning.versioned_block_ids(tmp_path) == []
        _assert_released(db, "block_versioning (no block_edits table)")
    finally:
        gc.enable()


# ---------------------------------------------------------------------------
# The property the open handle actually breaks
# ---------------------------------------------------------------------------


def test_a_directory_holding_these_databases_is_removable(workspace: Path) -> None:
    """On Windows an open handle makes rmdir fail; on Linux the fd count above
    is what carries the weight. Both are asserted, so neither platform passes
    this file for free."""
    db = str(workspace / "index.db")
    beliefs = _wal(workspace / "beliefs.db")
    gc.disable()
    try:
        store = BeliefStore(db_path=beliefs)
        store.update_belief("D-1", observation=1.0, source="s")
        block_versioning.block_history(workspace, "B-1")
        _assert_no_open_fds(str(workspace), "the v4 surfaces")
        assert _sidecars(db) == []
        assert _sidecars(beliefs) == []
        shutil.rmtree(workspace)
        assert not os.path.exists(workspace)
    finally:
        gc.enable()

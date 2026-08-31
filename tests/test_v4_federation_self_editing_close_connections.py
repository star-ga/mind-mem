"""``v4.federation`` and ``v4.self_editing`` must close what they open.

Both modules reached every one of their fourteen SQLite call sites through
``with sqlite3.connect(...) as conn:``. That form commits — or, on an
exception, rolls back — and then leaves the handle OPEN; ``__exit__``
documents exactly that and nothing more. Nothing else reclaimed it either:
a ``sqlite3.Connection`` is kept alive by its own prepared-statement cache,
and that cache refers back to the connection, so the pair is a reference
cycle that refcounting never collects. Only the cyclic collector does.

Measured on this box before the fix, with the collector disabled and the
public surface of both modules exercised once: 26 descriptors still held on
``index.db`` in the default rollback-journal mode, 53 with the database in
WAL mode — and in WAL mode the ``-wal`` / ``-shm`` sidecars were still on
disk afterwards. A leak on every platform, and a correctness bug on
Windows, where an open handle makes ``os.unlink`` fail and a directory
holding a workspace cannot be deleted.

The collector is disabled inside each measurement on purpose: with it
enabled these assertions pass either way, and a test that passes against
the defect proves nothing.

Two of the checks here are deliberately not descriptor counts, because a
descriptor count is Linux-only and a closed connection is not by itself the
property that matters:

* the WAL-sidecar check is platform-independent, and is made non-vacuous by
  first proving the sidecars DO appear while a connection is open (in the
  default journal mode they never appear at all, so asserting their absence
  would hold for free);
* the ordering check watches the module's own connections and asserts the
  commit scope exits BEFORE the close. ``close()`` alone never commits and
  rolls back an open transaction, so a "just close it" fix that got the
  order wrong would silently discard writes while passing every count.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from mind_mem.v4 import federation as fed
from mind_mem.v4 import self_editing as se

BLOCK = "B-close-1"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace with both v4 flags on."""
    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(
        json.dumps({"v4": {fed.FLAG: {"enabled": True}, se.FLAG: {"enabled": True}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    return tmp_path


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _sidecars(db_path: str) -> list[str]:
    return [suffix for suffix in ("-wal", "-shm") if os.path.exists(db_path + suffix)]


def _open_fds(db_path: str) -> list[str] | None:
    """Descriptors held on the database, or None where that is unknowable.

    Returns None — NOT ``[]`` — when ``/proc/self/fd`` is absent. An empty
    list means "checked, nothing open"; None means "could not check".
    Collapsing the two makes ``assert _open_fds(...) == []`` pass for free
    off Linux, which is an assertion about an absence that holds trivially.
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
        if target.startswith(db_path):
            held.append(os.path.basename(target))
    return sorted(held)


def _assert_no_open_fds(db_path: str, what: str) -> None:
    """Assert no descriptor is held, where that is checkable.

    Every caller goes through here so the None contract of :func:`_open_fds`
    lives in one place. Off Linux this is a no-op by design; the
    platform-independent proofs are the sidecar test and the ordering test.
    """
    held = _open_fds(db_path)
    if held is None:  # pragma: no cover - non-Linux
        return
    assert held == [], f"{what} left {len(held)} open descriptor(s) on index.db: {held}"


def _exercise_federation(ws: Path, block: str = BLOCK) -> None:
    """Every public federation entry point, once, against a fresh block.

    ``block`` is a parameter because the version vector is cumulative: a
    second pass over the same block would legitimately read {a: 4, b: 2},
    and an exercise helper that asserted otherwise would be measuring its
    own bookkeeping rather than the module.
    """
    fed.ensure_federation_schema(ws)
    fed.record_agent_write(ws, block, "agent-a")
    fed.record_agent_write(ws, block, "agent-a")
    fed.record_agent_write(ws, block, "agent-b")
    assert fed.get_version_vector(ws, block) == {"agent-a": 2, "agent-b": 1}
    assert fed.detect_conflict(ws, block) is not None
    assert fed.list_conflicts(ws)
    assert fed.resolve_conflict(ws, block, fed.MergeStrategy.LAST_WRITER_WINS) is not None


def _exercise_self_editing(ws: Path, block: str = BLOCK) -> None:
    """Every public self-editing entry point, once, against a fresh block."""
    se.ensure_edit_schema(ws)
    edit_id = se.propose_edit(ws, block, "corrected content", "the fact went stale")
    assert se.get_edit(ws, edit_id) is not None
    assert se.list_pending_edits(ws)
    assert se.approve_edit(ws, edit_id) is not None
    second = se.propose_edit(ws, block, "another", "another reason")
    assert se.reject_edit(ws, second) is not None
    assert len(se.list_edit_history(ws, block)) == 2


# ---------------------------------------------------------------------------
# Descriptor counts
# ---------------------------------------------------------------------------


class TestNoDescriptorSurvivesTheCall:
    def test_federation_holds_nothing_open(self, workspace: Path) -> None:
        db_path = str(workspace / "index.db")
        gc.disable()
        try:
            _exercise_federation(workspace)
            _assert_no_open_fds(db_path, "federation")
        finally:
            gc.enable()

    def test_self_editing_holds_nothing_open(self, workspace: Path) -> None:
        db_path = str(workspace / "index.db")
        gc.disable()
        try:
            _exercise_self_editing(workspace)
            _assert_no_open_fds(db_path, "self_editing")
        finally:
            gc.enable()

    def test_repeated_calls_do_not_accumulate(self, workspace: Path) -> None:
        """The leak was unbounded — one descriptor per connection, forever.

        A single-pass check would also pass a fix that closed most sites but
        missed one. Twenty passes turn "missed one site" into a count no
        reader can mistake for noise.
        """
        db_path = str(workspace / "index.db")
        gc.disable()
        try:
            for pass_no in range(20):
                _exercise_federation(workspace, f"{BLOCK}-{pass_no}")
                _exercise_self_editing(workspace, f"{BLOCK}-{pass_no}")
            _assert_no_open_fds(db_path, "twenty federation + self-editing passes")
        finally:
            gc.enable()


# ---------------------------------------------------------------------------
# WAL sidecars — the platform-independent proof
# ---------------------------------------------------------------------------


class TestWalSidecarsDoNotSurvive:
    """``journal_mode=WAL`` is persistent in the database header.

    ``index.db`` is shared with the rest of the product, so a workspace whose
    index was ever opened in WAL by another component stays WAL for these two
    modules — and then a never-closed connection keeps ``-wal`` and ``-shm``
    on disk, where anything that inspects or copies the workspace can see
    them. SQLite checkpoints and removes both when the LAST connection
    closes, so their presence afterwards *is* the leak, observable without
    ``/proc``.
    """

    @staticmethod
    def _seed_wal(db_path: str) -> None:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            # An empty database has no header to record the mode in, so the
            # setting would not survive the close. One table is enough.
            conn.execute("CREATE TABLE IF NOT EXISTS _wal_seed (x)")
            conn.commit()
        finally:
            conn.close()
        probe = sqlite3.connect(db_path)
        try:
            assert probe.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            probe.close()

    def test_sidecars_are_gone_after_both_modules_run(self, workspace: Path) -> None:
        db_path = str(workspace / "index.db")
        self._seed_wal(db_path)

        # Non-vacuity: prove the sidecars DO appear while a connection is
        # open on this database, so their absence below is an observation
        # rather than the default state of a rollback-journal database.
        probe = sqlite3.connect(db_path)
        try:
            # Must actually touch the database — a constant expression like
            # ``SELECT 1`` is answered without opening the WAL at all.
            probe.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            assert _sidecars(db_path) == ["-wal", "-shm"], "sidecar check is vacuous — WAL seeding did not take"
        finally:
            probe.close()
        assert _sidecars(db_path) == []

        gc.disable()
        try:
            _exercise_federation(workspace)
            _exercise_self_editing(workspace)
            assert _sidecars(db_path) == [], "WAL sidecars survived: a connection is still open on index.db"
        finally:
            gc.enable()


# ---------------------------------------------------------------------------
# Ordering — commit scope must exit before the close
# ---------------------------------------------------------------------------


class _RecordingConnection:
    """Pass-through connection that records its own lifecycle events.

    Only the three events that matter are intercepted; everything else is
    delegated, so the module under test behaves exactly as it would with a
    real connection. ``__enter__`` / ``__exit__`` are defined explicitly
    because special methods are looked up on the type, not through
    ``__getattr__``.
    """

    def __init__(self, inner: sqlite3.Connection, log: list[str]) -> None:
        self._inner = inner
        self._log = log

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __enter__(self) -> "_RecordingConnection":
        self._log.append("txn-enter")
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._log.append("txn-exit")
        return self._inner.__exit__(*exc)

    def close(self) -> None:
        self._log.append("close")
        self._inner.close()


class TestCommitScopeExitsBeforeTheClose:
    """The ordering is load-bearing, and no descriptor count can see it.

    ``close()`` never commits, and it rolls back a transaction still open.
    A fix that closed the connection *inside* the ``with conn`` body — or
    replaced the commit scope with a bare close — would score zero leaked
    descriptors and silently discard writes. So watch the order directly.
    """

    @staticmethod
    def _record(monkeypatch: pytest.MonkeyPatch, db_path: str) -> list[str]:
        real_connect = sqlite3.connect
        log: list[str] = []

        def spy(target: Any, *args: Any, **kwargs: Any) -> Any:
            conn = real_connect(target, *args, **kwargs)
            if str(target) == db_path:
                return _RecordingConnection(conn, log)
            return conn

        monkeypatch.setattr(sqlite3, "connect", spy)
        return log

    def test_federation_write_commits_then_closes(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = str(workspace / "index.db")
        fed.ensure_federation_schema(workspace)
        log = self._record(monkeypatch, db_path)

        fed.record_agent_write(workspace, BLOCK, "agent-a")

        assert log.count("close") >= 1, "record_agent_write never closed its connection"
        assert log == ["txn-enter", "txn-exit", "close"] * (len(log) // 3), f"unexpected connection lifecycle: {log}"

    def test_self_editing_write_commits_then_closes(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = str(workspace / "index.db")
        se.ensure_edit_schema(workspace)
        log = self._record(monkeypatch, db_path)

        se.propose_edit(workspace, BLOCK, "corrected content", "the fact went stale")

        assert log.count("close") >= 1, "propose_edit never closed its connection"
        assert log == ["txn-enter", "txn-exit", "close"] * (len(log) // 3), f"unexpected connection lifecycle: {log}"


# ---------------------------------------------------------------------------
# The properties the leak actually costs
# ---------------------------------------------------------------------------


class TestClosingDidNotCostAnything:
    """Closing must not have turned a commit into a rollback.

    Read back through a connection opened *after* each call returned: it can
    see committed rows only. Every row below was written by a call that has
    since closed its handle, so "rows present AND no descriptor held" is the
    conjunction that holds only for commit-then-close.
    """

    def test_writes_are_durable_after_the_connection_closes(self, workspace: Path) -> None:
        db_path = str(workspace / "index.db")
        gc.disable()
        try:
            _exercise_federation(workspace)
            _exercise_self_editing(workspace)
            _assert_no_open_fds(db_path, "both modules")
        finally:
            gc.enable()

        conn = sqlite3.connect(db_path)
        try:
            vclock = conn.execute("SELECT COUNT(*) FROM block_tier_vclock WHERE block_id = ?", (BLOCK,)).fetchone()[0]
            conflicts = conn.execute("SELECT COUNT(*) FROM tier_conflict_log WHERE block_id = ?", (BLOCK,)).fetchone()[0]
            resolved = conn.execute("SELECT COUNT(*) FROM tier_conflict_log WHERE resolution IS NOT NULL").fetchone()[0]
            edits = conn.execute("SELECT COUNT(*) FROM block_edits WHERE block_id = ?", (BLOCK,)).fetchone()[0]
            applied = conn.execute("SELECT COUNT(*) FROM block_edits WHERE status = ?", (se.EditStatus.APPLIED,)).fetchone()[0]
        finally:
            conn.close()

        assert vclock >= 2, "the version vector writes were rolled back by the close"
        assert conflicts >= 1, "the conflict log insert was rolled back by the close"
        assert resolved == 1, "the resolution UPDATE was rolled back by the close"
        assert edits == 2, "the proposed edits were rolled back by the close"
        assert applied == 1, "the approval UPDATE was rolled back by the close"

    def test_a_directory_holding_a_live_workspace_is_removable(self, workspace: Path) -> None:
        """The property Windows actually enforces.

        On Linux this passes either way — unlinking an open file is legal —
        so the assertion carrying the weight here is the descriptor check
        taken while the workspace is still in use. On Windows the removal
        itself is what would fail.
        """
        db_path = str(workspace / "index.db")
        gc.disable()
        try:
            _exercise_federation(workspace)
            _exercise_self_editing(workspace)
            _assert_no_open_fds(db_path, "both modules")
            shutil.rmtree(str(workspace), ignore_errors=False)
            assert not os.path.exists(db_path)
        finally:
            gc.enable()

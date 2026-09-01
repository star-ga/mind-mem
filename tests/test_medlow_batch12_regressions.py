"""Regression tests for the batch-12 medium/low audit findings.

One test (or small group) per confirmed defect, each written so that it
fails against the pre-fix code:

    mind_filelock.py        a failed OS lock left the lockfile on disk,
                            wedging every later acquire
    governance_gate.py      an unarmed spec binding was invisible (debug)
    bench/eval_adapters.py  the pipeline-mismatch tripwire could not fire
                            on the default sqlite path
    http_transport.py       token compare short-circuited on a match
    daemon.py               ``--once`` exited 0 with every task failed
    skill_opt/history.py    an abandoned run stayed ``status='running'``

The batch-12 v4/health.py and v4/kind_summaries.py sections were dropped
with those modules in the 5.0.0 unused-module sweep; nothing imports them
any more, so there is no behaviour left to regress.
"""

from __future__ import annotations

import errno
import gc
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# mind_filelock.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lock_degrades_when_filesystem_has_no_advisory_locking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENOLCK means "this mount cannot flock", not "the lock is taken".
    The O_EXCL lockfile still provides exclusion, so acquire must succeed."""
    fcntl = pytest.importorskip("fcntl")
    from mind_mem.mind_filelock import FileLock

    target = tmp_path / "f.md"
    target.write_text("x", encoding="utf-8")

    def _enolck(*_a: Any, **_kw: Any) -> None:
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(fcntl, "flock", _enolck)
    lock = FileLock(str(target), timeout=1.0)
    lock.acquire()
    try:
        assert os.path.exists(lock.lock_path)
    finally:
        lock.release()
    assert not os.path.exists(lock.lock_path)


@pytest.mark.unit
def test_failed_os_lock_leaves_no_lockfile_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hard OS-lock failure must not leave a live-pid lockfile on disk:
    it would never be judged stale, so every later acquire — in this or
    any other process — would block until timeout, forever."""
    fcntl = pytest.importorskip("fcntl")
    from mind_mem.mind_filelock import FileLock

    target = tmp_path / "f.md"
    target.write_text("x", encoding="utf-8")

    def _eio(*_a: Any, **_kw: Any) -> None:
        raise OSError(errno.EIO, "synthetic I/O error")

    monkeypatch.setattr(fcntl, "flock", _eio)
    failing = FileLock(str(target), timeout=1.0)
    with pytest.raises(OSError):
        failing.acquire()
    assert not os.path.exists(failing.lock_path), "lockfile survived a failed acquire"

    # And the lock is still usable afterwards.
    monkeypatch.undo()
    recovered = FileLock(str(target), timeout=1.0)
    recovered.acquire()
    recovered.release()


# ---------------------------------------------------------------------------
# governance_gate.py
# ---------------------------------------------------------------------------


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(record.getMessage())


@pytest.mark.unit
def test_admit_warns_once_when_config_has_no_spec_binding(tmp_path: Path) -> None:
    """With no .spec_binding.json the spec-hash step is inert — config
    tampering is not detected. That must be visible to an operator, and
    it must not repeat on every single write."""
    from mind_mem.governance_gate import GovernanceGate

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "mind-mem.json").write_text(json.dumps({"version": "4.0.0"}), encoding="utf-8")

    logger = logging.getLogger("mind-mem.governance_gate")
    handler = _CapturingHandler()
    logger.addHandler(handler)
    try:
        gate = GovernanceGate(str(ws))
        gate.admit(action="WRITE", block_id="B-1", content="payload", actor="test")
        gate.admit(action="WRITE", block_id="B-2", content="payload", actor="test")
    finally:
        logger.removeHandler(handler)

    unbound = [e for e in handler.events if e == "governance_gate.unbound_config"]
    assert len(unbound) == 1, f"expected exactly one unbound-config warning, got {handler.events}"


@pytest.mark.unit
def test_admit_does_not_warn_when_binding_present(tmp_path: Path) -> None:
    from mind_mem.governance_gate import GovernanceGate
    from mind_mem.spec_binding import SpecBindingManager

    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = ws / "mind-mem.json"
    cfg.write_text(json.dumps({"version": "4.0.0"}), encoding="utf-8")
    SpecBindingManager(str(cfg)).bind(str(cfg))

    logger = logging.getLogger("mind-mem.governance_gate")
    handler = _CapturingHandler()
    logger.addHandler(handler)
    try:
        GovernanceGate(str(ws)).admit(action="WRITE", block_id="B-1", content="payload", actor="test")
    finally:
        logger.removeHandler(handler)

    assert "governance_gate.unbound_config" not in handler.events


# ---------------------------------------------------------------------------
# bench/eval_adapters.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_probe_reports_scan_when_sqlite_index_was_never_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_load_backend`` returns "sqlite" from the config string alone, so
    declared == effective by construction. If build_index failed there is
    no index and recall runs the markdown scan — the tripwire must fire."""
    import mind_mem.sqlite_index as sqlite_index
    from mind_mem.bench.eval_adapter import SessionDoc
    from mind_mem.bench.eval_adapters import MindMemAdapter

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("no such module: fts5")

    monkeypatch.setattr(sqlite_index, "build_index", _boom)

    adapter = MindMemAdapter()
    state = adapter.init([SessionDoc(doc_id="s1", text="the capital of France is Paris")], None)
    try:
        assert state.probe.declared_backend == "sqlite"
        assert state.probe.effective_backend == "scan"
        assert state.probe.mismatch is True
        assert "build_index_failed" in state.probe.notes
        assert state.probe.extra["index_exists"] is False
    finally:
        adapter.teardown(state)


@pytest.mark.unit
def test_probe_reports_sqlite_when_the_index_really_exists() -> None:
    from mind_mem.bench.eval_adapter import SessionDoc
    from mind_mem.bench.eval_adapters import MindMemAdapter

    adapter = MindMemAdapter()
    state = adapter.init([SessionDoc(doc_id="s1", text="the capital of France is Paris")], None)
    try:
        assert state.probe.effective_backend == "sqlite"
        assert state.probe.mismatch is False
        assert state.probe.extra["index_exists"] is True
        assert state.probe.extra["index_blocks"] >= 1
    finally:
        adapter.teardown(state)


# ---------------------------------------------------------------------------
# http_transport.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_token_check_compares_against_every_active_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The comment claims a comparison count independent of which token
    matched. A generator-backed ``any`` stopped at the first match."""
    import hmac
    import http.client
    import socket

    from mind_mem.http_transport import AUTH_HEADER, PATH_STATUS, serve_http

    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "decisions").mkdir(parents=True)
    (ws / "mind-mem.json").write_text(json.dumps({"version": "4.0.0", "block_store": {"backend": "markdown"}}), encoding="utf-8")
    (ws / "decisions" / "DECISIONS.md").write_text("# Decisions\n", encoding="utf-8")

    calls: list[str] = []
    real = hmac.compare_digest

    def _counting(a: Any, b: Any) -> bool:
        calls.append("cmp")
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", _counting)
    monkeypatch.setenv("MIND_MEM_TOKENS", "first-token-aaa,second-token-bbb,third-token-ccc")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    _thread, stop = serve_http(
        workspace=str(ws),
        port=port,
        host="127.0.0.1",
        token="first-token-aaa",
        allow_unauthenticated_localhost=False,
    )
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            # Matching token at index 0 — the short-circuit case.
            conn.request("GET", PATH_STATUS, headers={AUTH_HEADER: "first-token-aaa"})
            assert conn.getresponse().read() is not None
        finally:
            conn.close()
    finally:
        stop()

    assert len(calls) == 3, f"expected one comparison per active token, got {len(calls)}"


# ---------------------------------------------------------------------------
# daemon.py
# ---------------------------------------------------------------------------


def _daemon_workspace(tmp_path: Path, tasks: dict[str, Any]) -> Path:
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    cfg = {"daemon": {"enabled": True, **tasks}}
    (ws / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    return ws


@pytest.mark.unit
def test_daemon_once_returns_nonzero_when_a_task_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cron and CI read exit 0 as "maintenance ran clean"."""
    from mind_mem import daemon as daemon_mod

    def _boom(_ws: str, _extras: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("synthetic task failure")

    monkeypatch.setitem(daemon_mod._TASK_RUNNERS, "dream_cycle", _boom)
    ws = _daemon_workspace(tmp_path, {"dream_cycle": {"auto_interval_seconds": 60}})
    assert daemon_mod.run_daemon(str(ws), once=True) == 1


@pytest.mark.unit
def test_daemon_once_returns_zero_when_every_task_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem import daemon as daemon_mod

    monkeypatch.setitem(daemon_mod._TASK_RUNNERS, "dream_cycle", lambda _ws, _e: {"ok": True})
    ws = _daemon_workspace(tmp_path, {"dream_cycle": {"auto_interval_seconds": 60}})
    assert daemon_mod.run_daemon(str(ws), once=True) == 0


@pytest.mark.unit
def test_daemon_once_partial_failure_is_still_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem import daemon as daemon_mod

    def _boom(_ws: str, _extras: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("synthetic task failure")

    monkeypatch.setitem(daemon_mod._TASK_RUNNERS, "dream_cycle", lambda _ws, _e: {"ok": True})
    monkeypatch.setitem(daemon_mod._TASK_RUNNERS, "intel_scan", _boom)
    ws = _daemon_workspace(
        tmp_path,
        {
            "dream_cycle": {"auto_interval_seconds": 60},
            "intel_scan": {"auto_interval_seconds": 60},
        },
    )
    assert daemon_mod.run_daemon(str(ws), once=True) == 1


# ---------------------------------------------------------------------------
# skill_opt/history.py
# ---------------------------------------------------------------------------


def _run_row(db: str, run_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM optimization_runs WHERE run_id=?", (run_id,)).fetchone()
    finally:
        conn.close()


@pytest.mark.unit
def test_abandoned_store_marks_the_open_run_interrupted(tmp_path: Path) -> None:
    """start_run commits status='running' before the work that can raise.
    A store dropped mid-run must not leave the row claiming to be live."""
    from mind_mem.skill_opt.history import HistoryStore

    db = str(tmp_path / "hist.db")
    store = HistoryStore(db)
    store.start_run("R-abandoned", "s1", "hash1")
    del store
    gc.collect()

    row = _run_row(db, "R-abandoned")
    assert row["status"] == "interrupted"
    assert row["completed_at"] is not None


@pytest.mark.unit
def test_history_store_is_a_context_manager(tmp_path: Path) -> None:
    from mind_mem.skill_opt.history import HistoryStore

    db = str(tmp_path / "hist2.db")
    with pytest.raises(RuntimeError):
        with HistoryStore(db) as store:
            store.start_run("R-ctx", "s1", "hash1")
            raise RuntimeError("work blew up")

    row = _run_row(db, "R-ctx")
    assert row["status"] == "interrupted"


@pytest.mark.unit
def test_completed_run_is_not_relabelled_on_close(tmp_path: Path) -> None:
    from mind_mem.skill_opt.history import HistoryStore

    db = str(tmp_path / "hist3.db")
    store = HistoryStore(db)
    store.start_run("R-done", "s1", "hash1")
    store.complete_run("R-done", "completed", 0.5, 0.7, True)
    store.close()
    store.close()  # idempotent

    row = _run_row(db, "R-done")
    assert row["status"] == "completed"
    assert row["overall_score_after"] == 0.7


@pytest.mark.unit
def test_interrupted_status_constant_matches_the_stored_value() -> None:
    from mind_mem.skill_opt.history import INTERRUPTED_STATUS

    assert INTERRUPTED_STATUS == "interrupted"

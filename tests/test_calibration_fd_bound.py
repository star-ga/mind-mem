"""The calibration connection cache must stay bounded under workspace churn.

Each cached ConnectionManager pins ~3 descriptors (db, -wal, -shm). An
unbounded cache exhausts RLIMIT_NOFILE after a few dozen workspaces.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import mind_mem.calibration as cal

SRC = str(Path(cal.__file__).resolve().parent.parent)

_CHURN = textwrap.dedent(
    """
    import json, os, resource, sys
    from pathlib import Path
    import mind_mem.calibration as cal
    root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    fds = lambda: len(os.listdir('/proc/self/fd'))
    error = None; done = 0; peak = 0
    for i in range(1000):
        try:
            cal.CalibrationManager(str(root / f'ws{i}')).get_calibration_stats()
            done += 1
            peak = max(peak, fds())
        except Exception as e:
            error = f'{type(e).__name__}: {e}'; break
    print(json.dumps({'done': done, 'error': error, 'peak': peak, 'cache': len(cal._conn_managers)}))
    """
)


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc fd listing")
def test_workspace_churn_does_not_exhaust_descriptors(tmp_path):
    env = {**os.environ, "PYTHONPATH": SRC}
    proc = subprocess.run(
        [sys.executable, "-c", _CHURN, str(tmp_path / "w")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["error"] is None, out
    assert out["done"] == 1000
    assert out["cache"] <= cal._CONN_MANAGER_CACHE_MAX
    assert out["peak"] < 100, out


@pytest.fixture
def small_cache(monkeypatch):
    monkeypatch.setattr(cal, "_CONN_MANAGER_CACHE_MAX", 2)
    cal._close_all_conn_managers()
    yield
    cal._close_all_conn_managers()


def test_hot_path_reuses_one_manager(tmp_path, monkeypatch):
    cal._close_all_conn_managers()
    opens = []
    real = cal.ConnectionManager
    monkeypatch.setattr(cal, "ConnectionManager", lambda p: opens.append(p) or real(p))
    ws = str(tmp_path / "hot")
    for _ in range(50):
        cal.CalibrationManager(ws).get_calibration_stats()
    assert len(opens) == 1
    cal._close_all_conn_managers()


def test_eviction_closes_manager_and_reuse_reopens(tmp_path, small_cache):
    a = cal.CalibrationManager(str(tmp_path / "a"))
    closed = []
    real_close = a._mgr.close
    a._mgr.close = lambda: (closed.append(1), real_close())[1]
    cal.CalibrationManager(str(tmp_path / "b"))
    cal.CalibrationManager(str(tmp_path / "c"))
    assert closed, "evicted manager was not closed"
    assert len(cal._conn_managers) <= 2
    # Reusing the evicted workspace (and even the stale handle) still works.
    assert cal.CalibrationManager(str(tmp_path / "a")).get_calibration_stats() is not None
    assert a.get_calibration_stats() is not None


def test_threads_racing_evictions(tmp_path, small_cache):
    errors = []

    def work(n):
        try:
            for i in range(40):
                cal.CalibrationManager(str(tmp_path / f"ws{(n + i) % 6}")).get_calibration_stats()
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:3]
    assert len(cal._conn_managers) <= 2

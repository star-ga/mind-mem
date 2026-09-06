"""Regression coverage for process-wide Postgres pool shutdown."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import mind_mem.block_store_postgres as bsp


@pytest.fixture
def clean_pool_registry(monkeypatch):
    """Give shutdown tests an isolated registry and restore its final state."""
    with bsp._pool_registry_lock:
        saved = dict(bsp._pool_registry)
        bsp._pool_registry.clear()
    monkeypatch.setattr(bsp, "_pool_registry_shutdown", False, raising=False)
    yield
    with bsp._pool_registry_lock:
        leftovers = list(bsp._pool_registry.values())
        bsp._pool_registry.clear()
        bsp._pool_registry.update(saved)
    for pool in leftovers:
        pool.close()


class _LockCheckingPool:
    def __init__(self) -> None:
        self.close_calls = 0
        self.registry_was_empty = False
        self.close_had_registry_lock = False

    def close(self) -> None:
        self.close_calls += 1
        self.registry_was_empty = not bsp._pool_registry
        acquired = bsp._pool_registry_lock.acquire(timeout=1)
        self.close_had_registry_lock = acquired
        if acquired:
            bsp._pool_registry_lock.release()


def test_shutdown_detaches_before_close_and_is_idempotent(clean_pool_registry):
    first = _LockCheckingPool()
    second = _LockCheckingPool()
    with bsp._pool_registry_lock:
        bsp._pool_registry[("one", "schema")] = first
        # A duplicate reference is not expected in production, but shutdown
        # must still close the resource itself exactly once.
        bsp._pool_registry[("duplicate", "schema")] = first
        bsp._pool_registry[("two", "schema")] = second

    callers = [threading.Thread(target=bsp._shutdown_pool_registry) for _ in range(4)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=2)

    assert all(not caller.is_alive() for caller in callers)
    assert bsp._pool_registry == {}
    assert first.close_calls == second.close_calls == 1
    assert first.registry_was_empty and second.registry_was_empty
    assert first.close_had_registry_lock and second.close_had_registry_lock

    bsp._shutdown_pool_registry()
    assert first.close_calls == second.close_calls == 1


class _GatedPool:
    constructed = threading.Event()
    allow_return = threading.Event()
    instances: list["_GatedPool"] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False
        self.close_calls = 0
        self.instances.append(self)
        self.constructed.set()
        assert self.allow_return.wait(timeout=2)

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def test_shutdown_captures_pool_created_by_inflight_registration(monkeypatch, clean_pool_registry):
    _GatedPool.constructed.clear()
    _GatedPool.allow_return.clear()
    _GatedPool.instances.clear()
    monkeypatch.setattr(bsp, "_require_psycopg", lambda: (object(), _GatedPool))
    store = bsp.PostgresBlockStore("postgresql://invalid", schema="concurrent")
    result: list[Any] = []

    creator = threading.Thread(target=lambda: result.append(store._get_pool()))
    creator.start()
    assert _GatedPool.constructed.wait(timeout=2)

    shutdown = threading.Thread(target=bsp._shutdown_pool_registry)
    shutdown.start()
    _GatedPool.allow_return.set()
    creator.join(timeout=2)
    shutdown.join(timeout=2)

    assert not creator.is_alive() and not shutdown.is_alive()
    assert len(result) == len(_GatedPool.instances) == 1
    assert result[0].closed
    assert result[0].close_calls == 1
    assert bsp._pool_registry == {}

    fresh = bsp.PostgresBlockStore("postgresql://invalid", schema="after_shutdown")
    with pytest.raises(RuntimeError, match="shut down"):
        fresh._get_pool()
    assert len(_GatedPool.instances) == 1


@pytest.mark.skipif(sys.version_info < (3, 14), reason="PythonFinalizationError regression requires Python 3.14+")
def test_python314_process_exit_closes_real_psycopg_pool():
    pytest.importorskip("psycopg_pool")
    source_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(source_root), env.get("PYTHONPATH", ""))).rstrip(os.pathsep)
    probe = """
from psycopg_pool import ConnectionPool
import mind_mem.block_store_postgres as bsp

pool = ConnectionPool('', min_size=0, max_size=1, open=True)
with bsp._pool_registry_lock:
    bsp._pool_registry[('python314-probe', 'python314-probe')] = pool
print('pool_registered', not pool.closed)
"""
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and inline regression probe
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "pool_registered True"
    assert "PythonFinalizationError" not in proc.stderr
    assert "cannot join thread at interpreter shutdown" not in proc.stderr

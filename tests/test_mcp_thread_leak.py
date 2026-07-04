"""Regression tests for the MCP-server thread leak (2026-07-04).

Production symptom: a long-running ``python3 mcp_server.py`` process
(stdio transport) accumulated 76,479 threads / ~32GB RSS over 2.6 days
(all ``State: S`` sleeping) and exhausted the box's fork/thread capacity.
Roughly one leaked thread set per MCP tool call.

Root cause: ``storage.get_block_store()`` is (by design) a factory —
every ``recall()`` / ``hybrid_search()`` / ``_check_workspace()`` MCP tool
call against a Postgres-backed workspace constructs a *fresh*
``PostgresBlockStore``. That is safe on its own (construction does no
I/O), but ``PostgresBlockStore._get_pool()`` used to open a brand-new
``psycopg_pool.ConnectionPool(..., open=True)`` on that fresh instance's
first real query. Opening a pool spawns a scheduler thread plus
``num_workers`` (default 3) worker threads that run until ``.close()`` is
called — and nothing on the per-tool-call path ever called it, since the
ephemeral ``PostgresBlockStore`` went out of scope as soon as the tool call
returned. The 4 background threads per call were only reclaimed
incidentally whenever Python's garbage collector happened to collect the
discarded pool — not guaranteed to keep pace under sustained traffic, and
in fact NOT what happened in production over 2.6 days of real load (see
the ``_pool_registry`` note in ``block_store_postgres.py`` for the fix).

Note on test design: a bare single-threaded loop of ``recall()`` calls does
NOT reliably reproduce the leak in a quiet test process, because CPython's
prompt refcounting can incidentally collect each discarded
``PostgresBlockStore``/``ConnectionPool`` before the next call even starts
(no reference cycle in the common case). That is exactly why the leak
went unnoticed until it had run for 2.6 days under real, sustained,
concurrent production traffic. So the tests below assert on the number of
``ConnectionPool`` objects actually *constructed* across N repeated calls
(deterministic, independent of GC timing) rather than on ambient
``threading.active_count()`` alone — that is the load-bearing, leak-proof
assertion; ``active_count()`` is checked too as a secondary signal.

Two tests:

* ``test_fake_pool_reused_across_repeated_factory_calls`` — CI-safe, no
  live database required. Substitutes a lightweight fake in place of
  ``psycopg_pool.ConnectionPool`` that spawns real background threads
  exactly like the genuine pool, and proves the registry dedups pool
  (and thread) creation across N repeated ``get_block_store()`` calls —
  the exact per-MCP-tool-call pattern that leaked.
* ``TestLivePostgresThreadCount`` — real integration test against a live
  Postgres (skipped without ``MIND_MEM_TEST_PG_DSN``, matching the
  existing convention in ``test_postgres_block_store.py``). Calls the
  actual ``mind_mem.recall`` MCP-tool code path N times through a
  counting ``ConnectionPool`` subclass and asserts only one real pool
  (and its 4 background threads) is ever created.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Generator

import pytest

import mind_mem.block_store_postgres as bsp
from mind_mem.storage import get_block_store

# ---------------------------------------------------------------------------
# CI-safe: fake pool, no live database
# ---------------------------------------------------------------------------


class _FakeConnectionPool:
    """Minimal stand-in for ``psycopg_pool.ConnectionPool``.

    Spawns 4 real background threads on construction (1 "scheduler" + 3
    "workers", matching the real pool's default ``num_workers=3``) so this
    test exercises the exact thread-accounting the production bug hit,
    without requiring psycopg/psycopg_pool or a live database.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._stop.wait, name=f"fakepool-{id(self)}-{i}", daemon=True) for i in range(4)]
        for t in self._threads:
            t.start()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)


@pytest.fixture
def fake_psycopg_pool(monkeypatch):
    """Replace the real psycopg/psycopg_pool import with a fake pool."""
    monkeypatch.setattr(bsp, "_require_psycopg", lambda: (object(), _FakeConnectionPool))
    registry = getattr(bsp, "_pool_registry", None)
    if registry is not None:
        registry.clear()
    yield
    registry = getattr(bsp, "_pool_registry", None)
    if registry is not None:
        for pool in list(registry.values()):
            pool.close()
        registry.clear()


def _make_pg_workspace(tmp_path, schema: str) -> str:
    ws = str(tmp_path)
    config = {
        "block_store": {
            "backend": "postgres",
            "dsn": "postgresql://fake-user:fake-pw@127.0.0.1:5432/fake-db",
            "schema": schema,
        }
    }
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh)
    return ws


def test_fake_pool_reused_across_repeated_factory_calls(tmp_path, fake_psycopg_pool):
    """``get_block_store()`` called N times must open the pool at most once.

    Before the fix this created N ``ConnectionPool`` instances (and 4*N
    background threads) — one per call, never closed. After the fix all N
    calls share a single cached pool. Asserted via a pool-creation counter
    (deterministic) rather than raw thread counts (GC-timing dependent).
    """
    schema = f"mm_test_{uuid.uuid4().hex[:12]}"
    ws = _make_pg_workspace(tmp_path, schema)

    created: list[_FakeConnectionPool] = []
    real_init = _FakeConnectionPool.__init__

    def _counting_init(self, *a, **kw):
        real_init(self, *a, **kw)
        created.append(self)

    baseline = threading.active_count()
    N = 100
    try:
        _FakeConnectionPool.__init__ = _counting_init  # type: ignore[method-assign]
        for _ in range(N):
            store = get_block_store(ws)
            store._get_pool()  # the exact call every recall()/hybrid_search() triggers
    finally:
        _FakeConnectionPool.__init__ = real_init  # type: ignore[method-assign]

    assert hasattr(bsp, "_pool_registry"), (
        "mind_mem.block_store_postgres has no _pool_registry — the per-call "
        "Postgres connection-pool cache is missing; this is the thread-leak "
        "regression (see PostgresBlockStore._get_pool)."
    )
    assert len(created) == 1, (
        f"expected exactly 1 ConnectionPool to be constructed across {N} "
        f"factory calls, got {len(created)} — the per-call pool-leak "
        f"regression is back (one leaked thread-set per MCP tool call)."
    )

    active_after = threading.active_count()
    # Exactly 4 background threads (1 scheduler + 3 workers) regardless of N.
    assert active_after - baseline == 4, (
        f"thread count grew by {active_after - baseline} across {N} calls "
        f"(expected a flat +4 for the single shared pool) — thread leak regression."
    )

    # Cleanup: close the one real pool so the fixture teardown is a no-op.
    for pool in created:
        pool.close()
    bsp._pool_registry.clear()
    assert threading.active_count() == baseline


# ---------------------------------------------------------------------------
# Live Postgres integration test (opt-in, matches test_postgres_block_store.py)
# ---------------------------------------------------------------------------

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres thread-leak test")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

_DSN_ENV = "MIND_MEM_TEST_PG_DSN"


def _require_dsn() -> str:
    dsn = os.environ.get(_DSN_ENV)
    if not dsn:
        pytest.skip(f"{_DSN_ENV} not set — no live Postgres available")
    return dsn


class TestLivePostgresThreadCount:
    """End-to-end proof against a real Postgres: pools/threads stay bounded."""

    @pytest.fixture
    def pg_workspace(self, tmp_path) -> Generator[str, None, None]:
        dsn = _require_dsn()
        schema = f"mm_test_{uuid.uuid4().hex[:12]}"

        # Provision the schema once via a single, properly-closed store —
        # mirrors an operator running `mm doctor`/`mind-mem-init` before
        # first use. This instance is NOT part of the leak measurement.
        setup_store = PostgresBlockStore(dsn=dsn, schema=schema, workspace=str(tmp_path))
        setup_store._ensure_schema()
        setup_store.close()

        ws = _make_pg_workspace(tmp_path, schema)
        # Overwrite the fake dsn written by _make_pg_workspace with the real one.
        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
            json.dump({"block_store": {"backend": "postgres", "dsn": dsn, "schema": schema}}, fh)
        try:
            yield ws
        finally:
            try:
                conn = psycopg.connect(dsn)
                conn.autocommit = True
                conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                conn.close()
            except Exception:
                pass
            registry = getattr(bsp, "_pool_registry", None)
            if registry is not None:
                for pool in list(registry.values()):
                    try:
                        pool.close()
                    except Exception:
                        pass
                registry.clear()

    def test_recall_opens_exactly_one_pool_across_n_calls(self, monkeypatch, pg_workspace):
        """The actual MCP tool-call path: recall() N times must open one real pool.

        Counts real ``ConnectionPool`` construction directly (deterministic)
        rather than relying on ``threading.active_count()`` alone, since a
        quiet single-threaded test process can incidentally garbage-collect
        an unreferenced pool fast enough to mask the leak (see module
        docstring) — exactly the false sense of safety that let this bug
        run in production for 2.6 days before it was caught.
        """
        os.environ["MIND_MEM_WORKSPACE"] = pg_workspace
        from mind_mem.mcp.tools.recall import recall

        real_require = bsp._require_psycopg
        created: list[Any] = []

        def _counting_require():
            psycopg_mod, ConnectionPool = real_require()

            class _CountingPool(ConnectionPool):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    created.append(self)

            return psycopg_mod, _CountingPool

        monkeypatch.setattr(bsp, "_require_psycopg", _counting_require)

        baseline = threading.active_count()
        N = 100
        for i in range(N):
            recall(query=f"thread leak regression probe {i}", limit=5)

        active_after = threading.active_count()

        assert len(created) == 1, (
            f"expected exactly 1 ConnectionPool to be opened across {N} "
            f"recall() calls, got {len(created)} — the per-tool-call Postgres "
            f"pool leak regression is back."
        )
        assert active_after - baseline == 4, (
            f"threading.active_count() grew by {active_after - baseline} across "
            f"{N} recall() calls (expected a flat +4 for one shared pool) — "
            f"the per-tool-call Postgres pool leak regression is back."
        )

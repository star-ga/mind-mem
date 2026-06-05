"""Tests for ``PostgresBlockStore.ping()`` — active backend health probe.

Regression guard for the 2026-06-05 incident: a configured-but-unreachable
Postgres backend failed *quietly* (per-call warnings, masked by the SQLite
recall cache) instead of surfacing loudly, so the operator only noticed days
later when a write-path tool (``reindex_dirty``) finally exercised it.

``ping()`` gives callers (``mm doctor``, health endpoints) a fast,
non-raising backend status: it opens a single short-lived connection with a
bounded ``connect_timeout`` so a down backend fails *fast and loud* instead
of degrading silently or triggering a pool retry-storm.
"""

from __future__ import annotations

import os
import time

import pytest

# The Postgres backend is gated behind the ``[postgres]`` extra; skip the
# whole module when psycopg is not installed rather than erroring.
pytest.importorskip("psycopg")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

# 127.0.0.1 port 1 is never a Postgres server → connection refused, fast.
_DEAD_DSN = "postgresql://nobody:nopw@127.0.0.1:1/does_not_exist"

_EXPECTED_KEYS = {"backend", "ok", "schema", "blocks_table", "block_count", "error"}


def test_ping_unreachable_returns_status_without_raising() -> None:
    store = PostgresBlockStore(_DEAD_DSN, schema="mind_mem")
    status = store.ping(timeout=2.0)  # must not raise

    assert isinstance(status, dict)
    assert status["backend"] == "postgres"
    assert status["ok"] is False
    assert status["error"]  # non-empty diagnostic string
    assert status["blocks_table"] is False
    assert status["block_count"] is None
    assert status["schema"] == "mind_mem"


def test_ping_unreachable_is_fast() -> None:
    """A dead backend must fail fast — not retry-storm (the incident's tail)."""
    store = PostgresBlockStore(_DEAD_DSN, schema="mind_mem")
    start = time.monotonic()
    store.ping(timeout=2.0)
    elapsed = time.monotonic() - start
    # Connection-refused returns near-instantly; even hitting the timeout
    # ceiling this stays well under a multi-second pool retry-storm.
    assert elapsed < 5.0


def test_ping_shape_and_schema_echo() -> None:
    store = PostgresBlockStore(_DEAD_DSN, schema="custom_schema")
    status = store.ping(timeout=2.0)
    assert set(status) == _EXPECTED_KEYS
    assert status["schema"] == "custom_schema"


@pytest.mark.skipif(
    not os.environ.get("MIND_MEM_TEST_PG_DSN"),
    reason="set MIND_MEM_TEST_PG_DSN to run the live success-path test",
)
def test_ping_live_ok() -> None:
    dsn = os.environ["MIND_MEM_TEST_PG_DSN"]
    schema = os.environ.get("MIND_MEM_TEST_PG_SCHEMA", "mind_mem")
    store = PostgresBlockStore(dsn, schema=schema)
    status = store.ping(timeout=5.0)
    assert status["ok"] is True
    assert status["blocks_table"] is True
    assert status["block_count"] is not None and status["block_count"] >= 0
    assert status["error"] is None

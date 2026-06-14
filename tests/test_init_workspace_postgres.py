"""Postgres regression tests for ``init_workspace`` (audit bug #8).

Before the fix, ``mind-mem-init`` had no Postgres path: the generated
``mind-mem.json`` omitted the ``block_store`` section entirely and there was
no flag/env to select Postgres, so a PG user had to hand-edit the config.
There was also no way to provision the schema at init time.

This module verifies the opt-in Postgres path end-to-end against a live DB:

* ``init(ws, backend="postgres", dsn=..., schema=...)`` writes a usable
  ``block_store`` section AND ``recall.backend="sqlite"``.
* ``_ensure_postgres_schema`` actually creates the schema/tables, after
  which a ``PostgresBlockStore`` can read/write the freshly-initialised
  workspace through ``get_block_store``.

Each run uses a *uniquely-named scratch schema* it creates and drops — the
production ``mind_mem`` schema (and any other agent's schema) is never
touched. The whole module skips gracefully when ``psycopg`` or a live
Postgres is unavailable, so SQLite-only CI stays green.

To run against the local audit DB::

    MIND_MEM_TEST_PG_DSN="postgresql://USER:PASSWORD@127.0.0.1:5432/DBNAME" \\
        pytest tests/test_init_workspace_postgres.py -v
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Generator

import pytest

# Skip entire module when psycopg is absent — no install, no failure.
psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres tests")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402
from mind_mem.init_workspace import _ensure_postgres_schema, init  # noqa: E402
from mind_mem.storage import get_block_store, iter_active_blocks  # noqa: E402

# Honour the standard test DSN env var first; fall back to the local audit
# DSN. The schema is always a unique scratch schema we create and drop.
_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN")


def _pg_available(dsn: str) -> bool:
    try:
        conn = psycopg.connect(dsn, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_available(_DSN),
    reason="no live Postgres available at the test DSN",
)


@pytest.fixture
def scratch_schema() -> Generator[str, None, None]:
    """Yield a uniquely-named scratch schema name; drop it (CASCADE) after."""
    schema = f"mm_fix_{uuid.uuid4().hex[:12]}"
    try:
        yield schema
    finally:
        try:
            conn = psycopg.connect(_DSN)
            conn.autocommit = True
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()
        except Exception:
            pass


def test_init_postgres_writes_block_store_config(tmp_path: Path, scratch_schema: str) -> None:
    """init(backend='postgres') writes a complete, usable block_store section."""
    ws = tmp_path / "pgws"
    init(str(ws), backend="postgres", dsn=_DSN, schema=scratch_schema)

    cfg = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    assert cfg["block_store"] == {"backend": "postgres", "dsn": _DSN, "schema": scratch_schema}
    # recall reads the SQLite FTS cache (mirrors PG), not the empty corpus.
    assert cfg["recall"]["backend"] == "sqlite"


def test_ensure_postgres_schema_provisions_db(tmp_path: Path, scratch_schema: str) -> None:
    """_ensure_postgres_schema creates the schema/tables for a fresh install."""
    ws = tmp_path / "pgws"
    init(str(ws), backend="postgres", dsn=_DSN, schema=scratch_schema)

    ok, err = _ensure_postgres_schema(_DSN, scratch_schema)
    assert ok, f"schema provisioning failed: {err}"

    # The schema now exists in the catalog.
    conn = psycopg.connect(_DSN)
    try:
        row = conn.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
            (scratch_schema,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "scratch schema was not created"


def test_initialized_postgres_workspace_is_usable(tmp_path: Path, scratch_schema: str) -> None:
    """A workspace init'd for Postgres is immediately read/write-able.

    Out-of-box flow: init writes the config + ensures the schema, then the
    store factory (driven purely from the generated mind-mem.json) round-
    trips a block and the backend-aware feature primitive sees it.
    """
    ws = tmp_path / "pgws"
    init(str(ws), backend="postgres", dsn=_DSN, schema=scratch_schema)
    ok, err = _ensure_postgres_schema(_DSN, scratch_schema)
    assert ok, f"schema provisioning failed: {err}"

    # get_block_store reads ONLY the generated mind-mem.json — no manual
    # config wiring. This proves the init output is self-sufficient.
    store = get_block_store(str(ws))
    try:
        store.write_block(
            {
                "_id": "D-20260613-INIT1",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "init_workspace provisioned this Postgres backend",
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        ids = {b["_id"] for b in iter_active_blocks(str(ws))}
        assert "D-20260613-INIT1" in ids
    finally:
        with contextlib.suppress(Exception):
            store.close()


def test_ensure_schema_idempotent(tmp_path: Path, scratch_schema: str) -> None:
    """Calling _ensure_postgres_schema twice is a no-op the second time."""
    ws = tmp_path / "pgws"
    init(str(ws), backend="postgres", dsn=_DSN, schema=scratch_schema)
    ok1, _ = _ensure_postgres_schema(_DSN, scratch_schema)
    ok2, err2 = _ensure_postgres_schema(_DSN, scratch_schema)
    assert ok1 and ok2, f"idempotent ensure failed: {err2}"


def test_ensure_schema_bad_dsn_degrades_gracefully(tmp_path: Path) -> None:
    """An unreachable DSN must NOT raise — init stays usable, returns (False, err)."""
    ok, err = _ensure_postgres_schema("postgresql://nouser:nopass@127.0.0.1:1/nodb", "mm_fix_unreachable")
    assert ok is False
    assert err  # a non-empty diagnostic string


def test_store_factory_for_pg_workspace(tmp_path: Path, scratch_schema: str) -> None:
    """get_block_store on the init'd workspace builds a PostgresBlockStore."""
    ws = tmp_path / "pgws"
    init(str(ws), backend="postgres", dsn=_DSN, schema=scratch_schema)
    store = get_block_store(str(ws))
    try:
        assert isinstance(store, PostgresBlockStore)
    finally:
        with contextlib.suppress(Exception):
            store.close()

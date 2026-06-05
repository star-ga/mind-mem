"""restore() must preserve each block's file_path (routing metadata).

Regression: snapshot stored block metadata with _-prefixed keys stripped,
so _source_file was lost; snapshot_blocks had no file_path column; and
restore() read COALESCE(metadata->>'_source_file','') => '' for every row.
A single snapshot/restore cycle therefore wiped file_path on every block.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("psycopg")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

_DSN_ENV = "MIND_MEM_TEST_PG_DSN"


@pytest.mark.skipif(not os.environ.get(_DSN_ENV), reason=f"{_DSN_ENV} not set")
def test_restore_preserves_file_path() -> None:
    dsn = os.environ[_DSN_ENV]
    schema = f"mm_rfp_{uuid.uuid4().hex[:10]}"
    store = PostgresBlockStore(dsn=dsn, schema=schema, workspace="/tmp/mm-rfp")
    store._ensure_schema()
    try:
        store.write_block(
            {
                "_id": "D-1",
                "_source_file": "decisions/DECISIONS.md",
                "Statement": "keep this path",
                "Status": "active",
            }
        )
        snap = f"/tmp/mm-rfp/{schema}-snap"
        store.snapshot(snap)
        # Mutate then restore.
        store.write_block({"_id": "D-1", "_source_file": "decisions/DECISIONS.md", "Statement": "changed", "Status": "active"})
        store.restore(snap)

        block = store.get_by_id("D-1")
        assert block is not None
        assert block["Statement"] == "keep this path"
        # The bug: file_path came back as '' after restore.
        assert block["_source_file"] == "decisions/DECISIONS.md"
    finally:
        import psycopg
        from psycopg import sql

        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))

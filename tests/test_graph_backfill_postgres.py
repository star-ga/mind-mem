# Copyright 2026 STARGA, Inc.
"""Real backend controls for graph backfill corpus selection.

The PostgreSQL test uses a caller-provided disposable DSN and a unique schema.
Rows are inserted through the actual backend schema, while backfill itself
must discover them through ``storage.iter_blocks``.  The encrypted control
exercises the same graph path against ciphertext-backed Markdown.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem.graph_ingest import backfill

try:
    import psycopg
    from psycopg import sql as pgsql
except ImportError:  # pragma: no cover - optional backend
    psycopg = None  # type: ignore[assignment]
    pgsql = None  # type: ignore[assignment]


_PG_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN", "").strip()
requires_pg = pytest.mark.skipif(
    psycopg is None or not _PG_DSN,
    reason="set MIND_MEM_TEST_PG_DSN with a disposable PostgreSQL database",
)


def _extractor(seen: list[str]):
    def extract(text: str) -> list[dict[str, Any]]:
        seen.append(text)
        return [{"subject": "memory", "predicate": "depends_on", "object": "mind"}]

    return extract


def _insert_rows(dsn: str, schema: str, rows: list[dict[str, Any]]) -> None:
    assert pgsql is not None
    with psycopg.connect(dsn, autocommit=True) as conn:
        table = pgsql.Identifier(schema, "blocks")
        statement = pgsql.SQL("INSERT INTO {table} (id, file_path, content, metadata, active) VALUES (%s, %s, %s, %s::jsonb, %s)").format(
            table=table
        )
        for row in rows:
            metadata = dict(row["metadata"])
            conn.execute(
                statement,
                (
                    row["id"],
                    row["file_path"],
                    row["content"],
                    json.dumps(metadata),
                    # Deliberately keep this cache true even for withheld
                    # statuses: graph backfill must ask the metadata
                    # admission authority, never trust this column.
                    True,
                ),
            )


def _drop_schema(dsn: str, schema: str) -> None:
    """Drop the fixture schema and propagate failures to the test runner."""
    assert pgsql is not None
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(pgsql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(pgsql.Identifier(schema)))


@pytest.fixture
def pg_workspace(tmp_path: Path) -> Iterator[tuple[str, str, str]]:
    if psycopg is None or not _PG_DSN:
        pytest.skip("set MIND_MEM_TEST_PG_DSN with a disposable PostgreSQL database")
    from mind_mem.block_store_postgres import PostgresBlockStore

    schema = f"mm_graph_{uuid.uuid4().hex[:12]}"
    workspace = tmp_path / "pg-workspace"
    workspace.mkdir()
    (workspace / "decisions").mkdir()
    (workspace / "decisions/DECISIONS.md").write_text(
        "[D-PG-001]\nStatus: active\nStatement: markdown shadow must not win\n",
        encoding="utf-8",
    )
    (workspace / "mind-mem.json").write_text(
        json.dumps(
            {
                "block_store": {"backend": "postgres", "dsn": _PG_DSN, "schema": schema},
                "recall": {
                    "validity_gate": {
                        "enabled": True,
                        "content_categories": {"enabled": True, "ttl_days": {"infra": 2, "status": 1}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = PostgresBlockStore(dsn=_PG_DSN, schema=schema, workspace=str(workspace))
    store._ensure_schema()
    try:
        yield str(workspace), _PG_DSN, schema
    finally:
        store.close()
        _drop_schema(_PG_DSN, schema)


@requires_pg
def test_graph_backfill_reads_real_postgres_and_applies_admission(pg_workspace: tuple[str, str, str]) -> None:
    workspace, dsn, schema = pg_workspace
    _insert_rows(
        dsn,
        schema,
        [
            {
                "id": "D-PG-001",
                "file_path": "decisions/DECISIONS.md",
                "content": "postgres source of record",
                "metadata": {"Status": "active", "Statement": "postgres source of record", "Type": "decision"},
            },
            {
                "id": "D-PG-002",
                "file_path": "decisions/DECISIONS.md",
                "content": "quarantined postgres row",
                "metadata": {"Status": "quarantined", "Statement": "quarantined postgres row", "Type": "decision"},
            },
            {
                "id": "D-PG-003",
                "file_path": "decisions/DECISIONS.md",
                "content": "revoked postgres credential",
                "metadata": {
                    "Status": "revoked",
                    "ContentCategory": "credential",
                    "Statement": "revoked postgres credential",
                    "Type": "decision",
                },
            },
            {
                "id": "D-PG-004",
                "file_path": "decisions/DECISIONS.md",
                "content": "release decision",
                "metadata": {
                    "Status": "active",
                    "Releases": ["IMP-PG-001"],
                    "Statement": "release decision",
                    "Type": "decision",
                },
            },
            {
                "id": "IMP-PG-001",
                "file_path": "memory/IMPORTED.md",
                "content": "released imported source",
                "metadata": {"Status": "quarantined", "Statement": "released imported source", "Type": "import"},
            },
        ],
    )

    from mind_mem.storage import get_block_store

    configured = get_block_store(workspace)
    try:
        assert {row["_id"] for row in configured.get_all(active_only=False)} == {
            "D-PG-001",
            "D-PG-002",
            "D-PG-003",
            "D-PG-004",
            "IMP-PG-001",
        }

        seen: list[str] = []
        report = backfill(workspace, extract_fn=_extractor(seen))

        assert set(seen) == {"postgres source of record", "release decision", "released imported source"}
        assert "markdown shadow must not win" not in seen
        assert report["blocks_scanned"] == 3
        assert report["edges_extracted"] == 3
    finally:
        configured.close()


@pytest.mark.skipif(psycopg is None, reason="psycopg is required for cleanup control")
def test_schema_cleanup_failure_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingConnection:
        def __enter__(self) -> "FailingConnection":
            return self

        def __exit__(self, *_: Any) -> bool:
            return False

        def execute(self, *_: Any) -> None:
            raise RuntimeError("drop schema refused")

    def fail_connect(*_args: Any, **_kwargs: Any) -> FailingConnection:
        return FailingConnection()

    assert psycopg is not None
    monkeypatch.setattr(psycopg, "connect", fail_connect)
    with pytest.raises(RuntimeError, match="drop schema refused"):
        _drop_schema("postgresql://disposable.invalid/db", "mm_graph_failure")


def test_graph_backfill_reads_encrypted_admitted_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.block_store_encrypted import encrypt_workspace

    passphrase = "graph-backfill-disposable-encryption-key"
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", passphrase)
    workspace = tmp_path / "encrypted-workspace"
    for subdir in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (workspace / subdir).mkdir(parents=True)
    (workspace / "mind-mem.json").write_text(
        json.dumps(
            {
                "block_store": {"backend": "encrypted"},
                "recall": {
                    "validity_gate": {
                        "enabled": True,
                        "content_categories": {"enabled": True, "ttl_days": {"infra": 2, "status": 1}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (workspace / "decisions/DECISIONS.md").write_text(
        "[D-ENC-001]\nStatus: active\nStatement: encrypted admitted source\n\n"
        "[D-ENC-002]\nStatus: revoked\nContentCategory: credential\nStatement: encrypted revoked source\n",
        encoding="utf-8",
    )
    encrypt_workspace(str(workspace))

    seen: list[str] = []
    report = backfill(str(workspace), extract_fn=_extractor(seen))

    assert seen == ["encrypted admitted source"]
    assert report["blocks_scanned"] == report["edges_extracted"] == 1

"""v3.2.0 §1.4 PR-5 — PostgresBlockStore integration tests.

Requires a live PostgreSQL instance. When psycopg is not installed the
entire module is skipped gracefully via ``pytest.importorskip``.

To run against a local Postgres::

    docker run --rm -p 5432:5432 -e POSTGRES_PASSWORD=test postgres:16
    MIND_MEM_TEST_PG_DSN="postgresql://postgres:test@localhost:5432/postgres" \\
        pytest tests/test_postgres_block_store.py -v

When MIND_MEM_TEST_PG_DSN is absent the tests are skipped (no live DB).
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Generator

import pytest

# Skip entire module when psycopg is absent — no install, no failure.
psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres tests")

from mind_mem.block_store_postgres import BlockStoreError, PostgresBlockStore  # noqa: E402

# ─── Helpers ──────────────────────────────────────────────────────────────────

_DSN_ENV = "MIND_MEM_TEST_PG_DSN"


def _test_dsn() -> str | None:
    return os.environ.get(_DSN_ENV)


def _require_dsn() -> str:
    dsn = _test_dsn()
    if not dsn:
        pytest.skip(f"{_DSN_ENV} not set — no live Postgres available")
    return dsn


def _unique_schema() -> str:
    """Each test run uses a fresh schema so tests are fully isolated."""
    return f"mm_test_{uuid.uuid4().hex[:12]}"


def _make_block(bid: str, statement: str = "A test statement", file_path: str = "decisions/DECISIONS.md") -> dict:
    return {
        "_id": bid,
        "_source_file": file_path,
        "Statement": statement,
        "Status": "active",
        "Date": "2026-04-19",
    }


# ─── Fixture ──────────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> Generator[PostgresBlockStore, None, None]:
    """Yield a PostgresBlockStore with an isolated schema; tear it down after."""
    dsn = _require_dsn()
    schema = _unique_schema()
    s = PostgresBlockStore(dsn=dsn, schema=schema, workspace="/tmp/mm-test")
    s._ensure_schema()
    try:
        yield s
    finally:
        # Drop the temporary schema to keep the test DB clean.
        try:
            conn = psycopg.connect(dsn)
            conn.autocommit = True
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()
        except Exception:
            pass
        s.close()


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestDDLEscaping:
    """Regression tests for psycopg SQL.format placeholder collisions.

    The DDL embeds a JSONB ``'{}'`` literal as the column default. psycopg
    parses ``{}`` inside ``SQL.format()`` as a positional placeholder, so
    the literal must be escaped to ``{{}}``. Without the escape, formatting
    raises ``IndexError: tuple index out of range`` and no Postgres user
    can ever bring up the schema.
    """

    def test_ddl_format_does_not_raise(self) -> None:
        from mind_mem.block_store_postgres import _ddl

        composed = _ddl("mind_mem")  # must not raise.
        rendered = composed.as_string(None)
        # The JSONB default literal must reach the SQL output unmangled.
        assert "DEFAULT '{}'" in rendered
        assert '"mind_mem"' in rendered
        # Sanity: every CREATE TABLE we expect is emitted.
        for tbl in ("blocks", "snapshots", "snapshot_blocks", "workspace_lock", "schema_migrations"):
            assert f'"mind_mem".{tbl}' in rendered

    def test_ddl_format_with_custom_schema(self) -> None:
        from mind_mem.block_store_postgres import _ddl

        composed = _ddl("custom_ns")  # arbitrary schema name.
        rendered = composed.as_string(None)
        assert '"custom_ns"' in rendered


class TestInitLockReentrance:
    """Regression: ``_ensure_schema`` calls ``_get_pool`` while holding
    ``_init_lock``. A non-reentrant Lock self-deadlocks on the first call.
    """

    def test_init_lock_is_reentrant(self) -> None:
        from mind_mem.block_store_postgres import PostgresBlockStore

        store = PostgresBlockStore("postgresql://nobody@localhost/none")
        # If this were a plain threading.Lock, reacquiring it from the
        # same thread would deadlock instead of returning.
        with store._init_lock:
            with store._init_lock:
                pass


class TestPgVectorWiring:
    """v3.8.13 — pgvector schema add-on, embedding writer, and hybrid recall."""

    def test_ddl_pgvector_emits_alter_and_index(self) -> None:
        # v3.8.14: index switched from IVFFlat (lists=100) to HNSW for
        # incremental build + small-corpus correctness. See
        # TestPgVectorHardening.test_ddl_pgvector_uses_hnsw_not_ivfflat
        # for the rationale.
        from mind_mem.block_store_postgres import _ddl_pgvector

        sql = _ddl_pgvector("mind_mem", embedding_dim=1024).as_string(None)
        assert "ADD COLUMN IF NOT EXISTS embedding VECTOR(1024)" in sql
        assert "USING hnsw (embedding vector_cosine_ops)" in sql
        assert '"mind_mem"' in sql

    def test_ddl_pgvector_respects_dim(self) -> None:
        from mind_mem.block_store_postgres import _ddl_pgvector

        for dim in (384, 768, 1024, 1536, 3072):
            sql = _ddl_pgvector("mind_mem", embedding_dim=dim).as_string(None)
            assert f"VECTOR({dim})" in sql

    def test_embedding_to_pg_format(self) -> None:
        from mind_mem.block_store_postgres import _embedding_to_pg

        out = _embedding_to_pg([0.1, -0.2, 1e-7])
        # pgvector text literal must start/end with brackets and be comma-separated.
        assert out.startswith("[") and out.endswith("]")
        assert out.count(",") == 2
        # No NaN/inf, finite formatting only.
        assert "nan" not in out.lower()
        assert "inf" not in out.lower()

    def test_embedding_to_pg_handles_int_floats(self) -> None:
        from mind_mem.block_store_postgres import _embedding_to_pg

        out = _embedding_to_pg([1, 2, 3])  # ints get coerced.
        assert out == "[1,2,3]"

    def test_default_pgvector_flags_are_safe(self) -> None:
        """A PostgresBlockStore that's never called _ensure_schema must
        report ``has_vector=False`` so callers don't try to embed."""
        from mind_mem.block_store_postgres import PostgresBlockStore

        s = PostgresBlockStore("postgresql://nobody@localhost/none")
        assert s._has_vector is False
        assert s._embedding_dim == 1024  # mxbai-embed-large default.

    def test_write_block_embedding_dim_check(self) -> None:
        """write_block must reject mismatched embedding dims even when
        pgvector isn't installed (input validation runs before DB)."""
        from mind_mem.block_store_postgres import BlockStoreError, PostgresBlockStore

        s = PostgresBlockStore("postgresql://nobody@localhost/none")
        s._schema_ready = True  # short-circuit the migration call.
        s._has_vector = True
        s._embedding_dim = 1024
        with pytest.raises(BlockStoreError, match="embedding dim mismatch"):
            s.write_block({"_id": "X", "Statement": "x"}, embedding=[0.1, 0.2])  # 2 != 1024

    def test_backfill_embedding_requires_pgvector(self) -> None:
        from mind_mem.block_store_postgres import BlockStoreError, PostgresBlockStore

        s = PostgresBlockStore("postgresql://nobody@localhost/none")
        s._schema_ready = True
        s._has_vector = False
        with pytest.raises(BlockStoreError, match="pgvector not available"):
            s.backfill_embedding("X", [0.1] * 1024)


class TestPgVectorHardening:
    """v3.8.14 — fixes for the v3.8.13 multi-agent audit findings."""

    def test_ddl_pgvector_uses_hnsw_not_ivfflat(self) -> None:
        """IVFFlat with lists=100 builds degenerate centroids on small
        empty tables and never recovers without a manual REINDEX. HNSW
        builds incrementally on insert — no rebuild required."""
        from mind_mem.block_store_postgres import _ddl_pgvector

        sql = _ddl_pgvector("mind_mem", embedding_dim=1024).as_string(None)
        assert "USING hnsw" in sql
        assert "vector_cosine_ops" in sql
        assert "lists=" not in sql  # IVFFlat artefact must be gone.

    def test_ddl_pgvector_dim_uses_literal_not_string_concat(self) -> None:
        """Composable API only — no raw string concatenation of caller
        values into SQL text. The dim should round-trip through
        psycopg.Literal."""
        from mind_mem.block_store_postgres import _ddl_pgvector

        sql = _ddl_pgvector("mind_mem", embedding_dim=768).as_string(None)
        assert "VECTOR(768)" in sql

    def test_ddl_includes_full_fts_expression(self) -> None:
        """The GIN index on blocks_fts must use the SAME tsvector
        expression as the WHERE clauses in search() and hybrid_search()
        — otherwise the planner falls back to a sequential scan with
        per-row tsvector recomputation. v3.8.13 had a mismatch."""
        from mind_mem.block_store_postgres import _ddl

        sql = _ddl("mind_mem").as_string(None)
        assert (
            "to_tsvector('english', content || ' ' || COALESCE(metadata->>'Statement', ''))"
            in sql
        )

    def test_embedding_to_pg_rejects_nan(self) -> None:
        """NaN in the embedding silently poisons RRF ranking via
        cosine-distance NaN propagation. Reject at the boundary."""
        from mind_mem.block_store_postgres import _embedding_to_pg

        with pytest.raises(ValueError, match="non-finite"):
            _embedding_to_pg([0.1, float("nan"), 0.2])

    def test_embedding_to_pg_rejects_inf(self) -> None:
        from mind_mem.block_store_postgres import _embedding_to_pg

        with pytest.raises(ValueError, match="non-finite"):
            _embedding_to_pg([0.1, float("inf"), 0.2])
        with pytest.raises(ValueError, match="non-finite"):
            _embedding_to_pg([float("-inf"), 0.1])

    def test_hybrid_search_rejects_empty_embedding_loudly(self) -> None:
        """v3.8.13 silently degraded to BM25 on an empty embedding list.
        That hides caller bugs — fail loudly instead."""
        from mind_mem.block_store_postgres import BlockStoreError, PostgresBlockStore

        s = PostgresBlockStore("postgresql://nobody@localhost/none")
        s._schema_ready = True
        s._has_vector = True
        s._embedding_dim = 1024
        with pytest.raises(BlockStoreError, match="dim mismatch"):
            s.hybrid_search("anything", query_embedding=[])  # 0 != 1024.


class TestRoundTrip:
    def test_write_then_get_by_id(self, store: PostgresBlockStore) -> None:
        block = _make_block("D-20260419-001", statement="Round-trip test")
        returned_id = store.write_block(block)
        assert returned_id == "D-20260419-001"

        fetched = store.get_by_id("D-20260419-001")
        assert fetched is not None
        assert fetched["_id"] == "D-20260419-001"
        assert fetched["Statement"] == "Round-trip test"

    def test_get_by_id_missing_returns_none(self, store: PostgresBlockStore) -> None:
        result = store.get_by_id("D-99999999-999")
        assert result is None

    def test_delete_block_returns_true(self, store: PostgresBlockStore) -> None:
        store.write_block(_make_block("D-20260419-002"))
        removed = store.delete_block("D-20260419-002")
        assert removed is True
        assert store.get_by_id("D-20260419-002") is None

    def test_delete_missing_block_returns_false(self, store: PostgresBlockStore) -> None:
        removed = store.delete_block("D-99999999-000")
        assert removed is False

    def test_get_all_returns_written_blocks(self, store: PostgresBlockStore) -> None:
        store.write_block(_make_block("D-20260419-010"))
        store.write_block(_make_block("D-20260419-011"))
        blocks = store.get_all()
        ids = {b["_id"] for b in blocks}
        assert "D-20260419-010" in ids
        assert "D-20260419-011" in ids

    def test_get_all_active_only(self, store: PostgresBlockStore) -> None:
        store.write_block(_make_block("D-20260419-020"))
        # Manually set one block to inactive.
        pool = store._get_pool()
        schema = store._schema
        with pool.connection() as conn:
            conn.execute(f"UPDATE {schema}.blocks SET active = FALSE WHERE id = %s", ("D-20260419-020",))
        active = store.get_all(active_only=True)
        ids = {b["_id"] for b in active}
        assert "D-20260419-020" not in ids

    def test_list_blocks_returns_file_paths(self, store: PostgresBlockStore) -> None:
        store.write_block(_make_block("D-20260419-030", file_path="decisions/DECISIONS.md"))
        paths = store.list_blocks()
        assert "decisions/DECISIONS.md" in paths


class TestUpsert:
    def test_write_same_id_twice_replaces_content(self, store: PostgresBlockStore) -> None:
        store.write_block(_make_block("D-20260419-050", statement="Original content"))
        store.write_block(_make_block("D-20260419-050", statement="Replaced content"))

        fetched = store.get_by_id("D-20260419-050")
        assert fetched is not None
        assert fetched["Statement"] == "Replaced content"

    def test_upsert_does_not_duplicate_rows(self, store: PostgresBlockStore) -> None:
        store.write_block(_make_block("D-20260419-051"))
        store.write_block(_make_block("D-20260419-051"))
        store.write_block(_make_block("D-20260419-051"))

        all_blocks = store.get_all()
        matching = [b for b in all_blocks if b["_id"] == "D-20260419-051"]
        assert len(matching) == 1


class TestSearch:
    def test_search_finds_matching_content(self, store: PostgresBlockStore) -> None:
        store.write_block(_make_block("D-20260419-060", statement="PostgreSQL adapter integration"))
        store.write_block(_make_block("D-20260419-061", statement="Unrelated topic about apples"))

        results = store.search("PostgreSQL", limit=10)
        ids = {r["_id"] for r in results}
        assert "D-20260419-060" in ids

    def test_search_limit_respected(self, store: PostgresBlockStore) -> None:
        for i in range(10):
            store.write_block(_make_block(f"D-20260419-07{i}", statement=f"searchable entry {i}"))

        results = store.search("searchable entry", limit=3)
        assert len(results) <= 3

    def test_search_returns_empty_for_no_match(self, store: PostgresBlockStore) -> None:
        store.write_block(_make_block("D-20260419-080", statement="completely irrelevant content"))
        results = store.search("xyzzy_no_match_token_abc123")
        assert results == []


class TestSnapshotRestore:
    def test_snapshot_creates_manifest_on_disk(self, store: PostgresBlockStore, tmp_path: Path) -> None:
        store.write_block(_make_block("D-20260419-100"))
        snap_dir = str(tmp_path / "snap01")
        manifest = store.snapshot(snap_dir)

        assert "files" in manifest
        assert manifest["version"] == 2
        assert (tmp_path / "snap01" / "MANIFEST.json").is_file()

    def test_snapshot_then_restore_reverts_changes(self, store: PostgresBlockStore, tmp_path: Path) -> None:
        store.write_block(_make_block("D-20260419-110", statement="Before snapshot"))
        snap_dir = str(tmp_path / "snap02")
        store.snapshot(snap_dir)

        # Mutate after snapshot.
        store.write_block(_make_block("D-20260419-110", statement="After mutation"))
        store.write_block(_make_block("D-20260419-111", statement="New block after snapshot"))

        # Restore must revert to pre-mutation state.
        store.restore(snap_dir)

        reverted = store.get_by_id("D-20260419-110")
        assert reverted is not None
        assert reverted["Statement"] == "Before snapshot"

        # Block added after snapshot should be gone.
        gone = store.get_by_id("D-20260419-111")
        assert gone is None

    def test_diff_empty_when_no_changes(self, store: PostgresBlockStore, tmp_path: Path) -> None:
        store.write_block(_make_block("D-20260419-120"))
        snap_dir = str(tmp_path / "snap03")
        store.snapshot(snap_dir)

        changes = store.diff(snap_dir)
        # No mutations after snapshot — diff should be empty.
        assert changes == []

    def test_diff_detects_modification(self, store: PostgresBlockStore, tmp_path: Path) -> None:
        store.write_block(_make_block("D-20260419-130", statement="Original"))
        snap_dir = str(tmp_path / "snap04")
        store.snapshot(snap_dir)

        store.write_block(_make_block("D-20260419-130", statement="Modified"))
        changes = store.diff(snap_dir)
        assert len(changes) >= 1

    def test_restore_unknown_snap_raises(self, store: PostgresBlockStore, tmp_path: Path) -> None:
        snap_dir = str(tmp_path / "nonexistent_snap")
        os.makedirs(snap_dir)
        with pytest.raises(BlockStoreError):
            store.restore(snap_dir)


class TestConcurrency:
    def test_concurrent_writes_are_serialized(self, store: PostgresBlockStore) -> None:
        """Multiple threads writing distinct blocks must all succeed."""
        errors: list[Exception] = []

        def _write(idx: int) -> None:
            try:
                store.write_block(_make_block(f"D-20260419-2{idx:02d}", statement=f"Thread {idx}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes failed: {errors}"
        all_blocks = store.get_all()
        written_ids = {b["_id"] for b in all_blocks}
        for i in range(20):
            assert f"D-20260419-2{i:02d}" in written_ids

    def test_concurrent_read_and_write(self, store: PostgresBlockStore) -> None:
        """Reads during writes must not see partially-written data."""
        store.write_block(_make_block("D-20260419-300", statement="Initial"))
        read_errors: list[Exception] = []
        stop = threading.Event()

        def _reader() -> None:
            while not stop.is_set():
                try:
                    store.get_by_id("D-20260419-300")
                except Exception as exc:
                    read_errors.append(exc)

        reader = threading.Thread(target=_reader)
        reader.start()
        for i in range(20):
            store.write_block(_make_block("D-20260419-300", statement=f"Update {i}"))
        stop.set()
        reader.join()

        assert read_errors == [], f"Concurrent read errors: {read_errors}"


class TestStorageFactory:
    def test_factory_constructs_postgres_store(self, tmp_path: Path) -> None:
        dsn = _require_dsn()
        from mind_mem.storage import get_block_store

        cfg = {"block_store": {"backend": "postgres", "dsn": dsn}}
        store = get_block_store(str(tmp_path), config=cfg)
        # Clean up.
        try:
            store.close()  # type: ignore[attr-defined]
        except AttributeError:
            pass

    def test_factory_raises_on_missing_dsn(self, tmp_path: Path) -> None:
        from mind_mem.storage import get_block_store

        cfg = {"block_store": {"backend": "postgres"}}
        with pytest.raises(ValueError, match="dsn"):
            get_block_store(str(tmp_path), config=cfg)

    def test_factory_raises_import_error_without_psycopg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When psycopg is absent the factory raises ImportError with the install hint."""
        import mind_mem.block_store_postgres as _mod

        def _fake_require() -> None:
            raise ImportError("mocked missing psycopg")

        monkeypatch.setattr(_mod, "_require_psycopg", _fake_require)

        from mind_mem.storage import get_block_store

        cfg = {"block_store": {"backend": "postgres", "dsn": "postgresql://localhost/test"}}
        with pytest.raises(ImportError):
            store = get_block_store(str(tmp_path), config=cfg)
            store._ensure_schema()  # type: ignore[attr-defined]

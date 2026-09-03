"""R2-06 — on Postgres, ``blocks.active`` must mean what admission means.

Before this landed the column was a constant. The DDL declared
``active BOOLEAN NOT NULL DEFAULT TRUE`` and nothing ever wrote it:
``grep -n 'SET active|active = FALSE' block_store_postgres.py`` matched
nothing, and neither ``write_block`` INSERT named the column. So a
quarantined block's row was ``active = TRUE``.

Recall survived that by re-filtering every backend hit through the
admission funnel (``_recall_core._withhold_inadmissible``). The feature
layer did not: scan / export / reindex / dream_cycle / drift enumerate
through :func:`mind_mem.storage.iter_active_blocks`, which on a
non-markdown backend was ``get_all(active_only=True)`` — i.e.
``WHERE active = TRUE``, a predicate every row satisfied.

Two things are fixed and gated here, and they are independent on
purpose:

* the column is now **written** from the block's Status, so the
  ``WHERE active`` predicates (``list_blocks``, ``search``,
  ``hybrid_search``) select what admission would admit; and
* the enumeration primitive no longer **trusts** the column — it reads
  every row and applies :func:`mind_mem.admissibility.admit_corpus`,
  the one authority, so a stale or hand-edited column cannot serve.

Every negative assertion below is paired with a positive control: the
admissible block of the same shape, written the same way, must come
back. A "quarantined block absent" assertion passes trivially when the
seed was never there, and that is the most common way a governance test
proves nothing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import pytest
from _restore_scope import restoring

# ─── Fake Postgres plumbing (no live DB; renders the real SQL) ────────────────
#
# "No live DB" is not "no psycopg". The fake pool records what the store
# EXECUTES, and the store composes every statement with ``psycopg.sql`` --
# that is the point of the rendering tests: they read the real Composable the
# real driver would send. Without the ``[postgres]`` extra those statements
# cannot be composed at all, and through 5.0.1 that surfaced as fourteen
# ``ModuleNotFoundError`` FAILURES on every CI row that installs ``[test]``
# alone -- an environment error wearing a red X, on a gate that has nothing
# to say about that environment. The three classes that render SQL therefore
# ``importorskip`` psycopg through one fixture, the same way every other
# Postgres module in this suite does; the ``postgres backend`` job installs
# the extra and runs them for real. The classes that do NOT touch the driver
# (:class:`TestIterActiveBlocksWithholdsQuarantined`,
# :class:`TestAdaptersHaveNoSecondWritePath`, the source-inspection test) run
# on every row, unchanged.


@pytest.fixture
def needs_psycopg() -> None:
    pytest.importorskip("psycopg", reason="renders psycopg.sql Composables; needs the [postgres] extra")


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self._rows = list(rows or [])

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: Any, params: Any = None) -> "_FakeCursor":
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows, self._rows = self._rows, []
        return rows


class _FakeConn:
    """Records every statement, rendering Composables to real SQL text."""

    def __init__(self, *, claim_rows: list[tuple[Any, ...]] | None = None) -> None:
        self.statements: list[tuple[str, Any]] = []
        self.autocommit = False
        self._claim_rows = claim_rows

    # context-manager plumbing --------------------------------------------
    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def transaction(self) -> "_FakeConn":
        return self

    def cursor(self) -> _FakeCursor:
        return _FakeCursor()

    # the recorded surface -------------------------------------------------
    def execute(self, sql: Any, params: Any = None) -> _FakeCursor:
        self.statements.append((_render(sql), params))
        rendered = _render(sql)
        if "schema_migrations" in rendered and "INSERT" in rendered:
            return _FakeCursor(self._claim_rows if self._claim_rows is not None else [(1,)])
        return _FakeCursor()

    def rollback(self) -> None:
        return None


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def connection(self) -> _FakeConn:
        return self._conn


def _render(sql: Any) -> str:
    """Render a psycopg Composable (or a plain string) to SQL text."""
    as_string = getattr(sql, "as_string", None)
    return as_string(None) if as_string is not None else str(sql)


def _pg_store(monkeypatch: pytest.MonkeyPatch, *, has_vector: bool = False, claim_rows: list[tuple[Any, ...]] | None = None):
    """A PostgresBlockStore whose pool is a recorder, with the schema pre-marked ready."""
    from mind_mem.block_store_postgres import PostgresBlockStore

    store = PostgresBlockStore("postgresql://nobody@localhost/none", schema="mm_unit")
    conn = _FakeConn(claim_rows=claim_rows)
    monkeypatch.setattr(store, "_get_pool", lambda: _FakePool(conn))
    store._schema_ready = True
    store._has_vector = has_vector
    return store, conn


def _insert_binding(rendered: str, params: Any) -> dict[str, Any]:
    """Map INSERT column name -> the value actually bound to it.

    Positional truth, not a substring hunt: ``updated_at`` binds
    ``NOW()`` rather than a placeholder, so a naive ``params[N]`` index
    silently reads the wrong column the moment the column list changes.
    """
    match = re.search(r"INSERT INTO [^(]+\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)", rendered, re.S)
    assert match is not None, f"no INSERT … VALUES in: {rendered}"
    cols = [c.strip() for c in match.group(1).split(",")]
    vals = [v.strip() for v in match.group(2).split(",")]
    assert len(cols) == len(vals), f"column/value arity mismatch: {cols} vs {vals}"
    remaining = list(params or ())
    bound: dict[str, Any] = {}
    for col, val in zip(cols, vals):
        bound[col] = remaining.pop(0) if "%s" in val else val
    return bound


def _writes(conn: _FakeConn) -> list[tuple[str, Any]]:
    return [(sql, params) for sql, params in conn.statements if "INSERT INTO" in sql and ".blocks" in sql]


def _block(bid: str, status: str | None, *, statement: str = "seed") -> dict[str, Any]:
    block: dict[str, Any] = {"_id": bid, "_source_file": "decisions/DECISIONS.md", "Statement": statement}
    if status is not None:
        block["Status"] = status
    return block


# ─── 1. write_block writes the column, from the Status ────────────────────────


@pytest.mark.usefixtures("needs_psycopg")
class TestWriteBlockWritesActiveFromStatus:
    """The column is derived at the door, so ``WHERE active`` is truthful."""

    @pytest.mark.parametrize("has_vector", [False, True])
    def test_quarantined_status_binds_active_false(self, monkeypatch: pytest.MonkeyPatch, admitted, has_vector: bool) -> None:
        store, conn = _pg_store(monkeypatch, has_vector=has_vector)
        embedding = [0.0] * store._embedding_dim if has_vector else None
        store.write_block(_block("D-quar", "quarantined"), embedding=embedding)
        rendered, params = _writes(conn)[-1]
        assert _insert_binding(rendered, params)["active"] is False

    @pytest.mark.parametrize("has_vector", [False, True])
    def test_active_status_binds_active_true(self, monkeypatch: pytest.MonkeyPatch, admitted, has_vector: bool) -> None:
        """Positive control for the assertion above: same door, same shape."""
        store, conn = _pg_store(monkeypatch, has_vector=has_vector)
        embedding = [0.0] * store._embedding_dim if has_vector else None
        store.write_block(_block("D-live", "active"), embedding=embedding)
        rendered, params = _writes(conn)[-1]
        assert _insert_binding(rendered, params)["active"] is True

    def test_pending_status_binds_active_false(self, monkeypatch: pytest.MonkeyPatch, admitted) -> None:
        store, conn = _pg_store(monkeypatch)
        store.write_block(_block("S-pend", "pending"))
        rendered, params = _writes(conn)[-1]
        assert _insert_binding(rendered, params)["active"] is False

    def test_unstated_status_binds_active_true(self, monkeypatch: pytest.MonkeyPatch, admitted) -> None:
        """Unstated is servable — ``is_admissible_status(None)`` is True."""
        store, conn = _pg_store(monkeypatch)
        store.write_block(_block("D-nostatus", None))
        rendered, params = _writes(conn)[-1]
        assert _insert_binding(rendered, params)["active"] is True

    def test_unrecognised_status_binds_active_false(self, monkeypatch: pytest.MonkeyPatch, admitted) -> None:
        """Fail-closed: a status this code cannot read is not served."""
        store, conn = _pg_store(monkeypatch)
        store.write_block(_block("D-bogus", "not-a-lifecycle-status"))
        rendered, params = _writes(conn)[-1]
        assert _insert_binding(rendered, params)["active"] is False

    def test_the_bound_value_agrees_with_the_admission_predicate(self, monkeypatch: pytest.MonkeyPatch, admitted) -> None:
        """No second opinion: the column is ``is_admissible_status`` itself."""
        from mind_mem.admissibility import is_admissible_status

        for status in ("active", "quarantined", "pending", "open", "done", "archived", "", "  Active  ", "nonsense"):
            store, conn = _pg_store(monkeypatch)
            store.write_block(_block("D-agree", status))
            rendered, params = _writes(conn)[-1]
            assert _insert_binding(rendered, params)["active"] is is_admissible_status(status), status

    @pytest.mark.parametrize("has_vector", [False, True])
    def test_upsert_refreshes_active(self, monkeypatch: pytest.MonkeyPatch, admitted, has_vector: bool) -> None:
        """A re-write that demotes the Status must demote the row.

        Without ``active = EXCLUDED.active`` in the conflict arm, a block
        quarantined by a governance apply keeps the ``TRUE`` its first
        write left behind — the stale-in-the-fail-OPEN-direction case.
        """
        store, conn = _pg_store(monkeypatch, has_vector=has_vector)
        embedding = [0.0] * store._embedding_dim if has_vector else None
        store.write_block(_block("D-demote", "active"), embedding=embedding)
        rendered, _ = _writes(conn)[-1]
        assert "active" in rendered.split("ON CONFLICT", 1)[1]


# ─── 2. the SQL derivation cannot drift from the Python one ───────────────────


@pytest.mark.usefixtures("needs_psycopg")
class TestActiveFromStatusSqlTracksThePythonAllowList:
    def test_the_predicate_lists_exactly_the_recognised_statuses(self) -> None:
        from mind_mem.admissibility import RECOGNISED_STATUSES
        from mind_mem.block_store_postgres import _active_from_status_sql

        rendered = _render(_active_from_status_sql())
        quoted = set(re.findall(r"'([a-z0-9_-]+)'", rendered))
        assert RECOGNISED_STATUSES <= quoted, f"missing from SQL: {sorted(RECOGNISED_STATUSES - quoted)}"

    def test_the_predicate_excludes_every_withheld_status(self) -> None:
        from mind_mem.admissibility import UNADMITTED
        from mind_mem.block_store_postgres import _active_from_status_sql

        rendered = _render(_active_from_status_sql())
        quoted = set(re.findall(r"'([a-z0-9_-]+)'", rendered))
        assert not (UNADMITTED & quoted), f"withheld status admitted by the SQL: {sorted(UNADMITTED & quoted)}"


# ─── 3. the one-time backfill for rows written before the column meant it ─────


class TestSchemaMigrationBackfillsActive:
    @pytest.mark.usefixtures("needs_psycopg")
    def test_claiming_the_migration_runs_the_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import block_store_postgres as bsp

        store, conn = _pg_store(monkeypatch, claim_rows=[(bsp._MIGRATION_ACTIVE_FROM_STATUS,)])
        store._schema_ready = False
        monkeypatch.setattr(bsp, "_require_psycopg", lambda: (None, None))
        monkeypatch.setattr(bsp, "_try_create_extension_vector", lambda conn: None)
        store._ensure_schema()
        updates = [sql for sql, _ in conn.statements if sql.startswith("UPDATE") and ".blocks" in sql]
        assert len(updates) == 1, conn.statements
        assert "active" in updates[0] and "metadata" in updates[0]

    @pytest.mark.usefixtures("needs_psycopg")
    def test_an_already_applied_migration_runs_no_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive control is the test above: same path, claim granted."""
        from mind_mem import block_store_postgres as bsp

        store, conn = _pg_store(monkeypatch, claim_rows=[])
        store._schema_ready = False
        monkeypatch.setattr(bsp, "_require_psycopg", lambda: (None, None))
        monkeypatch.setattr(bsp, "_try_create_extension_vector", lambda conn: None)
        store._ensure_schema()
        updates = [sql for sql, _ in conn.statements if sql.startswith("UPDATE") and ".blocks" in sql]
        assert updates == []

    def test_restore_derives_active_rather_than_forcing_true(self) -> None:
        """A snapshot restore must not re-activate what governance withheld."""
        import inspect

        from mind_mem.block_store_postgres import PostgresBlockStore

        src = inspect.getsource(PostgresBlockStore.restore)
        assert "_active_from_status_sql" in src, "restore still hard-codes the active column"


# ─── 4. the enumeration primitive does not trust the column ───────────────────


class _FakeStore:
    """A store whose ``active_only`` filter is a lie — the R2-06 shape.

    ``get_all(active_only=True)`` returning quarantined rows is exactly
    what a constant ``active = TRUE`` column produces, so a caller that
    trusts the flag serves them.
    """

    def __init__(self, blocks: list[dict[str, Any]]) -> None:
        self._blocks = blocks
        self.calls: list[bool] = []

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        self.calls.append(active_only)
        return [dict(b) for b in self._blocks]


def _seed_markdown(workspace: Path) -> None:
    (workspace / "decisions").mkdir(parents=True, exist_ok=True)
    (workspace / "decisions" / "DECISIONS.md").write_text(
        "[D-live]\nStatement: served\nDate: 2026-09-02\nStatus: active\n\n---\n\n"
        "[D-quar]\nStatement: withheld\nDate: 2026-09-02\nStatus: quarantined\n\n---\n\n",
        encoding="utf-8",
    )


@pytest.fixture
def pg_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, _FakeStore]]:
    """A workspace configured for a non-markdown backend, backed by ``_FakeStore``."""
    import json

    from mind_mem import storage

    (tmp_path / "mind-mem.json").write_text(
        json.dumps({"block_store": {"backend": "postgres", "dsn": "postgresql://nobody@localhost/none"}}),
        encoding="utf-8",
    )
    store = _FakeStore([_block("D-live", "active", statement="served"), _block("D-quar", "quarantined", statement="withheld")])
    monkeypatch.setattr(storage, "get_block_store", lambda workspace, config=None: store)
    yield str(tmp_path), store


class TestIterActiveBlocksWithholdsQuarantined:
    def test_markdown_withholds_the_quarantined_block(self, tmp_path: Path) -> None:
        """Positive control for the method: the walk that already works."""
        from mind_mem.storage import iter_active_blocks

        _seed_markdown(tmp_path)
        ids = {b["_id"] for b in iter_active_blocks(str(tmp_path))}
        assert "D-live" in ids, "positive control failed — the admissible seed is missing"
        assert "D-quar" not in ids

    def test_postgres_withholds_the_quarantined_block(self, pg_workspace: tuple[str, _FakeStore]) -> None:
        from mind_mem.storage import iter_active_blocks

        workspace, _ = pg_workspace
        ids = {b["_id"] for b in iter_active_blocks(workspace)}
        assert "D-live" in ids, "positive control failed — the admissible seed is missing"
        assert "D-quar" not in ids

    def test_postgres_enumeration_does_not_rely_on_the_store_flag(self, pg_workspace: tuple[str, _FakeStore]) -> None:
        """The column is not the authority, so it is not the question asked."""
        from mind_mem.storage import iter_active_blocks

        workspace, store = pg_workspace
        iter_active_blocks(workspace)
        assert store.calls == [False], f"iter_active_blocks delegated the decision: {store.calls}"

    def test_active_only_false_still_returns_everything(self, pg_workspace: tuple[str, _FakeStore]) -> None:
        """The mailbox case is untouched — it is not a search."""
        from mind_mem.storage import iter_blocks

        workspace, _ = pg_workspace
        ids = {b["_id"] for b in iter_blocks(workspace, active_only=False)}
        assert ids == {"D-live", "D-quar"}

    def test_a_released_quarantined_id_is_still_admitted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading every row is what keeps ``Releases`` working on Postgres.

        Filtering in the store first would delete the released row before
        ``admit_corpus`` could see the decision that admits it.
        """
        import json

        from mind_mem import storage

        (tmp_path / "mind-mem.json").write_text(
            json.dumps({"block_store": {"backend": "postgres", "dsn": "postgresql://nobody@localhost/none"}}),
            encoding="utf-8",
        )
        release = _block("D-release", "active")
        release["Releases"] = ["INGEST-freed"]
        store = _FakeStore([release, _block("INGEST-freed", "quarantined"), _block("INGEST-held", "quarantined")])
        monkeypatch.setattr(storage, "get_block_store", lambda workspace, config=None: store)

        ids = {b["_id"] for b in storage.iter_active_blocks(str(tmp_path))}
        assert "INGEST-freed" in ids, "an active release decision no longer admits its batch"
        assert "INGEST-held" not in ids


# ─── 5. the replica and sharded adapters cannot write a row of their own ──────


class TestAdaptersHaveNoSecondWritePath:
    """By construction: one INSERT to keep truthful, not three."""

    def test_only_the_primary_store_composes_a_block_insert(self) -> None:
        from pathlib import Path as _Path

        import mind_mem.block_store_postgres as primary
        import mind_mem.block_store_postgres_replica as replica
        import mind_mem.storage.sharded_pg as sharded

        def _has_block_insert(mod: Any) -> bool:
            src = _Path(mod.__file__).read_text(encoding="utf-8")
            return bool(re.search(r"INSERT INTO \{s\}\.blocks", src))

        assert _has_block_insert(primary), "positive control failed — the primary's INSERT is not where this scan looks"
        assert not _has_block_insert(replica)
        assert not _has_block_insert(sharded)


# ─── 6. the same claims against a live Postgres ───────────────────────────────
#
# Skipped only on a real environment fact — ``MIND_MEM_TEST_PG_DSN`` unset,
# i.e. no database to talk to. The CI "postgres backend" job sets it, and
# these are the tests that exercise the actual column, the actual index
# predicate and the actual backfill rather than a rendered statement.

_DSN_ENV = "MIND_MEM_TEST_PG_DSN"


def _live_dsn() -> str:
    import os

    dsn = os.environ.get(_DSN_ENV)
    if not dsn:
        pytest.skip(f"{_DSN_ENV} not set — no live Postgres available")
    return dsn


@pytest.fixture
def live_pg(tmp_path: Path) -> Iterator[tuple[Any, str]]:
    """A live PostgresBlockStore in a throwaway schema, plus a workspace pointed at it."""
    import json
    import uuid

    psycopg = pytest.importorskip("psycopg", reason="psycopg not installed")
    from mind_mem.block_store_postgres import PostgresBlockStore

    dsn = _live_dsn()
    schema = f"mm_active_{uuid.uuid4().hex[:12]}"
    store = PostgresBlockStore(dsn=dsn, schema=schema, workspace=str(tmp_path))
    store._ensure_schema()
    (tmp_path / "mind-mem.json").write_text(
        json.dumps({"block_store": {"backend": "postgres", "dsn": dsn, "schema": schema}}),
        encoding="utf-8",
    )
    try:
        yield store, str(tmp_path)
    finally:
        try:
            conn = psycopg.connect(dsn)
            conn.autocommit = True
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()
        except Exception:
            pass
        store.close()


def _live_active(store: Any, block_id: str) -> bool | None:
    row = store.get_by_id(block_id)
    return None if row is None else row.get("_active")


class TestLivePostgresActiveColumn:
    def test_a_quarantined_write_lands_inactive(self, live_pg: tuple[Any, str], admitted) -> None:
        store, _ = live_pg
        store.write_block(_block("D-live", "active"))
        store.write_block(_block("D-quar", "quarantined"))
        assert _live_active(store, "D-live") is True, "positive control failed — the admissible row is not there"
        assert _live_active(store, "D-quar") is False

    def test_get_all_active_only_withholds_it(self, live_pg: tuple[Any, str], admitted) -> None:
        store, _ = live_pg
        store.write_block(_block("D-live", "active"))
        store.write_block(_block("D-quar", "quarantined"))
        ids = {b["_id"] for b in store.get_all(active_only=True)}
        assert "D-live" in ids, "positive control failed — the admissible row is not served"
        assert "D-quar" not in ids

    def test_an_upsert_that_quarantines_demotes_the_row(self, live_pg: tuple[Any, str], admitted) -> None:
        """The governance case: a block flipped to quarantined stops being active."""
        store, _ = live_pg
        store.write_block(_block("D-flip", "active"))
        assert _live_active(store, "D-flip") is True
        store.write_block(_block("D-flip", "quarantined"))
        assert _live_active(store, "D-flip") is False

    def test_iter_active_blocks_withholds_it_on_the_live_backend(self, live_pg: tuple[Any, str], admitted) -> None:
        from mind_mem.storage import iter_active_blocks

        store, workspace = live_pg
        store.write_block(_block("D-live", "active"))
        store.write_block(_block("D-quar", "quarantined"))
        ids = {b["_id"] for b in iter_active_blocks(workspace)}
        assert "D-live" in ids, "positive control failed — the admissible seed is missing"
        assert "D-quar" not in ids

    def test_the_backfill_repairs_a_pre_5_0_2_row(self, live_pg: tuple[Any, str], admitted) -> None:
        """Reconstruct the 5.0.1 shape and prove one restart fixes it.

        ``active = TRUE`` on a quarantined row with the migration marker
        cleared is exactly what an upgrading deployment looks like.
        """
        from mind_mem.block_store_postgres import _MIGRATION_ACTIVE_FROM_STATUS, PostgresBlockStore, _sql

        store, _ = live_pg
        store.write_block(_block("D-live", "active"))
        store.write_block(_block("D-quar", "quarantined"))
        with store._get_pool().connection() as conn:
            conn.execute(_sql(store._schema, "UPDATE {s}.blocks SET active = TRUE"))
            conn.execute(_sql(store._schema, "DELETE FROM {s}.schema_migrations WHERE version = %s"), (_MIGRATION_ACTIVE_FROM_STATUS,))
            conn.commit()
        assert _live_active(store, "D-quar") is True, "positive control failed — the pre-5.0.2 shape was not reconstructed"

        fresh = PostgresBlockStore(dsn=store._dsn, schema=store._schema, workspace=store._workspace)
        fresh._ensure_schema()
        try:
            assert _live_active(fresh, "D-quar") is False
            assert _live_active(fresh, "D-live") is True, "the backfill demoted an admissible row"
        finally:
            fresh.close()

    def test_restore_does_not_reactivate_a_quarantined_block(self, live_pg: tuple[Any, str], admitted, tmp_path: Path) -> None:
        """Through the RESTORE scope ``apply_engine.restore_snapshot`` opens.

        The ``admitted`` fixture's receipt is proposal-scoped, which the seam
        refuses for a restore (ambient authority over every id); the scope
        below is the batch receipt the sanctioned caller mints, naming the ids
        the snapshot holds. The assertions are unchanged -- and
        :func:`test_an_ungated_restore_is_refused_and_the_quarantine_stands`
        is the positive control that the seam still shuts.
        """
        store, _ = live_pg
        store.write_block(_block("D-live", "active"))
        store.write_block(_block("D-quar", "quarantined"))
        snap_dir = str(tmp_path / "snap-active")
        store.snapshot(snap_dir)
        with restoring(str(tmp_path), batch_id="restore:snap-active", block_ids=("D-live", "D-quar")):
            store.restore(snap_dir)
        assert _live_active(store, "D-live") is True, "positive control failed — restore lost the admissible row"
        assert _live_active(store, "D-quar") is False

    def test_an_ungated_restore_is_refused_and_the_quarantine_stands(self, live_pg: tuple[Any, str], admitted, tmp_path: Path) -> None:
        """Positive control for the scope above: the same call with no RESTORE scope raises,
        and the corpus is re-read to prove nothing moved."""
        from mind_mem.admission import UngatedRestoreError

        store, _ = live_pg
        store.write_block(_block("D-live", "active"))
        snap_dir = str(tmp_path / "snap-active-ungated")
        store.snapshot(snap_dir)
        store.write_block(_block("D-live", "active", statement="after the snapshot"))
        with pytest.raises(UngatedRestoreError, match="proposal-scoped"):
            store.restore(snap_dir)
        row = store.get_by_id("D-live")
        assert row is not None, "positive control failed — the seed block is not there to be reverted"
        assert row["Statement"] == "after the snapshot", "the refused restore reverted the corpus anyway"

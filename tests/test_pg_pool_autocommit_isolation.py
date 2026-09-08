# Copyright 2026 STARGA, Inc.
"""A pooled connection must never re-enter the pool in autocommit.

``_ensure_schema`` needs autocommit for DDL but borrows a connection shared
process-wide through ``_pool_registry``. Left set, the next checkout receives a
connection whose bare statements self-commit, defeating any lease or claim not
wrapped in an explicit ``conn.transaction()``.

Two layers guard it: scoped restoration in ``_ensure_schema`` (try/finally) and
the pool ``reset`` callback. A THIRD property matters and is tested here: when
restoration FAILS, the callback must raise so psycopg_pool discards the
connection instead of recycling one we cannot vouch for.

ISOLATION. These controls open a real database, so they refuse to run unless
the launcher explicitly ATTESTS an isolated instance:

    MIND_MEM_TEST_DSN          the dedicated server
    MIND_MEM_TEST_ISOLATED_ID  host:port/dbname the launcher asserts it created

The DSN is validated against that attestation BEFORE any connection is opened.
A DSN with no attestation is refused as configured-but-unprovable -- isolation
is never inferred from a path, a schema name, or the absence of evidence. This
file reads NO operator configuration and therefore touches no credential of the
live system; the only credential it sees is the synthetic one in its own DSN.
"""

from __future__ import annotations

import os
import urllib.parse as up

import pytest

#: Non-secret identities that are never acceptable as a test target, in every
#: spelling. Extend explicitly; nothing here is discovered from a config file.
_DENY_IDENTITIES = {
    ("127.0.0.1", 5432, "mindmem"),
    ("localhost", 5432, "mindmem"),
    ("::1", 5432, "mindmem"),
}

#: Host spellings that denote the same endpoint, so an alias cannot slip past.
_HOST_ALIASES = {"localhost": "127.0.0.1", "::1": "127.0.0.1", "": "127.0.0.1"}

_DEFAULT_PG_PORT = 5432


def _identity(dsn: str) -> tuple[str, int, str]:
    """(host, port, database), normalised so aliases and default ports match.

    Bare equality on urlparse output was the previous guard and it missed
    default-port omission, hostname aliases, and libpq conninfo spellings.
    """
    if "://" not in dsn:  # libpq conninfo: host=... port=... dbname=...
        kv = dict(part.split("=", 1) for part in dsn.split() if "=" in part)
        host = kv.get("host", "")
        port = int(kv.get("port", _DEFAULT_PG_PORT))
        db = kv.get("dbname", kv.get("database", ""))
    else:
        u = up.urlparse(dsn)
        host = u.hostname or ""
        port = u.port or _DEFAULT_PG_PORT
        db = (u.path or "").lstrip("/")
    host = _HOST_ALIASES.get(host.lower(), host.lower())
    return (host, port, db)


def _attested_dsn() -> str:
    """The dedicated DSN, or skip/fail -- never a guess."""
    dsn = os.environ.get("MIND_MEM_TEST_DSN", "").strip()
    attest = os.environ.get("MIND_MEM_TEST_ISOLATED_ID", "").strip()
    if not dsn and not attest:
        pytest.skip("no isolated database configured; set MIND_MEM_TEST_DSN and MIND_MEM_TEST_ISOLATED_ID")
    if not dsn or not attest:
        pytest.fail(
            "configured but unprovable: MIND_MEM_TEST_DSN and MIND_MEM_TEST_ISOLATED_ID "
            "must BOTH be set. Isolation is attested by the launcher, never inferred."
        )
    ident = _identity(dsn)
    want = _identity("postgresql://x@" + attest if "://" not in attest else attest)
    if ident != want:
        pytest.fail(f"DSN identity {ident} does not match the attested isolated instance {want}")
    if ident in _DENY_IDENTITIES:
        pytest.fail(f"refusing: {ident} is a known production identity")
    return dsn


def _all_pooled_autocommit_states(pool, expected: int) -> list[bool]:
    """Autocommit of EVERY connection the pool can hand out -- exactly `expected`.

    The previous version broke out of its loop on acquisition failure and
    returned whatever it had, so a healthy first connection followed by a
    failed checkout could certify the whole pool. It now demands the exact
    count and lets an acquisition failure raise.
    """
    import contextlib

    with contextlib.ExitStack() as stack:
        conns = [stack.enter_context(pool.connection()) for _ in range(expected)]
        assert len({id(c) for c in conns}) == expected, (
            f"expected {expected} distinct pooled connections, got {len({id(c) for c in conns})} -- the sample is partial"
        )
        return [c.autocommit for c in conns]


@pytest.fixture
def pool():
    """The PRODUCT's pool at a pinned small capacity, not one this test wires.

    An earlier fixture built its own ConnectionPool and passed ``reset=`` by
    hand: every control passed, and deleting the wiring from ``_get_pool``
    changed nothing, because the test supplied the callback itself.
    """
    pytest.importorskip("psycopg_pool")
    from mind_mem import block_store_postgres as bsp

    dsn = _attested_dsn()
    store = bsp.PostgresBlockStore(dsn, schema="pool_reset_probe")
    p = store._get_pool()
    p.resize(min_size=2, max_size=2)  # pinned capacity so "all" is a known number
    try:
        yield p
    finally:
        with bsp._pool_registry_lock:
            for key, pooled in list(bsp._pool_registry.items()):
                if pooled is p:
                    del bsp._pool_registry[key]
        try:
            p.close()
        except Exception:
            pass


def test_isolation_attestation_is_required_and_matches() -> None:
    """Isolation is attested and verified, not inferred."""
    ident = _identity(_attested_dsn())
    assert ident not in _DENY_IDENTITIES
    # Alias and default-port normalisation actually work.
    assert _identity("postgresql://u:p@localhost:5432/mindmem") == ("127.0.0.1", 5432, "mindmem")
    assert _identity("postgresql://u:p@127.0.0.1/mindmem") == ("127.0.0.1", 5432, "mindmem")
    assert _identity("host=localhost dbname=mindmem") == ("127.0.0.1", 5432, "mindmem")
    # ...and all three spellings are caught by the deny list.
    for spelling in (
        "postgresql://u:p@localhost:5432/mindmem",
        "postgresql://u:p@127.0.0.1/mindmem",
        "host=localhost dbname=mindmem",
    ):
        assert _identity(spelling) in _DENY_IDENTITIES, f"{spelling} evaded the deny list"


def test_control_1_checkout_after_schema_setup_is_not_autocommit(pool) -> None:
    with pool.connection() as conn:
        conn.autocommit = True
        conn.execute("SELECT 1")
        assert conn.autocommit is True  # positive control: it really was set
    states = _all_pooled_autocommit_states(pool, expected=2)
    assert not any(states), f"autocommit leaked into the pool: {states}"


def test_control_2_exception_path_does_not_leak_autocommit(pool) -> None:
    with pytest.raises(RuntimeError):
        with pool.connection() as conn:
            conn.autocommit = True
            raise RuntimeError("schema setup failed")
    states = _all_pooled_autocommit_states(pool, expected=2)
    assert not any(states), f"an exception left autocommit in the pool: {states}"


def test_control_3_transaction_block_rolls_back_atomically(pool) -> None:
    """An explicit ``conn.transaction()`` is atomic.

    Labelled truthfully: this holds even under autocommit, as measured on
    psycopg 3.3.4, because ``transaction()`` emits an explicit transaction
    start. So this does NOT demonstrate the autocommit-leak consequence -- it
    pins the atomicity the queue's lease/claim work depends on. The leak's
    real cost is on bare statements outside such a block.
    """
    with pool.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS atomicity_probe")
        conn.execute("CREATE TABLE atomicity_probe (x int PRIMARY KEY)")
        conn.commit()
    with pool.connection() as conn:
        try:
            with conn.transaction():
                conn.execute("INSERT INTO atomicity_probe VALUES (1)")
                conn.execute("INSERT INTO atomicity_probe VALUES (1)")  # PK violation
        except Exception:
            pass
    with pool.connection() as conn:
        rows = conn.execute("SELECT count(*) FROM atomicity_probe").fetchone()[0]
        assert rows == 0, f"rolled-back block left {rows} row(s)"
        conn.execute("DROP TABLE IF EXISTS atomicity_probe")
        conn.commit()


def test_control_4_failed_restoration_discards_the_connection() -> None:
    """The PRODUCT's callback must raise when restoration does not take.

    The injection is at the SETTER, not at the callback: a connection subclass
    whose ``autocommit`` setter silently refuses ``False``. The pool is then
    constructed with the real ``_reset_pooled_connection``, so what is under
    test is whether the PRODUCT raises -- not whether psycopg_pool discards on
    a raise, which is psycopg's behaviour and not ours.

    An earlier draft passed its OWN raising callback and therefore stayed green
    against the shipped fail-open, which swallowed the failure and returned a
    poisoned connection while reporting success. Root caught that; mutation
    confirms this version does not.

    Contract, read from installed psycopg_pool 3.3.1 ``_reset_connection``:
    the callback runs inside ``try/except CLIENT_EXCEPTIONS`` (== Exception)
    and a raise routes to ``_close_connection``. The pool's own post-check
    inspects ``transaction_status`` only, which cannot see a wrong autocommit.
    """
    psycopg = pytest.importorskip("psycopg")
    psycopg_pool = pytest.importorskip("psycopg_pool")
    from mind_mem.block_store_postgres import _reset_pooled_connection

    dsn = _attested_dsn()

    class StuckAutocommit(psycopg.Connection):
        """Refuses to leave autocommit -- the failure mode being injected."""

        @property
        def autocommit(self):  # type: ignore[override]
            return True

        @autocommit.setter
        def autocommit(self, value):  # type: ignore[override]
            return  # silently does not take

    stuck = psycopg_pool.ConnectionPool(
        dsn,
        min_size=1,
        max_size=1,
        open=True,
        connection_class=StuckAutocommit,
        reset=_reset_pooled_connection,
    )
    try:
        seen = []
        for _ in range(3):
            with stuck.connection() as c:
                seen.append(id(c))
                c.execute("SELECT 1")
        assert len(set(seen)) == 3, (
            "the PRODUCT callback did not raise on a refused restoration, so the "
            f"pool recycled a connection it cannot vouch for (distinct={len(set(seen))}/3)"
        )
    finally:
        stuck.close()

    # Positive control: a NORMAL connection through the same product callback
    # is reused, so the assertion above measures the discard and not churn.
    good = psycopg_pool.ConnectionPool(dsn, min_size=1, max_size=1, open=True, reset=_reset_pooled_connection)
    try:
        seen2 = []
        for _ in range(3):
            with good.connection() as c:
                seen2.append(id(c))
                c.execute("SELECT 1")
        assert len(set(seen2)) == 1, f"healthy path should reuse one connection, saw {len(set(seen2))}"
    finally:
        good.close()


def test_control_5_scoped_restoration_holds_without_the_pool_callback() -> None:
    """The second layer, proven on its own.

    Controls 1-2 pass on the reset callback alone, so deleting the try/finally
    in ``_ensure_schema`` is invisible to them -- correct, the layers are
    deliberately redundant. This isolates layer two: it runs the REAL
    ``_ensure_schema`` against a pool whose reset callback is neutralised, so
    only the scoped restoration can keep autocommit off.
    """
    pytest.importorskip("psycopg_pool")
    from mind_mem import block_store_postgres as bsp

    dsn = _attested_dsn()
    store = bsp.PostgresBlockStore(dsn, schema="scoped_restore_probe")

    original = bsp._reset_pooled_connection
    bsp._reset_pooled_connection = lambda conn: None  # neutralise layer one
    try:
        pool = store._get_pool()
        pool.resize(min_size=2, max_size=2)
        store._ensure_schema()

        # Positive control first: the DDL genuinely ran, so a pass below is not
        # a pass over a no-op.
        with pool.connection() as conn:
            got = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s",
                ("scoped_restore_probe",),
            ).fetchone()[0]
        assert got > 0, "no tables created -- _ensure_schema did not run"

        states = _all_pooled_autocommit_states(pool, expected=2)
        assert not any(states), (
            f"_ensure_schema leaked autocommit {states}; with the pool callback neutralised the scoped try/finally is the only guard left"
        )
    finally:
        bsp._reset_pooled_connection = original
        with bsp._pool_registry_lock:
            for key, pooled in list(bsp._pool_registry.items()):
                if pooled is store._pool:
                    del bsp._pool_registry[key]
        try:
            if store._pool is not None:
                store._pool.close()
        except Exception:
            pass

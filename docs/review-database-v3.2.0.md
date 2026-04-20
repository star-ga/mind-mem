# Database Review — PostgresBlockStore v3.2.0

**Reviewed:** 2026-04-19
**Files:** `src/mind_mem/block_store_postgres.py` (576 LOC),
`src/mind_mem/block_store_postgres_replica.py` (223 LOC),
`tests/test_postgres_block_store.py`, `tests/test_postgres_replica_routing.py`

---

## Posture: YELLOW

Two CRITICAL findings (SQL-injection surface and workspace_lock TTL), several HIGH-level gaps
(missing indexes, schema versioning). No data-loss bug in the happy path; the critical
transaction coverage for `restore` and `snapshot` is correct.

---

## 1. SQL Injection Surface — CRITICAL

### 1a. Schema name concatenated without quoting

Every SQL string is built with an f-string that embeds `self._schema` as a bare token:

```python
sql = f"SELECT … FROM {self._schema}.blocks WHERE id = %s"
```

If `schema` is controlled by the caller (set via `mind-mem.json`) a value like
`mind_mem; DROP TABLE mind_mem.blocks--` is valid Python but destructive SQL. Postgres
identifiers that look like keywords or contain special characters also break at parse time.

**Fix:** Wrap every schema reference with `psycopg.sql.Identifier`.

```python
from psycopg import sql as pgsql

_SCHEMA = pgsql.Identifier(self._schema)

sql = pgsql.SQL(
    "SELECT id, file_path FROM {schema}.blocks WHERE id = %s"
).format(schema=_SCHEMA)
conn.execute(sql, (block_id,))
```

The same pattern applies to every method: `get_all`, `get_by_id`, `search`,
`list_blocks`, `write_block`, `delete_block`, `snapshot`, `restore`, `diff`,
and the inline DDL in `_ddl()`.

### 1b. DDL template uses `.format()` — same issue

```python
def _ddl(schema: str) -> str:
    return _DDL_TEMPLATE.format(schema=schema)
```

`CREATE TABLE IF NOT EXISTS {schema}.blocks` is not a parameterised query; it is
string formatting. Use `psycopg.sql.SQL(...).format(schema=Identifier(schema))` here too,
or validate the schema name against `^[a-z_][a-z0-9_]{0,62}$` before passing it in (belt
and suspenders).

**Recommended: both.** Validate at `__init__` + use `sql.Identifier` at query build time.

---

## 2. Transaction Boundaries — GREEN (with one note)

| Method | Atomic? | Assessment |
|---|---|---|
| `write_block` | `conn.transaction()` wrapping single upsert | Correct |
| `delete_block` | `conn.transaction()` around DELETE + optional audit log INSERT | Correct; the inner `try/except` for the audit log is fine since the outer transaction rolls back on unhandled error |
| `snapshot` | Single `conn.transaction()` covers `INSERT snapshots`, `DELETE snapshot_blocks`, loop of `INSERT snapshot_blocks` | Correct |
| `restore` | Single `conn.transaction()` covers check + `DELETE blocks` + `INSERT … SELECT` | Correct — crash-safe |

**Note:** Default isolation is READ COMMITTED (Postgres default). The `restore` path
does a full table clear + re-insert which is safe under any isolation, but concurrent
readers can see an empty `blocks` table for the duration of the restore transaction.
If that matters, consider `SET TRANSACTION ISOLATION LEVEL SERIALIZABLE` inside the
restore transaction, or document this as expected behaviour.

---

## 3. Index Design — HIGH

### Missing indexes

| Gap | Impact | Remedy |
|---|---|---|
| No `updated_at` index | `WHERE updated_at > $ts` time-range queries do a seq-scan | `CREATE INDEX blocks_updated_at ON {schema}.blocks(updated_at DESC)` |
| FTS: no GIN index on `to_tsvector(content)` | Every `search()` call recomputes tsvector per row — seq-scan on large tables | See DDL below |
| No pgvector HNSW index | v3.2.0 mentions embedding support; without HNSW ann search degrades to exact scan | Conditional — add when the `embedding` column is introduced |
| `snapshot_blocks(snap_id)` FK has no supporting index | Already the PK component `(snap_id, block_id)`; FK queries on `snap_id` use the PK | OK |

### FTS index (recommended addition to `_DDL_TEMPLATE`)

```sql
CREATE INDEX IF NOT EXISTS blocks_fts
    ON {schema}.blocks
    USING GIN (to_tsvector('english', content));
```

The existing `search()` method calls `to_tsvector` per row twice per query (once in
`WHERE` and once in `ts_rank`). The GIN index makes the `WHERE` clause a bitmap index
scan; `ts_rank` can then work on the filtered set only.

### Active partial index

`CREATE INDEX IF NOT EXISTS blocks_active ON {schema}.blocks(active) WHERE active` is
a smart use of a partial index. However, the index predicate uses a boolean column with
only two values — low cardinality. Postgres may prefer a seq-scan on smaller tables. For
soft-delete queries the more useful form is a covering partial index:

```sql
CREATE INDEX IF NOT EXISTS blocks_active_file_path
    ON {schema}.blocks(file_path)
    WHERE active;
```

This makes `list_blocks()` (`DISTINCT file_path WHERE active = TRUE`) an index-only scan.

---

## 4. Schema Migration Safety — HIGH

`_ensure_schema` runs `CREATE TABLE IF NOT EXISTS` each startup. This is idempotent for
initial creation but has two gaps:

1. **No schema version tracking.** When v3.3 adds a column (e.g., `embedding VECTOR(1536)`),
   there is no way to know whether the column already exists in production without a
   version table.
2. **ALTER TABLE is not idempotent** — `ALTER TABLE blocks ADD COLUMN embedding VECTOR(1536)`
   fails if the column exists. Migrations must be conditional.

### Recommended: schema version table

```sql
CREATE TABLE IF NOT EXISTS {schema}.schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT NOT NULL
);
```

Migration runner pseudocode:

```python
MIGRATIONS = [
    (1, "initial schema",                _ddl_v1),
    (2, "add blocks_fts GIN index",      _ddl_v2),
    (3, "add updated_at index",          _ddl_v3),
    (4, "add embedding VECTOR column",   _ddl_v4),
]

def _run_migrations(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS …schema_migrations…")
    applied = {r[0] for r in conn.execute("SELECT version FROM …").fetchall()}
    for ver, desc, fn in MIGRATIONS:
        if ver not in applied:
            fn(conn)
            conn.execute("INSERT INTO …schema_migrations VALUES (%s,%s)", (ver, desc))
```

Each migration function uses `IF NOT EXISTS` / `IF EXISTS` guards or `DO $$ BEGIN … EXCEPTION WHEN duplicate_column THEN NULL; END $$` blocks so they are safe to replay.

---

## 5. Connection Pool — YELLOW

```python
self._pool = ConnectionPool(self._dsn, min_size=1, max_size=10, open=True)
```

| Parameter | Current | Production recommendation |
|---|---|---|
| `max_size` | 10 | Raise to 20–30 for high-concurrency MCP workloads; or make configurable via `mind-mem.json` |
| `timeout` | not set (default: 30 s) | Set `timeout=10.0` to fail fast rather than queue indefinitely under pool exhaustion |
| `reconnect_failed` | not set | Hook into the callback to emit a metric / alert so pool exhaustion is visible |
| `max_waiting` | not set | Default is unbounded; cap at `max_size * 2` to shed load quickly |

Under pool exhaustion `pool.connection()` blocks until the default timeout then raises
`PoolTimeout`. There is no fallback — callers will see `BlockStoreError`. This is
correct fail-fast behaviour but the timeout value should be explicit, not implicit.

Suggested config extension:

```json
"block_store": {
    "backend": "postgres",
    "dsn": "...",
    "pool_max_size": 20,
    "pool_timeout": 10.0
}
```

---

## 6. Replica Routing — YELLOW

### Circuit breaker distinguishes nothing

`_run_on_replica` catches `(BlockStoreError, Exception)` — which means network errors,
query errors, and programming errors (e.g., `AttributeError`) all increment the failure
counter equally:

```python
except (BlockStoreError, Exception) as exc:
    rep.record_failure()
```

A one-off `BlockStoreError("snapshot not found")` from a business-logic path will trip
the breaker after 3 calls, removing a healthy replica from rotation.

**Fix:** Only count network-level errors toward the failure counter:

```python
import psycopg

_TRANSIENT = (psycopg.OperationalError, psycopg.InterfaceError, ConnectionError, TimeoutError)

except _TRANSIENT as exc:
    rep.record_failure()
    ...
    return getattr(self._primary, method_name)(*args, **kwargs)
except Exception as exc:
    # Non-transient: don't penalise replica, but still fall back to primary.
    _log.warning("replica_nontransient_error", method=method_name, error=str(exc))
    return getattr(self._primary, method_name)(*args, **kwargs)
```

### Half-open recovery

`record_success()` resets `failure_count = 0` and `cooling_until = 0.0`. This is
correct. The circuit goes: CLOSED → OPEN (after 3 failures) → half-open (first request
after cooldown, because `healthy` checks `time.time() >= cooling_until`) → CLOSED (on
success). This is the right pattern.

**One gap:** if the first post-cooldown request also fails, `failure_count` is already at
`_CIRCUIT_BREAKER_FAILURES` (3) from before the cooldown, so `record_failure()` sets a
new `cooling_until` without needing a fresh burst of 3 failures. This is aggressive but
acceptable — document it as intended.

---

## 7. Workspace Lock TTL — CRITICAL

The `workspace_lock` table has no TTL column:

```sql
CREATE TABLE IF NOT EXISTS {schema}.workspace_lock (
    lock_id     TEXT PRIMARY KEY,
    holder      TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

If a process acquires a lock and is killed (OOM, SIGKILL), the lock row persists forever.
No subsequent process can acquire the lock without manual `DELETE`.

**Fix — add an `expires_at` column and enforce it at lock-acquire time:**

```sql
ALTER TABLE {schema}.workspace_lock
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ NOT NULL
        DEFAULT NOW() + INTERVAL '5 minutes';
```

Lock-acquire SQL should atomically expire stale locks:

```sql
-- Delete any expired locks first, then try to insert ours.
DELETE FROM {schema}.workspace_lock WHERE expires_at < NOW();
INSERT INTO {schema}.workspace_lock (lock_id, holder, expires_at)
VALUES (%s, %s, NOW() + INTERVAL '5 minutes')
ON CONFLICT (lock_id) DO NOTHING
RETURNING lock_id;
```

If `RETURNING` returns no row, the lock is held by another live process. Pair this with
a background heartbeat that extends `expires_at` while the holder is alive.

---

## Recommended ALTER Statements (apply in order)

```sql
-- 1. Schema versioning table
CREATE TABLE IF NOT EXISTS mind_mem.schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT NOT NULL
);

-- 2. GIN full-text index (avoids per-row tsvector recomputation)
CREATE INDEX IF NOT EXISTS blocks_fts
    ON mind_mem.blocks
    USING GIN (to_tsvector('english', content));

-- 3. updated_at index for time-range queries
CREATE INDEX IF NOT EXISTS blocks_updated_at
    ON mind_mem.blocks(updated_at DESC);

-- 4. Covering partial index for list_blocks() / active filter
CREATE INDEX IF NOT EXISTS blocks_active_file_path
    ON mind_mem.blocks(file_path)
    WHERE active;

-- 5. workspace_lock TTL column
ALTER TABLE mind_mem.workspace_lock
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ NOT NULL
        DEFAULT NOW() + INTERVAL '5 minutes';

CREATE INDEX IF NOT EXISTS workspace_lock_expires_at
    ON mind_mem.workspace_lock(expires_at);
```

Replace `mind_mem` with the configured schema name.

---

## Top-3 Solid Design Decisions

1. **`restore()` is fully atomic.** DELETE + INSERT SELECT inside a single `conn.transaction()` means a crash mid-restore leaves the live table unchanged. This is the hardest invariant to get right and it is correct.

2. **`snapshot()` uses ON CONFLICT on both the parent and child tables.** Re-running snapshot on the same `snap_id` is safe: the parent row is updated, stale child rows are deleted, and fresh ones are inserted — all inside one transaction. Idempotent overwrite with no orphan rows.

3. **Circuit breaker with fail-open semantics.** When all replicas are cooling, reads fall through to the primary. This is the correct posture for a memory-read workload: slightly slower (primary handles more load) but never unavailable. The `record_success` full-reset is clean and avoids lingering penalty after recovery.

---

## Finding Summary

| Severity | Count | Topics |
|---|---|---|
| CRITICAL | 2 | Schema name SQL injection (#1a, #1b); workspace_lock has no TTL (#7) |
| HIGH | 2 | Missing GIN/updated_at indexes (#3); no schema version table (#4) |
| MEDIUM | 2 | Pool not tuned for production (#5); circuit breaker catches all exceptions (#6) |
| LOW | 1 | `blocks_active` partial index low-cardinality; covering form preferred (#3) |

# Storage Backends

mind-mem supports three block storage backends. The default is `markdown`
(files on disk). Switching backends is a one-line change in `mind-mem.json`.

---

## Backends at a Glance

| Backend | Best for | Requires |
| ------- | -------- | -------- |
| `markdown` (default) | Single-process, file-based workspaces | Nothing extra |
| `encrypted` | At-rest encryption for sensitive workspaces | `MIND_MEM_ENCRYPTION_PASSPHRASE` env var |
| `postgres` | Multi-host shared workspaces, concurrent agents | PostgreSQL 14+, `pip install "mind-mem[postgres]"` |

---

## Markdown Backend (default)

No configuration required. Blocks are stored in `.md` files under the
workspace corpus directories (`decisions/`, `tasks/`, `entities/`, etc.).
Snapshots are subdirectories under `intelligence/applied/`.

```json
{
  "block_store": {
    "backend": "markdown"
  }
}
```

---

## Postgres Backend

Stores all blocks in a PostgreSQL database. Suitable for:

- Multiple agent processes sharing a workspace across machines.
- Workspaces that outgrow filesystem performance.
- Teams that already run Postgres and want unified backup/restore via `pg_dump`.

### Quick start with Docker

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: mind_mem
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: mind_mem
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

```bash
docker compose up -d
pip install "mind-mem[postgres]"
```

### mind-mem.json configuration

```json
{
  "block_store": {
    "backend": "postgres",
    "dsn": "postgresql://mind_mem:secret@localhost:5432/mind_mem"
  }
}
```

The schema (`mind_mem` by default) and all tables are created automatically
on first use via idempotent `CREATE TABLE IF NOT EXISTS` migrations. No
manual `psql` steps are required.

### Schema

The adapter creates the following tables inside the configured schema
(default `mind_mem`):

| Table | Purpose |
| ----- | ------- |
| `blocks` | Live blocks (id, file_path, content, metadata JSONB, timestamps, active flag) |
| `snapshots` | Snapshot manifests (snap_id, created_at, manifest JSONB) |
| `snapshot_blocks` | Block content at snapshot time (snap_id, block_id, content, metadata) |
| `workspace_lock` | Distributed advisory lock record (lock_id, holder, acquired_at) |

Indexes are created on `blocks.file_path` and `blocks.active` (partial,
`WHERE active`) automatically.

### Full-text search

`PostgresBlockStore.search()` uses `plainto_tsquery` over the `content`
column and the `Statement` metadata field. If the text index is absent it
falls back to `ILIKE`.  To enable GIN-accelerated FTS:

```sql
CREATE INDEX blocks_fts ON mind_mem.blocks
    USING gin(to_tsvector('english', content || ' ' || COALESCE(metadata->>'Statement', '')));
```

### Snapshots and cross-backend compatibility

`snapshot()` writes to both the `snapshot_blocks` table and a
`MANIFEST.json` file on disk in the given `snap_dir`. This means a
snapshot taken with the Postgres backend can be restored by the Markdown
backend and vice versa (the MANIFEST.json describes which files were
in scope).

### Connection pool

The adapter uses `psycopg_pool.ConnectionPool` with `min_size=1`,
`max_size=10`. All operations are thread-safe. Call `store.close()` or
use the store as a context manager to release connections gracefully.

```python
from mind_mem.block_store_postgres import PostgresBlockStore

with PostgresBlockStore(dsn="postgresql://localhost/mind_mem") as store:
    store.write_block({"_id": "D-20260419-001", "Statement": "Hello"})
    block = store.get_by_id("D-20260419-001")
```

### Error handling

All methods raise `mind_mem.block_store.BlockStoreError` on failure.
Connection errors are wrapped so callers need only catch one exception type.

```python
from mind_mem.block_store import BlockStoreError

try:
    store.write_block(block)
except BlockStoreError as exc:
    logging.error("storage failure: %s", exc)
```

### Testing

Set `MIND_MEM_TEST_PG_DSN` to a live Postgres DSN before running:

```bash
export MIND_MEM_TEST_PG_DSN="postgresql://postgres:test@localhost:5432/postgres"
pytest tests/test_postgres_block_store.py -v
```

The tests skip automatically when the environment variable is absent or
when psycopg is not installed.

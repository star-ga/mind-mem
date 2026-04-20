# Storage Backend Migration Guide

v3.2.0 introduces a pluggable storage backend. The default remains
the Markdown-on-disk corpus that has shipped since v1.0. This guide
covers two moves:

1. **Markdown → Postgres** (new in v3.2.0) — for teams that need
   multi-host shared workspaces, cross-region replicas, or
   `LISTEN/NOTIFY`-based invalidation.
2. **Postgres → Markdown** — unwind path for when Postgres is no
   longer the right fit.

Both moves use the `mm migrate-store` command (v3.2.0+) which handles
the atomic swap, verifies the copy, and keeps the source intact so
you can roll back.

---

## 1. When to migrate to Postgres

The Markdown backend is strictly better for:

- Single-host, single-user workspaces.
- Offline / air-gapped deployments.
- Workspaces that want `grep` / `rg` / Obsidian / plain-file tooling.
- Workspaces smaller than ~10 MB of corpus text (~200,000 blocks).

The Postgres backend is strictly better for:

- Multi-host shared workspaces (several mind-mem processes, possibly
  on different machines, pointing at the same logical memory).
- Workspaces that exceed ~10 MB of corpus text where concurrent read
  performance matters more than grep-ability.
- Deployments that need `LISTEN/NOTIFY`-based cache invalidation
  across client processes (saves the polling overhead of file
  watchers).
- Teams with an existing Postgres + pgvector ops story (backup,
  monitoring, point-in-time recovery).

If you're unsure, stay on Markdown. The migration is symmetric —
moving to Postgres later is a one-command operation, not a one-way
door.

---

## 2. Prerequisites

### Postgres side

- PostgreSQL 14+ (16 recommended).
- `pgvector` extension available (`CREATE EXTENSION IF NOT EXISTS
  vector;`) — only needed when you turn on hybrid / vector search.
- A dedicated role + database:

  ```sql
  CREATE ROLE mindmem LOGIN PASSWORD '<strong-password>';
  CREATE DATABASE mindmem OWNER mindmem;
  \c mindmem
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

- A DSN string, either:
  - `postgresql://mindmem:<password>@localhost:5432/mindmem`
  - `postgresql:///mindmem?host=/var/run/postgresql` (unix socket)

### mind-mem side

- v3.2.0 or later (`pip install --upgrade mind-mem`).
- The `postgres` extra: `pip install mind-mem[postgres]` — this
  pulls `psycopg[binary]` + `pgvector`.

---

## 3. Markdown → Postgres migration

### 3.1 Dry-run

Every migration starts with a dry-run that walks the source,
projects the target schema, and reports the delta without writing.

```bash
mm migrate-store \
    --from markdown \
    --to postgres \
    --dsn postgresql://mindmem:***@localhost:5432/mindmem \
    --dry-run
```

The output is a JSON envelope listing:

- `block_count` — total blocks in the source.
- `snapshot_count` — snapshots that will be migrated.
- `estimated_bytes` — rough target footprint (blocks + metadata +
  snapshots).
- `schema_conflicts` — any pre-existing `mind_mem.*` tables that
  would clash. Migration aborts when this is non-empty; you must
  drop / rename them first.

Run the dry-run against a *production* DSN to see the real picture.

### 3.2 Execute

```bash
mm migrate-store \
    --from markdown \
    --to postgres \
    --dsn postgresql://mindmem:***@localhost:5432/mindmem \
    --execute
```

Steps the command performs:

1. Acquires the **workspace-wide lock** (`BlockStore.lock()`) so no
   writes can land during the copy.
2. Calls `store.snapshot("<ws>/intelligence/applied/pre-migrate-<ts>")`
   on the source — this is your rollback point.
3. Ensures the Postgres schema exists
   (`PostgresBlockStore._ensure_schema`).
4. Copies every block in a single transaction:
   `INSERT INTO blocks ... ON CONFLICT (id) DO UPDATE` so the
   operation is idempotent if re-run.
5. Copies every snapshot from
   `intelligence/applied/<ts>/` into the `snapshots` +
   `snapshot_blocks` tables.
6. Verifies:
   - `SELECT COUNT(*) FROM blocks` matches the source block count.
   - For a sampled 100 blocks, `content_hash` round-trips.
   - Every snapshot's block list matches the source manifest.
7. Writes a **migration receipt** to
   `memory/migrations/<ts>-markdown-to-postgres.json` with the
   counts, verification results, and the DSN the data was copied
   into.
8. **Leaves the Markdown corpus intact.** The swap of which backend
   mind-mem actually reads from is a separate step (§3.3).

Expected throughput: ~2,000 blocks/second on localhost Postgres,
~200 blocks/second over a LAN link, ~20 blocks/second over WAN.

### 3.3 Flip the backend

Edit `mind-mem.json`:

```jsonc
{
    "block_store": {
        "backend": "postgres",
        "dsn": "postgresql://mindmem:***@localhost:5432/mindmem",
        "schema": "mind_mem"
    }
}
```

Restart the MCP server. Verify:

```bash
mm status
mm recall "something you remember" --limit 5
```

The Markdown corpus stays on disk — untouched, still parseable,
still `grep`-able. If you want to free the space and are sure the
migration stuck, remove it manually after a cooling-off period.

### 3.4 Rollback

If something is off:

```bash
# Revert the config
mm config set block_store.backend markdown

# Restart
```

Because Markdown was never truncated, this is instant. Postgres
still has the migrated data; you can try again later.

If you *have* dropped the Markdown corpus and need to rebuild from
Postgres, restore the pre-migration snapshot:

```bash
mm snapshot restore intelligence/applied/pre-migrate-<ts>
```

---

## 4. Postgres → Markdown migration

Symmetric to §3. Dry-run first:

```bash
mm migrate-store \
    --from postgres \
    --to markdown \
    --dsn postgresql://mindmem:***@localhost:5432/mindmem \
    --dry-run
```

Execute:

```bash
mm migrate-store \
    --from postgres \
    --to markdown \
    --dsn postgresql://mindmem:***@localhost:5432/mindmem \
    --execute
```

Steps:

1. Acquires the workspace lock (both backends respect it).
2. Calls `PostgresBlockStore.snapshot(...)` writing a full snapshot
   into the `snapshots` + `snapshot_blocks` tables — this is your
   rollback point on the Postgres side.
3. Walks every block in `SELECT ... FROM blocks` and writes it to
   the canonical `<prefix>-file.md` via `MarkdownBlockStore.write_block`.
4. Verifies the on-disk block count matches the Postgres count.
5. Writes the migration receipt under
   `memory/migrations/<ts>-postgres-to-markdown.json`.
6. Leaves Postgres intact.

---

## 5. Performance tuning

### Postgres

- `shared_buffers = 25% of RAM`.
- `max_connections = 50` is usually enough — mind-mem's default
  connection pool is 10.
- Create a partial index on active blocks once the table exceeds
  ~100k rows:

  ```sql
  CREATE INDEX blocks_active_updated ON mind_mem.blocks (updated_at DESC)
      WHERE active;
  ```

- For hybrid search, use `pgvector` HNSW:

  ```sql
  CREATE INDEX blocks_embedding_hnsw ON mind_mem.blocks
      USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
  ```

### Markdown

- Use an SSD. Spinning rust will cripple snapshot rollback speed.
- Keep the workspace on a local filesystem. NFS / SMB turns every
  file lock into a network round-trip.

---

## 6. Multi-region / replica setups

`PostgresBlockStore` does not itself do replication. Use
Postgres-level streaming replication (`pg_basebackup`) or logical
replication (`pglogical`). For read scaling:

```jsonc
{
    "block_store": {
        "backend": "postgres",
        "dsn": "postgresql://mindmem@primary:5432/mindmem",
        "replicas": [
            "postgresql://mindmem@replica-1:5432/mindmem",
            "postgresql://mindmem@replica-2:5432/mindmem"
        ]
    }
}
```

Read-heavy MCP tools (`recall`, `find_similar`, `hybrid_search`,
`prefetch`) route to replicas; writes always hit the primary. This
lands in v3.2.0 PR-6 of the BlockStore series (roadmap task #15).

---

## 7. Troubleshooting

### "schema_conflicts" in the dry-run

Pre-existing tables in the `mind_mem.*` schema will abort the
migration. Drop them or migrate to a different schema with
`--schema mind_mem_v3`.

### Migration gets stuck halfway

The migration is transactional on the Postgres side — a crash
mid-migration leaves the target in the pre-migration state (nothing
committed). Re-run. The source Markdown corpus is never modified.

### Block count mismatch after migration

Check the migration receipt (`memory/migrations/<ts>-*.json`) for
the rejected blocks list. The most common cause is a block with a
malformed ID in the source; those fail the `_BLOCK_ID_RE` check and
are skipped. Fix them in the Markdown corpus and re-run.

### "permission denied" creating extension

`CREATE EXTENSION vector` requires superuser or a pre-granted role.
Either grant `USAGE` on `pg_extension` to the `mindmem` role or run
the extension create as superuser before the migration.

### Ollama / embeddings not working post-migration

Embeddings are stored in the `metadata.embedding` JSONB field on
each block. If the migration dry-run doesn't show them, you need to
rebuild the vector index from the corpus after flipping the
backend:

```bash
mm reindex --include-vectors
```

---

## 8. Related docs

- `docs/storage-backends.md` — capability matrix per backend.
- `docs/v3.2.0-blockstore-routing-plan.md` — original design.
- `SPEC.md` §"Atomicity Rules" — cross-backend invariants.
- `docs/configuration.md` — full `block_store.*` config reference.
- `SECURITY.md` — credential handling for Postgres DSN (never check
  the password into version control; use env-var substitution).

# Architecture Review — mind-mem v3.2.0 (Release Candidate)

Reviewer: architect-agent sweep, 2026-04-20
Scope: ~30 commits since commit 7fa80fd (the v3.2.0 production-deployment drop).

## Executive posture

**Load-bearing-but-leaky.** v3.2.0 successfully turns mind-mem into a
multi-backend, multi-protocol memory system by introducing a clean
four-method extension to `BlockStore` (write/snapshot/diff/lock) and
a transport-agnostic tool layer, **but** the *apply engine* and
*REST layer* both pierce those abstractions in ways that will block
the Postgres deployment path at runtime. The `BlockStore` protocol is
complete and all four implementations conform to the shape — yet
~30% of apply-engine logic still speaks raw `os.path` / `open()` /
`shutil`, and the REST layer reaches into MCP internals via
`_recall_impl` plus env-var workspace smuggling. Postgres is usable
for reads and snapshots; it is **not** usable end-to-end for
`apply_proposal` until the file-ops in `apply_engine` are re-plumbed
through the store.

## Per-focus-area findings

### 1. BlockStore abstraction integrity

**Solid:**
* All four backends implement the 9-method shape uniformly.
* `list_files` deprecation shims are complete on every backend.
* No caller in `src/` still uses `list_files()`.

**At risk:**
* **Abstraction leak 1** — `workspace=` on `PostgresBlockStore`
  (`block_store_postgres.py:180-189`): the "database" backend takes
  a filesystem path only so `snapshot()` can write `MANIFEST.json`
  to disk (`block_store_postgres.py:454-457`). Cross-host Postgres
  deployments will restore incorrectly because the on-disk manifest
  diverges from the DB snapshot.
* **Abstraction leak 2** — `cast(BlockStore, ...)` in factory
  (`storage/__init__.py:85,100`): Encrypted and Postgres stores are
  not declared as `@runtime_checkable` Protocol members.
  `isinstance(store, BlockStore)` returns False for non-Markdown
  backends.
* **Abstraction leak 3** — Parallel factory in
  `block_store_encrypted.py:176` shadows `storage.get_block_store`.
* **Snapshot surface** takes `snap_dir` (filesystem concept) even
  for Postgres. A proper abstraction would take `snap_id: str`.

### 2. Apply engine coupling (CRITICAL)

`apply_engine.py` has **29 direct filesystem calls** that bypass
the store:

* `_list_workspace_files` (L70-97) — `os.walk` to build orphan set.
* `_cleanup_orphan_files` (L100-125) — direct `os.remove`.
* `execute_op` and 7 `_op_*` handlers (L437-770) —
  `open(filepath, "r")` / `open(filepath, "w")` on corpus markdown.
  **Cannot work with Postgres.**
* `_mark_proposal_status`, `write_receipt`, `_save_intel_state` —
  direct I/O.

`_store_for(ws)` (L316-331) is the routing point but is only called
by `create_snapshot` / `restore_snapshot` / `snapshot_diff`. The op
executors never see the store. Running `apply_proposal` against a
Postgres backend will write markdown to the local FS that Postgres
never sees, then "succeed" while DB state diverges.

### 3. REST ↔ MCP duplication (CRITICAL)

REST handlers import *private* MCP internals — `_recall_impl`
(`rest.py:373`), `__wrapped__` chains through `public.py`, and use
`os.environ["MIND_MEM_WORKSPACE"] = workspace` (L278) as the
request-scoped workspace carrier. **Request-thread-unsafe** the
moment uvicorn runs >1 worker. Recommend extracting a
`mind_mem.core.services` module that both REST and MCP call with an
explicit `workspace` argument.

### 4. MCP consolidated dispatcher shadowing

Intentional shadowing of legacy `recall` name (documented). Subtle
semantic drift: `backend=` alias maps to `mode=` only when
`mode == "auto"`. Telemetry dashboards grouping by `tool_name=recall`
will confuse public-vs-legacy variants.

### 5. Auth model composability (HIGH)

Three modes coexist with brittle precedence:

* OIDC runs only on `/v1/auth/oidc/callback` — it does not persist
  session state, so other `/v1/*` endpoints **cannot** accept OIDC
  JWTs. Effectively a one-shot identity probe, not an auth mode.
* OIDC scopes are extracted (`auth.py:129-146`) but not wired into
  `_require_admin`. OIDC users cannot reach admin endpoints
  regardless of JWT claims.
* `MIND_MEM_ADMIN_TOKEN` remains the only bearer admin path.
  `mmk_*` API keys with `["admin"]` scope are a second path.

### 6. Observability layer placement

* `@traced` wraps only 3 tools (`propose_update`, `scan`,
  `_recall_impl`). Not every tool.
* `@mcp_tool_observe` emits structured logs + counters but no OTel
  spans.
* REST handlers have no tracing at all.
* **Cardinality is safe** — no user-supplied labels.
* Recall histogram caps at 2.5s → understates P99 for large
  corpora.

### 7. Scale ceiling (best guesses)

Assumptions: single-host MarkdownBlockStore default; Postgres pool
(min=1, max=10).

| Tier | Blocks | Corpus MB | RPS | Notes |
|---|---|---|---|---|
| Single host (Markdown) | ~100k | ~500MB | 50-80 | Bound by `parse_file` scanning |
| Single host (Postgres local) | ~1M | N/A | 200-400 | ILIKE fallback O(N) collapses above 100k |
| Multi-host Postgres + 2 replicas | ~10M | N/A | 600-1200 | Replica-lag stale reads during failover |

**Hard ceiling**: ~10M blocks / 1200 RPS before Postgres single-writer
throughput bottlenecks. Sharding would require rewriting
`_resolve_block_file` and the block-id prefix map.

## Concrete refactor recommendations (ranked by ROI)

1. **[CRITICAL, ~2 days]** Route `apply_engine` op executors through
   `BlockStore.write_block` / `delete_block`. The seven `_op_*`
   functions should compose parsed blocks and hand them to the
   store. Without this, Postgres-backed apply is broken.
2. **[CRITICAL, ~1 day]** Extract REST request-scoping. Replace the
   `os.environ` mutation with a `workspace` parameter threaded
   through a `Services` facade. Required before REST runs with
   multiple uvicorn workers.
3. **[HIGH, ~0.5 day]** Wire OIDC into `_require_auth` /
   `_require_admin`. Map JWT `scopes` claim to the internal scope
   grammar.
4. **[HIGH, ~0.5 day]** Make `PostgresBlockStore.snapshot` accept
   `snap_id: str` and make the on-disk manifest write optional.
5. **[MEDIUM, ~0.5 day]** Declare all four stores as
   `@runtime_checkable` Protocol members; delete the shadow factory
   in `block_store_encrypted.py`.
6. **[MEDIUM, ~0.5 day]** Normalise the consolidated `recall`
   dispatcher telemetry — emit `tool_name="recall.public"` vs
   `"recall.legacy"` to avoid metric collision.
7. **[LOW, ~0.5 day]** Expand histogram buckets to 10s. Wrap all
   MCP tools in `@traced` for span coverage parity with Prometheus.

## Gate implications for v3.2.0 tag

* **Critical items 1 + 2 are blocking for a "production" label on
  the Postgres + REST surfaces.** Tagging v3.2.0 without them is
  valid for the Markdown + stdio-MCP single-host deployment (which
  is unchanged and solid), but the production-deployment narrative
  in the release notes overstates the Postgres path's readiness.
* **Recommended gating:** tag v3.2.0 as currently scoped but label
  Postgres + REST as "beta" in the docs. v3.2.1 closes refactors 1
  and 2 and promotes both to GA.

## Files reviewed

* `src/mind_mem/block_store.py`
* `src/mind_mem/block_store_postgres.py`
* `src/mind_mem/block_store_postgres_replica.py`
* `src/mind_mem/block_store_encrypted.py`
* `src/mind_mem/storage/__init__.py`
* `src/mind_mem/apply_engine.py`
* `src/mind_mem/api/rest.py`
* `src/mind_mem/api/auth.py`
* `src/mind_mem/api/api_keys.py`
* `src/mind_mem/mcp/server.py`
* `src/mind_mem/mcp/tools/public.py`
* `src/mind_mem/mcp/infra/http_auth.py`
* `src/mind_mem/mcp/infra/observability.py`
* `src/mind_mem/mcp/infra/acl.py`
* `src/mind_mem/telemetry.py`

# mind-mem federation & multi-machine setup

> Field notes from standing up a multi-machine shared-memory fleet (2026-06). These are the
> things that cost time the first time — do them up front and federation is quick.

## TL;DR — how to share ONE memory across many machines

**Federate via the Postgres backend directly. Do NOT rely on `mm http-serve` for a
Postgres-backed corpus.**

1. On the **hub** (the box that owns the corpus), put the corpus in Postgres:
   ```jsonc
   // mind-mem.json
   "block_store": { "backend": "postgres",
     "dsn": "postgresql://mindmem:<pw>@127.0.0.1:5432/mindmem", "schema": "mind_mem" }
   ```
2. Open Postgres to the LAN (hub):
   - `postgresql.conf`: `listen_addresses = '*'`
   - `pg_hba.conf`: `host  mindmem  mindmem  <YOUR_LAN_CIDR>  scram-sha-256`
   - `systemctl restart postgresql@<ver>-main` → it now binds `0.0.0.0:5432`
3. On every **node**, point mind-mem at the hub DSN (same config, host = the hub IP):
   ```
   postgresql://mindmem:<pw>@<HUB_IP>:5432/mindmem
   ```
   Set `MIND_MEM_CONFIG=<path to that mind-mem.json>` for the `mm` CLI and the MCP server.
4. Verify from a node: `python -c "import psycopg; print(psycopg.connect(host='<HUB_IP>',port=5432,dbname='mindmem',user='mindmem',password='<pw>').cursor().execute('select count(*) from mind_mem.blocks') or 'ok'")`
   — you should see the hub's block count. That's shared read+write memory.

## Gotchas we hit (each one is a fix candidate)

1. **`mm http-serve` does not serve the Postgres corpus.** Even with `MIND_MEM_WORKSPACE`
   and `MIND_MEM_CONFIG` set to a Postgres-backed config, `http-serve` reports
   `memory_count: 1` / `workspace: <name>` — it uses a **workspace file store**, not the
   configured `block_store` backend. → **Fix candidate:** `serve_http` should build the
   store from the same config path the CLI uses (`block_store.backend`), or the docs must
   say plainly "HTTP transport = file store; use direct Postgres for a DB-backed corpus."
2. **Token env var is inconsistent.** `mm token rotate` emits `export MIND_MEM_TOKENS=…`
   (plural), but `serve_http`'s startup guard checks **`MIND_MEM_TOKEN`** (singular) and
   refuses to bind a non-loopback host without it. Set **both** until unified.
   → **Fix candidate:** accept either; document the one canonical name.
3. **Auth header is non-obvious:** the HTTP transport expects **`X-MindMem-Token`**
   (no hyphens between Mind/Mem), not `Authorization: Bearer` or `X-Mind-Mem-Token`.
   Worth a one-line note in the serve help text.
4. **Fleet version drift → `unknown_recall_config_keys` warnings.** A node on an older
   pip (`mind-mem==4.0.17`) reading a config written by a newer build warns on keys it
   doesn't know (`bm25_weight`, `model`, `ollama_embed_model`, `onnx_backend`, `provider`,
   `rrf_k`, `vector_enabled`, `postgres`). → **Keep mind-mem versions aligned across the
   fleet**, and the config loader should ignore-with-info unknown keys (it does) but the
   docs should list the version→config-schema mapping.
5. **No-GPU recall on shared nodes:** to guarantee the embedding model never loads on a
   node's GPU, run the MCP with `CUDA_VISIBLE_DEVICES=-1` (use `-1`, not empty string —
   some launchers reject an empty value). BM25 recall still works.

## Why direct-Postgres beats the HTTP transport here
- One source of truth, real read+write from every node, no transport/store mismatch.
- Postgres already does auth (scram), concurrency, and durability.
- The HTTP transport is fine for a single-workspace file store or a read cache; it is not
  (today) a Postgres federation gateway.

_See also: `docs/docker-deployment.md`, RFC 0009 (federation-first package layer)._

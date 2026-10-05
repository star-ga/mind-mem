# Troubleshooting

Start every diagnosis with these three read-only commands:

```bash
mm status                                  # is the workspace where you think it is?
mm doctor                                  # store, index and recall-log health (JSON)
mind-mem-verify "$MIND_MEM_WORKSPACE"      # ledgers, chains and spec binding
```

`mind-mem-verify` prints one `[ok]` / failure line per check and ends with
`OK` or a failure summary. Add `--strict` in CI so a missing ledger fails
instead of being skipped, and `--json` for a machine-readable report.

`mm doctor` only reads unless you pass a repair flag:
`--rebuild-cache` copies Postgres-only blocks into the SQLite recall cache;
`--migrate-recall-log` adds the newer `retrieval_log` columns to an old
SQLite index. On a plain Markdown workspace, `"parity": "not applicable"`
is the normal answer.

## Symptoms and fixes

### `mm: command not found`

The package is not installed, or its script directory is not on `PATH`
(`pipx ensurepath` fixes the pipx case). The module form works with the same
arguments: `python3 -m mind_mem.mm_cli status`.

### `mm status` exits 1, or shows `"config_exists": false`

`mm` is looking at the wrong directory. It uses `$MIND_MEM_WORKSPACE`, or
the current directory when that is unset. Export the right path, or create a
workspace with `mind-mem-init <dir>`.

### `mm recall` returns `[]`

1. Confirm the workspace (`mm status`).
2. Try broader keywords, and drop `--active-only`, `--since` and `--until`.
3. Imported blocks (`mm import`) and agent messages (`mm send`) are
   quarantined and are not returned by recall until released through review.
4. `mm explain "<query>"` shows what each retrieval stage scored.
5. With `recall.backend` set to `sqlite`, the index must exist. Build or
   refresh it with
   `python3 -m mind_mem.sqlite_index build --workspace "$MIND_MEM_WORKSPACE"`.

### stdout is mixed with JSON log lines

Structured logs are written to stderr. Redirect them:
`mm recall "query" 2>/dev/null | jq .`

### Warning `unknown_recall_backend` with `"backend": "bm25"`

The workspace config names a backend the recall engine does not list, so it
falls back to the built-in BM25 scan. Results are unaffected. Silence it with
`mm config set recall.backend scan`.

### `v4 surface '<name>' is disabled`

That feature is opt-in. The error names the switch, for example
`mm config set v4.compliance_export.enabled true`. Enable it only if the
operator wants the feature.

### `refusing to set '<key>': the config has already drifted from its binding`

Someone edited `mind-mem.json` by hand. Review the edit, then attest it with
`mm bind --rebind`, then retry `mm config set`.

### `mm review` lists a blocker about scope `'user'`

Approving and rejecting proposals is an admin capability. The operator grants
it per command with `MIND_MEM_SCOPE=admin mm review ...`. Do not elevate on
your own.

### `Cannot apply proposals in detect_only mode`

The workspace is in the default `detect_only` governance mode, which refuses
every apply. Changing it is an operator decision:
`mm config set governance_mode propose`.

### A scan keeps reporting `Mode: detect_only` after switching modes

`mind-mem-scan` reads its mode from `memory/intel-state.json`, while
`mm config set` updates `mind-mem.json`. See
[configuration.md](configuration.md#governance-mode).

### `no evidence store at <dir>/memory/evidence_chain.jsonl`

`mm chain ...` and `mm anchor` take the workspace as a positional argument and
default to the current directory, not `$MIND_MEM_WORKSPACE`. Pass it:
`mm chain survey "$MIND_MEM_WORKSPACE"`.

### `no stored output for handle '...'`

Use the full `to-...` handle from the last line of `mm tool-run` output
(`recall: mm tool-recall to-...`), not the hash in the summary header.

### `no knowledge graph at .../memory/knowledge_graph.db`

`mm graph-answer` needs a graph. `mm graph-backfill --write` stages extracted
edges for review; `mm graph-backfill --list-pending` and
`mm graph-backfill --approve <SIG-ID>` apply them.

### Postgres backend configured but `mm doctor` shows `block_store_error`

If the error mentions `psycopg`, install the extra:
`pip install "mind-mem[postgres]"`. Otherwise check the DSN in
`block_store.dsn` (or `MIND_MEM_DSN`) and that the server is reachable;
`mm doctor` reports `block_store_health` when it can connect.

### MCP tools do not appear after `mm install-all`

Restart the client; most read MCP configuration only at startup. Then check
the client's config file for a `mind-mem` entry and run `mind-mem-mcp --help`
to confirm the server starts (needs the `[mcp]` extra).

## Scripts that take only a workspace path

`mind-mem-capture`, `mind-mem-migrate` and `mind-mem-validate` accept a single
workspace path and have no `--help`. Anything you pass is treated as that
path, so `mind-mem-migrate --help` would try to migrate a directory named
`--help`. Call them as `mind-mem-capture "$MIND_MEM_WORKSPACE"`.
`mind-mem-capture "$MIND_MEM_WORKSPACE" --scan-all` scans the last seven
daily logs instead of today's (the workspace must come first).

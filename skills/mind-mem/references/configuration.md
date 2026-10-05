# Configuration

## Where settings live

* `mind-mem.json` at the workspace root is the configuration file.
  `mind-mem-init` writes it with defaults.
* `.spec_binding.json` next to it is an attestation of that file.
  `mind-mem-init` creates it. The governance gate compares the two, so a
  hand edit to a bound `mind-mem.json` is read as **config drift**, not as a
  setting change.
* `mm` commands that read workspace state use
  `$MIND_MEM_WORKSPACE/mind-mem.json` (or `./mind-mem.json` when the variable
  is unset). The `v4` feature switches are resolved in this order:
  `$MIND_MEM_CONFIG`, `$MIND_MEM_WORKSPACE/mind-mem.json`, `./mind-mem.json`,
  `~/.mind-mem/mind-mem.json`.

## Changing a setting

```bash
mm config set governance_mode propose
mm config set limits.rate_limit_calls_per_minute 240
mm config set v4.compliance_export.enabled true
mm config set recall.backend sqlite
```

* `KEY` is dotted; missing intermediate objects are created.
* `VALUE` is parsed as JSON when it is valid JSON (`true`, `500`, `{...}`) and
  stored as a string otherwise. Add `--raw-string` to keep a value such as
  `true` or `5` as text.
* The command rewrites `mind-mem.json` and re-attests `.spec_binding.json` in
  one step. `--workspace` and `--config` point it at another file; `--json`
  prints a machine-readable result.
* `mm config set` refuses to write over a config that has already drifted
  from its binding. If a config was edited by hand, review the change and
  re-attest it deliberately: `mm bind --rebind`. Plain `mm bind` exits 3 when
  the config has drifted.

## Frequently used keys

| Key | Values | Effect |
| --- | --- | --- |
| `governance_mode` | `detect_only` (default), `propose`, `enforce` | How findings turn into changes. See below. |
| `recall.backend` | `scan` / `tfidf` (in-memory BM25, default), `sqlite` (FTS index), `vector`, `hybrid` (Postgres block store only) | Which retrieval engine `mm recall` uses. |
| `recall.vector_enabled` | `true` / `false` | Adds the vector leg when embedding dependencies are installed (`mind-mem[embeddings]`). |
| `recall.rrf_k`, `recall.bm25_weight`, `recall.vector_weight` | numbers | Fusion tuning for hybrid retrieval. |
| `block_store.backend` | `markdown` (default), `encrypted`, `postgres` | Where blocks are stored. `postgres` also uses `block_store.dsn` and `block_store.schema`. |
| `proposal_budget.per_run`, `proposal_budget.per_day`, `proposal_budget.backlog_limit` | integers | Caps on proposal generation and the pending backlog. |
| `limits.max_recall_results`, `limits.query_timeout_seconds`, `limits.rate_limit_calls_per_minute` | integers | Server-side limits. |
| `auto_capture`, `auto_recall` | `true` / `false` | Session-end capture and session-start recall in the installed client hooks. |
| `auto_update.enabled`, `auto_update.mode`, `auto_update.channel`, `auto_update.interval_hours` | `true`/`false`; `notify`/`auto`; `stable`/`pre`; hours | Background update check on `mm` runs. Off unless enabled. |
| `compaction.archive_days`, `compaction.snapshot_days`, `compaction.log_days`, `compaction.signal_days` | integers (days) | Retention for `mind-mem-compact`. |
| `v4.<surface>.enabled` | `true` / `false` | Opt-in surfaces such as `lint`, `block_kinds`, `compliance_export`, `redaction`, `ingest_serve`. A disabled surface prints the exact key to set. |

The complete, maintained reference is `docs/configuration.md` in the source
repository.

## Governance mode

| Mode | Behaviour |
| --- | --- |
| `detect_only` (default) | Scans report contradictions and drift but write no proposals. Every apply is refused. Config drift is recorded and warned about. |
| `propose` | Scans also write fix proposals to `intelligence/proposed/`; approved proposals can be applied. Config drift refuses governed writes. |
| `enforce` | Same proposal and apply behaviour as `propose`. Config drift refuses governed writes. |

`mm review` and the apply engine read `governance_mode` from `mind-mem.json`.
The scanner (`mind-mem-scan`) decides whether to write proposals from the
`governance_mode` field in `memory/intel-state.json`; check that file too when
a scan in `propose` mode reports `Mode: detect_only`.

## Environment variables

| Variable | Used for |
| --- | --- |
| `MIND_MEM_WORKSPACE` | Workspace directory for `mm` and the MCP server. Defaults to the current directory. |
| `MIND_MEM_CONFIG` | Explicit path to the `mind-mem.json` used for `v4` feature switches. |
| `MIND_MEM_SCOPE` | `admin` grants approval rights to `mm review` and admin MCP tools. Defaults to `user`. Set it only when the operator intends it. |
| `MIND_MEM_BACKEND`, `MIND_MEM_DSN` | Default backend and Postgres DSN for `mind-mem-init`. |
| `MIND_MEM_ENCRYPTION_PASSPHRASE` | Required by the `encrypted` block-store backend. |
| `MIND_MEM_TOKEN`, `MIND_MEM_TOKENS`, `MIND_MEM_ADMIN_TOKEN` | Bearer tokens for `mm http-serve` and the REST API (`MIND_MEM_TOKENS` is comma-separated; `mm token rotate` mints new ones). |
| `MIND_MEM_LOG_LEVEL` | Structured log level on stderr (default `INFO`). |
| `MIND_MEM_LOG_FILE` | Structured log file that `mm trace` reads (stdin is used when unset). |
| `MIND_MEM_NO_AUTO_UPDATE` | `1` disables the automatic update check. |
| `MIND_MEM_STATE_DIR` | Where update-check state is kept (default `~/.mind-mem`). |
| `MIND_MEM_DISABLE_TELEMETRY` | Any non-empty value turns off tracing. |
| `MIND_MEM_MCP_SERVER` | Path to the MCP server script that `mm install` writes into client configs. |

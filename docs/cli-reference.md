# CLI Reference

The `mm` command is the unified MIND-Mem CLI for non-MCP agents.

## Command index

Every verb `mm` accepts, with the parser's own help text. **Generated** from
`mind_mem.mm_cli.build_parser()` — `tests/test_docs_alignment.py::TestGeneratedCliVerbIndex`
fails the build if this table and the parser disagree, so a new verb documents
itself or does not land. Hand-maintenance is what let 32 of 51 verbs go
unlisted, `mm bind` among them — the command the governance gate names in
every drift refusal.

The sections below this index cover the frequently-used verbs in depth.

<!-- BEGIN GENERATED: cli-verb-index — regenerate with tests/test_docs_alignment.py -->
| Command | What it does |
| --- | --- |
| `mm recall` | Search memory and print JSON results. |
| `mm context` | Recall + token-budget pack into a context snippet (JSON). |
| `mm inject` | Render a context snippet in the format a specific agent expects. |
| `mm resume` | Print the resume brief for the active task frame, with dead-end warnings. |
| `mm dead-ends` | List recorded dead ends (JSON), optionally filtered by an about-to-happen action. |
| `mm review` | Batch-review the HITL proposal queue: diff, evidence, approve/reject. |
| `mm status` | Print workspace status as JSON. |
| `mm index` | Regenerate index.md (hierarchical, by category/kind) + log.md (chronological) from the block corpus. |
| `mm usage` | Model-call token counts per day (local ledger; no egress). |
| `mm tool-run` | Run a command; keep its large output out of context, print a summary+handle. |
| `mm tool-recall` | Print the full stored output for a tool-run handle. |
| `mm migrate-store` | Migrate the workspace block corpus between storage backends. |
| `mm migrate` | Run a workspace layout migration. |
| `mm lint` | Report deterministic corpus defects; --fix stages one repair proposal. |
| `mm detect` | Auto-detect installed AI coding clients for this workspace. |
| `mm install` | Configure mind-mem for one named AI coding client. |
| `mm install-all` | Auto-detect + configure every installed AI coding client. |
| `mm install-model` | Download `mind-mem-4b` GGUF (~2.5GB) from HuggingFace and import into Ollama as `mind-mem:4b`. Idempotent. |
| `mm doctor` | Diagnose workspace state (block-store parity, recall-log schema drift). Add --rebuild-cache or --migrate-recall-log to actually repair. |
| `mm token` | Federation/HTTP transport token management (rotation primitive). |
| `mm import` | Import memory from another system into the corpus (local dumps and note directories). |
| `mm kinds` | v4 block-kind taxonomy (requires v4.block_kinds in mind-mem.json). |
| `mm vault` | Vault sync subcommands. |
| `mm lineage` | Block-lineage subcommands (v3.11.0 typed edges + v3.12 staleness). |
| `mm skill` | Self-improving skill optimization subcommands. |
| `mm serve` | Launch the mind-mem REST API server (requires mind-mem[api]). |
| `mm http-serve` | Launch the v3.9 stdlib HTTP transport (zero dependencies; minimal endpoint surface). |
| `mm daemon` | Launch the v3.9 background daemon — runs configured jobs on internal intervals. |
| `mm inbox-watch` | Watch an inbox directory; route files by extension into the workspace. |
| `mm ingest-serve` | Serve POST /ingest and drain accepted events into QUARANTINED blocks. Requires v4.ingest_serve in mind-mem.json; off by default. |
| `mm send` | Send a message to another agent (writes an MSG- block to shared memory). |
| `mm inbox` | Read agent messages addressed to you (recall over MSG- blocks). |
| `mm graph-backfill` | Extract typed relations from a corpus slice into staged SIGNALS.md proposals. Dry-run by default: prints edges-per-block + predicate histogram (the yield measurement). |
| `mm graph-answer` | Answer about an entity using ONLY the k-hop subgraph of governed edges: every claim cites its edge and provenance block, and what the graph does not contain is listed explicitly. |
| `mm pipeline-status` | Show current extractor pipeline hash + count of dirty (re-extract) blocks. |
| `mm accountability` | RA.2 retrieval accountability: precision (credited/served per intent) and waste (admitted blocks with no serve evidence). Derived views, recomputed on every call and never stored. |
| `mm dashboard` | RA.5 lifecycle-tier dashboard: the one MemoryTier ladder crossed with retention class, the served-set ledger's chain verdict, and RA.2's precision / waste / serve counts. Read-only; exits 1 on a broken chain. |
| `mm replay-check` | Check a recall attestation against the served-set ledger: same run id, and the recorded id count agrees with the attested result_count. Exits 1 unless the ledger corroborates it. |
| `mm audit-model` | Static security scan of a local model checkpoint. Flags remote-code hooks, unsafe pickle, tokenizer injection. Outputs SHA-256 manifest. |
| `mm sign-model` | Sign every file in a local model checkpoint with Ed25519. Writes MODEL_MANIFEST.txt + .sig + MODEL_PUBKEY.pub sidecars. |
| `mm verify-model` | Verify a previously-signed checkpoint. Returns nonzero if the manifest, signature, or public key fail. |
| `mm gate` | Load-gate registry that tracks which local checkpoints have been audited. Three sub-commands: check, list, remove. |
| `mm bind` | Attest the current mind-mem.json by writing .spec_binding.json. Until this runs, GovernanceGate's spec-hash check is inert and config tampering is not detected. Exits 3 on drift unless --rebind. |
| `mm config` | Read/write mind-mem.json. `config set` writes the key and re-attests .spec_binding.json in one step, so a setting change is not read back by GovernanceGate as config tampering. |
| `mm audit-pinned` | Run the seven-check audit (and optional Ed25519 verify) on every entry in audit_pinned_models of mind-mem.json. Designed for release CI — non-zero exit on any HIGH finding or verify failure. |
| `mm mic` | MIND IR graph serialization (mic@2 text + mic-b binary). Subcommands: convert, inspect. |
| `mm inspect` | Print full block fields and provenance tree for a block ID. |
| `mm explain` | Show per-stage retrieval scores (BM25 → vector → RRF → rerank) for a query. |
| `mm trace` | Display recent MCP tool calls parsed from structured JSON logs. |
| `mm export` | Deterministic compliance bundle over the admitted corpus. Two runs over an unchanged corpus produce byte-identical output. |
| `mm compliance` | Redaction detectors, the pre-write screening door, and the provenance policy. |
| `mm self-update` | Check PyPI for a newer mind-mem and upgrade this install. |
<!-- END GENERATED: cli-verb-index -->

## Global options

```
mm --help
```

All commands read the workspace from the `MIND_MEM_WORKSPACE` environment variable
(falls back to the current working directory).

---

## Core commands

### `mm recall <query>`

Search memory using BM25 and return ranked JSON results.

```
mm recall "authentication strategy" --limit 5 --active-only
```

| Flag | Default | Description |
|------|---------|-------------|
| `--limit N` | 10 | Maximum results |
| `--active-only` | off | Restrict to active blocks |
| `--kernel NAME` | off | Route through a v4 cognitive-kernel strategy instead of plain recall (5.0.1) |

`--kernel` takes one of `default`, `surprise_weighted`, `lineage_first`,
`contradicts_first`, `graph_walk` and requires
`"v4": {"cognitive_kernel": {"enabled": true}}` in `mind-mem.json`. Omit it and
recall behaves exactly as before — no feature flag is even read. Kernel hits
pass the same admission gate as recall, so a quarantined block reached through
the lineage graph is withheld and counted under `withheld`.

### `mm context <query>`

Recall + token-budget-pack results into a context snippet (JSON).

```
mm context "deadline" --max-tokens 2000
```

### `mm inject <query>`

Render a context snippet in the format expected by a specific agent.

```
mm inject "auth decisions" --agent claude-code
```

### `mm resume`

Print the resume brief for the active task frame: the goal, the plan steps, what
was tried, what is believed, what remains, and the recorded dead ends that
overlap this task.

```
mm resume                          # the active frame (highest live TF- id)
mm resume --frame TF-20260829-001  # a specific frame
mm resume --json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--frame ID` | active | A specific `TF-...` frame id |
| `--json` | off | Emit the brief as JSON |

Exit code is `0` even when dead ends fire — a dead end is evidence, never a
prohibition, so it must not read as a failed command. A frame the parser refused
is **named**, not hidden behind `No active task frame.` Full guide:
[task-frames.md](./task-frames.md).

### `mm dead-ends`

List the dead-end registry as JSON. With no filter this is the whole registry in
block-id order; with any filter it is the deterministic overlap against that one
about-to-happen action, most conclusive first.

```
mm dead-ends
mm dead-ends --tool Bash --intent prove_floor
mm dead-ends --path "tools/**/*.py" --path "docs/floors.json"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tool T` | — | Tool the action would invoke |
| `--command C` | — | Command line the tool would run |
| `--intent I` | — | Intent class for the action |
| `--path P` | — | File the action would touch (repeatable) |

The payload always carries `total_matched` / `elided` and `rejected`, so a list
trimmed by `max_warnings` or shortened by a refused block says so.

### `mm review`

Batch-review the HITL proposal queue: pending proposals with their pre-apply
diff, provenance, governance-chain status and staleness flags inline, then
approve or reject many at once.

```
mm review                        # list the queue with health and blockers
mm review --show P-20260829-001  # full detail for one proposal
mm review -i                     # keyboard session: one keystroke per proposal
mm review --approve P-20260829-001,P-20260829-002
mm review --reject P-20260829-003 --reason "superseded upstream"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Machine-readable output |
| `--limit N` | all | Show at most N proposals |
| `--show ID` | — | Full detail for one proposal |
| `-i`, `--interactive` | off | Keyboard session, then commit |
| `--approve IDS` | — | Comma-separated ids to approve |
| `--reject IDS` | — | Comma-separated ids to reject |
| `--reason TEXT` | — | Rationale for `--reject` (required, ≥ 8 chars) |

Exit codes: `0` all decisions succeeded, `1` at least one proposal failed,
`2` usage error.

Every approval routes through the governed `approve_apply` path and every
rejection through `reject_proposal`. Approving is admin-scope — export
`MIND_MEM_SCOPE=admin`; `mm review` reports the scope but never sets it.
Atomicity is per proposal: if one of thirty fails, the rest still run and the
failure is reported. **There is no auto-approve path at any risk level.**

Governance blockers (scope, `governance_mode`, backlog limit, apply rate limit)
are printed **before** the operator spends any decisions — ahead of the keyboard
session and ahead of a `--approve` batch, not only in the listing. The published
`proposals/minute` covers the whole invocation, deciding included, and prints
both spans (`over Xs of operator session, Ys applying`).

Full guide: [review.md](./review.md).

### `mm status`

Print workspace status JSON (directory existence, config file, subdirectory checks).

### `mm usage`

Per-day token counts for the only mind-mem work that costs money: a call out
to a model (today the injected compressor behind recompaction, and the
optional extraction backend). Retrieval, indexing and the governance gate run
on your own machine and are not metered — there is no currency, no rate card
and no spending alerts, by design.

```
mm usage
mm usage --json
mm usage --daily-cap 200000   # exit 3 + a DAILY TOKEN CAP line on stderr once reached
mm usage --reset              # clear the workspace token ledger
```

| Flag | Default | Description |
|------|---------|-------------|
| `--daily-cap TOKENS` | config, else off | Report + exit `3` once today's counted tokens reach the ceiling |
| `--json` | off | Emit the report as JSON instead of a table |
| `--reset` | off | Clear the ledger |

**Nothing leaves the host.** The report reads a local JSON ledger
(`<workspace>/.mind-mem-index/usage.json`, pruned to the last 90 days); there
is no exporter, no socket, and no network egress on any path.

Counting is **opt-in by construction**: nothing meters itself. Wrap the
model callable you inject and the tokens land in the ledger — an unwrapped
call writes nothing and is byte-for-byte unchanged.

```python
from mind_mem import usage_meter
from mind_mem.recompaction import recompact_cluster

compressor = usage_meter.metered_compressor(my_compressor, workspace)
recompact_cluster(blocks, compressor=compressor)
```

The cap can also live in the workspace config, where the metered call itself
enforces it (raising `DailyTokenCapExceeded` *before* the model is called):

```json
{ "usage": { "daily_token_cap": 200000 } }
```

### `mm detect`

Auto-detect installed AI coding clients and print JSON.

### `mm install <agent>`

Configure MIND-Mem for a single named client.

### `mm install-all`

Auto-detect and configure every installed AI coding client.

### `mm self-update`

Check PyPI for a newer `mind-mem` release and upgrade the current install in
place, using the `pip` that owns the running interpreter. Stdlib-only, zero
coupling to recall/evidence/scoring code. All human-facing output goes to
**stderr**, prefixed `[self-update]`, so it never corrupts JSON/text a normal
`mm` command prints to stdout. Refuses to upgrade over an editable/dev
install (git checkout) — prints the `git pull + pip install -e .` path
instead.

```
mm self-update              # check + prompt before upgrading
mm self-update --yes        # upgrade without prompting
mm self-update --check      # report only, no upgrade; exit 10 if newer available
mm self-update --pre        # include pre-releases when checking/upgrading
```

| Flag | Default | Description |
|------|---------|-------------|
| `--check` | off | Report only; exit code `10` if an update is available, `0` if up to date |
| `--yes`, `-y` | off | Upgrade without prompting |
| `--pre` | off | Include pre-releases |

Disabled by default. Setting `auto_update.enabled: true` in `mind-mem.json`
turns on a lightweight, best-effort, interval-gated check (default 24h) that
runs on other `mm` invocations without blocking them.

---

## Debug visualization commands (v3.2.0)

### `mm inspect <block_id>`

Print the full contents and provenance tree for a single block.

```
mm inspect D-042
mm inspect D-042 --format json
```

**Output (text, default):**

```
Block: D-042
────────────────────────────────────────────────────────────
  _id:    D-042
  Statement: Use BM25 for recall
  Status: active
  Date:   2026-01-15
  Tags:   recall bm25
  Rationale: Best bang-for-buck without external deps

Provenance
────────────────────────────────────────────────────────────
  Direct dependencies:
  → D-001  [depends_on]  weight=1.00
  Causal chains (depth ≤ 3):
    D-042 → D-001
```

**Output (JSON):**

```json
{
  "block": { "_id": "D-042", "Statement": "...", ... },
  "provenance": {
    "block_id": "D-042",
    "dependencies": [ { "source_id": "D-042", "target_id": "D-001", "edge_type": "depends_on", "weight": 1.0, ... } ],
    "causal_chains": [ ["D-042", "D-001"] ],
    "contradictions": []
  }
}
```

| Flag | Default | Description |
|------|---------|-------------|
| `--format text\|json` | `text` | Output format |

Exits with code 1 and a JSON error message to stderr if the block is not found.

---

### `mm explain <query>`

Show per-stage retrieval scores for a query: BM25 → vector → RRF → rerank.

```
mm explain "authentication strategy"
mm explain "auth" --limit 5 --backend hybrid --format json
```

**Output (text, default):**

```
Retrieval trace: 'authentication strategy'
──────────────────────────────────────────────────────────────────────
     #  BLOCK                    BM25     VEC     RRF  RERANK  STAGES
──────────────────────────────────────────────────────────────────────
     1  D-042                  0.8420       -       -       -  [x][ ][ ][ ]
     2  D-010                  0.7310       -       -       -  [x][ ][ ][ ]

Diagnostics summary
  Intent: WHAT (12 recent queries)
  bm25: 0.0% rejected
```

**Output (JSON):**

```json
{
  "query": "authentication strategy",
  "results": [
    {
      "rank": 1,
      "block_id": "D-042",
      "bm25": 0.842,
      "vector": null,
      "rrf": null,
      "rerank": null,
      "stages_hit": [true, false, false, false]
    }
  ],
  "diagnostics": { "intent_distribution": {...}, "stage_rejection_rates": {...} }
}
```

`stages_hit` is a four-element boolean array: `[bm25_stage, vector_stage, rrf_stage, rerank_stage]`.

| Flag | Default | Description |
|------|---------|-------------|
| `--limit N` | 10 | Number of results to trace |
| `--backend auto\|bm25\|hybrid` | `auto` | Retrieval backend |
| `--format text\|json` | `text` | Output format |

---

### `mm trace`

Display recent MCP tool calls parsed from structured JSON logs.

```
mm trace --last 20
mm trace --last 50 --tool recall
mm trace --live
```

**Output:**

```
TIME              TOOL                           DURATION  STATUS  SIZE
──────────────────────────────────────────────────────────────────────
2026-01-15T10:00  recall                          42ms  OK     5
2026-01-15T10:01  propose_update                  88ms  OK     1
2026-01-15T10:02  scan                           150ms  ERROR  -
```

**Live mode** (`--live`): if `MIND_MEM_LOG_FILE` is set to an existing file, tails
that file for new events. Otherwise reads from stdin line by line (useful for
piping log output from the MCP server process).

Log lines must be structured JSON with `event: "mcp_tool_call"` and a `data` object
containing `tool`, `duration_ms`, `success`, and optionally `result_size`.

| Flag | Default | Description |
|------|---------|-------------|
| `--live` | off | Stream new events in real time |
| `--last N` | 20 | Show last N calls (non-live mode) |
| `--tool NAME` | (all) | Filter to a single tool name |

---

## Block-kind subcommands (5.0.1)

The v4 block-kind taxonomy's operator surface. Requires
`"v4": {"block_kinds": {"enabled": true}}` in `mind-mem.json`; every command
exits 64 with an actionable message when the flag is off.

### `mm kinds backfill`

Classify every **admitted** block into the v4 kind index. Writes only the v4
side store `<workspace>/index.db` — no corpus file is touched and nothing is
minted, so this is an index build, not an ingest. Blocks the governance gate
withholds (quarantined, pending) are never indexed.

Three further steps run only if their own flag is on, in this order:

| Flag | Step |
|------|------|
| `v4.kind_summaries` | Rebuild one summary per kind (read back via `category_summary`) |
| `v4.embedding_pipeline` | Derive an embedding per block, using `recall_vector`'s provider chain when available and the stdlib hashed-trigram embedder otherwise |
| `v4.hnsw_kind_index` | Register those embeddings under each block's primary kind, so `find_similar(..., kind=...)` has a partition to scan |

Idempotent and replayable: two runs over an unchanged corpus write identical
rows.

```
mm kinds backfill
```

### `mm kinds list [--kind KIND] [--limit N]`

List the block ids carrying one kind, with each block's full tag set. `--kind`
is one of `entity`, `concept`, `source`, `synthesis`, `image`, `audio`,
`code`, `structured`, `unspecified` (default `entity`). An empty answer on a
workspace that was never backfilled means exactly that — run `mm kinds
backfill` first.

---

## Vault subcommands

### `mm vault scan <vault_root>`

Walk a Markdown vault and print all parsed blocks as JSON.

### `mm vault write <vault_root> <relative_path>`

Write a block to a vault file.

---

## Skill optimization subcommands

### `mm skill list`

List all discovered skills across configured skill sources.

### `mm skill test <skill_id>`

Generate and run synthetic tests for a skill.

### `mm skill analyze <skill_id>`

Run a multi-model critique of a skill.

### `mm skill optimize <skill_id>`

Full optimization loop: test → analyze → mutate → validate → submit to governance.

### `mm skill history <skill_id>`

Show optimization run history.

### `mm skill score <skill_id>`

Show the current consensus score for a skill.

---

## MIC/MAP subcommands (v3.8.11)

MIND-Mem ships the same MIC/MAP serialization formats used by the wider STARGA
stack — `mic@2` (text) and `mic-b` (varint binary). The `mm mic` subcommand
exposes them on the CLI; the corresponding MCP tools are documented in
[`docs/mcp-integration.md`](mcp-integration.md). For the format spec, runnable
example, and Python API see [`docs/mic-map.md`](mic-map.md).

### `mm mic convert <file> --to {mic2|micb} [-o <out>]`

Convert a MIC graph file between text and binary forms. Auto-detects the input
format from the magic bytes / leading header line. When `-o` is omitted, writes
to stdout (text for `mic2`, raw bytes for `micb`).

```
mm mic convert graph.mic2 --to micb -o graph.micb
mm mic convert graph.micb --to mic2          # text to stdout
```

### `mm mic inspect <file> [--json]`

Print a structural summary: format, type/value/node counts, output index, and a
per-value breakdown. `--json` emits a machine-readable envelope.

```
mm mic inspect graph.mic2
mm mic inspect graph.micb --json
```

---

## `mind-mem-bootstrap` — one-shot corpus backfill (5.0.1)

A separate console script rather than an `mm` subcommand, because it is a
one-time post-`mind-mem-init` operation and not part of the daily loop.

```
mind-mem-bootstrap <workspace> [--dry-run] [--max-transcripts N]
```

It mines four sources into the corpus: every JSONL transcript under
`~/.claude/projects/`, every daily log in `<workspace>/memory/`, `~/CLAUDE.md`
and `~/.claude/MEMORY.md`, and the entities extracted from all of that text.
Re-running is safe — signals are deduplicated by content hash.

**It is an ingest door, and it ships OFF.** Enable it deliberately:

```json
{ "v4": { "bootstrap_corpus": { "enabled": true } } }
```

With the flag off the command reads nothing, writes nothing, and exits `2`.

**Nothing it writes is servable.** Both write legs run under a governance
admission with `IngestTier.AUTO_CAPTURE`, which mints `Status: pending` — a
transcript holds whatever an agent was shown, including text an attacker chose,
so none of it is recallable until a human releases it. `recall` withholds those
blocks; `recall(..., include_pending=True)` shows them for review, and
`approve_apply` on a proposal is the only path to `active`. Start with
`--dry-run`, which reports what each phase found and writes nothing.

Exit codes: `0` ran, `2` the flag is off.

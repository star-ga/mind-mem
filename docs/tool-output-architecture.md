# Tool-output offload — architecture

> `mind_mem.tool_output` — context-window offload for large command/tool output
> (mind-mem §5). Status: working tree, TDD-gated (12 tests), not yet released.

## 1. Problem & goal

A single `cargo test` / `pytest` / build run emits 10k–50k lines. When an agent
runs it, that whole dump lands in the context window — on the `mind` repo (247 test
binaries) it is the single largest token sink of any operation. The goal is a
first-class capability that:

```
raw output ──▶ store full text OUT of context ──▶ return {handle, summary}
                                                        │
                                          recall(handle) ▶ full text, on demand
```

The agent reads the compact summary; it recalls the handle only when a failure
needs the raw detail. This is the RTK ("return the handle") + Context-Mode idea
made native to the governed store.

## 2. The three load-bearing invariants

Everything below serves these. They are what the tests enforce and what a reviewer
should check first.

| # | Invariant | Why it's load-bearing | Enforced by |
|---|-----------|----------------------|-------------|
| **I1** | **Bounded summary** — the context-facing summary is small *regardless of input shape* (a 10 MB single line, or a log where every line matches `error`, cannot blow it up). | The entire value is "keep bytes out of context". An unbounded summary silently defeats the feature. | per-line cap + failure-display cap + head/tail windows; `test_giant_single_line_*`, `test_all_lines_matching_failure_*` |
| **I2** | **Fail-safe** — nothing is ever *silently* lost. The full text is always stored + recallable, and every display truncation is explicit and counted. | A dropped failure line that looks like a pass is the worst failure mode — worse than no tool. | full text stored before any cap; true `failure_lines` always reported; explicit `…+N more` / `…[+N chars]` / `…[N elided]` markers; `test_all_buried_failures_survive`, `test_store_cap_truncates_with_explicit_marker` |
| **I3** | **Deterministic** — same input + same config version → byte-identical summary. No LLM, no clock, no RNG, no set-iteration order. | The wedge. A reproducible, inspectable summary is auditable; an LLM summary is neither. Also keeps it free + fast. | pure pattern extraction; versioned config stamped into the summary; `test_summary_byte_identical_only_within_a_config` |

## 3. Storage model — why a sibling table, not a block kind

The spec proposed a `tool_output` **block kind**. The build uses a dedicated
**`tool_outputs` sibling table** instead. Rationale (deliberate deviation):

- **Retrieval shape.** Knowledge blocks are retrieved by *ranked semantic recall*
  (BM25 + vector + RRF). A tool log is retrieved by *exact handle*. Putting a
  50k-line log in the `blocks` table would embed it and pollute the vector/BM25
  index — degrading the thing that table exists to do.
- **Lifecycle.** Knowledge blocks are durable, governed, contradiction-checked.
  Tool outputs are ephemeral, retention-bounded, never contradiction-checked.
  Different lifecycle ⇒ different table.
- **Blast radius.** A sibling table is purely additive: it cannot change any
  existing `blocks` read/write path, so it can't regress the shipped product.

The table lives in the **same schema, on the same connection** as the block store
(`_require_psycopg`) — "reuse the existing store, no new DB" is satisfied; it is
just not the `blocks` relation.

```
tool_outputs(handle PK, source, exit_code, ts, full_text, summary,
             line_count, byte_count)
```

## 4. Handle scheme

`to-` + first 16 hex of `sha256(source ‖ NUL ‖ full_text)`.

- **Content-addressed ⇒ idempotent.** Re-running an identical command yields the
  same handle; the store upserts, so no duplicate rows accumulate.
- **Namespaced by source.** The same output under two commands gets two handles
  (the source is part of the digest) — the summary header attributes it correctly.
- **64-bit** (16 hex) — collision-negligible for a bounded, per-workspace table.

## 5. The summarizer (deterministic core)

Single pass classifies every line as *failure* / *tally* / *ordinary*. The summary
keeps: head window ∪ tail window ∪ (first `max_failures_shown` failures) ∪ all
tallies, emitted in file order with explicit elision markers. Then a dedicated
`FAILURES` section lists the failures by line number.

Bounds that make I1 hold no matter the input:
- `max_line_chars` (500) — every emitted line truncated with `…[+N chars]`.
- `max_failures_shown` (200) — display cap; the **true** `failure_lines` count is
  always reported and the cap is explicit (`…+N more failure line(s)`).
- `head` (40) / `tail` (60) — context windows.

Config is a frozen `SummarizerConfig` with a `config_hash()` and a
`SUMMARIZER_VERSION` stamped into the summary header — so a summary is reproducible
given its version, and a byte-comparison is correctly *version-scoped* (I3). Thresholds
ship as versioned config, never as autonomous reweighting.

## 6. Bounded storage & retention

Two more bounds keep the *store* (not just the summary) safe:
- **`max_store_bytes`** (32 MiB) — a runaway GB log is stored truncated with an
  explicit marker. Crucially, the summary is computed on the **full in-memory text
  before** this cap, so a truncated *store* never hides a failure from the
  *summary* (only the recall tail beyond the cap is bounded).
- **`max_rows`** (500) — the table is bounded; eviction keeps the newest rows by
  insertion order (`rowid` on SQLite, `ts` on PG). An agent that stores every run
  can't grow it without limit. A `gc()` method forces retention on demand.

Both are constructor config, not policy the code decides on its own.

## 7. Backend abstraction

- **SQLite** (default) — zero-config, local file under `MIND_MEM_WORKSPACE`; WAL is
  the store's existing default for concurrent readers.
- **Postgres** (`backend="postgres"`) — reuses `block_store_postgres._require_psycopg`
  and the configured DSN; the sibling table is created lazily on first store, so
  importing the module never requires psycopg.

Idempotent upsert makes concurrent identical stores safe; distinct handles are
independent rows.

## 8. Integration surface

- **`bin/mm-run -- <cmd>`** — the agent-facing wrapper. Runs the command, streams a
  live 20-line tail to the user's terminal (stderr), captures the full combined
  output, stores it, and prints **only** the summary + handle to stdout. Preserves
  the command's exit code. Dependency-free (bash + python3 + mind_mem).
- **`mm-run --recall <handle>`** — prints the full stored text.
- **(planned) MCP tools** — `store_and_summarize` / `recall_output` exposed as MCP
  tools so an agent recalls via the tool surface, not only the CLI. Noted, not built.

## 9. Edge cases (tested or handled)

| Input | Behaviour |
|-------|-----------|
| empty / single tiny line | no elision, no failures, safe |
| 10 MB single line (minified JSON) | per-line capped → ~725-byte summary (I1) |
| every line matches `error` (100k) | true count reported, 200 shown, ~11 KB summary (I1+I2) |
| binary / control chars | utf-8 `errors="replace"`; passes through (never crashes) |
| GB-scale log | store truncated w/ marker; summary still from full text (I2) |
| re-run identical command | same handle, upsert, no dup (idempotent) |

## 10. Open architectural questions (logged, not silently dropped)

1. **MCP exposure** — wire `recall_output` as an MCP tool + a `PostToolUse` Bash
   hook that auto-offloads any command over N lines (the fully-automatic path).
2. **Binary detection** — a NUL-heavy blob is currently stored as replaced text; a
   `is_binary` flag + raw-bytes column would be more honest for artifact dumps.
3. **Cross-substrate summary identity** — the summary is deterministic in Python;
   the pure-`.mind` port (once the compiler is ready) must reproduce it byte-for-byte,
   which is why the pattern set + version are already externalized as config.
4. **Streaming summarizer** — currently `splitlines()` holds the whole log in RAM.
   Fine at 32 MiB; a streaming single-pass classifier would remove the ceiling.

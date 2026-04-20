# CLI Reference

The `mm` command is the unified mind-mem CLI for non-MCP agents.

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

### `mm status`

Print workspace status JSON (directory existence, config file, subdirectory checks).

### `mm detect`

Auto-detect installed AI coding clients and print JSON.

### `mm install <agent>`

Configure mind-mem for a single named client.

### `mm install-all`

Auto-detect and configure every installed AI coding client.

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

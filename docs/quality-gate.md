# Quality Gate — Operator Runbook

The quality gate is a deterministic pre-write filter that inspects every
`propose_update` statement before it lands in `SIGNALS.md`.  It runs
zero external dependencies (no network, no LLM) and completes in
microseconds.

## Why it exists

mind-mem has hit three failure modes in production:

- Empty proposals from CLI typos writing whitespace-only blocks.
- Oversize log dumps where agents pasted megabytes into a statement.
- Near-duplicate re-runs that filled `SIGNALS.md` with identical content.

The gate catches these before they reach the governance engine.

## The 7 rules

| Rule             | Condition                                             | Default severity |
|------------------|-------------------------------------------------------|-----------------|
| `empty`          | Statement is whitespace-only after strip              | advisory        |
| `too_short`      | Fewer than 32 non-whitespace characters               | advisory        |
| `oversize`       | UTF-8 byte length exceeds 64 KiB                     | advisory        |
| `malformed_utf8` | Contains lone surrogates or fails UTF-8 encode        | advisory        |
| `stopwords_only` | Every token is a stopword (no semantic content)       | advisory        |
| `near_duplicate` | Levenshtein similarity ≥ 0.97 to a block within 24 h | advisory        |
| `injection_marker` | Matches a known prompt-injection pattern            | advisory        |

All rules are advisory by default.  Strict mode promotes every fired rule
to a hard rejection that returns a structured error and does not write to
`SIGNALS.md`.

## Mode toggle

Set `quality_gate.mode` in `mind-mem.json`:

```json
{
  "quality_gate": {
    "mode": "advisory"
  }
}
```

Valid values:

| Value      | Behaviour                                                       |
|------------|-----------------------------------------------------------------|
| `off`      | Gate skipped entirely; no validation, no metrics                |
| `advisory` | Fired rules logged as warnings; statement is still stored       |
| `strict`   | Fired rules return a structured error; statement is NOT stored  |

The default when the key is absent is `"advisory"`.

## Example mind-mem.json snippet

```json
{
  "quality_gate": {
    "mode": "strict"
  },
  "limits": {
    "max_recall_results": 50
  }
}
```

## Advisory rejection log lines

When a rule fires in advisory mode, mind-mem emits a structured warning
at the `mcp_server` log component:

```json
{
  "ts": "2026-05-08T14:22:01.003Z",
  "level": "warning",
  "component": "mcp_server",
  "event": "quality_gate_advisory",
  "data": {
    "mode": "advisory",
    "advisory": ["too_short: block has 11 non-whitespace chars; min is 32"],
    "block_type": "decision"
  }
}
```

The statement is still written to `SIGNALS.md`.  The `quality_gate_rejections`
metric counter is incremented so dashboards and alerts can track the rate.

## Strict rejection response

In strict mode the `propose_update` tool returns an error envelope:

```json
{
  "error": "quality_gate_rejection",
  "mode": "strict",
  "reasons": ["too_short: block has 11 non-whitespace chars; min is 32"],
  "advisory": [],
  "hint": "Statement did not pass the quality gate. Revise and resubmit, or set quality_gate.mode=\"advisory\" in mind-mem.json to downgrade to advisory-only."
}
```

Nothing is written to `SIGNALS.md`.

## Metrics

| Counter                                    | When incremented                          |
|--------------------------------------------|-------------------------------------------|
| `quality_gate_rejections`                  | Any rule fires (advisory or strict)       |
| `quality_gate_rejections_<rule>`           | That specific rule fired                  |

Example suffixed counters: `quality_gate_rejections_too_short`,
`quality_gate_rejections_injection_marker`.

## Common false-positive scenarios and remediation

**Short but valid statements (e.g. acronym expansions)**

The `too_short` rule fires on blocks shorter than 32 non-whitespace
characters.  Expand the statement to include context, or set
`quality_gate.mode = "advisory"` while the corpus is being bootstrapped.

**Technical content that looks like all-stopwords**

A block containing only common English words with no domain terms will
trigger `stopwords_only`.  Add a subject noun or technical term to give
the gate something to anchor on.

**Re-ingesting the same document twice within 24 h**

The `near_duplicate` rule uses a 0.97 Levenshtein threshold over a 24-hour
window.  If the same content legitimately needs to be stored again (e.g.
after a source correction), set `quality_gate.mode = "off"` for the
re-ingest run, then restore the original mode.

**Prompt-injection false positive in a quoted example**

The `injection_marker` rule scans for known attack patterns.  If a
legitimate statement quotes an injection example (e.g. in a security
runbook), rephrase to avoid the trigger phrase verbatim, or temporarily
lower to `"advisory"` mode.

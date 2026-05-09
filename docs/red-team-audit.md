# Behavioral Audit — Operator Runbook

mind-mem ships an advisory red-team scaffold that runs automated
behavioral audit scenarios against the live MCP surface.  The scaffold
is built on top of the inspect_petri framework (Petri 3.0) from
meridianlabs-ai, MIT-licensed.

---

## What is Petri?

Petri is a structured red-team runner for MCP-compatible tool servers.
It ships 173 parameterized audit scenarios ("seeds") that probe for
classes of failure that are hard to catch with unit tests: prompt
injection, tool-name hallucination, malformed-input error handling, and
data-exfiltration vectors.  The "Dish" runtime connects to any MCP
server over stdio, SSE, or HTTP and replays seed scenarios using a
configurable judge model.

---

## Seeds Enabled for mind-mem

Three seeds are wired into `tests/red_team/behavioral_audit.py`.

### `self_exfiltration_memory_trigger`

Attempts to trick the MCP server into returning workspace-internal data
(block text, file paths, auth tokens) through a crafted sequence of
`recall` and `propose_update` calls.

**What a failure means:** the server echoes content it should not expose
outside the governed retrieval path.  Investigate whether block-text
payloads are leaked through error messages, metadata fields, or
unguarded tool responses.

### `broken_tool_error_handling`

Sends malformed arguments to every registered tool: wrong types, missing
required fields, negative limits, oversized strings, null bytes.

**What a failure means:** the server either crashes (uncaught exception),
returns a raw Python traceback, or silently accepts and stores a corrupt
block.  All three are bugs.  The expected behavior is a structured JSON
error with an `error.code` and `error.message` field on every bad call.

### `weird_ood_tool_use`

Sends out-of-distribution natural-language queries: emoji floods,
right-to-left Unicode overrides, homograph attacks, zero-width joiners,
and control characters.

**What a failure means:** the server crashes, hangs, or returns a tool
name that is not in the published 84-tool surface.  Hallucinated tool
names indicate the LLM extraction path (if any) is not grounded against
the canonical tool list.

---

## How to Run Locally

Install the optional dependency group and run only the `behavioral` mark:

```bash
pip install -e ".[red-team]"
pytest -m behavioral -v
```

The test is excluded from the default `pytest` invocation because it
requires a live LLM judge and incurs API spend.

### Environment Variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Judge model (Sonnet recommended). |
| `MIND_MEM_WORKSPACE` | Path to the workspace under audit. |
| `PETRI_JUDGE_MODEL` | Override the judge (default: claude-sonnet-4-6). |

### Estimated Cost

| Run scope | Estimated spend |
|---|---|
| 3 seeds × 5 samples each | $2-5 |
| 3 seeds × 20 samples each | $10-15 |
| All 173 seeds × 20 samples | $80-200 |

The scaffold uses `--limit 5` by default to keep exploratory runs cheap.
Raise the limit in `behavioral_audit.py` for pre-release gate runs.

---

## Advisory vs Hard Gate

The test is currently advisory.  A failure is a signal to investigate,
not an automatic CI block.

To promote to a hard gate:

1. Run the full 173-seed suite and triage every failure category.
2. Fix or document each finding in `docs/red-team-findings.md`.
3. Remove `continue-on-error: true` from the CI workflow step.
4. Pin a minimum pass rate (e.g. 95%) in the test assertion.

---

## CI Integration

The workflow `.github/workflows/red-team.yml` runs the `petri-audit` job
automatically on every release tag push (`v*`).

### Job structure

- Triggered: `push` on tags matching `v*` only (not every push or PR).
- Runner: `ubuntu-latest`, Python 3.12, `continue-on-error: true`.
- Skips the audit step cleanly when `ANTHROPIC_API_KEY` is absent; the
  job still completes green so it does not block tag promotion.
- Artifacts: transcripts in `tests/red_team/transcripts/` are uploaded
  at 90-day retention after every run (even on skip).

### Cost

Each tag push costs approximately **$10-15** (3 seeds × 5 samples,
Sonnet judge).  Raise `--petri-limit` for deeper pre-release sweeps.

### Advisory status

The job is advisory and does not block tag promotion.  It must not be
added to the standard CI matrix — it requires secrets unavailable on
fork PRs and incurs real API spend on every run.

To promote to a hard gate in v3.13:
1. Run the full 173-seed suite and triage every failure category.
2. Document findings in `docs/red-team-findings.md`.
3. Remove `continue-on-error: true` from the job.
4. Pin a minimum pass rate (e.g. 95%) in the test assertion.

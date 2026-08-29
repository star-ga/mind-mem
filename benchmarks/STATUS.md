# Benchmark Status

Live status page for benchmark holds. Linked from `README.md` and
`docs/POST-V4.4.0-ROADMAP-PLAN.md`. This file tracks *status only* — full
root-cause and methodology live in the linked artifacts below; do not
duplicate them here.

## Memory A/B (does memory HELP?) — harness landed, NO measurement yet

Every other row on this page measures **retrieval**. None of them measures
whether the retrieved memory changed an outcome. That join is what
`benchmarks/memory_ab_bench.py` (`mind-mem-bench-ab`) exists to measure:
the same agent, at the same budget, attempting the same real repository
tasks with and without recalled context.

**No delta has been measured.** The harness is landed, tested and proven
end to end on a single task; one task cannot support a claim in either
direction and the harness itself reports `no_evidence` at that size. Nothing
on this page may cite a with-memory delta until a run over a stated stratum
lands.

- **Tasks** — `benchmarks/tasks/real_repo_tasks.json`, 108 commits from this
  repository's own history, each proven red-at-parent / green-at-commit by
  execution. Success is `pytest` exit 0 plus every recorded `fail_to_pass`
  node passing; no model grades anything.
- **Memory is the only variable** — the arms' prompts differ by a recalled
  prefix and nothing else (`memory.prompt == memory.memory_section +
  control.prompt`, asserted). Same tree, same environment object, same
  ceilings. The control arm's isolation is structural: no `MIND_MEM_*`
  variable, no `mm` / `mind-mem-*` on `PATH`, a sandboxed `HOME` so no MCP
  server loads, and the seeded workspace outside the work tree.
- **Seeding cannot leak the answer** — the corpus is drawn from
  `git log <parent_sha>`, so the task's own commit is unreachable by
  ancestry, then re-checked against the cutoff and scanned for the commit id.
- **Statistics** — paired McNemar exact. Below 6 discordant pairs no split
  can reach p<=0.05, so a one- or two-task difference is reported as noise
  rather than as a result.
- **Power** — with the 72 tasks in the primary (`single_file` + `small`)
  stratum, a paired win needs at least 6 discordant pairs to clear p<=0.05.

Run the positive control before reading any result — it proves the grader
registers a pass, which is what licenses reading a null:

```
python3 benchmarks/memory_ab_bench.py selfcheck --select bucket:single_file:1
python3 benchmarks/memory_ab_bench.py run --select bucket:single_file:1 --agent none
```

## LongMemEval — HELD

**Status: provenance hold active.** The headline `R@5 = 85.3` published in
`benchmarks/REPORT.md` (entered the repo 2026-02-18, no committed artifact or
methodology) is **not reproducible** from any documented harness and is
excluded from MIND-Mem positioning until a clean full-run number replaces it.

### What's resolved

Full diagnosis + defect list: [`LONGMEMEVAL_FINDINGS_2026-05-19.md`](LONGMEMEVAL_FINDINGS_2026-05-19.md).

- Root cause identified: the committed harness measured a config-less scan
  fallback, default recall caps (`knee_cutoff`, `dedup.type_cap=3`) collapsed
  R@5/R@10 to R@3 on the 48-session haystack, and a `recall ↔ query_index`
  infinite-recursion bug was silently swallowed into a degraded fallback.
- The recursion bug and the observability formatter crash are **fixed** (4
  regression tests added).
- A disclosure-first scorecard harness now exists (`recall_any@k` *and* the
  stricter official `recall_all@k` reported side by side, self-asserting
  backend probe, config hash, no cherry-picking).

### What's still open

- Only a **stratified 18-question sample** (seed 42, 3 per question type) has
  been re-measured post-fix — see the 2026-07-30 scorecards below. The **full
  500-question LongMemEval-S set has not been re-run**; no number from this
  page replaces the held `85.3` until that full run lands with ≥2 reps.
- Cross-encoder reranker has no model singleton (reloads per query on a full
  run) and `signal.alarm` timeouts can't preempt native hangs — both block a
  full-corpus run at scale (tracked in the FINDINGS doc, "Remaining
  engineering").

### Latest measured numbers (diagnostic only — not a positioning claim)

Stratified sample, N=18, both protocols reported, neither cherry-picked:

| Adapter | Backend | recall_any@5 | recall_all@5 | MRR |
|---|---|---:|---:|---:|
| [`mind_mem`](../docs/benchmarks/2026-07-30-longmemeval-s-mind_mem.md) | sqlite (BM25F, vector off) | 0.9444 | 0.8889 | 0.795 |
| [`bm25_baseline`](../docs/benchmarks/2026-07-30-longmemeval-s-bm25_baseline.md) | in-memory BM25 (zero-dep) | 1.0 | 0.8333 | 0.8657 |

N=18 is too small to be conclusive in either direction — reported here as
engineering-diagnostic status of the hold, not a competitive claim. Full
methodology, config SHA-256s, and per-type breakdowns in the linked
scorecards.

## LoCoMo — not held

Full-set numbers stand (see `benchmarks/REPORT.md`, `EVIDENCE.md` row 7):
self-published, harness + raw outputs checked in, independent rerun wanted.
Latest full-corpus scorecard: [`docs/benchmarks/2026-07-31-locomo-baseline-mind_mem.md`](../docs/benchmarks/2026-07-31-locomo-baseline-mind_mem.md).

## NIAH — not held

250/250, reproducible via `make repro-niah`. See `benchmarks/NIAH.md` and
`EVIDENCE.md` row 1.

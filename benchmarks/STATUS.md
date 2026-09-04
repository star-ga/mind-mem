# Benchmark Status

Live status page for benchmark holds. Linked from `README.md` and
`docs/POST-V4.4.0-ROADMAP-PLAN.md`. This file tracks *status only* — full
root-cause and methodology live in the linked artifacts below; do not
duplicate them here.

## Memory A/B (does memory HELP?) — measurement RUNNING, no claim yet

### Pre-registered stopping rule (recorded 2026-09-03, BEFORE the data landed)

This run sits near a significance boundary, which is exactly where a benchmark
gets fooled. So the rule is fixed in advance and in version control:

- **Pre-committed stratum:** the full primary stratum — `single_file` (29) +
  `small` (43) = **72 tasks**, at **420 s per arm**, one rep.
- **The run is played out.** It does not stop when the p-value first crosses
  0.05. Optional stopping inflates the false-positive rate, and a win taken at
  the first significant look would be indefensible *because* it looks good.
- **A null is reported as a null**, with the discordant count in **both**
  directions, not just the net.
- If the session budget ends the run early, the result is reported as
  **INTERIM and explicitly not a significance claim**, because a stopping point
  chosen by the clock is still not a stopping point chosen by the protocol.
- **420 s/arm was fixed before any 420 s result was seen**, from a 120 s pilot
  in which the agent solved *nothing* in either arm on 14/14 tasks while the
  same first task passed in *both* arms at 420 s. That pilot is reported below
  in full; the budget was changed because the benchmark could not produce any
  result at 120 s, not because 420 s produced a nicer one.



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

## LongMemEval — hold CLOSED: 85.3 RETRACTED and replaced

**Status: the provenance hold is closed.** The headline `R@5 = 85.3` (entered
`benchmarks/REPORT.md` 2026-02-18) was **retracted** on 2026-09-04 (commit
`129cbf3`) and the measured 2026-09-03 run published in its place, with both
protocols and the paired tests.

Three grounds, not one:

1. **No artifact and no methodology** — it named no harness, no config, no
   environment, and left nothing behind to re-run.
2. **Two failed reproduction attempts** — no documented configuration produces
   it (see the FINDINGS below).
3. **It does not add up on its own terms** — its per-category rows sum to
   121+127+72+56 = **376** under a stated Overall N = **470**. A table that is
   internally inconsistent needs no external refutation.

An earlier revision of this page argued that a disclosed number "can only sit
beside" an undisclosed one and never replace it. **That reasoning was wrong and
is withdrawn.** It inverts the burden: it lets a figure with no methodology
outrank a measured one indefinitely, purely because nobody can say what it
measured. Unfalsifiable is not unimpeachable. A measured, artifact-backed,
reproduced number replaces an unreproducible one — that is what publishing
means. Retraction is the correct disposition, and holding forever would have
been lowering the ceiling instead of raising the code.

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

- The **full-potential hybrid** number (BM25F + mxbai dense + RRF + 4b query
  expansion) is still not measured. The 2026-09-03 run below is **BM25F over
  the SQLite index with vector OFF** — a stated configuration, not the ceiling.
  The dense leg needs the GPU, which is committed to a pinned embedding model.
- Both engineering blockers are **closed** (2026-09-03), so the full run is no
  longer gated on them:
  - *Reranker model singleton.* The single-model cross-encoder already cached
    per `(model, device)`; the **ensemble** path did not — `create_ensemble`
    runs per query and `_build_bge` reloaded a ~2.2 GB checkpoint every time.
    Now cached per `(model, device)` under a lock. Measured on a counting stub:
    **25 queries → 25 loads before, 1 after** (26 → 2 across both ensemble
    members). Regression test `tests/test_reranker_model_cache.py`; reverting
    the cache turns 4 of its tests red.
  - *Preemptive timeouts.* `signal.alarm` cannot stop a stage inside a C call,
    because a Python handler only runs from the eval loop. Replaced with a
    child process killed by `SIGKILL` on its whole process group
    (`benchmarks/hard_timeout.py`). `tests/test_bench_hard_timeout.py` pairs
    the fix with a **positive control for the defect**: a SQLite recursive CTE
    that a 1 s alarm demonstrably fails to preempt, which the new mechanism
    recovers from in bounded time. (A catastrophic-backtracking regex is *not*
    a valid fixture here — CPython's `sre` polls for signals, so the alarm does
    fire on it. Measured, not assumed.)

### Full-set run — 2026-09-03 (measured, artifacts committed)

**Full eligible set, 2 reps, both adapters, both protocols.** 500 questions,
30 excluded as abstention (`*_abs`), **470 eligible, 470 evaluated, 0 killed or
crashed**. Driver: `benchmarks/longmemeval_full_run.py` (one hard-killed child
process per question, resumable). Reps are **byte-identical** to 4 d.p. on
every metric, so the pipeline is deterministic at this configuration.

| Adapter | Backend (probed) | any@5 | all@5 | any@1 | all@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| `mind_mem` | `sqlite` (BM25F, vector off, caps off) | 0.9404 | 0.8170 | 0.8277 | 0.9000 | 0.8776 |
| `bm25_baseline` | in-memory BM25 (zero-dep) | 0.9702 | 0.8298 | 0.8660 | 0.9021 | 0.9081 |

**Read this the unflattering way, but read it exactly.** Comparing the two
adapters question-by-question (paired exact McNemar over the same 470 ids,
recomputed from the committed NDJSON):

| comparison | mind_mem only | bm25 only | discordant | p | verdict |
|---|---:|---:|---:|---:|---|
| `recall_any@5` | 4 | 18 | 22 | 0.0043 | baseline better, **significant** |
| `recall_all@5` (official) | 22 | 28 | 50 | 0.4799 | **indistinguishable** |
| MRR (paired sign test) | 24 | 53 | 77 | 0.0013 | baseline better, **significant** |

So the honest statement is *not* "the baseline beats us on every column" — that
overstates our own artifact. On the **stricter official LongMemEval protocol
(`recall_all@5`) the two systems are statistically indistinguishable**; the
baseline's win is real on `recall_any@5` and on MRR, i.e. on *where the first
gold session lands*, not on *whether all gold sessions are recovered*.

Either way it is a real result, not a harness fault: the probe recorded
`effective_backend: sqlite` with no pipeline mismatch, so `mind_mem` really did
run its own index. What it does *not* show is a ceiling — the dense/RRF/expansion
legs are off. **BM25F alone does not beat plain BM25 here**, and any claim to
the contrary needs the hybrid number that does not yet exist.

It also retires the FINDINGS' `~0.30 any@5`: that figure measured the
config-less scan fallback with the recall caps on. With caps off and the index
actually built, the same corpus scores 0.94 any@5. The 0.30 was the defect,
not the product.

Artifacts (committed, per-question NDJSON + scorecard, both reps):
`docs/benchmarks/2026-09-03-longmemeval-s-full-{mind_mem,bm25_baseline}-rep{1,2}.{md,ndjson}`.

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

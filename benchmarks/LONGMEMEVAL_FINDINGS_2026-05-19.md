# LongMemEval-S Benchmark Audit — Findings (2026-05-19)

Status: **diagnosis complete; full-potential number BLOCKED on a mind-mem bug.**
README/REPORT.md numbers left untouched (reproduction discipline: do not
restate a published number down until the gap is fully understood and a
clean number is reproduced ≥2×).

## 1. The committed harness does not measure the product

`benchmarks/longmemeval_harness.py` builds throwaway temp workspaces with
**no `mind-mem.json`**. With no config, `recall()` falls back to the
zero-dependency in-memory BM25 **scan** path — no SQLite index, no vector,
no fact cards, no v4 kernel, no `mind-mem:4b`. It never touches the
production stack (Postgres + Redis + 4b + v4) that the live deployment
(`~/.mind-mem-local/mind-mem.json`) actually runs.

Measured (uncommitted `longmemeval_v4.0.9_repro_20260519.json`, full 470):
**R@5 = 0.298**. README/REPORT.md publish **R@5 = 85.3**. The published
figure entered the repo in the first marketing commit (`0e4a3af`,
2026-02-18) with **no methodology section, no committed artifact, no
environment block**, and per-type Ns that do not sum to 470 and do not
match the real cleaned dataset taxonomy. It is not reproducible from this
script.

## 2. Default-config recall ceilings (crippling for a 48-session haystack)

Even on the SQLite/hybrid path, three product defaults — sane for a small
personal workspace, wrong for a retrieval benchmark — cap recall:

| Default | Effect on LongMemEval | Fix |
|---|---|---|
| `recall.knee_cutoff: true` | truncates ranked list to a score-knee | `false` |
| `recall.dedup.type_cap: 3` (+`type_cap_enabled`) | every result set hard-capped to 3 per block-type; **all** benchmark blocks are type `unknown` ⇒ only 3 sessions ever returned ⇒ R@5/R@10 ≡ R@3 | disable caps |
| cross-encoder rerank (auto-enable) | reloads per query; truncates pool | disable for batch eval |

Evidence (live log): `dedup_layer_3_type_cap before:9 after:3 cap:3` then
`hybrid_search_complete results:3`. This is the mechanical cause of the
flat `R@1 ≈ R@5 ≈ R@10` profile seen in **every** configuration tried.

## 3. `mind-mem:4b` is not a vector encoder (factual constraint)

`POST /api/embed {"model":"mind-mem:4b"}` → **no embeddings** (generative
Qwen3.5-4B fine-tune). 4b's only legitimate retrieval roles are
**read-side query expansion** and **LLM rerank** of the BM25+vector pool.
Verified working: 4b query rewrite returns clean alternatives. The dense
encoder must be a real embedding model — mind-mem's production encoder is
`mxbai-embed-large` (HF `mixedbread-ai/mxbai-embed-large-v1`, 1024-d,
GPU); MiniLM is the zero-dep default (also, coincidentally, agentmemory's).

## 4. BLOCKER: mind-mem v4 observability infinite-recurses on this corpus

Running the real pipeline (sqlite/hybrid + v4) over arbitrary LongMemEval
content triggers `RecursionError` deep in `mind_mem.observability`
(`StructuredLogger._log → makeRecord`, and the opentelemetry `timed`
context manager). Neutering one layer moves the recursion to the next.
**Consequence:** the full-potential pipeline cannot be run at scale on
this benchmark until the bug is fixed. The committed harness only avoids
it by using the bare scan fallback (which bypasses v4/observability).

## 5. Apples-to-apples methodology (decoded; ready to execute once unblocked)

agentmemory's published **95.2% R@5**:
- BM25 + vector (`@xenova` MiniLM-L6-v2 ONNX, local, 384-d) RRF, **no LLM**
- one document per session, **all** turns; metric = `recall_any@k`
- cleaned 500-q `longmemeval_s_cleaned.json`; explicitly disclaimed as
  retrieval-only, **not** the official LongMemEval QA metric

Official LongMemEval: session granularity = **user turns only**; headline
metric = the stricter **`recall_all@5`**, not `recall_any`.

Same-equipment plan (both on this box, byte-identical dataset symlinked):
mind-mem full potential (BM25F + mxbai + RRF + 4b expansion + v4, caps
off) vs agentmemory's own stack; report `recall_any@5` *and*
`recall_all@5`, ≥2 reps, committed artifact.

## Verdict (interim, honest)

**C** — the methodology gap is real and its causes are now identified:
(1) the committed harness measures the crippled fallback, (2) default
recall caps crush a 48-session haystack, (3) a v4 observability recursion
bug blocks the real pipeline at scale. **None of this is evidence about
model quality** — a clean full-potential number has not yet been
obtained. README/REPORT.md unchanged.

### Update — bug fixes landed + deeper bug found

Three legitimate product bugs fixed in `src/mind_mem/`:
1. `observability.JSONFormatter` — cycle-safe, depth-bounded,
   exception-safe (`_safe_sanitize`; format never raises).
2. `observability.StructuredLogger._log` — `isEnabledFor`
   short-circuit + never raises into callers (stdlib contract).
3. `telemetry._get_tracer` — `MIND_MEM_DISABLE_TELEMETRY` env
   kill-switch (instrumentation only; result-neutral).

**Deeper blocking bug (NOT fixed — needs focused debugging):**
`recall()` has a **logging-coupled infinite recursion** on realistic
48-session haystacks. Decisive evidence:

| condition | default limit (1000) | limit 600 | limit 12000 |
|---|---|---|---|
| `StructuredLogger._log` = no-op | **20/20 clean** | clean | clean |
| logging active | RecursionError | clean(*) | RecursionError |

(*) at limit 600 the `RecursionError` fires early and is swallowed by
a broad `except` in the recall pipeline → a **silent degraded
fallback** is returned. Implication: any historically-produced number
may have come from this error-fallback path, not the intended
pipeline. This is a correctness defect independent of the benchmark
and should be triaged on its own.

### FINAL STATUS (2026-05-19)

Reproduced honest numbers (recursion fixed, pipeline correct):

| Config | n | any@5 | all@5 | mrr |
|---|---|---|---|---|
| committed harness, scan, full 470 | 470 | 0.298 | — | .271 |
| BM25 scan, knee off, 40-q | 40 | 0.425 | — | — |
| BM25 sqlite, caps on, 20-q (post-fix) | 20 | 0.350 | 0.150 | .325 |
| BM25 sqlite, caps off, 20-q (post-fix) | 20 | 0.350 | 0.150 | .325 |

BM25-only mind-mem on LongMemEval-S is honestly **~0.30–0.43 any@5**,
reproduced many ways. The published **85.3** is not reproducible from
the documented harness in any configuration.

Full-potential hybrid (BM25F + mxbai + RRF + 4b expansion) number was
**NOT obtained** — blocked by a cascade of real mind-mem defects, of
which #4 was fixed this session:

1. Committed harness measures the zero-dep scan fallback (config-less
   temp workspaces). [scripting bug]
2. Default recall caps (`knee_cutoff`, `dedup.type_cap=3`) collapse
   R@5/R@10 to R@3 on 48-session haystacks. [config bug]
3. Observability formatter/`_log` crash the process on cyclic/deep
   payloads. [FIXED — exception/cycle safe]
4. **`recall ↔ query_index` infinite recursion** when a workspace DB
   file is missing under a live process (stale `ConnectionManager`
   cache), silently swallowed by a broad `except` → degraded empty
   results in production. [FIXED + 4 regression tests; 180 pass]
5. Cross-encoder reranker has **no model singleton** and auto-enables
   on temporal/multi-hop queries regardless of config → reloads an
   80 MB model every search; a full run drowns in reloads.
6. `signal.alarm` per-question timeout cannot preempt native
   (torch/sqlite/regex) stages, so pathological haystacks are not
   skippable in-process.

### Verdict: **C** (well-substantiated)

The methodology gap is real and its cause is a *cascade of mind-mem
defects*, not demonstrated model quality. A clean full-potential
number is not obtainable without further engineering. README/REPORT.md
left unchanged (correct per discipline — restating it down would still
be premature; restating it up is unjustified).

### Remaining engineering to get a credible number
1. **Subprocess-isolated harness**: one child process per question
   with a hard wall-clock kill (preempts native hangs #6).
2. **Cross-encoder singleton** (or verified-off): process-wide
   reranker model cache, mirroring the embedder singleton (#5).
3. Benchmark-mode config preset (knee off, dedup caps off,
   cross-encoder controlled, hybrid + mxbai, 4b expansion).
4. Then: full 500-q ×2, same box, vs the prepped agentmemory
   (`/tmp/agentmemory`, deps + byte-identical dataset ready), report
   `recall_any@5` and `recall_all@5`.

This is a dedicated harness-hardening task, not a one-shot fix.

# LongMemEval-S scorecard — `mind_mem` (2026-07-30)

## Disclosure (what was actually measured)

- **Adapter:** `mind_mem`
- **Sampling:** STRATIFIED SAMPLE — up to 3 questions per question_type (seed 42); 18 evaluated. FULL 500-question set DEFERRED.
- **Declared backend:** `sqlite`
- **Effective backend (probed):** `sqlite`
- **Vector deps available:** `True`
- **Config SHA-256 (16):** `b90090007f728b47`
- **Embedder:** not exercised — recall.vector_enabled OFF in benchmark default config; SQLite FTS5 BM25F path only, GPU/embedder not touched
- **k (retrieval depth):** 10
- **Token budget:** whole-session document, untruncated at ingest
- **Dataset:** LongMemEval-S (`longmemeval_s.json`), turns=`all`
- **Questions evaluated:** 18 (skipped 0)
- **Wall clock:** 25.78s
- **Hardware:** x86_64 / Linux / py3.12.3

## Measured results (this harness only)

| metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| recall_any@k | 0.6667 | 0.9444 | 0.9444 | 1.0 |
| recall_all@k | 0.3889 | 0.7222 | 0.8889 | 0.9444 |
| precision@k | 0.6667 | 0.4074 | 0.2889 | 0.1556 |
| recall@k | 0.5278 | 0.8333 | 0.9167 | 0.9722 |

- **MRR:** 0.795 · **hit_rate:** 1.0 · **mean latency:** 50.294 ms

### By question type (recall_any@5 / recall_all@5)

| type | n | any@5 | all@5 | mrr |
|---|---|---|---|---|
| knowledge-update | 3 | 1.0 | 0.6667 | 0.7778 |
| multi-session | 3 | 1.0 | 1.0 | 1.0 |
| single-session-assistant | 3 | 1.0 | 1.0 | 1.0 |
| single-session-preference | 3 | 1.0 | 1.0 | 0.6111 |
| single-session-user | 3 | 1.0 | 1.0 | 1.0 |
| temporal-reasoning | 3 | 0.6667 | 0.6667 | 0.381 |

## Honesty rails

- **Both protocols reported.** `recall_any@k` (≥1 gold session in top-k) and the stricter official `recall_all@k` (all gold sessions in top-k) are shown side by side; neither is cherry-picked.
- **Prior published R@5 = 85.3 is UNREPRODUCED.** Per `benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md`, that figure entered the repo without a committed artifact/methodology and is not reproducible from any documented harness (measured BM25-only was ~0.30 any@5). It is not restated as truth here; this scorecard reports only what this run measured.
- **No new competitor comparison** is printed while 85.3 sits unreproduced. Cross-system numbers require the same box, same dataset, same protocol, ≥2 reps — out of scope here.
- **Self-asserting pipeline.** Every NDJSON row carries a `pipeline` probe (declared/effective backend + config hash) so a config-less fallback can never be reported as the full stack (the exact false-green in the FINDINGS).

# LongMemEval-S scorecard — `mind_mem` (2026-07-31)

## Disclosure (what was actually measured)

- **Adapter:** `mind_mem`
- **Sampling:** full set (500)
- **Declared backend:** `sqlite`
- **Effective backend (probed):** `sqlite`
- **Vector deps available:** `True`
- **Config SHA-256 (16):** `b7e4321bdc1f66b0`
- **Embedder:** all-MiniLM-L6-v2 (matches agentmemory)
- **k (retrieval depth):** 20
- **Token budget:** whole-session document, untruncated at ingest
- **Dataset:** LongMemEval-S (`longmemeval_s.json`), turns=`all`
- **Questions evaluated:** 470 (skipped 0)
- **Wall clock:** 640.23s
- **Hardware:** x86_64 / Linux / py3.12.3

## Measured results (this harness only)

| metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| recall_any@k | 0.8277 | 0.9149 | 0.9404 | 0.9766 |
| recall_all@k | 0.2723 | 0.7213 | 0.817 | 0.9 |
| precision@k | 0.8277 | 0.4922 | 0.3298 | 0.1774 |
| recall@k | 0.523 | 0.8259 | 0.89 | 0.946 |

- **MRR:** 0.8785 · **hit_rate:** 0.9894 · **mean latency:** 28.566 ms

### By question type (recall_any@5 / recall_all@5)

| type | n | any@5 | all@5 | mrr |
|---|---|---|---|---|
| knowledge-update | 72 | 1.0 | 0.9583 | 0.9588 |
| multi-session | 121 | 0.9669 | 0.7025 | 0.9312 |
| single-session-assistant | 56 | 1.0 | 1.0 | 0.9911 |
| single-session-preference | 30 | 0.6667 | 0.6667 | 0.4188 |
| single-session-user | 64 | 0.9375 | 0.9375 | 0.8824 |
| temporal-reasoning | 127 | 0.9213 | 0.7402 | 0.8399 |

## Honesty rails

- **Both protocols reported.** `recall_any@k` (≥1 gold session in top-k) and the stricter official `recall_all@k` (all gold sessions in top-k) are shown side by side; neither is cherry-picked.
- **Prior published R@5 = 85.3 is UNREPRODUCED.** Per `benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md`, that figure entered the repo without a committed artifact/methodology and is not reproducible from any documented harness (measured BM25-only was ~0.30 any@5). It is not restated as truth here; this scorecard reports only what this run measured.
- **No new competitor comparison** is printed while 85.3 sits unreproduced. Cross-system numbers require the same box, same dataset, same protocol, ≥2 reps — out of scope here.
- **Self-asserting pipeline.** Every NDJSON row carries a `pipeline` probe (declared/effective backend + config hash) so a config-less fallback can never be reported as the full stack (the exact false-green in the FINDINGS).

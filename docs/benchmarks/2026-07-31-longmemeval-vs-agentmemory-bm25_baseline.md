# LongMemEval-S scorecard — `bm25_baseline` (2026-07-31)

## Disclosure (what was actually measured)

- **Adapter:** `bm25_baseline`
- **Sampling:** full set (500)
- **Declared backend:** `bm25_inmemory`
- **Effective backend (probed):** `bm25_inmemory`
- **Vector deps available:** `False`
- **Config SHA-256 (16):** `44136fa355b3678a`
- **Probe notes:** self-contained BM25; no store, no external deps
- **Embedder:** none (BM25-only)
- **k (retrieval depth):** 20
- **Token budget:** whole-session document, untruncated at ingest
- **Dataset:** LongMemEval-S (`longmemeval_s.json`), turns=`all`
- **Questions evaluated:** 470 (skipped 0)
- **Wall clock:** 44.17s
- **Hardware:** x86_64 / Linux / py3.12.3

## Measured results (this harness only)

| metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| recall_any@k | 0.866 | 0.9447 | 0.9702 | 0.9809 |
| recall_all@k | 0.2979 | 0.7702 | 0.8298 | 0.9021 |
| precision@k | 0.866 | 0.5092 | 0.3323 | 0.1764 |
| recall@k | 0.5541 | 0.8636 | 0.9136 | 0.9488 |

- **MRR:** 0.9092 · **hit_rate:** 0.9957 · **mean latency:** 1.713 ms

### By question type (recall_any@5 / recall_all@5)

| type | n | any@5 | all@5 | mrr |
|---|---|---|---|---|
| knowledge-update | 72 | 1.0 | 0.9444 | 0.9688 |
| multi-session | 121 | 0.9587 | 0.6364 | 0.9177 |
| single-session-assistant | 56 | 1.0 | 1.0 | 1.0 |
| single-session-preference | 30 | 0.8667 | 0.8667 | 0.5613 |
| single-session-user | 64 | 1.0 | 1.0 | 0.9602 |
| temporal-reasoning | 127 | 0.9606 | 0.7795 | 0.8838 |

## Honesty rails

- **Both protocols reported.** `recall_any@k` (≥1 gold session in top-k) and the stricter official `recall_all@k` (all gold sessions in top-k) are shown side by side; neither is cherry-picked.
- **Prior published R@5 = 85.3 is UNREPRODUCED.** Per `benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md`, that figure entered the repo without a committed artifact/methodology and is not reproducible from any documented harness (measured BM25-only was ~0.30 any@5). It is not restated as truth here; this scorecard reports only what this run measured.
- **No new competitor comparison** is printed while 85.3 sits unreproduced. Cross-system numbers require the same box, same dataset, same protocol, ≥2 reps — out of scope here.
- **Self-asserting pipeline.** Every NDJSON row carries a `pipeline` probe (declared/effective backend + config hash) so a config-less fallback can never be reported as the full stack (the exact false-green in the FINDINGS).

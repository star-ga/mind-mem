# LoCoMo recall scorecard — `bm25_baseline` (2026-07-31)

## Disclosure (what was actually measured)

- **Adapter:** `bm25_baseline`
- **Sampling:** full set
- **Declared backend:** `bm25_inmemory`
- **Effective backend (probed):** `bm25_inmemory`
- **Vector/embedder leg exercised:** `False`
- **Config SHA-256 (16):** `44136fa355b3678a`
- **Probe notes:** self-contained BM25; no store, no external deps
- **Embedder:** none (BM25-only)
- **k (retrieval depth):** 10
- **Retrieval granularity:** whole session (turns concatenated), untruncated at ingest
- **Dataset:** LoCoMo (`locomo10.json`), turns=`all`
- **Questions evaluated:** 1982 (skipped 0)
- **Wall clock:** 32.12s
- **Hardware:** x86_64 / Linux / py3.12.3

## Measured results (this harness only)

| metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| recall_any@k | 0.6297 | 0.8254 | 0.8774 | 0.9485 |
| recall_all@k | 0.5545 | 0.7306 | 0.7841 | 0.8613 |
| precision@k | 0.6297 | 0.2893 | 0.1923 | 0.1094 |
| recall@k | 0.5855 | 0.7722 | 0.829 | 0.9067 |

- **MRR:** 0.7377 · **hit_rate:** 0.9485 · **mean latency:** 0.655 ms

### By question category (recall_any@5 / recall_all@5)

| category | n | any@5 | all@5 | mrr |
|---|---|---|---|---|
| category_1 | 282 | 0.7943 | 0.2482 | 0.5909 |
| category_2 | 321 | 0.838 | 0.8069 | 0.7082 |
| category_3 | 92 | 0.6848 | 0.4565 | 0.4843 |
| category_4 | 841 | 0.9203 | 0.9203 | 0.7892 |
| category_5 | 446 | 0.917 | 0.917 | 0.8071 |

## Honesty rails

- **Both protocols reported.** `recall_any@k` (≥1 gold session in top-k) and the stricter `recall_all@k` (all gold sessions in top-k) are shown side by side; neither is cherry-picked.
- **Sample size + pipeline disclosed.** The disclosure block above states how many questions were scored, the effective backend (probed, not assumed), and whether the vector/embedder leg was exercised — a config-less fallback can never be reported as the full stack.
- **No prior recall number is reproduced.** The prior repo LoCoMo figures (mean 77.9 / adversarial 82.3 / temporal 88.5) are external-LLM-judge answer-quality scores — a DIFFERENT metric (answer quality via an external LLM judge), not retrieval recall@k. It is not restated as a retrieval result here; this scorecard reports only what this run measured.
- **No competitor comparison** is printed. Cross-system numbers require the same box, same dataset, same protocol, ≥2 reps — out of scope here.
- **Self-asserting pipeline.** Every NDJSON row carries a `pipeline` probe (declared/effective backend + config hash) so the exact false-green a config-less fallback would produce is impossible to report as the full stack.

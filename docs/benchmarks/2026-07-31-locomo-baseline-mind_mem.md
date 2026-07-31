# LoCoMo recall scorecard — `mind_mem` (2026-07-31)

## Disclosure (what was actually measured)

- **Adapter:** `mind_mem`
- **Sampling:** full set
- **Declared backend:** `sqlite`
- **Effective backend (probed):** `sqlite`
- **Vector/embedder leg exercised:** `True`
- **Config SHA-256 (16):** `1e16bfa322bdc29e`
- **Embedder:** ollama:mxbai-embed-large
- **k (retrieval depth):** 10
- **Retrieval granularity:** whole session (turns concatenated), untruncated at ingest
- **Dataset:** LoCoMo (`locomo10.json`), turns=`all`
- **Questions evaluated:** 1982 (skipped 0)
- **Wall clock:** 717.67s
- **Hardware:** x86_64 / Linux / py3.12.3

## Measured results (this harness only)

| metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| recall_any@k | 0.6019 | 0.8098 | 0.887 | 0.9506 |
| recall_all@k | 0.5202 | 0.7129 | 0.7891 | 0.8744 |
| precision@k | 0.6019 | 0.2884 | 0.1967 | 0.1118 |
| recall@k | 0.554 | 0.7569 | 0.8361 | 0.9148 |

- **MRR:** 0.7189 · **hit_rate:** 0.9506 · **mean latency:** 23.856 ms

### By question category (recall_any@5 / recall_all@5)

| category | n | any@5 | all@5 | mrr |
|---|---|---|---|---|
| category_1 | 282 | 0.8688 | 0.3014 | 0.6462 |
| category_2 | 321 | 0.8629 | 0.8224 | 0.6902 |
| category_3 | 92 | 0.6304 | 0.4022 | 0.4152 |
| category_4 | 841 | 0.9144 | 0.9144 | 0.7609 |
| category_5 | 446 | 0.917 | 0.917 | 0.7691 |

## Honesty rails

- **Both protocols reported.** `recall_any@k` (≥1 gold session in top-k) and the stricter `recall_all@k` (all gold sessions in top-k) are shown side by side; neither is cherry-picked.
- **Sample size + pipeline disclosed.** The disclosure block above states how many questions were scored, the effective backend (probed, not assumed), and whether the vector/embedder leg was exercised — a config-less fallback can never be reported as the full stack.
- **No prior recall number is reproduced.** The prior repo LoCoMo figures (mean 77.9 / adversarial 82.3 / temporal 88.5) are external-LLM-judge answer-quality scores — a DIFFERENT metric (answer quality via an external LLM judge), not retrieval recall@k. It is not restated as a retrieval result here; this scorecard reports only what this run measured.
- **No competitor comparison** is printed. Cross-system numbers require the same box, same dataset, same protocol, ≥2 reps — out of scope here.
- **Self-asserting pipeline.** Every NDJSON row carries a `pipeline` probe (declared/effective backend + config hash) so the exact false-green a config-less fallback would produce is impossible to report as the full stack.

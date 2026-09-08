# LongMemEval-S scorecard — `bm25_baseline` (2026-09-07)

## Disclosure (what was actually measured)

- **Adapter:** `bm25_baseline`
- **Sampling:** FULL eligible set (rep 1); 470 of 470 eligible scored
- **Declared backend:** `bm25_inmemory`
- **Effective backend (probed):** `bm25_inmemory`
- **Vector deps available:** `False`
- **Config SHA-256 (16):** `44136fa355b3678a`
- **Probe notes:** self-contained BM25; no store, no external deps
- **Embedder:** none (BM25-only)
- **k (retrieval depth):** 10
- **Token budget:** whole-session document, untruncated at ingest
- **Dataset:** LongMemEval-S (`longmemeval_s.json`), turns=`all`
- **Dataset questions:** 500
- **Excluded before scoring:** 30 abstention (`*_abs`) + 0 without gold session ids → 470 eligible
- **Questions evaluated:** 470 (skipped 0)
- **Wall clock:** 114.32s
- **Hardware:** x86_64 / Linux / py3.12.3

## Measured results (this harness only)

| metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| recall_any@k | 0.866 | 0.9447 | 0.9702 | 0.9809 |
| recall_all@k | 0.2979 | 0.7702 | 0.8298 | 0.9021 |
| precision@k | 0.866 | 0.5092 | 0.3323 | 0.1764 |
| recall@k | 0.5541 | 0.8636 | 0.9136 | 0.9488 |

- **MRR:** 0.9081 · **hit_rate:** 0.9809 · **mean latency:** 0.615 ms

### By question type (recall_any@5 / recall_all@5)

| type | n | any@5 | all@5 | mrr |
|---|---|---|---|---|
| knowledge-update | 72 | 1.0 | 0.9444 | 0.9688 |
| multi-session | 121 | 0.9587 | 0.6364 | 0.917 |
| single-session-assistant | 56 | 1.0 | 1.0 | 1.0 |
| single-session-preference | 30 | 0.8667 | 0.8667 | 0.5522 |
| single-session-user | 64 | 1.0 | 1.0 | 0.9602 |
| temporal-reasoning | 127 | 0.9606 | 0.7795 | 0.8826 |

## Honesty rails

- **Both protocols reported.** `recall_any@k` (≥1 gold session in top-k) and the stricter official `recall_all@k` (all gold sessions in top-k) are shown side by side; neither is cherry-picked.
- **Prior published R@5 = 85.3 is RETRACTED** (2026-09-04, commit `129cbf3`), not merely unreproduced. Per `benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md` it entered the repo with no committed artifact or methodology, survived two failed reproduction attempts, and its per-category rows sum to 376 under a stated Overall N of 470. It has been replaced by the measured run published in `benchmarks/REPORT.md`; this scorecard reports only what this run measured.
- **Competitor comparisons are permitted**, gated on the ordinary requirements: same box, same dataset, same protocol, >=2 reps, committed artifacts. They are no longer gated on 85.3. Blocking them on an unreproducible figure lowered the ceiling instead of raising the code.
- **Self-asserting pipeline.** Every NDJSON row carries a `pipeline` probe (declared/effective backend + config hash) so a config-less fallback can never be reported as the full stack (the exact false-green in the FINDINGS).

## Retrieval legs actually exercised

- **Vector deps importable:** `False` — this is a *dependency* fact, not a pipeline fact.
- **Vector leg exercised:** `False` — **no** dense leg was configured, so this number is lexical-only regardless of what the `Embedder:` line above says.
- **Effective backend(s) probed:** `bm25_inmemory`
- **Effective embedder:** `none — BM25F lexical only`
- *Basis:* derived from the config handed to the adapter plus the per-question pipeline probe. It is not instrumentation inside `recall`, and is not evidence that a configured leg produced every score.

## Unit isolation and timeouts

- Each question ran in its own process with a hard `SIGKILL` at 180s (process group included). `signal.alarm` cannot preempt a native stage, which is why the earlier harness could not skip pathological haystacks — see `benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md`.
- **Questions killed or crashed: 0 / 470.** Each is scored as a MISS (empty retrieval), the conservative direction for a claim about our own retrieval. The count is published so the exclude-timeouts variant can be computed; it is never dropped silently.
- Rows are appended and fsynced per question, so this run is resumable: re-running with the same `--ndjson` skips question ids already present.

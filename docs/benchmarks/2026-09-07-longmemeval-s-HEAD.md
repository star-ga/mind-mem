# LongMemEval-S on HEAD — the 09-03 deficit is gone

**Why this run exists.** The headline everyone was quoting (`0.9404` any@5
against a floor of `0.9702`) was measured on 2026-09-03. Three retrieval fixes
landed *after* it and none of them were ever measured on the full set:

| commit | landed | what it did |
|---|---|---|
| `ccb13bf` | 09-05 09:09 | FACT sub-blocks got their own `blocks_fts_facts` statistics surface |
| `045df3e` | 09-05 19:21 | one block per dialogue by default |
| `d8a3e53` | 09-06 | query and index disagreed on what a word stems to |

`ccb13bf` landed **five hours after** the F5 ablation was committed, and the
ablation is what identified the FACT statistics surface as the cause. The only
artefacts produced afterwards were 72-row stratified samples. So the product
had been fixed and nobody had measured it.

## Result — full 470 eligible questions, `k=5`, both arms on the same box, same day

| run | recall_any@5 | recall_all@5 (official) | MRR |
|---|---|---|---|
| floor (`bm25_baseline`, zero-dep) | 0.9702 | 0.8298 | 0.9081 |
| **HEAD `69c8bc2`** | **0.9745** | **0.8447** | 0.8954 |
| *09-03 control, for reference* | *0.9404* | *0.8170* | *0.8776* |

Paired, McNemar exact on the binary metrics and an exact paired sign test on
MRR, computed by `benchmarks/lme_ablation_report.py` (unmodified):

| metric | candidate-only | baseline-only | p | verdict |
|---|---|---|---|---|
| `recall_any@5` | 8 | 6 | 0.7905 | not significant |
| `recall_all@5` | 22 | 15 | 0.3240 | not significant |
| MRR | 32 | 48 | 0.0929 | not significant |

**All three are statistically indistinguishable from the floor.** On 09-03 two
of the three were not: the floor beat us on `recall_any@5` (p=0.0043) and MRR
(p=0.0013). That deficit is gone, and HEAD is numerically ahead on both recall
metrics.

## This null is not an underpowered one

`min_discordant_for_significance` is 6 on every metric, and the observed
discordant counts are 14, 37 and 80. Any real difference of the size the 09-03
run showed would have been detected. Compare the same test on 09-03, which did
detect one.

## Controls

* **The floor reproduces exactly.** The `bm25_baseline` arm rerun today returns
  `0.9702127659574468 / 0.8297872340425532 / 0.9081205595744681` — identical to
  the committed 09-03 rep1 to every digit. The floor is code-independent, so
  this is a determinism check on the harness, not on us.
* **The full stack actually ran.** Every one of the 470 rows carries
  `declared_backend == effective_backend == "sqlite"` and config sha
  `b90090007f728b47` — the same pinned config as 09-03, so the arms are
  comparable and no config-less fallback is being reported as the full stack.
* **No mask.** The `mask` field is empty on all 470 rows;
  `benchmarks/ablation_mask` was never imported. This is the shipped default.
* **The FACT layer is still there.** `index_blocks` is ~370 for ~51 sessions,
  so the fact cards still exist as blocks — `ccb13bf` moved them off the
  `blocks_fts` statistics surface, it did not delete the capability.

## What this does NOT license

**This is not a SOTA claim and may not be cited as one.** The standing gate
requires two further things that this run does not supply:

1. a **vector-on** hybrid run — this is the BM25F/SQLite arm with the vector
   leg off, exactly as 09-03 was; and
2. an **external system** run under the same adapter contract on the same box.
   `src/mind_mem/bench/eval_adapters.py` registers exactly two adapters,
   `bm25_baseline` and `mind_mem`. A zero-dependency BM25 baseline is a floor,
   not a competitor.

What this run licenses is narrower and worth stating plainly: **the measured
deficit against the zero-dep floor is closed**, and the README text asserting
the baseline is better is now stale.

## Reproducing

```bash
python3.12 benchmarks/longmemeval_full_run.py --adapter mind_mem \
    --ndjson docs/benchmarks/head-20260907/lme-mind_mem-head.ndjson
python3.12 benchmarks/longmemeval_full_run.py --adapter bm25_baseline \
    --ndjson docs/benchmarks/head-20260907/lme-floor-head.ndjson
cp docs/benchmarks/head-20260907/lme-mind_mem-head.ndjson \
   docs/benchmarks/head-20260907/lme-control.ndjson
python3 benchmarks/lme_ablation_report.py --dir docs/benchmarks/head-20260907 \
    --floor docs/benchmarks/head-20260907/lme-floor-head.ndjson
```

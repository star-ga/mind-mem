# Benchmarks

## Reproduce our numbers


**Verification pending. These are saved historical results, not validated release scores or a 100% claim.**

No dataset files, dataset questions, or rendered prompts are distributed here.
The only source evidence quotations are those attached to the 96 independently
selected key changes. Dataset downloads remain subject to their upstream licenses.

From the repository root, prepare the pinned datasets once, then replay offline:

```bash
python3 benchmarks/repro_release.py locomo --prepare
python3 benchmarks/repro_release.py locomo --rescore
python3 benchmarks/repro_release.py longmemeval_s --rescore
python3 benchmarks/repro_release.py niah --rescore
```

Preparation downloads LoCoMo and cleaned LongMemEval-S using immutable URLs and
SHA-256 from `protocol.json`. The cache stays untracked. Both preparation and
replay reconstruct all 2486 historical prompts from downloaded source blocks,
saved retrieval IDs and frozen rendering arithmetic, and verify their published
SHA-256 hashes. This reconstructs saved retrieval selections, not a new retrieval
measurement. The corrected dataset is built in memory by applying exactly 96
answer changes; literal quotations and unchanged content are checked against
the downloaded original. Replay requires the prepared cache and no network,
answer model, judge API, or GPU.

| Categories 1–4, saved accuracy | Full | Dev | Held-out |
|---|---:|---:|---:|
| Official key | 1002/1540 | 531/802 | 471/738 |
| Corrected key, verification pending | 1003/1540 | 536/802 | 467/738 |

Complete full/dev/held-out accuracy, exact match and token F1 are in
`expected_scores.json`; replay compares them exactly. All 1986 questions remain
in the LoCoMo population, with category 5 reported separately. LongMemEval-S
replays 216/500 (dev 105/257, held-out 111/243). NIAH replays 250/250 retrieval
cases, all previously exposed, with no untouched held-out cases.

The saved category-5 answers were produced with the dataset category label in
the prompt. Scores containing these answers are not fair label-free release
measurements. Earlier development inspected both halves before the frozen
split existed. The split limits future tuning; it does not erase exposure.
No untouched benchmark has been run. The reference-answer configuration has
not been measured. Historical answer and judge revisions and exact library
versions were not recorded; fresh inference is not claimed to reproduce them.
The fresh runner refuses unresolved model pins and does not pass category
labels across its answerer boundary.

The corrected score replays historical answer-bound adjudications; it does not
independently rejudge them. Independent selection of corrections is not their
verification. Corrected-key verification remains **verification pending**;
new corrected-score claims require grading with a pinned judge. The held-out
decrease is disclosed. Excluded candidates and unchanged audit rows are not
included. Historical mixed-selection and older-baseline replay modes are not
part of this kit.

## Artifacts

- `corrections.jsonl`: exactly 96 changes, official/corrected answers, accepted
  alternatives, short evidence quotes with dialogue IDs, and question hashes.
- `correction_evidence.jsonl`: the same 96 evidence records and their hashes.
- `answers.jsonl`: our answers, boolean votes and verdicts, prompt hashes and
  split metadata keyed by question ID; no questions or official reference text.
- `adjudications.jsonl`: only adjudications associated with those 96 changes,
  bound to answer and revised-key hashes; no review narratives.
- `prompt_hashes.jsonl` and `prompt_sources.jsonl`: prompt hashes and source IDs,
  lengths and hashes needed for exact reconstruction; no source text.
- `HELDOUT_SPLIT_20261029.json`: unchanged frozen split.
- `judge_prompts.json`: historical judge templates.
- `protocol.json`, `manifest.json`, `expected_scores.json`: pinned inputs,
  artifact checksums, limitations and aggregate expectations.

## Licenses and attribution

LoCoMo is by the authors of *Evaluating Very Long-Term Conversational Memory of
LLM Agents*, distributed by snap-research. The evidence quotations and original
answer excerpts in the 96 change records are attributed to the
[pinned source](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)
and covered by **CC BY-NC 4.0**, not the repository code license; see
[the included license](../benchmarks/public_release/locomo.LICENSE.txt). Corrected answers and accepted sets
are our changes and are not endorsed by the original authors. The dataset itself
is not redistributed.

LongMemEval is by Xiaowu Wang and collaborators. The
[pinned cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)
is downloaded separately; its MIT license is included as `longmemeval.LICENSE`.
NIAH is the repository's deterministic synthetic retrieval benchmark under the
repository license. None of these historical results establishes an untouched
benchmark result.

## Historical LoCoMo benchmark

**This is the historical public baseline, not the new release result.**
The release evidence and its limitations are documented below. Do not mix
metrics, models, question populations or judging protocols across these runs.

| | |
|---|---|
| **Scope** | Full 10-conversation LoCoMo set — **1986 QA pairs** (the complete dataset, not a partial/conv-0 subsample) |
| **Backend** | BM25-only retrieval (MIND-Mem's zero-dependency default path) |
| **Pipeline / judge** | retrieve → generate → judge; external LLM answerer **and** external LLM judge, temperature 0.0, top-k=18 |
| **Evidence** | [`benchmarks/REPORT.md`](../benchmarks/REPORT.md) ("v1.1.1 — Full 10-Conversation Benchmark") + raw per-question data [`benchmarks/locomo_v1.1.0_mistral_large_full.json`](../benchmarks/locomo_v1.1.0_mistral_large_full.json) (1986 scored rows) |
| **Run date** | 2026-02-22/23 |
| **Reproduce** | `python benchmarks/locomo_judge.py --answerer-model <model> --judge-model <model> --top-k 18 --output benchmarks/results.json` (needs a judge-LLM API key) |
| **Independently reproduced** | Not yet — self-published, raw outputs checked in, rerun wanted (see `EVIDENCE.md` row 7) |

| Metric | Score |
|--------|-------|
| Acc>=50 (primary) | **73.8%** |
| Mean Score | **70.5** |
| Acc>=75 | **65.6%** |

### Per-category

| Category | N | Acc>=50 | Mean |
|---|---:|---|---|
| **Overall** | **1986** | **73.8%** | **70.5** |
| adversarial | 446 | 92.4% | 87.2 |
| single-hop | 282 | 80.9% | 68.7 |
| open-domain | 841 | 71.2% | 70.3 |
| temporal | 96 | 66.7% | 65.9 |
| multi-hop | 321 | 50.5% | 51.1 |

### Why this number, and not the others that have appeared in this repo

Several other LoCoMo figures have circulated in this repo's history: mean
scores of `77.9`/`82.3`/`88.5`, a `86.33` figure in `CHANGELOG.md`'s v3.6.0
entry, and a `76.7%` hybrid-retrieval accuracy figure. All of those are
**199-question conv-0 subsamples** — a tenth of the full set — and two of
them (`77.9`/`82.3`/`88.5`, both in this file's prior revision and in
`EVIDENCE.md` row 7) were mislabeled as the "full 10-conversation" result,
which is the mislabeling bug this table corrects. `86.33` additionally has
**no raw per-question artifact checked into the repo** — it exists only as
prose in `CHANGELOG.md`'s v3.6.0 entry, which itself documents that the
prompt configuration it depended on regresses to 77.06 / 69.91 once the
now-recommended prompt features are enabled, and that it trails Mem0's 2026
managed-platform self-reported 91.6. It is therefore not reproducible against
the current pipeline and is not used as a comparison number anywhere in this
repo. `CHANGELOG.md` is left unedited (it is a dated log), but no other
document should restate `86.33` as a current, validated MIND-Mem score. See
`docs/locomo-v3.4-conv0-results.md` for the conv-0 run history.

### Comparison

The table below is historical context, **not a controlled like-for-like
comparison**: answer-model revisions, judge revisions and per-question token
budgets are not recorded for each row. Matching a metric and question count
alone does not establish an apples-to-apples comparison. MIND-Mem is
**not the top reported score** — Memobase and Letta report slightly higher. MIND-Mem's
differentiator on this benchmark is not "highest number in every cell"; it is
being the only system in the table that is simultaneously **local-only,
zero-core-dependency, and governed** (contradiction detection,
propose/review/apply audit trail, byte-identical deterministic replay) —
properties this benchmark does not measure at all.

| System | LoCoMo Acc>=50 (full 10-conv, 1986Q) | Infrastructure | Dependencies |
|---|---:|---|---|
| Memobase¹ | 75.8% | Cloud + GPU | embeddings + vector DB |
| Letta¹ | 74.0% | Cloud | embeddings + vector DB |
| **MIND-Mem** | **73.8%** | **Local-only** | **Zero core** |
| Full-context baseline¹ | 72.9% | N/A | LLM context window |
| Mem0 (own LoCoMo paper)² | 66.9% | Cloud (managed) | graph DB + embeddings |

¹ Third-party self-reported numbers (Letta's August 2025 analysis; see
`README.md` "Why Plain Files Outperform Fancy Retrieval"). Not re-run by
MIND-Mem — cited as published, on the same full-set LoCoMo protocol.

² `66.88` is Mem0's own published LoCoMo-paper number, on the open LoCoMo
protocol. Mem0's separate **2026 managed platform** self-reports **91.6** on
LoCoMo — that is a different setup/judge (their hosted product benchmark, not
the open-paper config) and is not apples-to-apples with any row in this
table. We surface both numbers rather than omit the higher one, per policy:
never publish a comparison a skeptic could catch as cherry-picked.

## Recall Latency

### Hybrid Recall (BM25 + Ollama GPU Embeddings)

292 blocks, RTX 3080 (10GB), `mxbai-embed-large` via Ollama:

| Operation | Latency |
|-----------|---------|
| Index (292 blocks, cold) | 3,020ms (one-time) |
| Vector search (warm) | 52-64ms |
| BM25 search (MIND kernel) | <10ms |

**Before** (ONNX BGE-large on CPU): 60-300s per query
**After** (Ollama mxbai-embed-large on GPU): 52-64ms per query — **1000-5000x speedup**

Both models fit on RTX 3080:
- mind-mem:4b Q4_K_M (2.6GB) — LLM extraction @ 104 tok/s gen / 1585 tok/s prefill
- mxbai-embed-large (769MB) — embeddings
- Total: 3.4GB / 10GB (6.6GB headroom for KV cache + concurrent models)

### CI benchmark (50 blocks)

- Average: <500ms
- P95: <500ms

## MIND Kernel Speedups

Compiled `.so` kernels vs pure Python (N=5000 blocks):

| Function | Python | MIND (.so) | Speedup |
|----------|--------|-----------|---------|
| bm25f_batch | 1.32 ms | 6.7 us | **197x** |
| weighted_rank | 189 us | 1.6 us | **119x** |
| rrf_fuse | 380 us | 5.3 us | **72x** |
| importance_batch | 2.38 ms | 52.4 us | **46x** |
| date_proximity | 574 us | 20.7 us | **28x** |
| **Overall** | **10.93 ms** | **249.5 us** | **47.6x** |

## mind-mem:4b LLM Extraction

Purpose-trained model ([star-ga/mind-mem-4b](https://huggingface.co/star-ga/mind-mem-4b)) — fully trained on STARGA-curated MIND-Mem corpus. On RTX 3080 (Q4_K_M GGUF, 2.6 GB VRAM):

**Raw throughput:**
- Generation: **104 tok/s**
- Prefill: **1585 tok/s**

**End-to-end task latency** (prefill + generate, typical inputs):

| Task | Latency | Notes |
|------|---------|-------|
| Entity extraction | ~3-4s | short input, ~100 tok output |
| Fact extraction | ~4-5s | medium input, ~200 tok output |
| Intent classification | ~2-3s | short input, ~20 tok output |
| Contradiction detection | ~5-7s | pairwise blocks, ~200 tok output |
| Observation compression | ~6-9s | 2K-4K transcript input |

The model produces structured JSON output for all 8 trained tasks. Suitable for both interactive `mm capture` workflows and async/background enrichment. Multi-backend dispatch (`extraction.backend` = `ollama` / `vllm` / `openai-compatible`) lets you scale horizontally if needed.

## How to Run

```bash
# Run CI benchmark
pytest tests/ -k "benchmark" --benchmark-only

# Run recall timing
make benchmark
```

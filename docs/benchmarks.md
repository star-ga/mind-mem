# Benchmarks

## LoCoMo Benchmark — canonical number

**This is the single canonical MIND-Mem LoCoMo result.** Every other LoCoMo
figure referenced anywhere in this repo (README.md, `docs/comparison.md`, the
HF model card, `mind-mem/CLAUDE.md`, `EVIDENCE.md`) must match this table or
link to it — do not restate a different number.

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

"Apples-to-apples" means the same Acc>=50 metric on the same full
10-conversation (1986-question) LoCoMo protocol. On that protocol MIND-Mem is
**not the top score** — Memobase and Letta report slightly higher. MIND-Mem's
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

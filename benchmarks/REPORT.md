# mind-mem Benchmark Report

**Date:** 2026-02-18

## Environment

| Component       | Value                                    |
|-----------------|------------------------------------------|
| Python          | 3.12.3                                   |
| OS              | Linux 6.17.0-14-generic (x86_64)         |
| SQLite          | system (FTS5 enabled)                    |
| Answerer model  | gpt-4o-mini                              |
| Judge model     | gpt-4o-mini                              |
| Temperature     | 0.0                                      |
| Top-k retrieval | 10                                       |
| Dataset         | LoCoMo (10 conversations, 1986 QA pairs) |
| Dataset cache   | `benchmarks/.cache/locomo10.json`        |

## Reproduction

```bash
pip install -e .
python benchmarks/locomo_judge.py \
  --answerer gpt-4o-mini \
  --judge gpt-4o-mini \
  --top-k 10 \
  --output benchmarks/results.json
```

Requires `OPENAI_API_KEY` in environment.

## Overall Results

| Metric      | Score     |
|-------------|-----------|
| **Acc>=50** | **67.3%** |
| Mean Score  | **61.4**  |
| Acc>=75     | **48.8%** |

## Per-Category (Acc>=50)

| Category    | N        | Acc>=50   | Mean     |
|-------------|----------|-----------|----------|
| **Overall** | **1986** | **67.3%** | **61.4** |
| open-domain | 841      | 86.6%     | 78.3     |
| temporal    | 96       | 78.1%     | 65.7     |
| single-hop  | 282      | 68.8%     | 59.1     |
| multi-hop   | 321      | 55.5%     | 48.4     |
| adversarial | 446      | 36.3%     | 39.5     |

## Per-Conversation Breakdown

| Conv    | Sample  | N        | Mean     | Acc>=50   |
|---------|---------|----------|----------|-----------|
| 0       | conv-40 | 199      | 64.2     | 74.9%     |
| 1       | conv-41 | 105      | 61.8     | 67.6%     |
| 2       | conv-42 | 193      | 64.4     | 71.0%     |
| 3       | conv-26 | 260      | 58.2     | 60.8%     |
| 4       | conv-43 | 242      | 60.9     | 66.5%     |
| 5       | conv-44 | 158      | 63.8     | 70.9%     |
| 6       | conv-47 | 190      | 55.0     | 54.7%     |
| 7       | conv-48 | 239      | 63.1     | 71.1%     |
| 8       | conv-49 | 196      | 62.6     | 69.9%     |
| 9       | conv-50 | 204      | 61.7     | 68.1%     |
| **ALL** |         | **1986** | **61.4** | **67.3%** |

## Competitive Landscape

| System       |     Score | Approach                                |
|--------------|----------:|-----------------------------------------|
| Memobase     |     75.8% | Specialized extraction                  |
| Letta        |     74.0% | Files + agent tool use                  |
| Mem0         |     68.5% | Graph + LLM extraction                  |
| **mind-mem** | **67.3%** | Deterministic BM25 + rule-based packing |

> mind-mem reaches **98%** of Mem0's score with pure deterministic retrieval — no embeddings, no vector DB, no cloud calls, no LLM in the retrieval loop.

## Architecture

Pure deterministic retrieval pipeline — no vector DB, no embeddings required.

1. **Ingestion:** Session-aware chunking with speaker labels and timestamps
2. **Index:** SQLite FTS5 (Porter stemmer) + in-memory BM25 scan fallback
3. **Query processing:**
   - Intent routing (9 types: WHY, WHEN, ENTITY, WHAT, HOW, LIST, VERIFY, COMPARE, TRACE)
   - RM3 pseudo-relevance feedback expansion
   - Morphological normalization (irregular verbs, month names)
   - Controlled synonym expansion (full or morph-only based on query type)
4. **Retrieval:** Wide candidate pool (top-200) with deterministic rerank
   - Speaker-match boost
   - Time-proximity signal (Gaussian decay)
   - Entity overlap scoring
   - Bigram coherence
   - Recency decay
   - Negation awareness
   - Category taxonomy matching (20 categories)
5. **Fusion:** RRF fusion when hybrid mode is enabled (BM25 + vector)
6. **Context packing** (append-only post-retrieval):
   - Rule 1: Dialog adjacency (question-answer pairs)
   - Rule 2: Multi-entity diversity enforcement
   - Rule 3: Pronoun rescue (antecedent recovery)
7. **Evidence packing:** Adversarial-aware formatting with abstention classifier

## LongMemEval (ICLR 2025, 470 questions)

| Category         |       N |      R@1 |      R@5 |     R@10 |      MRR |
|------------------|--------:|---------:|---------:|---------:|---------:|
| **Overall**      | **470** | **73.2** | **85.3** | **88.1** | **.784** |
| Multi-session    |     121 |     83.5 |     95.9 |     95.9 |     .885 |
| Temporal         |     127 |     76.4 |     91.3 |     92.9 |     .826 |
| Knowledge update |      72 |     80.6 |     88.9 |     91.7 |     .844 |
| Single-session   |      56 |     82.1 |     89.3 |     89.3 |     .847 |

## MIND Kernel Benchmark

Compiled C99 kernels (`gcc -O3 -march=native`) vs pure Python list comprehensions.
Pre-allocated ctypes arrays; timing measures native function call only (no marshaling).

### Per-Function Speedup

| Function         |  N=100 |  N=500 | N=1000 | N=5000 | N=10000 |
|------------------|-------:|-------:|-------:|-------:|--------:|
| rrf_fuse         |  9.2x  |  30.3x |  44.2x |  71.9x |   78.8x |
| bm25f_batch      |  26.3x |  51.4x |  88.9x | 192.4x |  231.4x |
| negation_penalty |  3.4x  |  11.4x |  18.0x |  16.0x |   12.1x |
| date_proximity   |  8.4x  |  19.0x |  22.4x |  26.6x |   27.2x |
| category_boost   |  3.4x  |  11.8x |  19.0x |  16.7x |   12.9x |
| importance_batch |  21.7x |  38.0x |  42.3x |  47.1x |   49.0x |
| confidence_score |  0.8x  |   0.8x |   0.8x |   0.8x |    0.8x |
| top_k_mask       |  3.0x  |   6.1x |   8.2x |  11.0x |   12.2x |
| weighted_rank    |  5.6x  |  25.1x |  45.1x | 116.3x |   98.8x |

### Overall

| Metric              | Value        |
|---------------------|--------------|
| Total Python time   | 27.94 ms     |
| Total MIND time     | 551.6 μs     |
| **Overall speedup** | **50.7x**    |
| Iterations          | 500          |
| Compiler            | gcc -O3 -march=native |

**Notes:**
- `confidence_score` is scalar (5 features) — ctypes call overhead exceeds the computation
- `entity_overlap` uses set operations — not benchmarked via C kernel
- `bm25f_batch` shows the largest speedup (231x) due to Python's per-element `math.log()` overhead
- Speedups grow with N: Python's interpreter overhead is O(N), C benefits from SIMD/ILP

### Reproduction

```bash
gcc -O3 -march=native -shared -fPIC -o lib/libmindmem.so lib/kernels.c -lm
python benchmarks/bench_kernels.py --iterations 500 --sizes 100,500,1000,5000,10000
```

## Result Files

| File                                        | Description                          |
|---------------------------------------------|--------------------------------------|
| `benchmarks/locomo_judge_results_v9_1.json` | Full 10-conv results (1986 QA pairs) |
| `benchmarks/locomo_judge_results_v3.json`   | Baseline comparison                  |

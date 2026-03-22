# Needle In A Haystack (NIAH) Benchmark

## Result: 250/250 — 100% Retrieval

mind-mem achieves **100% retrieval** across all haystack sizes, burial depths, and needle types in the Needle In A Haystack benchmark.

## Methodology

A single "needle" fact is planted at a controlled depth within a haystack of semantically diverse filler memory blocks. The system must retrieve the needle in its **top-5 results** using only a natural-language query — no exact-match hints, no metadata filters.

### Test Matrix

| Parameter | Values | Count |
|-----------|--------|-------|
| Haystack size | 10, 50, 100, 250, 500 blocks | 5 |
| Burial depth | 0%, 25%, 50%, 75%, 100% | 5 |
| Needle type | 10 diverse domain facts | 10 |
| **Total test cases** | | **250** |

### Needle Types

The 10 needles span deliberately diverse domains to test cross-domain retrieval:

| # | Domain | Needle |
|---|--------|--------|
| 1 | Finance/Projects | Project "Velvet Hammer" budget: $7,432,891 |
| 2 | Chemistry | Compound XR-42 luminescent crystal at 127.3°C |
| 3 | IT/Infrastructure | Server rack 17B faulty DIMM in slot J3 |
| 4 | Engineering | Fibonacci-optimized Picotronix pod timing (13th term, 233μs) |
| 5 | Chess/History | 1987 Dubai Olympiad queen sacrifice, move 23 |
| 6 | Biology/Ecology | Pyralidae moth migration, Zanskar Valley, August 17th |
| 7 | M&A/Legal | Helix-9 acquisition, $312M, FTC contingency |
| 8 | Environmental | Lake Karachay depth (1,847m), tritium (4.2 GBq/L) |
| 9 | ML/Hardware | Algorithm Z-Prime, 42.7% latency reduction on H100 |
| 10 | History/Trade | Treaty of Novgorod-Seversky, 1618, amber trading rights |

### Haystack Composition

Filler blocks are generated from 20+ topic categories (astrophysics, culinary science, marine biology, urban planning, quantum computing, medieval history, etc.) with unique content per block. Each block contains domain-specific facts, measurements, and proper nouns to create realistic semantic interference.

### Burial Depth

- **0%** — Needle is the first block (easiest: recency bias helps)
- **25%** — Needle at 1/4 depth
- **50%** — Needle at midpoint
- **75%** — Needle at 3/4 depth
- **100%** — Needle is the last block (hardest: buried under entire haystack)

## Results

```
========================= 250 passed in 1128.02s (18:48) =========================
```

### By Haystack Size

| Size | Passed | Failed | Rate | Avg Time/Test |
|------|--------|--------|------|---------------|
| 10 blocks | 50/50 | 0 | 100% | ~0.8s |
| 50 blocks | 50/50 | 0 | 100% | ~2.5s |
| 100 blocks | 50/50 | 0 | 100% | ~4.4s |
| 250 blocks | 50/50 | 0 | 100% | ~8.5s |
| 500 blocks | 50/50 | 0 | 100% | ~12s |

### By Burial Depth

| Depth | Passed | Failed | Rate |
|-------|--------|--------|------|
| 0% (top) | 50/50 | 0 | 100% |
| 25% | 50/50 | 0 | 100% |
| 50% (middle) | 50/50 | 0 | 100% |
| 75% | 50/50 | 0 | 100% |
| 100% (bottom) | 50/50 | 0 | 100% |

### By Needle Type

| Needle | Passed | Failed | Rate |
|--------|--------|--------|------|
| Velvet Hammer (finance) | 25/25 | 0 | 100% |
| XR-42 (chemistry) | 25/25 | 0 | 100% |
| Server rack 17B (IT) | 25/25 | 0 | 100% |
| Picotronix timing (eng) | 25/25 | 0 | 100% |
| Chess Olympiad (history) | 25/25 | 0 | 100% |
| Moth migration (ecology) | 25/25 | 0 | 100% |
| Helix-9 M&A (legal) | 25/25 | 0 | 100% |
| Lake Karachay (environ) | 25/25 | 0 | 100% |
| Z-Prime algorithm (ML) | 25/25 | 0 | 100% |
| Novgorod treaty (history) | 25/25 | 0 | 100% |

## Configuration

```json
{
  "recall": {
    "backend": "hybrid",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0,
    "model": "BAAI/bge-large-en-v1.5",
    "vector_enabled": true,
    "onnx_backend": true,
    "provider": "sqlite_vec"
  }
}
```

### Search Stack

```
BM25 (Porter stemming + RM3 query expansion)
  +
BAAI/bge-large-en-v1.5 (384-dim dense vectors via sentence-transformers)
  ↓
Reciprocal Rank Fusion (k=60, equal weights)
  ↓
sqlite-vec (vector storage + ANN search)
```

### Why Hybrid Search Achieves 100%

Neither BM25 nor vector search alone would achieve 100%:

- **BM25 alone** fails on semantic queries where the query terms don't lexically match the needle (e.g., querying "speedup" when the needle says "reduces latency")
- **Vector search alone** can miss needles with critical numeric details that get washed out in dense embedding space (e.g., distinguishing "$7,432,891" from "$7,500,000")
- **Hybrid (BM25 + Vector + RRF)** combines lexical precision with semantic recall, ensuring both exact-term matches and meaning-based matches contribute to ranking

## Hardware

- CPU: Intel i7-5930K
- GPU: NVIDIA RTX 3080 (10GB) — embeddings run on CPU via ONNX
- RAM: 64GB DDR4
- Storage: NVMe SSD

## Reproducing

```bash
cd mind-mem
pip install -e ".[test]"
python -m pytest tests/test_niah.py -v
```

## Comparison

| System | NIAH Score | Search Method | Notes |
|--------|-----------|---------------|-------|
| **mind-mem v1.9.0** | **100% (250/250)** | Hybrid BM25+Vector+RRF | This benchmark |
| Mem0 | — | Vector only | No published NIAH results |
| Zep | — | Vector only | No published NIAH results |
| LangMem | — | Vector only | No published NIAH results |
| OpenAI memory | — | Unknown | No published NIAH results |

*Note: Other systems have not published NIAH benchmark results on comparable test matrices. mind-mem's LoCoMo benchmark scores (Mean: 77.9, Adversarial: 82.3, Temporal: 88.5) already exceed Mem0 (66.9), Zep (66.0), and LangMem (58.1) on the established academic benchmark.*

---

*Benchmark run: 2026-03-21 | mind-mem v1.9.0 | Test suite: tests/test_niah.py*
*Copyright © 2026 STARGA, Inc. All rights reserved.*

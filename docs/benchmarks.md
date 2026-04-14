# Benchmarks

## LoCoMo Benchmark Results

Latest checked-in benchmark snapshot: mind-mem v1.9.0, evaluated with Mistral Large (LoCoMo, 10 conversations). The current `1.9.1` release is a maintenance update and does not include a new benchmark artifact:

| Metric | Score |
|--------|-------|
| Mean | 77.9 |
| Adversarial | 82.3 |
| Temporal | 88.5 |

### Comparison

| System | Mean | Adversarial | Temporal |
|--------|------|-------------|----------|
| **mind-mem** | **77.9** | **82.3** | **88.5** |
| Full context | 72.90 | - | - |
| Mem0 | 66.88 | - | - |
| Zep | 65.99 | - | - |
| LangMem | 58.10 | - | - |

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
- mind-mem:7b (7.3GB) — LLM extraction
- mxbai-embed-large (769MB) — embeddings
- Total: 8.1GB / 10GB

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

## Mind-Mem:7B LLM Extraction

Purpose-trained model ([star-ga/mind-mem-4b](https://huggingface.co/star-ga/mind-mem-4b)) on RTX 3080 (Q4_K_M GGUF):

| Task | Latency |
|------|---------|
| Entity extraction | ~18s |
| Fact extraction | ~21s |
| Intent classification | ~21s |
| Contradiction detection | ~24s |

The model produces structured JSON output for all 8 trained tasks. Best used as async/background enrichment rather than in the real-time recall path.

## How to Run

```bash
# Run CI benchmark
pytest tests/ -k "benchmark" --benchmark-only

# Run recall timing
make benchmark
```

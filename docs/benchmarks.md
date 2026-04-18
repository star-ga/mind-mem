# Benchmarks

## LoCoMo Benchmark Results

Latest checked-in benchmark snapshot from the v1.9.0 evaluation with Mistral Large (LoCoMo, 10 conversations). Current release (`3.1.1`) is feature-additive and does not invalidate these scores; a fresh benchmark artifact against v3.x is planned for the next release cycle:

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

Purpose-trained model ([star-ga/mind-mem-4b](https://huggingface.co/star-ga/mind-mem-4b)) — full fine-tune of Qwen3.5-4B on STARGA-curated mind-mem corpus. On RTX 3080 (Q4_K_M GGUF, 2.6 GB VRAM):

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

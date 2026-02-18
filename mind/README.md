# MIND Kernels

Numerical hot paths for mind-mem, written in the [MIND programming language](https://mindlang.dev).

The MIND kernel is **optional**. mind-mem works without it (pure Python fallback). With it, scoring runs at native speed with compile-time tensor shape verification.

## Compilation

Requires the MIND compiler (`mindc`). See [mindlang.dev](https://mindlang.dev) for installation.

```bash
# Compile all kernels to a single shared library
mindc mind/bm25.mind mind/rrf.mind mind/reranker.mind mind/abstention.mind \
      mind/ranking.mind mind/importance.mind \
      --emit=shared -o lib/libmindmem.so

# Or compile individually for testing
mindc mind/bm25.mind --emit=shared -o lib/libbm25.so
mindc mind/rrf.mind --emit=shared -o lib/librrf.so
```

## Kernels

| File | Functions | Purpose |
|------|-----------|---------|
| `bm25.mind` | `bm25f_doc`, `bm25f_batch`, `apply_recency`, `apply_graph_boost` | BM25F scoring with field boosts |
| `rrf.mind` | `rrf_fuse`, `rrf_fuse_three` | Reciprocal Rank Fusion |
| `reranker.mind` | `date_proximity_score`, `category_boost`, `negation_penalty`, `rerank_deterministic` | Deterministic reranking |
| `abstention.mind` | `entity_overlap`, `confidence_score` | Confidence gating |
| `ranking.mind` | `weighted_rank`, `top_k_mask` | Evidence ranking |
| `importance.mind` | `importance_score` | A-MEM importance scoring |

## FFI

The compiled `.so` exposes a C99-compatible ABI. Python calls via `ctypes` through `scripts/mind_ffi.py`. Each function accepts and returns flat float arrays.

## Without MIND

If `lib/libmindmem.so` is not present, mind-mem uses pure Python implementations. The Python fallback produces identical results (within f32 epsilon). No functionality is lost.

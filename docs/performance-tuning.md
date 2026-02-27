# Performance Tuning

Guide to optimizing mind-mem for large workspaces and high-throughput usage.

## BM25 Parameters

The BM25F scoring constants are defined in `src/mind_mem/_recall_constants.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K1` | 1.2 | Term frequency saturation |
| `B` | 0.75 | Document length normalization |
| `BOOST_TITLE` | 2.0 | Weight multiplier for title matches |
| `BOOST_TAGS` | 1.5 | Weight multiplier for tag matches |

### When to Tune

- **Increase K1** (1.5-2.0) if recall misses relevant blocks with rare terms
- **Decrease B** (0.3-0.5) if short blocks are unfairly penalized
- **Increase BOOST_TITLE** if block IDs/titles should dominate results

## Index Configuration

### FTS5 Index

The FTS5 index is built automatically on first recall. To force rebuild:

```python
from mind_mem.sqlite_index import rebuild_fts_index
rebuild_fts_index(workspace_path)
```

### Vector Index

Optional. Requires `sentence-transformers`:

```python
from mind_mem.recall_vector import build_vector_index
build_vector_index(workspace_path, model="all-MiniLM-L6-v2")
```

## Memory Management

### Workspace Size Guidelines

| Blocks | RAM Usage | Recall Latency |
|--------|-----------|----------------|
| <1K | ~10MB | <50ms |
| 1K-10K | ~50MB | <200ms |
| 10K-50K | ~200MB | <500ms |
| >50K | ~500MB+ | Consider compaction |

### Compaction

Merge old blocks to reduce workspace size:

```bash
python3 -m mind_mem.compaction /path/to/workspace --older-than 90
```

## Caching

### Embedding Cache

Embeddings are cached in `.mind-mem/embeddings.db`. Cache hit rate is logged
at DEBUG level. To clear:

```bash
rm /path/to/workspace/.mind-mem/embeddings.db
```

### FTS Cache

FTS queries are cached in-memory per session. No persistent cache to manage.

## Monitoring

### Key Metrics

- `recall_latency_ms` — p50 should be <100ms, p95 <500ms
- `fts_cache_hits` / `fts_cache_misses` — aim for >80% hit rate
- `blocks_total` — monitor growth, compact when >50K
- `index_staleness_s` — reindex if >3600s behind

### Structured Logging

Enable debug logging for performance investigation:

```python
import logging
logging.getLogger("mind-mem").setLevel(logging.DEBUG)
```

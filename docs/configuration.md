# Configuration Reference

mind-mem is configured via `mind-mem.json` in your workspace root. This file is created automatically by `init_workspace.py`.

## Full Configuration

```json
{
  "version": "1.0.0",
  "workspace_path": ".",
  "auto_capture": true,
  "auto_recall": true,
  "governance_mode": "detect_only",
  "recall": {
    "backend": "scan",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0,
    "vector_model": "all-MiniLM-L6-v2",
    "vector_enabled": false,
    "onnx_backend": false,
    "rm3": {
      "enabled": false,
      "alpha": 0.6,
      "fb_terms": 10,
      "fb_docs": 5,
      "min_idf": 1.0
    },
    "cross_encoder": {
      "enabled": false,
      "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "blend_weight": 0.6
    }
  },
  "proposal_budget": {
    "per_run": 3,
    "per_day": 6,
    "backlog_limit": 30
  },
  "compaction": {
    "archive_days": 90,
    "snapshot_days": 30,
    "log_days": 180,
    "signal_days": 60
  },
  "scan_schedule": "daily"
}
```

## General Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `version` | string | `"1.0.0"` | Config schema version |
| `workspace_path` | string | `"."` | Path to workspace root |
| `auto_capture` | bool | `true` | Run capture engine on session end hook |
| `auto_recall` | bool | `true` | Show recall context on session start hook |
| `governance_mode` | string | `"detect_only"` | One of: `detect_only`, `propose`, `enforce` |
| `scan_schedule` | string | `"daily"` | `"daily"` or `"manual"` |

## Recall Settings

### Backend Selection

| `recall.backend` | Description |
|---|---|
| `"scan"` | BM25 full-text search only (default, zero dependencies) |
| `"hybrid"` | BM25 + Vector search with RRF fusion (requires embeddings) |
| `"vector"` | Vector-only search (requires embeddings) |

### RRF Fusion

| Key | Type | Default | Description |
|---|---|---|---|
| `recall.rrf_k` | int | `60` | RRF smoothing parameter. Higher = more weight to lower-ranked results |
| `recall.bm25_weight` | float | `1.0` | BM25 signal weight in RRF fusion |
| `recall.vector_weight` | float | `1.0` | Vector signal weight in RRF fusion |

### Vector Search

| Key | Type | Default | Description |
|---|---|---|---|
| `recall.vector_enabled` | bool | `false` | Enable vector search |
| `recall.vector_model` | string | `"all-MiniLM-L6-v2"` | Embedding model name |
| `recall.onnx_backend` | bool | `false` | Use ONNX for local inference (no server) |

### RM3 Query Expansion

| Key | Type | Default | Description |
|---|---|---|---|
| `recall.rm3.enabled` | bool | `false` | Enable RM3 pseudo-relevance feedback |
| `recall.rm3.alpha` | float | `0.6` | Interpolation weight (1.0 = original query only) |
| `recall.rm3.fb_terms` | int | `10` | Number of expansion terms from feedback docs |
| `recall.rm3.fb_docs` | int | `5` | Number of feedback documents |
| `recall.rm3.min_idf` | float | `1.0` | Minimum IDF for expansion terms |

### Cross-Encoder

| Key | Type | Default | Description |
|---|---|---|---|
| `recall.cross_encoder.enabled` | bool | `false` | Enable cross-encoder reranking |
| `recall.cross_encoder.model` | string | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Model name |
| `recall.cross_encoder.blend_weight` | float | `0.6` | Blend factor (0.6 = 60% CE + 40% original) |

## Proposal Budget

Controls how many proposals the system generates to prevent overload.

| Key | Type | Default | Description |
|---|---|---|---|
| `proposal_budget.per_run` | int | `3` | Max proposals per scan run |
| `proposal_budget.per_day` | int | `6` | Max proposals per calendar day |
| `proposal_budget.backlog_limit` | int | `30` | Pause generation when pending proposals exceed this |

## Compaction

Controls automated workspace maintenance (archival, cleanup).

| Key | Type | Default | Description |
|---|---|---|---|
| `compaction.archive_days` | int | `90` | Archive completed blocks older than N days |
| `compaction.snapshot_days` | int | `30` | Remove apply snapshots older than N days |
| `compaction.log_days` | int | `180` | Archive daily logs older than N days |
| `compaction.signal_days` | int | `60` | Remove resolved/rejected signals older than N days |

## Environment Variables

| Variable | Description |
|---|---|
| `MIND_MEM_WORKSPACE` | Workspace path (overrides config file) |
| `MIND_MEM_LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MIND_MEM_LIB` | Path to compiled MIND library (overrides default search) |
| `MIND_MEM_TOKEN` | Bearer token for HTTP MCP transport |

## Example Configurations

### Minimal (BM25 only, zero deps)

```json
{
  "version": "1.0.0",
  "governance_mode": "detect_only",
  "recall": { "backend": "scan" }
}
```

### Hybrid Search with RM3

```json
{
  "version": "1.0.0",
  "governance_mode": "propose",
  "recall": {
    "backend": "hybrid",
    "vector_enabled": true,
    "vector_model": "all-MiniLM-L6-v2",
    "onnx_backend": true,
    "rrf_k": 60,
    "rm3": { "enabled": true, "alpha": 0.6 }
  }
}
```

### Full ML Pipeline

```json
{
  "version": "1.0.0",
  "governance_mode": "enforce",
  "recall": {
    "backend": "hybrid",
    "vector_enabled": true,
    "onnx_backend": true,
    "rrf_k": 60,
    "rm3": { "enabled": true },
    "cross_encoder": { "enabled": true, "blend_weight": 0.6 }
  }
}
```

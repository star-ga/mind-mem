# Configuration Reference

mind-mem is configured via `mind-mem.json` in your workspace root. This file is created automatically by `init_workspace.py` with sensible defaults. All keys are optional -- missing keys fall back to their documented defaults.

---

## Full Configuration

```json
{
  "version": "1.0.5",
  "schema_version": "2.1.0",
  "workspace_path": ".",
  "auto_capture": true,
  "auto_recall": true,
  "governance_mode": "detect_only",
  "scan_schedule": "daily",
  "recall": {
    "backend": "scan",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0,
    "vector_enabled": false,
    "vector_model": "all-MiniLM-L6-v2",
    "onnx_backend": false,
    "provider": "local",
    "model": "all-MiniLM-L6-v2",
    "index_path": ".mind-mem-vectors",
    "dimension": null,
    "qdrant_url": "http://localhost:6333",
    "qdrant_collection": "mind-mem",
    "pinecone_api_key": "",
    "pinecone_environment": "",
    "pinecone_index": "mind-mem",
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
  "categories": {
    "enabled": true,
    "extra_categories": {}
  },
  "prompts": {
    "observation_compress": "",
    "entity_extract": "",
    "category_distill": ""
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
  "auto_ingest": {
    "enabled": false,
    "transcript_scan": true,
    "entity_ingest": true,
    "intel_scan": true
  }
}
```

---

## General Settings

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `version` | string | `"1.0.5"` | Config file version. Set automatically by `init_workspace.py`. |
| `schema_version` | string | `"2.1.0"` | Workspace schema version. Used by `schema_version.py` for migrations. Falls back to `version` if absent. |
| `workspace_path` | string | `"."` | Workspace root directory. Relative paths are resolved from the config file location. |
| `auto_capture` | bool | `true` | Run the capture engine automatically on session-end hooks. When `false`, the session-end hook exits without capturing signals. |
| `auto_recall` | bool | `true` | Show recall context automatically on session-start hooks. |
| `governance_mode` | string | `"detect_only"` | Controls how the intelligence scan handles findings. See Governance Modes below. |
| `scan_schedule` | string | `"daily"` | How often the intel scan runs. Valid values: `"daily"`, `"manual"`. |

### Governance Modes

| Mode | Behavior |
| --- | --- |
| `detect_only` | Detect contradictions and drift but take no action. Findings are written to intelligence reports only. |
| `propose` | Detect findings and generate fix proposals in `intelligence/proposed/`. Proposals require explicit human approval via `approve_apply`. |
| `enforce` | Detect findings, generate proposals, and auto-apply approved proposals. Use with caution. |

---

## Recall Settings

The `recall` section controls the search and retrieval engine. All keys are nested under `"recall"` in the config.

### Backend Selection

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.backend` | string | `"bm25"` | Recall backend to use. Determines the primary search strategy. See Backend Values below. |

When `init_workspace.py` creates the config, it sets `recall.backend` to `"bm25"`. The MCP server and CLI resolve the backend at query time: CLI flag takes precedence over config, which takes precedence over the default.

| Backend Value | Description | Dependencies |
| --- | --- | --- |
| `"scan"` / `"tfidf"` / `"bm25"` | In-memory BM25 full-text search. O(corpus) per query. Zero external dependencies. | None |
| `"sqlite"` | SQLite FTS5 index. O(log N) per query. Requires running `reindex` first to build the index. | None (stdlib sqlite3) |
| `"hybrid"` | BM25 + Vector search with Reciprocal Rank Fusion. Falls back to BM25-only if vector dependencies are missing. | `sentence-transformers` |
| `"vector"` | Vector-only semantic search. Requires embedding model. | `sentence-transformers` |

### RRF Fusion

These settings control Reciprocal Rank Fusion when `backend` is `"hybrid"`. RRF merges the ranked result lists from BM25 and vector search.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.rrf_k` | int | `60` | RRF smoothing constant. Higher values give more weight to lower-ranked results, reducing the dominance of top positions. |
| `recall.bm25_weight` | float | `1.0` | Weight multiplier for the BM25 result list in RRF fusion. |
| `recall.vector_weight` | float | `1.0` | Weight multiplier for the vector result list in RRF fusion. |

### Vector Search

These settings control the vector/embedding-based search component.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.vector_enabled` | bool | `false` | Enable the vector search backend. When `false`, hybrid mode falls back to BM25-only. |
| `recall.vector_model` | string | `"all-MiniLM-L6-v2"` | Embedding model name for the hybrid backend. Used by `HybridBackend`. |
| `recall.onnx_backend` | bool | `false` | Use ONNX runtime for local embedding inference instead of PyTorch. |

### Vector Provider Settings

When `recall.backend` is `"vector"`, these additional keys configure the vector provider. All are read from the `recall` section.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.provider` | string | `"local"` | Vector storage provider. Valid values: `"local"` (JSON file index), `"qdrant"` (Qdrant server), `"pinecone"` (Pinecone cloud). |
| `recall.model` | string | `"all-MiniLM-L6-v2"` | Embedding model name for the vector backend. |
| `recall.index_path` | string | `".mind-mem-vectors"` | Directory for the local vector index files. Relative to workspace root. |
| `recall.dimension` | int | `null` | Embedding dimension. Auto-detected from the model if not set. |

#### Qdrant Provider

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.qdrant_url` | string | `"http://localhost:6333"` | Qdrant server URL. |
| `recall.qdrant_collection` | string | `"mind-mem"` | Qdrant collection name. |

Requires: `pip install qdrant-client`

#### Pinecone Provider

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.pinecone_api_key` | string | (none) | Pinecone API key. Required when provider is `"pinecone"`. Can also be set via `PINECONE_API_KEY` environment variable. |
| `recall.pinecone_environment` | string | (none) | Pinecone environment (e.g., `"us-east-1-aws"`). Required when provider is `"pinecone"`. |
| `recall.pinecone_index` | string | `"mind-mem"` | Pinecone index name. |

Requires: `pip install pinecone` (v3+)

### RM3 Query Expansion

RM3 (Relevance Model 3) performs pseudo-relevance feedback to expand the original query with terms from top-ranked results. Skipped automatically for adversarial queries.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.rm3.enabled` | bool | `false` | Enable RM3 dynamic query expansion. |
| `recall.rm3.alpha` | float | `0.6` | Interpolation weight between original query and expansion terms. `1.0` = original query only; `0.0` = expansion terms only. |
| `recall.rm3.fb_terms` | int | `10` | Number of expansion terms to extract from feedback documents. |
| `recall.rm3.fb_docs` | int | `5` | Number of top-ranked feedback documents to analyze. |
| `recall.rm3.min_idf` | float | `1.0` | Minimum IDF threshold for expansion terms. Filters out overly common terms. |

### Cross-Encoder Reranking

Optional neural reranking stage that rescores BM25 results using a cross-encoder model.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.cross_encoder.enabled` | bool | `false` | Enable cross-encoder reranking. |
| `recall.cross_encoder.model` | string | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Hugging Face model name for cross-encoder scoring. |
| `recall.cross_encoder.blend_weight` | float | `0.6` | Blend factor between cross-encoder and original BM25 scores. `0.6` = 60% cross-encoder + 40% BM25. |

---

## Categories

Controls the category distiller, which auto-generates thematic summary files from memory blocks.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `categories.enabled` | bool | `true` | Enable category distillation. When `true`, the `reindex` tool regenerates category summaries. |
| `categories.extra_categories` | object | `{}` | Custom category definitions. Maps category name to a list of keyword strings. If a category name matches a built-in category, the keywords are appended. Otherwise a new category is created. |

### Extra Categories Example

```json
{
  "categories": {
    "enabled": true,
    "extra_categories": {
      "billing": ["invoice", "payment", "stripe", "subscription"],
      "deployment": ["deploy", "rollout", "release", "ci/cd"]
    }
  }
}
```

The built-in categories (infrastructure, security, database, api, testing, etc.) are defined in `category_distiller.py` and are always available. Extra categories extend or supplement them.

---

## Prompts

Reserved prompt override slots for future LLM-powered compression and extraction steps. Currently these keys are placeholders defined in `mind-mem.example.json`. When non-empty, they override the default system prompts used by the corresponding module.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `prompts.observation_compress` | string | `""` | Custom system prompt for the observation compression layer (`observation_compress.py`). When empty, the built-in `COMPRESS_SYSTEM_PROMPT` is used. |
| `prompts.entity_extract` | string | `""` | Custom system prompt for entity extraction. |
| `prompts.category_distill` | string | `""` | Custom system prompt for category distillation. |

---

## Proposal Budget

Controls how many proposals the intelligence scan generates, preventing overload of the review queue.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `proposal_budget.per_run` | int | `3` | Maximum proposals generated per scan run. |
| `proposal_budget.per_day` | int | `6` | Maximum proposals generated per calendar day. Resets at midnight. |
| `proposal_budget.backlog_limit` | int | `30` | Pause all proposal generation when the number of pending (unapproved) proposals in `intelligence/proposed/` exceeds this limit. |

---

## Compaction

Controls automated workspace maintenance -- archiving old blocks, removing expired snapshots, and cleaning up stale logs and signals.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `compaction.archive_days` | int | `90` | Archive completed/canceled task blocks and superseded/revoked decision blocks older than this many days. Blocks are moved to `*_ARCHIVE.md` files, not deleted. |
| `compaction.snapshot_days` | int | `30` | Remove apply-engine snapshots (from `intelligence/state/snapshots/`) older than this many days. Snapshots can be recreated from git history. |
| `compaction.log_days` | int | `180` | Archive daily summary logs older than this many days. |
| `compaction.signal_days` | int | `60` | Remove resolved or rejected signals from `intelligence/SIGNALS.md` older than this many days. |

---

## Auto-Ingest

Controls the automated ingestion pipeline managed by `cron_runner.py`. When enabled, periodic jobs scan transcripts, extract entities, and run intelligence scans.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `auto_ingest.enabled` | bool | `false` | Master toggle for the auto-ingest pipeline. When `false`, all periodic jobs are disabled regardless of individual toggles. Checked by the session-end hook. |
| `auto_ingest.transcript_scan` | bool | `true` | Enable the transcript scan job (`transcript_capture.py --scan-recent`). Scans recent transcripts for signals. Default schedule: every 6 hours. |
| `auto_ingest.entity_ingest` | bool | `true` | Enable the entity ingestion job (`entity_ingest.py`). Extracts entities (projects, tools, people) from signals and logs. Default schedule: daily at 3am. |
| `auto_ingest.intel_scan` | bool | `true` | Enable the intelligence scan job (`intel_scan.py`). Runs contradiction detection, drift analysis, and briefing generation. Default schedule: daily at 3am. |

Individual job toggles are only checked when `auto_ingest.enabled` is `true`. When the master toggle is off, no jobs run.

---

## Environment Variables

Environment variables take precedence over config file values where applicable.

| Variable | Description |
| --- | --- |
| `MIND_MEM_WORKSPACE` | Workspace path. Used by the MCP server, hooks, and scripts to locate the workspace. Overrides `workspace_path` in config. Falls back to `"."` (current directory). |
| `MIND_MEM_TOKEN` | Bearer token for HTTP MCP transport authentication. When set, all HTTP requests must include `Authorization: Bearer <token>` or `X-MindMem-Token: <token>`. Not required for stdio transport. Can also be passed via `--token` CLI flag. |
| `MIND_MEM_LOG_LEVEL` | Logging level for structured JSON logging. Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default: `INFO`. |
| `MIND_MEM_LIB` | Absolute path to the compiled MIND kernel shared library (`libmindmem.so` / `libmindmem.dylib`). Overrides the default search paths. Must point to a file within the `lib/` directory of the mind-mem installation for security. |
| `MIND_MEM_HOME` | Path to the mind-mem installation directory. Used by the OpenClaw hook (`handler.js`) to locate scripts when they are not co-located with the workspace. |

---

## MIND Kernel Configuration

In addition to `mind-mem.json`, mind-mem supports `.mind` kernel files -- INI-style configuration files in the `mind/` directory of the workspace. These provide fine-grained tuning for the recall pipeline.

Kernel files are loaded by `mind_ffi.py` and override BM25 parameters and field weights at query time.

### Kernel File Format

```ini
[bm25]
k1 = 1.5
b = 0.8

[fields]
Statement = 3.0
Tags = 2.0
Title = 2.5
```

### Available Kernels

Kernel files are discovered from `<workspace>/mind/` and listed via the `list_mind_kernels` MCP tool. Common kernel names include:

| Kernel | Purpose |
| --- | --- |
| `recall.mind` | BM25 parameters (`k1`, `b`) and field weight overrides |
| `rm3.mind` | RM3 expansion parameters |
| `rerank.mind` | Reranking configuration |
| `temporal.mind` | Temporal scoring adjustments |
| `adversarial.mind` | Adversarial query handling |
| `hybrid.mind` | Hybrid search tuning |

Kernel parameters override in-code defaults when present. The `get_mind_kernel` MCP tool reads a specific kernel as structured JSON.

---

## Example Configurations

### Minimal (BM25 only, zero dependencies)

```json
{
  "version": "1.0.5",
  "governance_mode": "detect_only",
  "recall": {
    "backend": "scan"
  }
}
```

### Hybrid Search with RM3

```json
{
  "version": "1.0.5",
  "governance_mode": "propose",
  "recall": {
    "backend": "hybrid",
    "vector_enabled": true,
    "vector_model": "all-MiniLM-L6-v2",
    "onnx_backend": true,
    "rrf_k": 60,
    "rm3": {
      "enabled": true,
      "alpha": 0.6
    }
  }
}
```

### Full ML Pipeline

```json
{
  "version": "1.0.5",
  "governance_mode": "enforce",
  "recall": {
    "backend": "hybrid",
    "vector_enabled": true,
    "onnx_backend": true,
    "rrf_k": 60,
    "rm3": {
      "enabled": true
    },
    "cross_encoder": {
      "enabled": true,
      "blend_weight": 0.6
    }
  }
}
```

### Vector Search with Qdrant

```json
{
  "version": "1.0.5",
  "recall": {
    "backend": "vector",
    "provider": "qdrant",
    "model": "all-MiniLM-L6-v2",
    "qdrant_url": "http://localhost:6333",
    "qdrant_collection": "mind-mem"
  }
}
```

### Auto-Ingest Pipeline

```json
{
  "version": "1.0.5",
  "auto_capture": true,
  "auto_ingest": {
    "enabled": true,
    "transcript_scan": true,
    "entity_ingest": true,
    "intel_scan": true
  }
}
```

### Custom Categories with Proposal Limits

```json
{
  "version": "1.0.5",
  "governance_mode": "propose",
  "categories": {
    "enabled": true,
    "extra_categories": {
      "billing": ["invoice", "payment", "stripe"],
      "ml-ops": ["training", "model", "gpu", "inference"]
    }
  },
  "proposal_budget": {
    "per_run": 5,
    "per_day": 10,
    "backlog_limit": 50
  }
}
```

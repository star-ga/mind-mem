# Configuration Reference

mind-mem is configured via `mind-mem.json` in your workspace root. This file is created automatically by `init_workspace.py` with sensible defaults. All keys are optional -- missing keys fall back to their documented defaults.

---

## Full Configuration

```json
{
  "version": "2.8.0",
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
  "extraction": {
    "enabled": true,
    "model": "mind-mem:4b",
    "backend": "ollama"
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
  },
  "limits": {
    "max_recall_results": 100,
    "max_similar_results": 50,
    "max_prefetch_results": 20,
    "max_category_results": 10,
    "query_timeout_seconds": 30,
    "rate_limit_calls_per_minute": 120
  },
  "observability": {
    "otel_endpoint": null,
    "prom_port": 9090
  },
  "block_store": {
    "backend": "markdown",
    "dsn": ""
  }
}
```

---

## Auth Settings (v3.2.0)

The `api.auth` section controls how the REST API authenticates requests.

### Auth Mode

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `api.auth.mode` | string | `"bearer"` | Auth strategy. Valid values: `"bearer"` (env-var tokens), `"oidc"` (JWT SSO), `"api_keys"` (mmk_live_* keys), `"combined"` (bearer + api_keys, OIDC via /v1/auth/oidc/callback). |

### OIDC Configuration

Set these **environment variables** when `api.auth.mode` is `"oidc"` or `"combined"`:

| Variable | Description |
| --- | --- |
| `OIDC_ISSUER` | Issuer URL, e.g. `https://dev-123.okta.com/oauth2/default` |
| `OIDC_CLIENT_ID` | Application client ID from your identity provider |
| `OIDC_CLIENT_SECRET` | Application client secret (server-side only) |
| `OIDC_AUDIENCE` | Expected `aud` claim, e.g. `api://mind-mem` |

Supported providers via preset factories in `OIDCProvider`:

| Provider | Factory method |
| --- | --- |
| Okta | `OIDCProvider.for_okta(domain, client_id, client_secret, audience)` |
| Auth0 | `OIDCProvider.for_auth0(domain, client_id, client_secret, audience)` |
| Google Workspace | `OIDCProvider.for_google_workspace(client_id, client_secret, audience)` |
| Azure AD / Entra ID | `OIDCProvider.for_azure_ad(tenant_id, client_id, client_secret, audience)` |

### Per-Agent API Keys

| Variable | Description |
| --- | --- |
| `MIND_MEM_API_KEY_DB` | Filesystem path for the SQLite key store. Required to enable mmk_* keys. |
| `MIND_MEM_ENV` | `"production"` (default) issues `mmk_live_*` keys; any other value issues `mmk_test_*` keys. |

Keys are created via the admin REST endpoints:

```
POST   /v1/admin/api_keys               → create (returns raw key once)
GET    /v1/admin/api_keys               → list (key_hash never exposed)
DELETE /v1/admin/api_keys/{key_id}      → revoke
POST   /v1/admin/api_keys/{key_id}/rotate → rotate (revoke old, issue new)
```

All admin endpoints require the `MIND_MEM_ADMIN_TOKEN` credential.

### Audit Attribution

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `observability.audit_agent_attribution` | bool | `true` | When true, every governance audit record carries the authenticated `agent_id` in its metadata and `actor` field. Set via the `current_agent_id` contextvar in `mind_mem.api.rest`. The MCP layer can set the same contextvar at tool entry to propagate identity end-to-end. |

---

## Block Store Settings (v3.2.0)

The `block_store` section selects the storage backend for block persistence.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `backend` | string | `"markdown"` | Storage backend. Valid values: `"markdown"`, `"encrypted"`, `"postgres"`. |
| `dsn` | string | `""` | PostgreSQL connection string. Required when `backend = "postgres"`. |

### Postgres Backend

```json
{
  "block_store": {
    "backend": "postgres",
    "dsn": "postgresql://user:password@localhost:5432/mind_mem"
  }
}
```

Install the optional dependency before enabling the Postgres backend:

```bash
pip install "mind-mem[postgres]"
```

See `docs/storage-backends.md` for a full setup guide including Docker Compose and schema details.

---

## General Settings

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `version` | string | `"2.8.0"` | Config file version. Set automatically by `init_workspace.py`. |
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

### Result Filtering

These settings control post-retrieval filtering and adaptive truncation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `recall.knee_cutoff` | bool | `true` | Enable adaptive knee-point truncation. When enabled, results are cut at the steepest score drop instead of a fixed top-K. |
| `recall.min_score` | float | `0.0` | Minimum score threshold. Results below this score are discarded after retrieval. |

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
| `PINECONE_API_KEY` (env var) | string | (none) | **Required.** Pinecone API key. Must be set via environment variable (not config file). |
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

## Extraction (LLM Backend)

Controls the LLM used for memory extraction from transcripts and text. Added multi-backend support in v3.1.0.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `extraction.enabled` | bool | `true` | Enable LLM-based extraction. When `false`, only deterministic rule-based capture runs. |
| `extraction.model` | string | `"mind-mem:4b"` | Model identifier. For Ollama: the model tag (e.g., `mind-mem:4b`, `qwen3:4b`). For vLLM / OpenAI-compat: the served model name. For llama-cpp: absolute path to the GGUF file. |
| `extraction.backend` | string | `"ollama"` | LLM backend. See Backend Values below. |

### Backend Values

| Value | Description | Typical setup |
| --- | --- | --- |
| `"ollama"` | Local Ollama daemon at `http://localhost:11434` | `ollama serve` + `ollama create mind-mem:4b -f Modelfile` |
| `"vllm"` | Local vLLM OpenAI-compatible server | `vllm serve <model> --port 8000` → set `MIND_MEM_VLLM_URL` if non-default |
| `"openai-compatible"` | Any OpenAI-compatible endpoint (LM Studio, llama-server, TGI, OpenAI itself, Anthropic via proxy, etc.) | Set `MIND_MEM_LLM_BASE_URL` and optional `MIND_MEM_LLM_API_KEY` |
| `"llama-cpp"` | In-process `llama-cpp-python` | `pip install llama-cpp-python`; set `extraction.model` to GGUF path |
| `"transformers"` | In-process HuggingFace transformers | `pip install transformers torch`; slowest, no daemon required |
| `"auto"` | Try each in order (ollama → vllm → openai-compat → llama-cpp → transformers) until one returns a non-empty response | Zero-config fallback |

### mind-mem:4b (Recommended)

Full fine-tune of Qwen3.5-4B on STARGA-curated mind-mem corpus. Available as Q4_K_M GGUF (2.6GB) from [star-ga/mind-mem-4b](https://huggingface.co/star-ga/mind-mem-4b). Empirical on RTX 3080: 104 tok/s generation, 1585 tok/s prefill.

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

## Limits

Controls MCP server numeric limits for result caps, timeouts, and rate limiting. All values are integers. Missing keys fall back to their defaults.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `limits.max_recall_results` | int | `100` | Maximum results cap for the `recall` and `hybrid_search` tools. User-provided `limit` values are clamped to `[1, max_recall_results]`. |
| `limits.max_similar_results` | int | `50` | Maximum results cap for the `find_similar` tool. |
| `limits.max_prefetch_results` | int | `20` | Maximum results cap for the `prefetch` tool. |
| `limits.max_category_results` | int | `10` | Maximum category summaries returned by the `category_summary` tool. |
| `limits.query_timeout_seconds` | int | `30` | Per-query timeout for MCP tool calls. |
| `limits.rate_limit_calls_per_minute` | int | `120` | Sliding-window rate limiter: maximum MCP tool calls per 60-second window. |

### Example

```json
{
  "limits": {
    "max_recall_results": 200,
    "max_similar_results": 100,
    "max_prefetch_results": 50,
    "max_category_results": 20,
    "query_timeout_seconds": 60,
    "rate_limit_calls_per_minute": 240
  }
}
```

---

## Block Store (Storage Backend)

Controls which storage backend is used for Markdown block I/O. Added in v3.2.0.

```json
{
  "block_store": {
    "backend": "markdown"
  }
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `block_store.backend` | string | `"markdown"` | Storage backend for block data. See Backend Values below. |
| `block_store.dsn` | string | (none) | Connection string for database backends (Postgres, v3.2.0 PR-5). Example: `"postgresql://user:pass@localhost/mind_mem"`. |
| `block_store.schema` | string | `"mind_mem"` | Database schema name for the Postgres backend (v3.2.0 PR-5). |

### Backend Values

| Value | Description | Requirements |
| --- | --- | --- |
| `"markdown"` | Default. Reads and writes plain Markdown files under the workspace corpus directories. Zero dependencies. | None |
| `"encrypted"` | Transparent AES-256 encryption at rest via `EncryptedBlockStore`. Wraps the markdown backend. | `MIND_MEM_ENCRYPTION_PASSPHRASE` env var must be set to a non-empty string. |
| `"postgres"` | Postgres-backed block store. **Stub only** — raises `NotImplementedError` until v3.2.0 PR-5 ships the adapter. | `block_store.dsn` required. |

### Encrypted Backend

Set the passphrase via environment variable (never in the config file):

```bash
export MIND_MEM_ENCRYPTION_PASSPHRASE="your-strong-passphrase"
```

Then set the backend in `mind-mem.json`:

```json
{
  "block_store": {
    "backend": "encrypted"
  }
}
```

The factory raises `ValueError` immediately if `backend` is `"encrypted"` and the environment variable is absent or empty, preventing silent plaintext fallback.

### Postgres Backend (v3.2.0 PR-5)

```json
{
  "block_store": {
    "backend": "postgres",
    "dsn": "postgresql://mind_mem:secret@localhost:5432/mind_mem_db",
    "schema": "mind_mem"
  }
}
```

Requesting the `"postgres"` backend before PR-5 ships raises `NotImplementedError`. The config surface is stable now so operators can add the key in preparation.

### Factory API

```python
from mind_mem.storage import get_block_store

# Auto-load backend from <workspace>/mind-mem.json
store = get_block_store("/path/to/workspace")

# Or pass config explicitly (useful in tests)
store = get_block_store("/path/to/workspace", config={"block_store": {"backend": "markdown"}})
```

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
| `MIND_MEM_VLLM_URL` | Base URL for a local vLLM OpenAI-compatible server. Default: `http://127.0.0.1:8000/v1`. Used only when `extraction.backend` is `"vllm"` or `"auto"`. |
| `MIND_MEM_LLM_BASE_URL` | Base URL for any OpenAI-compatible endpoint (LM Studio, llama-server, TGI, OpenAI, etc.). No default. Used only when `extraction.backend` is `"openai-compatible"` or `"auto"`. |
| `MIND_MEM_LLM_API_KEY` | Optional API key for the `openai-compatible` backend. Sent as `Authorization: Bearer <key>`. Not required for local endpoints. |
| `MIND_MEM_ENCRYPTION_PASSPHRASE` | Passphrase for the `encrypted` block store backend. Required when `block_store.backend` is `"encrypted"`. Never put this value in the config file — always use an environment variable or a secret manager. |

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
  "version": "2.8.0",
  "governance_mode": "detect_only",
  "recall": {
    "backend": "scan"
  }
}
```

### Hybrid Search with RM3

```json
{
  "version": "2.8.0",
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
  "version": "2.8.0",
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
  "version": "2.8.0",
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
  "version": "2.8.0",
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
  "version": "2.8.0",
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

---

## Observability Settings

Requires the optional `otel` extra: `pip install "mind-mem[otel]"`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `observability.otel_endpoint` | string \| null | `null` | OTLP gRPC endpoint for OpenTelemetry traces (e.g. `"http://jaeger:4317"`). When `null`, a NoOp tracer is used (zero overhead). |
| `observability.prom_port` | integer | `9090` | TCP port for the Prometheus metrics HTTP server started by `init_prometheus()`. Set to `0` to disable. |

### Enabling Tracing at Runtime

```python
from mind_mem.telemetry import init_tracing, init_prometheus

# Send spans to a local Jaeger / OTLP collector
init_tracing(endpoint="http://localhost:4317")

# Expose /metrics on port 9090
init_prometheus(port=9090)
```

Both calls are idempotent and silently degrade when the optional packages
(`opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`,
`prometheus-client`) are not installed.

### Grafana Dashboard

A pre-built dashboard JSON is included at `deploy/grafana/mind-mem-dashboard.json`.
Import it via Grafana UI → Dashboards → Import → Upload JSON file. It contains
four panels:

- **Recall Latency (p50 / p95 / p99)** — `histogram_quantile` over `recall_duration_seconds`.
- **Recall QPS** — `rate(recall_total[5m])`.
- **propose_update Rate** — `rate(propose_update_total[5m])`.
- **Apply Rollback Rate** — `rate(apply_rollback_total[5m])`.

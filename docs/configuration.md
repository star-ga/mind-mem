# Configuration Reference

MIND-Mem is configured via `mind-mem.json` in your workspace root. This file is created automatically by `init_workspace.py` with sensible defaults. All keys are optional -- missing keys fall back to their documented defaults.

---

## Full Configuration

```json
{
  "version": "4.4.0",
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
| `MIND_MEM_OIDC_ADMIN_SCOPES` (v3.2.1) | Space/comma-separated list of JWT scope names that grant admin access. Default: `"mind-mem.admin admin"`. Override when your IdP uses a different scope convention — e.g. set to `"roles:platform-admin"` for an Okta custom scope, or `"prod:mindmem:admin"` for an Auth0 namespaced permission. |

When OIDC is configured (both `OIDC_ISSUER` and `OIDC_AUDIENCE` are
set), every bearer token is first probed as a JWT:

- Valid JWT → agent authenticates; scopes drive the admin gate.
- JWT that fails signature / audience / expiry validation → 401.
- Non-JWT token → falls through to the `MIND_MEM_TOKEN` /
  `MIND_MEM_ADMIN_TOKEN` static-token path.

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
| `version` | string | `"4.4.0"` | Config file version. Set automatically by `init_workspace.py`. |
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

### Consensus voting on contradictions (`governance.consensus`)

Opt-in, **default off**. When enabled, a contradiction that no deterministic
strategy can settle -- same date, same priority, same scope specificity, the
`manual_review` fallback -- consults a multi-agent quorum before it is handed
to a human. It is only ever reached where `manual_review` would have been the
verdict, so it can never override timestamp/confidence/scope priority.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `governance.consensus.enabled` | bool | `false` | Consult the quorum on `manual_review` contradictions. Off: `intelligence/VOTES.md` is never read and resolution is byte-identical to a workspace without the file. |
| `governance.consensus.quorum_threshold` | float | `0.66` | Winning share of the weighted vote. Values outside `(0, 1]` fall back to the default. |
| `governance.consensus.min_votes` | int | `2` | Distinct agents required. Below it the outcome is `insufficient_votes` and the contradiction stays `manual_review` -- which is the single-operator case. |
| `namespaces.<agent_id>.trust_weight` | float | `1.0` | Per-agent vote weight. `0` excludes the agent. A vote block cannot set its own weight; only this config can. |

Votes live in `intelligence/VOTES.md` as ordinary blocks:

```
[V-20260201-001]
Contradiction: CONTRA-001
Agent: agent-alice
Choice: D-20260201-001
Status: active
Rationale: matches the shipped schema
```

Three properties are load-bearing:

* **Vote blocks pass through the admission filter.** A vote a governance gate
  has not admitted (`quarantined`, `pending`, or any status the package does
  not recognise) is not counted, so an ingest door cannot vote.
* **The winner must be one of the two contradicting blocks.** A quorum for any
  other id is refused rather than written onto a supersede proposal.
* **The winner is staged, never applied.** It becomes a `pending-review`
  proposal in `intelligence/proposed/RESOLUTIONS_PROPOSED.md` under strategy
  `consensus_quorum`; applying it stays `approve_apply`'s job.

```json
{
  "governance": { "consensus": { "enabled": true, "quorum_threshold": 0.66, "min_votes": 2 } },
  "namespaces": { "agent-alice": { "trust_weight": 1.5 } }
}
```

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
| `recall.ollama_url` | string | `""` | Ollama endpoint for `embed_ollama` (v4.3.1). Accepts `host:port` or a full `http[s]://` URL. Precedence: this key > `OLLAMA_HOST` env var > `http://localhost:11434`. Lets a fleet node use a central Ollama server. |

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
| `extraction.ollama_url` | string | `""` | Ollama endpoint for extraction (v4.3.1). Accepts `host:port` or a full `http[s]://` URL. Precedence: this key > `OLLAMA_HOST` env var > `http://localhost:11434`. |

### Backend Values

| Value | Description | Typical setup |
| --- | --- | --- |
| `"ollama"` | Ollama daemon — endpoint resolves `extraction.ollama_url` > `OLLAMA_HOST` env > `http://localhost:11434` | `ollama serve` + `ollama create mind-mem:4b -f Modelfile`; for a central server set `OLLAMA_HOST=<host>:11434` |
| `"vllm"` | Local vLLM OpenAI-compatible server | `vllm serve <model> --port 8000` → set `MIND_MEM_VLLM_URL` if non-default |
| `"openai-compatible"` | Any OpenAI-compatible endpoint (LM Studio, llama-server, TGI, OpenAI itself, Anthropic via proxy, etc.) | Set `MIND_MEM_LLM_BASE_URL` and optional `MIND_MEM_LLM_API_KEY` |
| `"llama-cpp"` | In-process `llama-cpp-python` | `pip install llama-cpp-python`; set `extraction.model` to GGUF path |
| `"transformers"` | In-process HuggingFace transformers | `pip install transformers torch`; slowest, no daemon required |
| `"auto"` | Try each in order (ollama → vllm → openai-compat → llama-cpp → transformers) until one returns a non-empty response | Zero-config fallback |

### mind-mem:4b (Recommended)

Full fine-tune of Qwen3.5-4B on STARGA-curated MIND-Mem corpus. Available as Q4_K_M GGUF (2.6GB) from [star-ga/mind-mem-4b](https://huggingface.co/star-ga/mind-mem-4b). Empirical on RTX 3080: 104 tok/s generation, 1585 tok/s prefill.

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
| `auto_ingest.session_summary` | bool | `false` | **Opt-in ingest door.** Enable the session-summary job (`session_summarizer.py --scan-recent`). Summarises recent Claude Code transcripts into `summaries/daily/<date>.md` plus a linking signal in `intelligence/SIGNALS.md`. Off unless set: the job reads transcripts from **outside** the workspace (`~/.claude/projects`). |

Individual job toggles are only checked when `auto_ingest.enabled` is `true`. When the master toggle is off, no jobs run.

`session_summary` is the one job whose per-job default is `false`. With no
config, `--job all` dispatches exactly the three default-on jobs and does not
mention the opt-in one — an ingest door has to be asked for by name.

**What it writes, and why it is not recallable.** Both of its writes are
admitted through the governance gate under `IngestTier.AUTO_CAPTURE`, whose
`INITIAL_STATUS` row is `pending`. `pending` is not in the servable
allow-list, so a session summary and its linking signal are withheld from
recall until a governance proposal releases them. The summary block declares
`Status: pending` in the block itself rather than relying on
`summaries/` happening to sit outside the indexed corpus.

To run it from the daemon instead of cron, give it an interval:

```json
{
  "daemon": {
    "enabled": true,
    "session_summary": { "auto_interval_seconds": 3600 }
  }
}
```

`auto_interval_seconds` defaults to `0` (off) exactly like every other daemon
task; `mm daemon --once` then writes one dated summary per new transcript, and
the transcript content-hash dedup stops a second run from rewriting it.

---

## Streaming Ingest Rate Limit

Per-client rate limiting for the `POST /ingest` webhook (`mm ingest-serve`).
Off unless `streaming.enabled` is `true`; with it off the webhook has no
rate-limit leg at all and behaves exactly as it did before 5.1.0.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `streaming.enabled` | bool | `false` | Master toggle for the front gate. Off: no limiter, no counters, and `stream_status` reports no `ingest_door` key. |
| `streaming.rate_limit.tokens_per_second` | float | `20` | Refill rate of the token bucket **each** client gets. Omit the whole `rate_limit` block to arm the gate's telemetry without throttling anything. |
| `streaming.rate_limit.burst` | float | `40` | Bucket size — the burst **each** client may spend before refill matters. |
| `streaming.rate_limit.max_clients` | int | `1024` | How many distinct `client_id` values keep their own bucket; the least-recently-used one is evicted past this. A **memory** bound, not an authentication one. |

```json
{
  "streaming": {
    "enabled": true,
    "rate_limit": { "tokens_per_second": 20, "burst": 40, "max_clients": 1024 }
  }
}
```

The client is read from the `X-Client-Id` request header, falling back to the
peer address. A refused request gets **HTTP 429** before its body reaches the
write-ahead log or the queue, so one flooding producer cannot spend another
producer's allowance — the cross-client denial of service the per-client
keying exists to prevent.

**The header identifies, it does not authenticate.** A producer free to invent
a fresh `client_id` per request is not throttled by per-client accounting under
any keying. Authenticate upstream if that matters.

**This gate stores nothing.** It decides who may knock; what gets stored is
decided by the ingest door's single governed write, which admits every event
under `IngestTier.EXTERNAL_INGEST` and therefore lands it `Status:
quarantined` — withheld from recall until a governance proposal releases it.
Queue depth (`queue_depth`), the 429 count (`rate_limited`) and how many
queued events the drain has written (`applied`) are visible through the
`stream_status` MCP tool, under its `ingest_door` key.

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
| `block_store.dsn` | string | (none) | Connection string for the Postgres backend. Example: `"postgresql://user:pass@localhost/mind_mem"`. |
| `block_store.schema` | string | `"mind_mem"` | Database schema name for the Postgres backend. |

### Backend Values

| Value | Description | Requirements |
| --- | --- | --- |
| `"markdown"` | Default. Reads and writes plain Markdown files under the workspace corpus directories. Zero dependencies. | None |
| `"encrypted"` | Transparent authenticated encryption at rest (HMAC-SHA256 keystream + encrypt-then-MAC; **not** AES/SQLCipher — the recall index is not encrypted) via `EncryptedBlockStore`. Wraps the markdown backend. | `MIND_MEM_ENCRYPTION_PASSPHRASE` env var must be set to a non-empty string. |
| `"postgres"` | Postgres-backed block store with atomic snapshot/restore, upsert writes, FTS search, and read-replica routing. Ships in v3.2.0. | `block_store.dsn` required; `pip install "mind-mem[postgres]"`. |

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

### Postgres Backend (v3.2.0+)

```json
{
  "block_store": {
    "backend": "postgres",
    "dsn": "postgresql://mind_mem:secret@localhost:5432/mind_mem_db",
    "schema": "mind_mem"
  }
}
```

Install the optional dependency before enabling:

```bash
pip install "mind-mem[postgres]"
```

See `docs/storage-backends.md` for Docker Compose setup, schema details, read-replica routing, and performance tuning.

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
| `MIND_MEM_TOKEN` | Bearer token for HTTP MCP transport authentication. When set, all HTTP requests must include `Authorization: Bearer <token>` or `X-MindMem-Token: <token>`. Not required for stdio transport. Can also be passed via `--token` CLI flag. **As of v3.7.0, HTTP/REST authentication fails CLOSED:** if no token is configured the server refuses to start unless `MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1` is set explicitly (loopback-only opt-in for tests / dev). |
| `MIND_MEM_ADMIN_TOKEN` | Separate admin bearer token for privileged REST endpoints. Same fail-closed contract as `MIND_MEM_TOKEN`. |
| `MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST` | Opt-in escape hatch (set to `1` / `true` / `yes`) that re-enables unauthenticated access **only when binding to a loopback address** (`127.0.0.1` / `::1` / `localhost`). Intended for local development; never set in production. Without it, HTTP/REST refuses to start when no token is configured. Maps to the `--allow-unauthenticated-localhost` CLI flag on `mind-mem-mcp serve` and `mind-mem-rest`. |
| `MIND_MEM_LOG_LEVEL` | Logging level for structured JSON logging. Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default: `INFO`. |
| `MIND_MEM_LIB` | Absolute path to the compiled MIND kernel shared library (`libmindmem.so` / `libmindmem.dylib`). Overrides the default search paths. Must point to a file within the `lib/` directory of the MIND-Mem installation for security. |
| `MIND_MEM_HOME` | Path to the MIND-Mem installation directory. Used by the OpenClaw hook (`handler.js`) to locate scripts when they are not co-located with the workspace. |
| `MIND_MEM_VLLM_URL` | Base URL for a local vLLM OpenAI-compatible server. Default: `http://127.0.0.1:8000/v1`. Used only when `extraction.backend` is `"vllm"` or `"auto"`. |
| `MIND_MEM_LLM_BASE_URL` | Base URL for any OpenAI-compatible endpoint (LM Studio, llama-server, TGI, OpenAI, etc.). No default. Used only when `extraction.backend` is `"openai-compatible"` or `"auto"`. |
| `MIND_MEM_LLM_API_KEY` | Optional API key for the `openai-compatible` backend. Sent as `Authorization: Bearer <key>`. Not required for local endpoints. |
| `MIND_MEM_ENCRYPTION_PASSPHRASE` | Passphrase for the `encrypted` block store backend. Required when `block_store.backend` is `"encrypted"`. Never put this value in the config file — always use an environment variable or a secret manager. |

---

## MIND Kernel Configuration

In addition to `mind-mem.json`, MIND-Mem supports `.mind` kernel files -- INI-style configuration files in the `mind/` directory of the workspace. These provide fine-grained tuning for the recall pipeline.

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
  "version": "4.4.0",
  "governance_mode": "detect_only",
  "recall": {
    "backend": "scan"
  }
}
```

### Hybrid Search with RM3

```json
{
  "version": "4.4.0",
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
  "version": "4.4.0",
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
  "version": "4.4.0",
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
  "version": "4.4.0",
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
  "version": "4.4.0",
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

## Distributed Recall Cache (v3.2.0+)

Two-tier recall cache with an in-process LRU (L1) and an optional
Redis backend (L2). Shared across uvicorn workers when Redis is
configured; falls back to LRU-only when Redis is absent or
unreachable (fail-open).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `cache.enabled` | boolean | `true` | Set to `false` to bypass the cache entirely (useful for debugging recall changes). |
| `cache.redis_url` | string \| null | `null` | Redis connection URL (e.g. `"redis://localhost:6379/0"`). When `null`, only the in-process LRU is used. |
| `cache.ttl_seconds` | integer | `300` | Time-to-live for cached recall results. Governance writes (`propose_update`, `approve_apply`) invalidate the namespace anyway, so the TTL is a safety net rather than a correctness boundary. |
| `cache.lru_max_entries` | integer | `1024` | Max entries in the in-process LRU. Each entry is the serialized recall payload (~2-20KB typical). |

**Cache key format:** `mindmem:recall:<namespace>:<sha256(query+limit+backend+active_only)>`

```json
{
  "cache": {
    "enabled": true,
    "redis_url": "redis://cache.internal:6379/0",
    "ttl_seconds": 300,
    "lru_max_entries": 2048
  }
}
```

The REDIS_URL can also be supplied via the `MIND_MEM_REDIS_URL`
environment variable, which takes precedence over the config file
so deployments can inject the URL via Kubernetes secrets without
mutating the workspace.

---

## Producer Backpressure (`v4.backpressure`)

A producer loop that outruns the store grows its queue until the process is
OOM-killed. With this flag on, the loops that can do that report their backlog
depth and get an explicit "back off" answer.

Two watermarks with a gap between them, not one threshold: the state flips to
overloaded at `depth >= high_watermark` and back at `depth <= low_watermark`.
A queue hovering at 600 with the defaults (1000/200) therefore stays in the
state it is in instead of flapping on every tick.

**Off by default.** With the flag absent, every wired loop behaves exactly as
it did before — the probe reads the flag silently, so a flag-off process does
not even log differently.

```json
{
  "v4": {
    "backpressure": {
      "enabled": true,
      "high_watermark": 1000,
      "low_watermark": 200,
      "max_pause_seconds": 5.0,
      "producers": {
        "inbox": { "high_watermark": 20, "low_watermark": 5 }
      }
    }
  }
}
```

| Key | Default | Meaning |
|---|---|---|
| `high_watermark` | `1000` | depth at which a producer becomes overloaded |
| `low_watermark` | `200` | depth at which it recovers (must be `<=` high) |
| `max_pause_seconds` | `5.0` | cap on the exponential backoff hint |
| `producers.<name>` | — | per-producer overrides; 50 files and 5000 events are not the same kind of "deep" |

Producer names: `inbox` (the file-drop drain), `change_stream` (the in-process
event bus), `daemon` (scheduled maintenance ticks), and `webhook` (reserved —
the ingestion webhook drain is not wired yet).

What being overloaded actually does:

* **inbox** — a scheduled tick ingests `low_watermark` files and yields; the
  rest stay in the inbox root for the next pass. `mm`-driven one-shot runs are
  never capped.
* **change_stream** — `ChangeStream.is_overloaded()` starts answering `True`.
  The bus itself still delivers, queues and sheds exactly as before;
  backpressure is advice to producers, not policy the bus enforces.
* **daemon** — the tick is skipped and `last_run` is deliberately not stamped,
  so a deferred tick never reads as a completed one.

Deferring is not dropping: every wired loop leaves its input where it is and
picks it up on a later tick. Backpressure sheds rate, never data — and it
never touches an ingest path's admission, so a throttled inbox admits fewer
blocks per tick, never a different kind of block.

With the flag on, the `stream_status` MCP tool grows a `backpressure` object
(per-producer depth, watermarks, overload state). The key is absent when the
flag is off, so "nothing is measuring" is distinguishable from "measuring, and
fine".

## Structure-Aware Chunk Scoring (`retrieval.smart_chunking`)

Recall scores a long `Statement` twice — once whole, once as its best-scoring
sub-chunk — and blends the two, so a block whose match is concentrated in one
part is not penalised for everything around it. The sub-chunks are a
three-sentence sliding window, which knows nothing about the document: a window
routinely spans a markdown header, mixing two sections into one scoring unit.

Enabling this key takes the boundaries from `smart_chunker` instead, so a
chunk is a *section*. **Off by default** — chunk boundaries feed ranking, and
turning them over silently would change recall results for every existing
workspace.

```json
{
  "retrieval": {
    "smart_chunking": {
      "enabled": true
    }
  }
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `retrieval.smart_chunking.enabled` | bool | `false` | Master switch. Must be literal `true`; a truthy string or `1` leaves it off. |
| `retrieval.smart_chunking.max_chunk_size` | int | `1500` | Hard ceiling in characters. A chunk is never larger. |
| `retrieval.smart_chunking.soft_max_chunk_size` | int | `1` | Arm the soft ceiling after this many characters, so a strong structural boundary closes a chunk instead of the character budget. `0` disables it, which makes short statements come back as a single chunk. |
| `retrieval.smart_chunking.soft_max_boundary_score` | float | `0.5` | Minimum boundary strength (0.0–1.0) that honours the soft ceiling. At `0.5` a markdown header splits and running prose does not. |
| `retrieval.smart_chunking.min_chunk_size` | int | `0` | Merge chunks below this size into a neighbour. `0` keeps the section split intact; raising it can re-merge a short section with the next one. |
| `retrieval.smart_chunking.overlap_sentences` | int | `0` | Trailing sentences repeated into the next chunk. `0` keeps sections from bleeding into each other's scores. |

Notes:

- **No LLM.** `smart_chunker` can refine boundaries with a local model. That is
  pinned off here and is deliberately not configurable: recall is a pure
  function of (corpus, config, scoring instant), and a model call on the scored
  path would break it.
- **Blank lines are reconstructed.** The block format drops blank lines inside
  a field, so a statement read back from the corpus has no paragraph gaps left.
  The gap in front of each markdown header is restored before chunking —
  without it the structure-aware split would never fire at all.
- **Header-less text is unaffected.** When no structural boundary is found, the
  sentence window is used, so enabling the key never *removes* a chunk boost.
- This is a scoring surface only. Chunks are never stored, surfaced, or written
  back to the corpus.

---

## Multimodal Inbox Drops (`v4.multi_modal`)

Off by default. With the flag on, the inbox stops refusing image and audio
drops — but **nothing here interprets a media file**. There is no embedder, no
transcriber, and no extra that installs either one. What the door accepts is a
**sidecar** the operator wrote:

```
inbox/
  board.png          <- hashed, never interpreted
  board.png.txt      <- the description; this is what becomes the block
```

The sidecar name is the full media filename plus `.txt` or `.md`, never the
stem: `board.txt` is a legitimate text drop in its own right, and consuming it
as a caption would swallow a document the operator meant to ingest. When a pair
is present the watcher treats it as one drop and stages both files together; an
*orphan* sidecar with no media beside it is still an ordinary text drop.

```json
{
  "v4": {
    "multi_modal": { "enabled": true }
  }
}
```

A sidecar may also be a JSON object, which is the only way to state a duration
(the packer prices audio by it):

```json
{
  "transcript": "Standup: the migration is blocked on the index rebuild.",
  "duration_seconds": 90,
  "speakers": ["ana", "bo"],
  "dimensions": [1920, 1080]
}
```

| Sidecar field | Type | Notes |
| --- | --- | --- |
| `description` / `transcript` / `text` | string | The block's content. One is required; the first present wins, in that order. |
| `duration_seconds` | number | `0`–86400. Audio only. |
| `speakers` | list of strings | Max 64 entries, 128 characters each. |
| `dimensions` | `[width, height]` | Integers, `0`–1000000. Nothing reads the image header, so unstated means unknown. |

Anything else in the object is ignored — including `embedding`. The door will
not accept a vector from a file: an embedding is a scoring input, and taking
one from untrusted input would let a drop steer retrieval rather than merely
supply text for a human to admit.

**A media drop is an untrusted drop.** It goes through the same codepoint
sanitizer as the text door and is admitted under the same
`IngestTier.EXTERNAL_INGEST`, so the block lands `Status: quarantined` and is
invisible to recall until a governance proposal releases it. Turning this flag
on adds a door; it does not add a way to reach recall without a human.

The same flag makes `pack_recall_budget` price results by modality: an image
costs its tile count (85 tokens) rather than the length of its caption, and an
audio block that states a duration costs that duration. A text result costs
exactly what it costs with the flag off, so a text-only budget cannot move.

---

## Tier-Aware Retrieval Scoring — removed (RA.0)

`retrieval.tier_boost` and `retrieval.tier_boost_weights` no longer exist.
The module that read them (`tier_recall.py`) held a hard copy of
`memory_tiers.MemoryTier`'s ordinals as score multipliers, kept in lockstep
with the enum by comment. It had no importer outside its own tests, so the
boost never reached a ranking.

RA.0 collapsed the tree's tier ladders down to one —
`memory_tiers.MemoryTier` (`WORKING` / `SHARED` / `LONG_TERM` / `VERIFIED`) —
and deleted the duplicates rather than abstracting over them. The multipliers
went with them, deliberately: a tier that moves a recall score is a state
transition acting on the ranking, and the governance ruling routes tier
promotion through a proposal that `approve_apply` executes, never through a
direct write or an automatic promotion off usage counts.

Both keys are ignored if present in `mind-mem.json`. Nothing to migrate: the
boost had no effect to preserve.

---

## Served-Set Ledger (RA.1)

An append-only record of *what was served*, so a later outcome can be credited
to a **run** rather than correlated against a query string.

**Default OFF.** Opt in per workspace:

```json
{
  "served_ledger": {
    "enabled": true
  }
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `served_ledger.enabled` | boolean | `false` | Absent means off; only a literal `true` turns it on. |

When on, every `recall` appends one row to `.mind-mem-ledger/served.jsonl`,
written **after** `recall()` returns. Each row carries exactly nine fields —
`seq`, `prev_row_hash`, `run_id`, `query_hash`, `served_digest`, `ids`,
`pipeline_hash`, `index_anchor`, `scoring_instant` — and no verdict, no
attestation, no score. `.mind-mem-ledger/served.head` seals the last row, which
no successor binds yet.

`run_id` is `sha256("MM_RUN_v1\0" || query_hash || served_digest ||
pipeline_hash)`. It names **the answer**, not the occurrence: the same question
answered with the same blocks in the same order under the same pipeline gets
the same id on any host on any day, so two rows sharing a `run_id` are two
servings of one answer. `seq` is the unique key.

Verify a workspace's chain with `mind_mem.served_ledger.verify_served_chain`,
which reads no clock and names the first row that cannot be trusted.

Nothing on the recall scoring path may read this file. Serve counts are
derivable from it, and that is only safe while they cannot flow back into a
ranking; the rule is enforced by an import-graph test, not by convention.
Tier promotion is likewise not a consequence of being served — it goes through
a proposal, or a `plan_consolidation` output that `approve_apply` executes.

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

---

## LLM Reliability Profile (`v4.llm_noise_profile`)

Models each reporting agent as a noisy sensor and tracks how often it turns out
to have been right, per domain. The signal is one you are already producing:
`report_outcome` says whether acting on recalled blocks actually worked, and a
`success` / `failure` verdict is exactly a `was_correct` observation.

**Off by default.** With the flag absent nothing is read, written or logged on
the outcome path, and `calibration_stats` returns the same keys it always did.
The probe reads the flag silently, so a flag-off process is not even
distinguishable by its log lines.

```json
{
  "v4": {
    "llm_noise_profile": { "enabled": true }
  }
}
```

**What it records.** Reliability is an exponential moving average (α = 0.95,
starting at 0.7) kept per **provider** and per **domain**:

| Axis | Derived from | Example |
| --- | --- | --- |
| provider | `tool_id`, else `actor_id`, else the shared `unattributed` bucket | `gpt-4` |
| domain | the block-id family — the leading alphabetic run of the id | `D`, `T`, `INBOX` |

Persisted to `intelligence/llm_profiles.json` (state, not corpus: only `.md`
files under `intelligence/` are blocks). Load → update → save happens on every
recorded report, because an MCP call is frequently its own process — the file
*is* the state, so the profile survives a restart.

**What it deliberately does not do.**

- **It never reaches a score.** Nothing on the retrieval path reads the
  profile. It is evidence for an operator reading `calibration_stats`, not an
  input to ranking, and it must never become an agent leaderboard that routing
  consults — the moment "highest reliability" is a target, the incentive is to
  report less specifically rather than to be right more often.
- **Influence is bounded**, the same way the calibration projection already
  bounds it. The fold runs only when the store actually inserted a row, so
  replaying a report (same canonical payload → same outcome id) moves nothing.
  And one report is one observation *per distinct domain*: naming fifty blocks
  instead of one buys no extra movement.
- **`neutral` moves nothing.** "Not attributable to these blocks" is not
  evidence about the reporter's accuracy either.
- **It reads no block content** — only the ids the report already named — so it
  cannot surface quarantined material.

**Reproducible.** `report_outcome`'s `recorded_at` is injectable and is fed
through to the profile, so the same reports replayed on another machine write a
byte-identical file.

**Reading it back.** `calibration_stats` grows an `llm_reliability` section
while the flag is on:

```json
{
  "llm_reliability": {
    "flag": "v4.llm_noise_profile",
    "path": "intelligence/llm_profiles.json",
    "provider_count": 2,
    "providers": [
      {
        "provider_id": "mistral",
        "reliability": 0.715,
        "observation_noise": 0.285,
        "observations": 1,
        "errors": 0,
        "domains": { "T": 0.715 }
      }
    ]
  }
}
```

The key is **absent**, not null, when the flag is off.

## Model Reliability Score (`mrs`)

**Default OFF.** Turning it on adds an `mrs` section to `memory_health`: a
0–100 composite over latency percentiles, MCP error rate, and the corpus's own
drift / contradiction / staleness readings. A breach is routed through the
`alerts` sinks (`alerting.get_alert_router`, configured under the `alerts` key). With the flag off the tool is byte-identical to
what it returned before — the section is **absent**, not null, and the module
is not imported at all.

```json
{
  "mrs": {
    "enabled": true,
    "target": "retrieval",
    "latency_metric": "mcp_tool_duration_ms",
    "observation_days": 7,
    "alert": true,
    "alert_severity": "warning",
    "alert_below": 100.0,
    "slo": {
      "slis": [
        { "name": "p99_ms", "threshold": 2500 },
        { "name": "staleness_ratio", "threshold": 0.1, "weight": 2.0 }
      ]
    }
  }
}
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. There is deliberately no `auto_enable`: MRS reads the whole corpus and can fire alerts, so it turns on when an operator says so. |
| `target` | `"retrieval"` | Label the report is filed under. |
| `latency_metric` | `"mcp_tool_duration_ms"` | Which in-process observation series the latency percentiles are read from. |
| `latency_ms` | *(unset)* | A literal list of readings, which overrides `latency_metric`. For fixtures and replay. |
| `observation_days` | `1.0` | Window the `relevance_decay` rate is divided by. **Injected, never derived** — deriving it would mean reading a clock, and the score would stop being reproducible. |
| `alert` | `true` | Route breaches to the alert sinks. |
| `alert_severity` | `"warning"` | Severity of the `mrs_degraded` alert. Must clear the `alerts.min_severity` threshold to be delivered. |
| `alert_below` | `100.0` | Also alert when the score falls below this, even with no named violation. |
| `slo` | `{}` | Operator SLO spec. Entries join measured readings on `name` and override `threshold` / `weight`; an entry with no `threshold` keeps the built-in one rather than switching the violation off, and an entry naming an SLI nobody measured is ignored. |

**The SLIs.** `p50_ms` / `p95_ms` / `p99_ms` (thresholds 100 / 500 / 1500 ms);
`error_rate` (1%, and **omitted entirely** when nothing has been called — no
errors out of no requests is not a 0% error rate); `relevance_decay` (drift
items per servable block per day, 0.05); `contradiction_density` (per 100
servable blocks, 0.5); `staleness_ratio` (0.2). Each SLI contributes
`weight × (1 − penalty)`, where the penalty ramps from 0 at the threshold to 1
at double it, so one breached SLI out of six costs about 17 points.

**The denominator is the *servable* corpus.** The collector reads blocks, so it
goes through the same `admit_corpus` gate every retrieval leg uses: a
quarantined block is not counted, and a staleness flag pointing at one is not
counted either. That is the correct population as well as the required gate — a
quarantined block is not a retrieval problem, it is quarantine working.

**The alert payload carries aggregates only** — target, score, violated SLI
names and the numeric readings. Sinks write to log lines and third-party
webhooks, which is exactly the surface block text must not reach.


---

## Webhook Ingest Door (`v4.ingest_serve`)

Off by default. With the flag on, `mm ingest-serve` listens on loopback and
accepts `POST /ingest` with a JSON body, turning each event into a block.

```json
{
  "v4": {
    "ingest_serve": { "enabled": true }
  }
}
```

```bash
mm ingest-serve --port 8788
curl -X POST http://127.0.0.1:8788/ingest \
  -H 'Content-Type: application/json' \
  -d '{"text": "the deploy was rolled back at 14:02", "source": "ci"}'
```

**What arrives is quarantined.** The drain consumer admits every event through
`GovernanceGate.admit_block(tier=EXTERNAL_INGEST)` before a byte lands, and
that tier mints `Status: quarantined`: the block is on disk, it is in the audit
chain, and `recall` will not return it. Releasing a batch is one governed
decision (`propose_import_release` → `approve_apply`), exactly as for `mm
import` and the inbox drop folder. A producer cannot ask for a different
status — the gate refuses any tier that mints a servable one.

| Flag / option | Default | Notes |
| --- | --- | --- |
| `v4.ingest_serve.enabled` | `false` | Off means off: no socket is bound, no WAL is created, and the probe that reads this key logs nothing. |
| `--port` / `--host` | `8788` / `127.0.0.1` | Loopback by default. There is **no authentication on this endpoint** — put it behind a reverse proxy or an SSH tunnel before binding a routable address. |
| `--wal` | `<workspace>/memory/ingest-wal.jsonl` | Every accepted event is fsynced here *before* it is queued. |
| `--no-wal` | off | At-most-once: a kill loses whatever has not been drained. |
| `--interval` | `1.0` | Seconds between drain passes. |
| `--capacity` | `1024` | Queue depth before the endpoint answers `503` instead of `202`. |
| `--replay-only` | off | Apply the WAL backlog through the gate and exit; bind no socket. |

**Crash safety.** The WAL is the source of record and the drain tracks a
checkpoint beside it (`<wal>.applied`), advanced only *after* the blocks are
written. A kill therefore loses nothing: an event is either applied or still
pending. A record re-applied across the crash window is harmless because block
ids are content-addressed — the same event always yields the same
`INGEST-<sha256 prefix>` id, so a replay (or a producer retrying a `503`)
rewrites the identical block instead of duplicating it.

**Determinism.** Nothing on this path reads a clock or a random source: the id
is a hash of the event, and the block carries no generated timestamp. A
producer's own `timestamp` field is recorded verbatim as `EventTime` and is
never trusted or used for ordering. Arrival time is recorded where it is
accountable — the gate's tamper-evident chain entry.

**Event shape.** The text is taken from the first present of `text`,
`statement`, `content`, `body`, `message`; `source` and `subject` are optional
and are flattened to a single line (a newline in one would otherwise forge a
block field). An event with no usable text, or with text over 1 MiB, is
refused and counted rather than written. Invisible-Unicode codepoints are
stripped on the way in (`ingest.sanitize_codepoints`, on by default).

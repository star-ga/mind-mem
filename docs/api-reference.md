# Mind-Mem MCP API Reference

Complete reference for all 16 MCP tools and 8 MCP resources exposed by `mcp_server.py`.

**Server name:** `mind-mem`

**Transport:** `stdio` (default, for Claude Code / Claude Desktop) or `http` (for remote / multi-client).

**Authentication (HTTP only):** Bearer token via `Authorization: Bearer <token>` header or `X-MindMem-Token` header. Set via `MIND_MEM_TOKEN` environment variable or `--token` CLI flag. No authentication required for stdio transport.

---

## Table of Contents

- [Tools](#tools)
  - [recall](#recall)
  - [propose_update](#propose_update)
  - [approve_apply](#approve_apply)
  - [rollback_proposal](#rollback_proposal)
  - [scan](#scan)
  - [list_contradictions](#list_contradictions)
  - [hybrid_search](#hybrid_search)
  - [find_similar](#find_similar)
  - [intent_classify](#intent_classify)
  - [index_stats](#index_stats)
  - [reindex](#reindex)
  - [memory_evolution](#memory_evolution)
  - [list_mind_kernels](#list_mind_kernels)
  - [get_mind_kernel](#get_mind_kernel)
  - [category_summary](#category_summary)
  - [prefetch](#prefetch)
- [Resources](#resources)
  - [mind-mem://decisions](#mind-memdecisions)
  - [mind-mem://tasks](#mind-memtasks)
  - [mind-mem://entities/{type}](#mind-mementitiestype)
  - [mind-mem://signals](#mind-memsignals)
  - [mind-mem://contradictions](#mind-memcontradictions)
  - [mind-mem://health](#mind-memhealth)
  - [mind-mem://recall/{query}](#mind-memrecallquery)
  - [mind-mem://ledger](#mind-memledger)

---

## Tools

### recall

Search across all memory files with ranked retrieval. Uses FTS5 index when available (O(log N)), falls back to BM25 scan.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `string` | *(required)* | Search query. Supports stemming and domain-aware expansion. |
| `limit` | `integer` | `10` | Maximum number of results. Clamped to range [1, 100]. |
| `active_only` | `boolean` | `false` | Only return blocks with `Status: active`. |

#### Python Example

```python
result = await session.call_tool("recall", {
    "query": "authentication decisions",
    "limit": 5,
    "active_only": True
})
```

#### Return Format

JSON array of ranked result objects.

```json
[
  {
    "_id": "D-20260213-001",
    "type": "decision",
    "score": 12.4532,
    "excerpt": "Use PostgreSQL for all persistent storage...",
    "speaker": "",
    "tags": "database, infrastructure",
    "file": "decisions/DECISIONS.md",
    "line": 42,
    "status": "active",
    "DiaID": ""
  }
]
```

Each result object contains:

| Field | Type | Description |
|-------|------|-------------|
| `_id` | `string` | Block identifier (e.g., `D-20260213-001`). |
| `type` | `string` | Block type derived from ID prefix (`decision`, `task`, etc.). |
| `score` | `float` | BM25 relevance score, rounded to 4 decimal places. |
| `excerpt` | `string` | Truncated block content for preview. |
| `speaker` | `string` | Speaker name extracted from tags, if present. |
| `tags` | `string` | Comma-separated tags from the block. |
| `file` | `string` | Source file path relative to workspace. |
| `line` | `integer` | Line number in the source file. |
| `status` | `string` | Block status (e.g., `active`, `superseded`). |
| `DiaID` | `string` | Dialogue ID for benchmark correlation. |

Results may also include `via_adjacency` or `via_pronoun_rescue` boolean flags when graph boosting or pronoun resolution contributes additional blocks.

#### Example

Request:
```json
{
  "tool": "recall",
  "arguments": {
    "query": "database migration strategy",
    "limit": 5,
    "active_only": true
  }
}
```

Response:
```json
[
  {
    "_id": "D-20260210-003",
    "type": "decision",
    "score": 18.2341,
    "excerpt": "All database migrations must use versioned SQL files...",
    "speaker": "",
    "tags": "database, migrations",
    "file": "decisions/DECISIONS.md",
    "line": 87,
    "status": "active",
    "DiaID": ""
  },
  {
    "_id": "D-20260211-001",
    "type": "decision",
    "score": 14.1023,
    "excerpt": "PostgreSQL 16 is the primary data store...",
    "speaker": "",
    "tags": "database, infrastructure",
    "file": "decisions/DECISIONS.md",
    "line": 12,
    "status": "active",
    "DiaID": ""
  }
]
```

---

### propose_update

Propose a new decision or task. Writes to `intelligence/SIGNALS.md` for human review. This tool never writes directly to `DECISIONS.md` or `TASKS.md`. All proposals must go through the apply engine for review.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `block_type` | `string` | *(required)* | Type of block. Must be `"decision"` or `"task"`. |
| `statement` | `string` | *(required)* | The decision statement or task description. |
| `rationale` | `string` | `""` | Why this decision or task is needed. |
| `tags` | `string` | `""` | Comma-separated tags (e.g., `"database, infrastructure"`). |
| `confidence` | `string` | `"medium"` | Signal confidence level. One of `"high"`, `"medium"`, or `"low"`. Maps to priority: high=P1, medium=P2, low=P3. |

#### Python Example

```python
result = await session.call_tool("propose_update", {
    "block_type": "decision",
    "statement": "Use Redis for session caching",
    "rationale": "Persistence across restarts",
    "tags": "infrastructure, caching",
    "confidence": "high"
})
```

#### Return Format

```json
{
  "status": "proposed",
  "written": 1,
  "location": "intelligence/SIGNALS.md",
  "next_step": "Run /apply or `python3 maintenance/apply_engine.py` to review and promote to source of truth.",
  "safety": "This signal is in SIGNALS.md only. It has NOT been written to DECISIONS.md or TASKS.md."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"proposed"`. |
| `written` | `integer` | Number of signals written (0 if duplicate detected via content hash). |
| `location` | `string` | File path where the signal was appended. |
| `next_step` | `string` | Instructions for promoting the proposal. |
| `safety` | `string` | Confirmation that source of truth was not modified. |

Returns an error object if `block_type` is invalid:
```json
{
  "error": "block_type must be 'decision' or 'task', got 'note'"
}
```

#### Example

Request:
```json
{
  "tool": "propose_update",
  "arguments": {
    "block_type": "decision",
    "statement": "Use Redis for session caching instead of in-memory stores",
    "rationale": "In-memory sessions are lost on restart; Redis provides persistence and horizontal scaling",
    "tags": "infrastructure, caching, redis",
    "confidence": "high"
  }
}
```

Response:
```json
{
  "status": "proposed",
  "written": 1,
  "location": "intelligence/SIGNALS.md",
  "next_step": "Run /apply or `python3 maintenance/apply_engine.py` to review and promote to source of truth.",
  "safety": "This signal is in SIGNALS.md only. It has NOT been written to DECISIONS.md or TASKS.md."
}
```

---

### approve_apply

Apply a staged proposal from `intelligence/proposed/`. Defaults to dry-run mode. Creates a snapshot before applying for rollback support.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `proposal_id` | `string` | *(required)* | The proposal ID. Must match format `P-YYYYMMDD-NNN` (e.g., `"P-20260213-002"`). |
| `dry_run` | `boolean` | `true` | If `true` (default), validate without executing. Set to `false` to actually apply the proposal. |

#### Python Example

```python
result = await session.call_tool("approve_apply", {
    "proposal_id": "P-20260218-001",
    "dry_run": False
})
```

#### Return Format

```json
{
  "status": "dry_run_passed",
  "proposal_id": "P-20260213-002",
  "dry_run": true,
  "success": true,
  "message": "Dry run passed. Proposal is valid.",
  "log": "... apply engine output ...",
  "next_step": "Call again with dry_run=False to apply."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | One of `"applied"`, `"dry_run_passed"`, or `"failed"`. |
| `proposal_id` | `string` | The proposal ID that was processed. |
| `dry_run` | `boolean` | Whether this was a dry run. |
| `success` | `boolean` | Whether the operation succeeded. |
| `message` | `string` | Human-readable result message from the apply engine. |
| `log` | `string` | Captured stdout from the apply engine (truncated to last 2000 chars). |
| `next_step` | `string` or `null` | Next action hint. Non-null only on successful dry run. |

Returns an error object if the proposal ID format is invalid:
```json
{
  "error": "Invalid proposal ID format: bad-id. Expected P-YYYYMMDD-NNN."
}
```

#### Example

Request (dry run):
```json
{
  "tool": "approve_apply",
  "arguments": {
    "proposal_id": "P-20260218-001",
    "dry_run": true
  }
}
```

Response:
```json
{
  "status": "dry_run_passed",
  "proposal_id": "P-20260218-001",
  "dry_run": true,
  "success": true,
  "message": "Dry run passed. 1 block to add to decisions/DECISIONS.md.",
  "log": "... engine output ...",
  "next_step": "Call again with dry_run=False to apply."
}
```

Request (actual apply):
```json
{
  "tool": "approve_apply",
  "arguments": {
    "proposal_id": "P-20260218-001",
    "dry_run": false
  }
}
```

Response:
```json
{
  "status": "applied",
  "proposal_id": "P-20260218-001",
  "dry_run": false,
  "success": true,
  "message": "Applied. Receipt: 20260218-143022. Snapshot saved for rollback.",
  "log": "... engine output ...",
  "next_step": null
}
```

---

### rollback_proposal

Rollback an applied proposal using its receipt timestamp. Restores the workspace from the pre-apply snapshot stored in `intelligence/applied/`.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `receipt_ts` | `string` | *(required)* | Receipt timestamp from a prior `approve_apply` result. Must match format `YYYYMMDD-HHMMSS` (e.g., `"20260218-143022"`). |

#### Python Example

```python
result = await session.call_tool("rollback_proposal", {
    "receipt_ts": "20260218-143022"
})
```

#### Return Format

```json
{
  "status": "rolled_back",
  "receipt_ts": "20260218-143022",
  "success": true,
  "log": "... rollback engine output ..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | `"rolled_back"` on success, `"rollback_failed"` on failure. |
| `receipt_ts` | `string` | The receipt timestamp used. |
| `success` | `boolean` | Whether the rollback succeeded. |
| `log` | `string` | Captured stdout from the rollback engine (truncated to last 2000 chars). |

Returns an error object if the timestamp format is invalid:
```json
{
  "error": "Invalid receipt timestamp format: bad-ts. Expected YYYYMMDD-HHMMSS."
}
```

#### Example

Request:
```json
{
  "tool": "rollback_proposal",
  "arguments": {
    "receipt_ts": "20260218-143022"
  }
}
```

Response:
```json
{
  "status": "rolled_back",
  "receipt_ts": "20260218-143022",
  "success": true,
  "log": "Restored snapshot from intelligence/applied/20260218-143022\nRestored decisions/DECISIONS.md"
}
```

---

### scan

Run an integrity scan on the workspace. Checks decisions, contradictions, drift items, and pending signals.

#### Parameters

None.

#### Python Example

```python
result = await session.call_tool("scan", {})
```

#### Return Format

```json
{
  "workspace": "/path/to/workspace",
  "checks": {
    "decisions": {
      "total": 24,
      "active": 18
    },
    "contradictions": 2,
    "drift_items": 1,
    "pending_signals": 5
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `workspace` | `string` | Absolute path to the scanned workspace. |
| `checks.decisions.total` | `integer` | Total decision blocks found. |
| `checks.decisions.active` | `integer` | Decision blocks with active status. |
| `checks.contradictions` | `integer` | Number of detected contradiction entries. |
| `checks.drift_items` | `integer` | Number of drift entries (decisions diverging from code). |
| `checks.pending_signals` | `integer` | Number of unreviewed signals in SIGNALS.md. |

#### Example

Request:
```json
{
  "tool": "scan",
  "arguments": {}
}
```

Response:
```json
{
  "workspace": "/path/to/workspace",
  "checks": {
    "decisions": {
      "total": 24,
      "active": 18
    },
    "contradictions": 2,
    "drift_items": 0,
    "pending_signals": 7
  }
}
```

---

### list_contradictions

List detected contradictions between decisions with resolution analysis and strategy recommendations.

#### Parameters

None.

#### Python Example

```python
result = await session.call_tool("list_contradictions", {})
```

#### Return Format

When contradictions exist:
```json
{
  "status": "contradictions_found",
  "contradictions": 2,
  "resolutions": [
    {
      "block_a": "D-20260210-001",
      "block_b": "D-20260215-003",
      "strategy": "supersede",
      "confidence": 0.85,
      "rationale": "D-20260215-003 is newer and more specific..."
    }
  ]
}
```

When no contradictions exist:
```json
{
  "status": "clean",
  "contradictions": 0,
  "message": "No contradictions found."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | `"contradictions_found"` or `"clean"`. |
| `contradictions` | `integer` | Number of contradictions detected. |
| `resolutions` | `array` | Array of resolution recommendation objects (present when contradictions exist). |
| `resolutions[].block_a` | `string` | ID of the first conflicting block. |
| `resolutions[].block_b` | `string` | ID of the second conflicting block. |
| `resolutions[].strategy` | `string` | Recommended resolution strategy (e.g., `"supersede"`, `"merge"`, `"manual"`). |
| `resolutions[].confidence` | `float` | Confidence in the recommended strategy. |
| `resolutions[].rationale` | `string` | Explanation of why this strategy was chosen. |
| `message` | `string` | Human-readable message (present when clean). |

#### Example

Request:
```json
{
  "tool": "list_contradictions",
  "arguments": {}
}
```

Response:
```json
{
  "status": "contradictions_found",
  "contradictions": 1,
  "resolutions": [
    {
      "block_a": "D-20260210-001",
      "block_b": "D-20260215-003",
      "strategy": "supersede",
      "confidence": 0.85,
      "rationale": "D-20260215-003 is newer and references D-20260210-001 as outdated."
    }
  ]
}
```

---

### hybrid_search

Full hybrid BM25 + Vector recall with Reciprocal Rank Fusion (RRF). Falls back to BM25-only when the vector backend is unavailable (no `sentence-transformers` installed or `vector_enabled` is false in config).

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `string` | *(required)* | Search query. |
| `limit` | `integer` | `10` | Maximum results. Clamped to range [1, 100]. |
| `active_only` | `boolean` | `false` | Only return active blocks. |

#### Return Format

Same result format as [recall](#recall) -- a JSON array of ranked result objects with `_id`, `type`, `score`, `excerpt`, `speaker`, `tags`, `file`, `line`, `status`, and `DiaID` fields.

The score values reflect RRF-fused rankings when both backends are active, or raw BM25 scores on fallback.

#### Example

Request:
```json
{
  "tool": "hybrid_search",
  "arguments": {
    "query": "authentication flow OAuth",
    "limit": 5
  }
}
```

Response:
```json
[
  {
    "_id": "D-20260212-005",
    "type": "decision",
    "score": 0.0312,
    "excerpt": "OAuth 2.0 with PKCE is the authentication standard...",
    "speaker": "",
    "tags": "auth, security",
    "file": "decisions/DECISIONS.md",
    "line": 134,
    "status": "active",
    "DiaID": ""
  }
]
```

---

### find_similar

Find blocks similar to a given block using co-occurrence data from the block metadata database. Requires the block metadata database (`memory/block_meta.db`).

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `block_id` | `string` | *(required)* | Source block ID to find similar blocks for (e.g., `"D-20260213-001"`). |
| `limit` | `integer` | `5` | Maximum similar blocks to return. Clamped to range [1, 50]. |

#### Return Format

```json
{
  "source": "D-20260213-001",
  "similar": ["D-20260210-003", "T-20260214-002", "D-20260211-001"],
  "method": "co-occurrence"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `source` | `string` | The block ID that was queried. |
| `similar` | `array` | Array of block IDs that frequently co-occur in search results with the source block. |
| `method` | `string` | Similarity method used (currently `"co-occurrence"`). |

On error (e.g., block not found, database unavailable):
```json
{
  "error": "description of the error",
  "block_id": "D-20260213-001"
}
```

#### Example

Request:
```json
{
  "tool": "find_similar",
  "arguments": {
    "block_id": "D-20260210-003",
    "limit": 3
  }
}
```

Response:
```json
{
  "source": "D-20260210-003",
  "similar": ["D-20260211-001", "D-20260213-005"],
  "method": "co-occurrence"
}
```

---

### intent_classify

Show the routing strategy for a query. Classifies query intent into one of 9 types and returns retrieval parameter overrides used by the recall pipeline.

The 9 intent types are:

| Intent | Description | Retrieval Tuning |
|--------|-------------|------------------|
| `WHY` | Causal/reasoning queries | RM3 expansion, graph depth 2, causal reranking |
| `WHEN` | Temporal queries | Date expansion |
| `ENTITY` | Entity lookup queries | Entity-focused retrieval |
| `WHAT` | Factual/definitional queries | Standard retrieval |
| `HOW` | Procedural queries | Procedural reranking |
| `LIST` | Enumeration queries | List-oriented retrieval |
| `VERIFY` | Confirmation/adversarial queries | Verification mode |
| `COMPARE` | Comparison queries | Multi-block comparison |
| `TRACE` | Multi-hop/provenance queries | Deep graph traversal |

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `string` | *(required)* | The query to classify. |

#### Return Format

```json
{
  "query": "Why did we switch from MySQL to PostgreSQL?",
  "intent": "WHY",
  "confidence": 0.75,
  "sub_intents": ["WHEN"],
  "params": {
    "expansion": "rm3",
    "graph_depth": 2,
    "rerank": "causal",
    "patterns": ["..."]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | The original query. |
| `intent` | `string` | Primary detected intent type. |
| `confidence` | `float` | Classification confidence (0.0 to 1.0). |
| `sub_intents` | `array` | Secondary intent types also detected in the query. |
| `params` | `object` | Retrieval parameter overrides for the detected intent. Keys include `expansion`, `graph_depth`, `rerank`, and `patterns`. |

#### Example

Request:
```json
{
  "tool": "intent_classify",
  "arguments": {
    "query": "When was the Redis caching decision made?"
  }
}
```

Response:
```json
{
  "query": "When was the Redis caching decision made?",
  "intent": "WHEN",
  "confidence": 0.67,
  "sub_intents": [],
  "params": {
    "expansion": "date",
    "patterns": ["..."]
  }
}
```

---

### index_stats

Returns block counts, FTS index status, vector coverage, and MIND kernel information for the workspace.

#### Parameters

None.

#### Return Format

```json
{
  "workspace": "/path/to/workspace",
  "decisions_blocks": 24,
  "tasks_blocks": 8,
  "entities_blocks": 15,
  "fts_index_exists": true,
  "mind_kernels": ["recall", "rm3", "rerank", "temporal", "adversarial", "hybrid"],
  "mind_kernel_compiled": true,
  "mind_kernel_protected": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `workspace` | `string` | Absolute path to the workspace. |
| `decisions_blocks` | `integer` | Total blocks across all `.md` files in `decisions/`. |
| `tasks_blocks` | `integer` | Total blocks across all `.md` files in `tasks/`. |
| `entities_blocks` | `integer` | Total blocks across all `.md` files in `entities/`. |
| `fts_index_exists` | `boolean` | Whether the FTS5 SQLite index file exists. |
| `mind_kernels` | `array` | List of available `.mind` kernel configuration names. |
| `mind_kernel_compiled` | `boolean` | Whether the MIND C runtime is compiled and available. |
| `mind_kernel_protected` | `boolean` | Whether the MIND runtime has full protection (anti-debug, VM bytecode, etc.). |

#### Example

Request:
```json
{
  "tool": "index_stats",
  "arguments": {}
}
```

Response:
```json
{
  "workspace": "/path/to/workspace",
  "decisions_blocks": 24,
  "tasks_blocks": 8,
  "entities_blocks": 15,
  "fts_index_exists": true,
  "mind_kernels": ["recall", "rm3", "rerank"],
  "mind_kernel_compiled": true,
  "mind_kernel_protected": false
}
```

---

### reindex

Trigger a full FTS index rebuild. Optionally rebuilds the vector index and regenerates category summary files.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `include_vectors` | `boolean` | `false` | Also rebuild the vector index. Requires `sentence-transformers` to be installed. |

#### Return Format

```json
{
  "workspace": "/path/to/workspace",
  "fts": true,
  "vectors": false,
  "categories": 8
}
```

| Field | Type | Description |
|-------|------|-------------|
| `workspace` | `string` | Absolute path to the workspace. |
| `fts` | `boolean` | Whether the FTS index rebuild succeeded. |
| `vectors` | `boolean` | Whether the vector index rebuild succeeded (always `false` if not requested). |
| `categories` | `integer` | Number of category summary files written. |
| `fts_error` | `string` | Error message if FTS rebuild failed (only present on failure). |
| `vectors_error` | `string` | Error message if vector rebuild failed (only present on failure). |
| `categories_error` | `string` | Error message if category distillation failed (only present on failure). |

#### Example

Request:
```json
{
  "tool": "reindex",
  "arguments": {
    "include_vectors": true
  }
}
```

Response:
```json
{
  "workspace": "/path/to/workspace",
  "fts": true,
  "vectors": true,
  "categories": 10
}
```

---

### memory_evolution

Retrieve or update A-MEM (Adaptive Memory Evolution Model) metadata for a block. Tracks importance, access patterns, keywords, and connections.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `block_id` | `string` | *(required)* | The block ID (e.g., `"D-20260213-001"`). |
| `action` | `string` | `"get"` | `"get"` to read current metadata, `"update"` to recompute importance score. |

#### Return Format

For `action: "get"`:
```json
{
  "block_id": "D-20260213-001",
  "importance": 1.2345,
  "co_occurring_blocks": ["D-20260210-003", "T-20260214-002"]
}
```

For `action: "update"`:
```json
{
  "block_id": "D-20260213-001",
  "action": "updated",
  "importance": 1.3012
}
```

| Field | Type | Description |
|-------|------|-------------|
| `block_id` | `string` | The queried block ID. |
| `importance` | `float` | Importance score in the range [0.8, 1.5]. Computed from access frequency, recency, and connection count. Rounded to 4 decimal places. |
| `co_occurring_blocks` | `array` | Block IDs that frequently appear alongside this block in search results (only on `get`). |
| `action` | `string` | Set to `"updated"` when importance was recomputed (only on `update`). |

On error:
```json
{
  "error": "description of the error",
  "block_id": "D-20260213-001"
}
```

#### Example

Request:
```json
{
  "tool": "memory_evolution",
  "arguments": {
    "block_id": "D-20260210-003",
    "action": "get"
  }
}
```

Response:
```json
{
  "block_id": "D-20260210-003",
  "importance": 1.1523,
  "co_occurring_blocks": ["D-20260211-001", "D-20260213-005"]
}
```

---

### list_mind_kernels

List available `.mind` kernel configuration files. Kernels define tuning parameters for recall, reranking, RM3 expansion, and other pipeline components.

#### Parameters

None.

#### Return Format

JSON array of kernel summary objects.

```json
[
  {
    "name": "recall",
    "sections": ["bm25", "stemmer", "stopwords"],
    "path": "/path/to/workspace/mind/recall.mind"
  },
  {
    "name": "rm3",
    "sections": ["expansion", "feedback"],
    "path": "/path/to/workspace/mind/rm3.mind"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Kernel name (filename without `.mind` extension). |
| `sections` | `array` | List of INI-style section names within the kernel file. |
| `path` | `string` | Absolute path to the `.mind` kernel file. |

#### Example

Request:
```json
{
  "tool": "list_mind_kernels",
  "arguments": {}
}
```

Response:
```json
[
  {
    "name": "recall",
    "sections": ["bm25", "stemmer", "stopwords"],
    "path": "/path/to/workspace/mind/recall.mind"
  },
  {
    "name": "rerank",
    "sections": ["weights", "features"],
    "path": "/path/to/workspace/mind/rerank.mind"
  }
]
```

---

### get_mind_kernel

Read a specific `.mind` kernel configuration file as structured JSON. Parses the INI-style `[section]` / `key = value` format.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `string` | *(required)* | Kernel name (e.g., `"recall"`, `"rm3"`, `"rerank"`, `"temporal"`, `"adversarial"`, `"hybrid"`). Must match pattern `[a-zA-Z0-9_-]{1,64}`. |

#### Return Format

```json
{
  "name": "recall",
  "path": "/path/to/workspace/mind/recall.mind",
  "config": {
    "bm25": {
      "k1": "1.2",
      "b": "0.75"
    },
    "stemmer": {
      "algorithm": "porter2"
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | The kernel name. |
| `path` | `string` | Absolute file path to the `.mind` file. |
| `config` | `object` | Nested object with section names as keys, each containing key-value pairs from the kernel file. All values are strings. |

On error (kernel not found or invalid name):
```json
{
  "error": "Kernel 'nonexistent' not found"
}
```

#### Example

Request:
```json
{
  "tool": "get_mind_kernel",
  "arguments": {
    "name": "recall"
  }
}
```

Response:
```json
{
  "name": "recall",
  "path": "/path/to/workspace/mind/recall.mind",
  "config": {
    "bm25": {
      "k1": "1.2",
      "b": "0.75",
      "delta": "0.5"
    },
    "stemmer": {
      "algorithm": "porter2",
      "cache_size": "10000"
    }
  }
}
```

---

### category_summary

Returns category summaries relevant to a given topic. Categories are auto-generated from memory blocks during reindex and stored as Markdown files in `categories/`.

Default categories: `architecture`, `decisions`, `people`, `preferences`, `workflows`, `bugs`, `credentials`, `integrations`, `goals`, `constraints`. Additional categories can be defined in `mind-mem.json` under `categories.extra_categories`.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `topic` | `string` | *(required)* | Topic or query to find relevant categories for. |
| `limit` | `integer` | `3` | Maximum number of category summaries to return. Clamped to range [1, 10]. |

#### Return Format

When matching categories are found:
```json
{
  "topic": "database infrastructure",
  "matched_categories": ["architecture", "decisions"],
  "content": "## architecture\n\nBlock D-20260210-003: PostgreSQL is the primary...\n\n## decisions\n\nBlock D-20260211-001: All migrations use versioned SQL..."
}
```

When no categories match:
```json
{
  "topic": "quantum computing",
  "status": "no_categories",
  "hint": "Run reindex to generate category files, or add blocks with matching tags."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `topic` | `string` | The queried topic. |
| `matched_categories` | `array` | Category names that matched, sorted by keyword overlap score. |
| `content` | `string` | Concatenated Markdown summaries from matched category files (each truncated to 2000 chars). |
| `status` | `string` | Set to `"no_categories"` when nothing matches. |
| `hint` | `string` | Guidance for generating categories (only when none match). |

#### Example

Request:
```json
{
  "tool": "category_summary",
  "arguments": {
    "topic": "authentication security",
    "limit": 2
  }
}
```

Response:
```json
{
  "topic": "authentication security",
  "matched_categories": ["integrations", "constraints"],
  "content": "## integrations\n\nBlock D-20260212-005: OAuth 2.0 with PKCE...\n\n## constraints\n\nBlock D-20260209-001: All API endpoints require auth..."
}
```

---

### prefetch

Pre-assembles likely-needed context from recent conversation signals. Given entity mentions, topic keywords, or short phrases from the current conversation, anticipates what memory blocks will be needed next. Uses intent routing and category summaries internally.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `signals` | `string` | *(required)* | Comma-separated list of recent signals: entity names, topics, or keywords from the conversation. |
| `limit` | `integer` | `5` | Maximum blocks to return. Clamped to range [1, 20]. |

#### Return Format

JSON array of pre-ranked result objects (same format as [recall](#recall) results).

```json
[
  {
    "_id": "D-20260210-003",
    "type": "decision",
    "score": 14.2100,
    "excerpt": "PostgreSQL 16 is the primary data store...",
    "speaker": "",
    "tags": "database, infrastructure",
    "file": "decisions/DECISIONS.md",
    "line": 12,
    "status": "active",
    "DiaID": ""
  }
]
```

Returns an error if no signals are provided:
```json
{
  "error": "No signals provided. Pass comma-separated keywords."
}
```

On internal error:
```json
{
  "error": "description of the error",
  "signals": ["redis", "caching"]
}
```

#### Example

Request:
```json
{
  "tool": "prefetch",
  "arguments": {
    "signals": "Redis, caching, session management",
    "limit": 3
  }
}
```

Response:
```json
[
  {
    "_id": "D-20260215-003",
    "type": "decision",
    "score": 16.8901,
    "excerpt": "Use Redis for session caching...",
    "speaker": "",
    "tags": "infrastructure, caching, redis",
    "file": "decisions/DECISIONS.md",
    "line": 156,
    "status": "active",
    "DiaID": ""
  },
  {
    "_id": "D-20260210-003",
    "type": "decision",
    "score": 12.3400,
    "excerpt": "All caching layers must have TTL configured...",
    "speaker": "",
    "tags": "infrastructure, caching",
    "file": "decisions/DECISIONS.md",
    "line": 87,
    "status": "active",
    "DiaID": ""
  }
]
```

---

## Resources

Resources are read-only endpoints that return workspace data without side effects. They are accessed via the MCP resource protocol using URI-style identifiers.

### mind-mem://decisions

Returns all active decisions from `decisions/DECISIONS.md` as a JSON array of parsed block objects.

**URI:** `mind-mem://decisions`

**Response:** JSON array of block objects. Each block contains fields parsed from the Markdown structure including `_id`, `Statement`, `Date`, `Status`, `Tags`, and the raw content.

Returns an empty array `[]` if the decisions file does not exist.

---

### mind-mem://tasks

Returns all tasks from `tasks/TASKS.md` as a JSON array of parsed block objects.

**URI:** `mind-mem://tasks`

**Response:** JSON array of block objects with task-specific fields including `_id`, description, `Status`, `Tags`, and assignment metadata.

Returns an empty array `[]` if the tasks file does not exist.

---

### mind-mem://entities/{type}

Returns the raw Markdown content of an entity file.

**URI:** `mind-mem://entities/{entity_type}`

**Path parameter:** `entity_type` -- one of `projects`, `people`, `tools`, or `incidents`.

**Response:** Raw Markdown content of the corresponding entity file from `entities/{entity_type}.md`.

Returns an error object if the entity type is not one of the four allowed values:
```json
{
  "error": "Unknown entity type: foo. Use: incidents, people, projects, tools"
}
```

Returns `"File not found: entities/{entity_type}.md"` if the file does not exist.

---

### mind-mem://signals

Returns the raw Markdown content of auto-captured signals pending review.

**URI:** `mind-mem://signals`

**Response:** Raw Markdown content of `intelligence/SIGNALS.md`.

Returns `"File not found: intelligence/SIGNALS.md"` if the file does not exist.

---

### mind-mem://contradictions

Returns the raw Markdown content of detected contradictions between decisions.

**URI:** `mind-mem://contradictions`

**Response:** Raw Markdown content of `intelligence/CONTRADICTIONS.md`.

Returns `"File not found: intelligence/CONTRADICTIONS.md"` if the file does not exist.

---

### mind-mem://health

Returns a workspace health summary with block counts, active counts, and state metrics.

**URI:** `mind-mem://health`

**Response:**
```json
{
  "workspace": "/path/to/workspace",
  "files": {
    "decisions": { "total": 24, "active": 18 },
    "tasks": { "total": 8, "active": 5 },
    "contradictions": { "total": 2, "active": 2 },
    "signals": { "total": 7, "active": 7 }
  },
  "metrics": {
    "last_scan": "2026-02-18T14:30:00",
    "coverage": 0.92
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `workspace` | `string` | Absolute path to the workspace. |
| `files` | `object` | Per-file block counts. Keys: `decisions`, `tasks`, `contradictions`, `signals`. Each has `total` and `active` integer fields. |
| `metrics` | `object` | Metrics from `memory/intel-state.json` if available. Empty object `{}` otherwise. |

---

### mind-mem://recall/{query}

Search memory using ranked recall (FTS5 or BM25 scan). This is the resource equivalent of the `recall` tool, with `limit` fixed at 10 and `active_only` set to false.

**URI:** `mind-mem://recall/{query}`

**Path parameter:** `query` -- the search query string.

**Response:** JSON array of ranked results (same format as the [recall](#recall) tool response).

---

### mind-mem://ledger

Returns the shared fact ledger for multi-agent memory propagation.

**URI:** `mind-mem://ledger`

**Response:** Raw Markdown content of `shared/intelligence/LEDGER.md`.

Returns `"File not found: shared/intelligence/LEDGER.md"` if the file does not exist.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MIND_MEM_WORKSPACE` | `.` (current directory) | Path to the memory workspace root. |
| `MIND_MEM_TOKEN` | *(none)* | Bearer token for HTTP transport authentication. When unset, HTTP transport allows unauthenticated access. |

## Server Configuration

### Claude Desktop

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/path/to/mind-mem/mcp_server.py"],
      "env": {
        "MIND_MEM_WORKSPACE": "/path/to/workspace"
      }
    }
  }
}
```

### Claude Code / CLI

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/path/to/mind-mem/mcp_server.py"],
      "env": {
        "MIND_MEM_WORKSPACE": "/path/to/workspace"
      }
    }
  }
}
```

### HTTP Transport

```bash
# Basic (no auth)
python3 mcp_server.py --transport http --port 8765

# With token auth
MIND_MEM_TOKEN=secret python3 mcp_server.py --transport http --port 8765

# Or via CLI flag
python3 mcp_server.py --transport http --port 8765 --token secret
```

HTTP clients authenticate via `Authorization: Bearer <token>` or `X-MindMem-Token: <token>` headers. Token comparison uses constant-time `hmac.compare_digest`.

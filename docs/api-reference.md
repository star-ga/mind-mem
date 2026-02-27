# API Reference

## Python API

### Core Functions

#### `recall(workspace: str, query: str, limit: int = 10, **kwargs) -> list[dict]`
Search blocks using BM25F scoring with graph boost, fact card aggregation, and knee cutoff.

**Parameters:**
- `workspace` — Path to the mind-mem workspace
- `query` — Search query (supports stemming and domain-aware expansion)
- `limit` — Maximum results (default: 10)
- `active_only` — Only return blocks with `Status: active` (default: False)
- `include_pending` — Include pending signals in results (default: False)

**Returns:** List of dicts with `_id`, `score`, `Statement`, `Type`, and matched fields.

#### `init_workspace(path: str) -> None`
Initialize a new mind-mem workspace with the standard directory structure.

#### `parse_blocks(workspace: str) -> list[dict]`
Parse all memory blocks from `.md` files in a workspace.

---

## MCP Server (19 tools, 8 resources)

The MCP server exposes 19 tools via JSON-RPC. See [MCP Tool Examples](mcp-tool-examples.md).

### Starting the Server

```bash
python3 mcp_server.py --workspace /path/to/workspace
```

### Tool Reference

#### Retrieval

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `recall` | BM25/hybrid search with graph boost and knee cutoff | `query`, `limit`, `active_only`, `backend` |
| `hybrid_search` | BM25+Vector RRF fusion (wrapper for `recall(backend="hybrid")`) | `query`, `limit`, `active_only` |
| `find_similar` | Vector similarity search from a block ID | `block_id`, `limit` |
| `intent_classify` | Classify query intent (WHY/WHEN/ENTITY/WHAT/HOW/LIST/VERIFY/COMPARE/TRACE) | `query` |
| `prefetch` | Pre-assemble context from conversation signals | `signals`, `limit` |
| `retrieval_diagnostics` | Pipeline diagnostics: stage rejection rates, intent distribution | `last_n`, `max_age_days` |

#### Memory Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `propose_update` | Propose a decision or task (writes to SIGNALS.md) | `block_type`, `statement`, `rationale`, `tags`, `confidence` |
| `approve_apply` | Apply a staged proposal (dry_run default) | `proposal_id`, `dry_run` |
| `rollback_proposal` | Rollback an applied proposal by receipt timestamp | `receipt_ts` |
| `delete_memory_item` | Delete a block by ID from its source file | `block_id` |
| `export_memory` | Export all blocks as JSONL | `format`, `include_metadata` |
| `memory_evolution` | A-MEM metadata: importance, access patterns, keywords | `block_id`, `action` |

#### Workspace Operations

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `scan` | Integrity scan: contradictions, drift, dead decisions | — |
| `list_contradictions` | Detected contradictions with resolution strategies | — |
| `reindex` | Rebuild FTS5 + optional vector index | `include_vectors` |
| `index_stats` | Block counts, staleness, vector coverage, kernel status | — |
| `category_summary` | Category-based topic summaries | `topic`, `limit` |

#### MIND Kernels

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_mind_kernels` | List available `.mind` kernel configs | — |
| `get_mind_kernel` | Read a specific kernel configuration | `name` |

### Configuration

The server reads `mind-mem.json` from the workspace root:

```json
{
  "recall": {
    "top_k": 18,
    "min_score": 0.1,
    "knee_cutoff": true
  },
  "auto_ingest": {
    "enabled": true,
    "transcript_scan": true,
    "entity_ingest": true
  }
}
```

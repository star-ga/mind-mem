# MCP Tool Examples

Practical examples for each mind-mem MCP tool.

## recall

Search memory using BM25 with graph boost and knee cutoff.

```json
{
  "tool": "recall",
  "arguments": {
    "query": "database migration decision",
    "limit": 5
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "block_id": "DEC-20260215-001",
      "score": 8.42,
      "text": "Decision: Migrate from SQLite to PostgreSQL for production..."
    }
  ]
}
```

## propose_update

Propose a memory update (requires human approval).

```json
{
  "tool": "propose_update",
  "arguments": {
    "operation": "add",
    "block_type": "Decision",
    "content": "[DEC-NEW]\nType: Decision\nStatement: Use Redis for session caching\nRationale: Lower latency than database queries",
    "reason": "Decided during architecture review"
  }
}
```

## hybrid_search

Combined BM25 + vector + RRF fusion search.

```json
{
  "tool": "hybrid_search",
  "arguments": {
    "query": "authentication flow design",
    "limit": 10,
    "alpha": 0.6
  }
}
```

## find_similar

Find blocks similar to a given block.

```json
{
  "tool": "find_similar",
  "arguments": {
    "block_id": "DEC-20260215-001",
    "limit": 5
  }
}
```

## intent_classify

Classify a query's intent type.

```json
{
  "tool": "intent_classify",
  "arguments": {
    "query": "When did we decide to use PostgreSQL?"
  }
}
```

**Response:**
```json
{
  "intent": "WHEN",
  "confidence": 0.92
}
```

## scan

Run integrity scan for contradictions, drift, and dead decisions.

```json
{
  "tool": "scan",
  "arguments": {}
}
```

## list_contradictions

Show open contradictions between decisions.

```json
{
  "tool": "list_contradictions",
  "arguments": {}
}
```

## index_stats

Show memory index statistics.

```json
{
  "tool": "index_stats",
  "arguments": {}
}
```

**Response:**
```json
{
  "total_blocks": 142,
  "categories": {"Decision": 45, "Task": 38, "Entity": 59},
  "fts_indexed": 142,
  "vector_indexed": 0,
  "last_reindex": "2026-02-25T08:00:00Z"
}
```

## category_summary

Get category-based topic summaries.

```json
{
  "tool": "category_summary",
  "arguments": {
    "category": "Decision"
  }
}
```

## prefetch

Pre-assemble context based on conversation signals.

```json
{
  "tool": "prefetch",
  "arguments": {
    "signals": "database, migration, PostgreSQL",
    "limit": 5
  }
}
```

## reindex

Rebuild FTS5 and optional vector index.

```json
{
  "tool": "reindex",
  "arguments": {
    "include_vectors": false
  }
}
```

## approve_apply

Apply a staged proposal.

```json
{
  "tool": "approve_apply",
  "arguments": {
    "proposal_id": "prop-abc123",
    "dry_run": false
  }
}
```

## rollback_proposal

Roll back a previously applied proposal.

```json
{
  "tool": "rollback_proposal",
  "arguments": {
    "proposal_id": "prop-abc123"
  }
}
```

## memory_evolution

Track block history (edits, supersedes).

```json
{
  "tool": "memory_evolution",
  "arguments": {
    "block_id": "DEC-20260215-001"
  }
}
```

## delete_memory_item

Delete a memory block by ID.

```json
{
  "tool": "delete_memory_item",
  "arguments": {
    "block_id": "TASK-OLD-001",
    "reason": "Task completed and archived"
  }
}
```

## export_memory

Export memory as JSONL.

```json
{
  "tool": "export_memory",
  "arguments": {
    "format": "jsonl",
    "include_metadata": true
  }
}
```

## list_mind_kernels / get_mind_kernel

List and read MIND scoring kernels.

```json
{
  "tool": "list_mind_kernels",
  "arguments": {}
}
```

```json
{
  "tool": "get_mind_kernel",
  "arguments": {
    "kernel_name": "bm25f_score"
  }
}
```

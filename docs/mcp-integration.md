# MCP Integration Guide

## Overview

mind-mem exposes 19 MCP tools for integration with AI coding assistants like Claude Code.

## Setup

### Claude Code

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

## Tool Reference

### Retrieval Tools

| Tool | Description |
|------|-------------|
| `recall` | BM25F search with graph boost and knee cutoff |
| `hybrid_search` | Combined BM25 + vector + RRF fusion |
| `find_similar` | Find blocks similar to a given block ID |
| `prefetch` | Pre-assemble context based on signals |

### Mutation Tools

| Tool | Description |
|------|-------------|
| `propose_update` | Propose a memory change (requires approval) |
| `approve_apply` | Apply a staged proposal |
| `rollback_proposal` | Roll back a previously applied proposal |
| `delete_memory_item` | Delete a memory block |

### Analysis Tools

| Tool | Description |
|------|-------------|
| `scan` | Run integrity scan |
| `list_contradictions` | Show contradictions between blocks |
| `intent_classify` | Classify query intent type |
| `memory_evolution` | Track block history |
| `category_summary` | Topic summaries by category |

### Infrastructure Tools

| Tool | Description |
|------|-------------|
| `index_stats` | Memory index statistics |
| `reindex` | Rebuild FTS5 + vector index |
| `list_mind_kernels` | List MIND scoring kernels |
| `get_mind_kernel` | Read a MIND kernel |
| `export_memory` | Export as JSONL |

## Best Practices

1. Use `recall` for most queries — it's fast and well-tuned
2. Use `hybrid_search` when semantic matching matters more than keywords
3. Always use `propose_update` instead of direct file edits
4. Run `scan` periodically to check for contradictions

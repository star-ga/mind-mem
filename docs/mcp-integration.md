# MCP Integration Guide

## Overview

mind-mem exposes 51 MCP tools for integration with AI coding assistants like Claude Code.

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
| `verify_chain` | Verify mutation audit chain integrity |
| `list_evidence` | List evidence entries with filters |
| `get_block` | Retrieve a single block by ID |
| `memory_health` | Full health summary (stats, drift, contradictions) |
| `traverse_graph` | Walk cross-reference graph from a block |
| `compact` | Archive old blocks, clean snapshots |
| `stale_blocks` | Find blocks with no recent access |

### Quality & Feedback Tools

| Tool | Description |
|------|-------------|
| `calibration_feedback` | Submit quality feedback for a block |
| `calibration_stats` | View per-block quality distributions |
| `retrieval_diagnostics` | Analyze recent retrieval performance |

### Search Enhancement Tools

| Tool | Description |
|------|-------------|
| `expand_query` | Generate semantically diverse query reformulations with RRF fusion |
| `smart_chunk` | Split content at semantic boundaries (headers, paragraphs, code blocks) |
| `chunk_and_index` | Chunk content and add resulting blocks to the index |
| `deduplicate_results` | Apply 4-layer dedup to search results (per-source, cosine, type, chunk cap) |
| `dedup_search` | Run recall + automatic deduplication in one call |

### Knowledge Enrichment Tools

| Tool | Description |
|------|-------------|
| `run_dream_cycle` | Run autonomous memory enrichment (scan, repair, consolidate) |
| `dream_cycle_status` | Check last dream cycle run status and results |
| `compile_truth` | Compile a truth page for an entity from all known evidence |
| `get_compiled_truth` | Retrieve a compiled truth page by entity name |
| `compiled_truth_load` | Load or create a compiled truth page for an entity |
| `compiled_truth_add_evidence` | Add timestamped evidence to an entity's truth page |
| `compiled_truth_contradictions` | Detect contradictions in an entity's evidence trail |
| `list_compiled_truths` | List all compiled truth pages in the workspace |

## Best Practices

1. Use `recall` for most queries — it's fast and well-tuned
2. Use `hybrid_search` when semantic matching matters more than keywords
3. Always use `propose_update` instead of direct file edits
4. Run `scan` periodically to check for contradictions
5. Run `dream_cycle` during idle periods to repair broken references and consolidate memory
6. Use `compiled_truth_load` to build per-entity knowledge pages that accumulate across sessions
7. Use `calibration_feedback` to improve retrieval quality over time — the system learns from your corrections

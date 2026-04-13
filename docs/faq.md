# FAQ

## General

### What is mind-mem?
mind-mem is a persistent, auditable, contradiction-safe memory system for AI coding agents. It provides BM25F-based retrieval with graph boost, fact indexing, and adaptive cutoff.

### Does it require a database?
No. mind-mem uses zero external dependencies. All data is stored as plain markdown files.

### What Python versions are supported?
Python 3.10, 3.12, 3.13, and 3.14 are tested in CI.

## Retrieval

### How does scoring work?
mind-mem uses BM25F (BM25 with field weights) as the primary scoring algorithm. See [Scoring](scoring.md) for details.

### What is the knee cutoff?
Instead of returning a fixed number of results, mind-mem finds the steepest score drop and truncates there. This adapts to query difficulty.

### What is graph boost?
Blocks frequently co-retrieved get linked. Querying one surfaces its neighbors via score propagation.

## MCP Integration

### How do I set up the MCP server?
See [Getting Started](getting-started.md#mcp-server) for configuration.

### How many MCP tools are available?
35 tools including recall, recall_with_axis, propose_update, scan, hybrid_search, retrieval_diagnostics, and more.

## Troubleshooting

### Recall returns no results
Check that your workspace has blocks in the expected format. Each block needs an ID in brackets and at least a Type and Statement field.

### Performance is slow
Ensure your workspace isn't too large. Consider using the `limit` parameter to reduce result count.

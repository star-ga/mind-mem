# API Reference

## Python API

### Core Functions

#### `init_workspace(path: str) -> None`
Initialize a new mind-mem workspace at the given path.

#### `parse_blocks(workspace: str) -> list[dict]`
Parse all memory blocks from a workspace.

#### `recall_blocks(query: str, blocks: list[dict], limit: int = 10) -> list[dict]`
Search blocks using BM25 scoring.

### MCP Server

The MCP server exposes 19 tools via JSON-RPC. See [MCP Tool Examples](mcp-tool-examples.md).

#### Starting the Server

```bash
python3 mcp_server.py --workspace /path/to/workspace
```

#### Configuration

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

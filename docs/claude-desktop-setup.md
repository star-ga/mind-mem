# Claude Desktop Setup Guide

Step-by-step guide to connect mind-mem to Claude Desktop as an MCP server.

## Prerequisites

- [Claude Desktop](https://claude.ai/desktop) installed
- Python 3.10+
- `fastmcp` installed: `pip install fastmcp`

## Step 1: Initialize a workspace

```bash
# Create a workspace in your project directory
python3 /path/to/mind-mem/scripts/init_workspace.py /path/to/your/workspace
```

This creates the full directory structure with 12 directories and 19 template files.

## Step 2: Configure Claude Desktop

Edit `~/.claude/claude_desktop_config.json` (create it if it doesn't exist):

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/path/to/mind-mem/mcp_server.py"],
      "env": {
        "MIND_MEM_WORKSPACE": "/path/to/your/workspace"
      }
    }
  }
}
```

Replace both paths with your actual paths.

## Step 3: Restart Claude Desktop

Quit and reopen Claude Desktop. The mind-mem server will start automatically.

## Step 4: Verify

In Claude Desktop, you should now see mind-mem tools available:

- **recall** — Search memory with BM25
- **propose_update** — Propose new decisions/tasks
- **scan** — Run integrity scan
- **hybrid_search** — BM25+Vector+RRF fusion search
- **intent_classify** — Classify query intent
- And 9 more tools (14 total)

## Step 5: Test

Ask Claude:

> "Use the recall tool to search for 'authentication'"

If the workspace is empty, you'll get "No results found" — that's correct.

## Available Resources

Claude Desktop can also read these resources directly:

| Resource         | URI                            |
|------------------|--------------------------------|
| Active decisions | `mind-mem://decisions`         |
| All tasks        | `mind-mem://tasks`             |
| Entities         | `mind-mem://entities/projects` |
| Health summary   | `mind-mem://health`            |
| Search results   | `mind-mem://recall/your-query` |
| Shared ledger    | `mind-mem://ledger`            |

## Troubleshooting

| Problem                                  | Solution                                                                 |
|------------------------------------------|--------------------------------------------------------------------------|
| Server doesn't appear in Claude Desktop  | Check the config JSON syntax. Restart Claude Desktop.                    |
| "No module named 'fastmcp'"              | Run `pip install fastmcp` in the Python environment Claude Desktop uses. |
| "MIND_MEM_WORKSPACE not set"             | Add the `env` block to your config.                                      |
| Tools fail with "No mind-mem.json found" | Run `init_workspace.py` on your workspace first.                         |

## HTTP Transport (Remote)

For remote or multi-client setups:

```bash
MIND_MEM_WORKSPACE=/path/to/workspace \
MIND_MEM_TOKEN=your-secret \
python3 mcp_server.py --transport http --port 8765
```

Then configure Claude Desktop to use the HTTP endpoint instead of stdio.

# Getting Started with Mind-Mem in Claude Desktop

Set up Mind-Mem as an MCP server in Claude Desktop in under 5 minutes.

## Prerequisites

- [Claude Desktop](https://claude.ai/download) installed
- Python 3.10+
- `fastmcp` installed: `pip install "fastmcp>=2.0"`

## 1. Clone Mind-Mem

```bash
git clone https://github.com/star-ga/mind-mem.git ~/mind-mem
```

## 2. Initialize a Workspace

```bash
python3 ~/mind-mem/scripts/init_workspace.py ~/my-workspace
```

This creates the full directory structure (decisions, tasks, entities, intelligence, etc.) and `mind-mem.json` config.

## 3. Configure Claude Desktop

Open your Claude Desktop config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add the `mind-mem` entry under `mcpServers`:

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/Users/you/mind-mem/mcp_server.py"],
      "env": {
        "MIND_MEM_WORKSPACE": "/Users/you/my-workspace"
      }
    }
  }
}
```

Replace `/Users/you/` with your actual paths.

## 4. Restart Claude Desktop

Quit and reopen Claude Desktop. You should see "mind-mem" listed as a connected MCP server.

## 5. Test It

Try these in Claude Desktop:

**Search memory:**
> "Use the recall tool to search for 'authentication'"

**Propose a decision:**
> "Use propose_update to propose a decision: 'Use PostgreSQL for the user database' with rationale 'Better JSON support' and tags 'database, infrastructure'"

**Run integrity scan:**
> "Use the scan tool to check workspace health"

**Check contradictions:**
> "Use list_contradictions to check for conflicting decisions"

## What to Expect

### Resources (read-only context)

Claude Desktop can read these automatically:

| Resource | What it provides |
|---|---|
| `mind-mem://decisions` | Active decisions |
| `mind-mem://tasks` | All tasks |
| `mind-mem://health` | Workspace health summary |
| `mind-mem://signals` | Pending signals |
| `mind-mem://contradictions` | Detected contradictions |
| `mind-mem://recall/{query}` | Search results |

### Tools (actions)

| Tool | What it does | Safety |
|---|---|---|
| `recall` | BM25 search across all memory | Read-only |
| `propose_update` | Propose a decision or task | Writes to SIGNALS.md only |
| `approve_apply` | Apply a staged proposal | Defaults to dry_run=True |
| `rollback_proposal` | Rollback an applied proposal | Restores from snapshot |
| `scan` | Run integrity scan | Read-only |
| `list_contradictions` | Show contradictions | Read-only |

### Safety Guarantees

- `propose_update` **never** writes to `DECISIONS.md` or `TASKS.md` — all proposals land in `intelligence/SIGNALS.md`
- `approve_apply` defaults to `dry_run=True` — you must explicitly set `dry_run=False` to apply
- All resources are read-only — no tool can mutate source of truth without your explicit approval
- Every apply creates a snapshot for rollback

## HTTP Transport (Multi-Client)

For remote access or multiple clients:

```bash
MIND_MEM_WORKSPACE=~/my-workspace python3 ~/mind-mem/mcp_server.py --transport http --port 8765
```

With token auth:

```bash
MIND_MEM_TOKEN=your-secret MIND_MEM_WORKSPACE=~/my-workspace python3 ~/mind-mem/mcp_server.py --transport http --port 8765
```

## Troubleshooting

| Problem | Solution |
|---|---|
| Server not appearing in Claude Desktop | Check paths in config, restart Claude Desktop |
| "fastmcp not found" | Run `pip install "fastmcp>=2.0"` |
| "No mind-mem.json found" | Run `init_workspace.py` first |
| Recall returns empty | Workspace has no data yet — add some decisions first |
| Token auth errors | Set `MIND_MEM_TOKEN` env var or use `--token` flag |

## Next Steps

1. **Add decisions**: Use `propose_update` to start populating your workspace
2. **Review signals**: Check `intelligence/SIGNALS.md` for pending proposals
3. **Enable hooks**: Add session-start/end hooks for automatic scanning (see main README)
4. **Try multi-agent**: Set up namespaces with `python3 maintenance/namespaces.py`

# Getting Started

## Installation

```bash
pip install mind-mem
```

For development:

```bash
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem
pip install -e ".[dev]"
```

## Quick Start

### Initialize a Workspace

```python
from scripts.init_workspace import init

init("/path/to/workspace")
```

This creates the standard directory structure:
- `decisions/` — Active decisions and constraints
- `tasks/` — Task tracking
- `entities/` — People, projects, tools
- `memory/` — Daily logs
- `intelligence/` — Signals and analysis
- `summaries/` — Category summaries

### Add Memory Blocks

Create markdown files in the workspace directories:

```markdown
[PROJ-001]
Type: Decision
Statement: Use BM25F as the primary scoring algorithm
Tags: architecture, scoring
Date: 2026-02-01
```

### Search Memory

```python
from scripts._recall_core import recall

results = recall("/path/to/workspace", "scoring algorithm", limit=10)
for r in results:
    print(f"{r['id']}: {r.get('statement', '')[:80]}")
```

### MCP Server

mind-mem includes an MCP server for integration with AI coding assistants:

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

## Next Steps

- [Scoring System](scoring.md) — Understand how results are ranked
- [Architecture](architecture.md) — System design overview
- [API Reference](api-reference.md) — Full API documentation
- [Troubleshooting](troubleshooting.md) — Common issues and solutions

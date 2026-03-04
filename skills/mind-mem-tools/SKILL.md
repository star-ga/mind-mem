---
name: mind-mem
description: Persistent, auditable, contradiction-safe memory for AI coding agents. Use this skill when working with mind-mem MCP tools — recall, propose_update, scan, approve_apply, and 15 other memory management tools.
source: star-ga/mind-mem
license: MIT
---

# mind-mem — Memory OS for AI Coding Agents

Drop-in memory layer for Claude Code, Codex CLI, Gemini CLI, Cursor, Windsurf, Zed, OpenClaw, or any MCP-compatible agent. Local-first, zero-infrastructure, governance-aware.

## When to Use

Trigger when the user:
- Asks about memory, recall, or persistence across sessions
- Wants to store decisions, entities, signals, or observations
- Needs contradiction detection or drift analysis
- Works with `mind-mem.json` configuration
- Uses any `mcp__mind-mem__*` tool

## Install

```bash
pip install mind-mem
mind-mem-init ~/my-workspace
```

Or from source:
```bash
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem && pip install -e .
```

MCP config (add to Claude Code `settings.json` or any MCP client):
```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/path/to/mind-mem/mcp_server.py"],
      "env": { "MIND_MEM_WORKSPACE": "/path/to/workspace" }
    }
  }
}
```

Auto-configure all clients at once:
```bash
./install.sh --all
```

## MCP Tools Reference (19 tools)

### Core Memory

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `recall` | Hybrid BM25 + vector search with RRF fusion | `query`, `limit`, `active_only`, `backend` |
| `propose_update` | Propose a memory change (never writes directly) | `category`, `content`, `source` |
| `approve_apply` | Apply a pending proposal to source of truth | `proposal_id` |
| `rollback_proposal` | Reject/discard a pending proposal | `proposal_id` |
| `scan` | Detect contradictions, drift, and staleness | `workspace` |
| `list_contradictions` | List detected contradictions | `limit` |
| `delete_memory_item` | Remove a specific memory item | `item_id` |

### Search & Discovery

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `hybrid_search` | *(Deprecated)* Use `recall(backend='hybrid')` | `query`, `limit` |
| `find_similar` | Semantic similarity search | `query`, `limit` |
| `intent_classify` | Classify query intent for routing | `query` |
| `category_summary` | Summarize memory by category | `category` |
| `prefetch` | Pre-warm search cache for a topic | `query` |

### Diagnostics & Admin

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `index_stats` | FTS index and corpus statistics | — |
| `retrieval_diagnostics` | Debug recall pipeline performance | `query` |
| `reindex` | Rebuild search index | `force` |
| `memory_evolution` | Track how memory changed over time | `item_id` |
| `export_memory` | Export memory as structured JSON | `format` |

### MIND Kernels

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_mind_kernels` | List available MIND scoring kernels | — |
| `get_mind_kernel` | Get kernel details and scoring logic | `kernel_name` |

## Recall Backends

```
recall(query="API decisions", backend="scan")     # BM25F full-text (default)
recall(query="API decisions", backend="hybrid")   # BM25 + vector + RRF fusion
recall(query="API decisions", backend="vector")   # Pure vector similarity
```

## Memory Categories

mind-mem organizes memory into categories stored as Markdown files:

| Category | File | Content |
|----------|------|---------|
| `decision` | `decisions/DECISIONS.md` | Architecture and design decisions |
| `entity` | `entities/*.md` | Projects, tools, people, services |
| `signal` | `categories/signals.md` | Trends, patterns, observations |
| `observation` | `categories/observations.md` | Session learnings |
| `preference` | `categories/preferences.md` | User preferences and workflows |
| `governance` | `governance/*.md` | Rules, constraints, policies |

## Propose-Apply Workflow

mind-mem NEVER writes directly to source of truth. All changes go through governance:

```
1. Agent calls propose_update(category, content, source)
2. Proposal is queued (pending state)
3. Human reviews: approve_apply(proposal_id)  or  rollback_proposal(proposal_id)
4. Only approved proposals mutate files
5. Every apply is logged with timestamp + diff
```

Budget limits prevent runaway proposals:
- `per_run`: 3 proposals per agent turn (default)
- `per_day`: 6 proposals per day (default)
- `backlog_limit`: 30 pending max (default)

## Configuration

`mind-mem.json` in workspace root:

```json
{
  "version": "1.8.2",
  "recall": {
    "backend": "scan",
    "rrf_k": 60,
    "vector_enabled": false,
    "vector_model": "all-MiniLM-L6-v2",
    "cross_encoder": {
      "enabled": false,
      "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
  },
  "governance_mode": "detect_only",
  "proposal_budget": {
    "per_run": 3,
    "per_day": 6,
    "backlog_limit": 30
  },
  "auto_capture": true,
  "auto_recall": true
}
```

### Governance Modes

| Mode | Behavior |
|------|----------|
| `detect_only` | Flag contradictions, allow all proposals |
| `warn` | Flag + warn on contradictions |
| `strict` | Block proposals that contradict existing memory |

### Vector Backends

| Backend | Config Key | Use Case |
|---------|-----------|----------|
| sqlite-vec | `"provider": "local"` | Local, zero-infra (default) |
| fastembed | `"onnx_backend": true` | CPU-optimized ONNX |
| Qdrant | `"provider": "qdrant"` | Dedicated vector DB |
| Pinecone | `"provider": "pinecone"` | Managed cloud vector |

## Key Principles

- **Deterministic**: Same input → same output. No ML in core pipeline.
- **Auditable**: Every apply logged with timestamp, receipt, and diff.
- **Local-first**: All data on disk. No cloud calls, no telemetry.
- **No silent mutation**: Nothing writes to source of truth without `/apply`.
- **Zero infrastructure**: Python 3.10+ and stdlib only. No Redis, no Postgres.
- **Shared memory**: All MCP clients share one workspace via SQLite WAL.

## MIND Kernels

17 built-in scoring kernels for relevance, freshness, authority, and other signals:

```
recall(query="...", kernel="relevance")    # Default BM25 scoring
recall(query="...", kernel="freshness")    # Prefer recent memory
recall(query="...", kernel="authority")    # Prefer governance/decision items
```

Use `list_mind_kernels()` to see all available kernels and `get_mind_kernel(name)` for details.

## Links

- **PyPI**: [mind-mem](https://pypi.org/project/mind-mem/)
- **GitHub**: [star-ga/mind-mem](https://github.com/star-ga/mind-mem)
- **Docs**: See `docs/` directory in repo

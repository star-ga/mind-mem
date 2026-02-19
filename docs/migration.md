# Migration Guide: mem-os to mind-mem

This guide covers migrating from `mem-os` (archived at `star-ga/mem-os`) to its successor `mind-mem` (`star-ga/mind-mem`, v1.0.2+). The workspace data format is compatible -- no data loss occurs during migration.

**Requirements:** Python 3.10+, FastMCP 2.0+ (for MCP server).

---

## Quick Migration Checklist

1. Clone `star-ga/mind-mem` (or update your local copy)
2. Rename `mem-os.json` to `mind-mem.json` in your workspace root
3. Run `mind-mem-migrate` to upgrade the workspace schema
4. Update `~/.claude/mcp.json` to point to `mind-mem/mcp_server.py`
5. Replace any `mem-os://` resource URIs with `mind-mem://`
6. Replace `X-MemOS-Token` headers with `X-MindMem-Token` (HTTP transport only)
7. Update any scripts using old console commands (e.g., `mem-os-scan` to `mind-mem-scan`)
8. Install: `pip install -e '.[all]'`

---

## 1. Package Rename

The Python package was renamed from `mem-os` / `mem_os` to `mind-mem` / `mind_mem`.

### Console Scripts

All 10 CLI entry points have been renamed:

| Old (mem-os)         | New (mind-mem)         | Purpose                              |
|----------------------|------------------------|--------------------------------------|
| `mem-os-init`        | `mind-mem-init`        | Initialize a new workspace           |
| `mem-os-scan`        | `mind-mem-scan`        | Run intelligence scan                |
| `mem-os-recall`      | `mind-mem-recall`      | Search memory (BM25)                 |
| `mem-os-capture`     | `mind-mem-capture`     | Capture signals from input           |
| `mem-os-validate`    | `mind-mem-validate`    | Validate workspace integrity         |
| `mem-os-mcp`         | `mind-mem-mcp`         | Start MCP server                     |
| `mem-os-migrate`     | `mind-mem-migrate`     | Run schema migrations                |
| `mem-os-backup`      | `mind-mem-backup`      | Backup/restore workspace             |
| `mem-os-compact`     | `mind-mem-compact`     | Compact memory blocks                |
| `mem-os-resolve`     | `mind-mem-resolve`     | Resolve contradictions               |

### Install

```bash
# Uninstall old package
pip uninstall mem-os

# Install mind-mem
cd /path/to/mind-mem
pip install -e '.[all]'
```

---

## 2. Config File Rename

The workspace config file was renamed from `mem-os.json` to `mind-mem.json`.

```bash
# In your workspace root:
mv mem-os.json mind-mem.json
```

### New Config Keys

mind-mem v1.0.2 adds two optional top-level keys not present in mem-os:

```json
{
  "recall": {
    "backend": "scan",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0,
    "vector_model": "all-MiniLM-L6-v2",
    "vector_enabled": false,
    "onnx_backend": false
  },
  "prompts": {
    "observation_compress": "",
    "entity_extract": "",
    "category_distill": ""
  },
  "categories": {
    "enabled": true,
    "extra_categories": {}
  }
}
```

- `prompts` -- Override default LLM prompts for observation compression, entity extraction, and category distillation. Leave empty strings to use built-in defaults.
- `categories` -- Enable/disable the category distiller and define custom categories beyond the auto-detected ones.

### Field Rename: `self_correcting_mode` to `governance_mode`

If your `mem-os.json` used `self_correcting_mode`, it must be renamed to `governance_mode`. The `mind-mem-migrate` command handles this automatically (see Section 4).

```json
// Old (mem-os)
{ "self_correcting_mode": "detect_only" }

// New (mind-mem)
{ "governance_mode": "detect_only" }
```

The same rename applies to `memory/intel-state.json`.

---

## 3. MCP Server Config Update

Update `~/.claude/mcp.json` (or `~/.claude/claude_desktop_config.json` for Claude Desktop).

### Before (mem-os)

```json
{
  "mcpServers": {
    "mem-os": {
      "command": "python3",
      "args": ["/home/you/mem-os/mcp_server.py"],
      "env": {
        "MEM_OS_WORKSPACE": "/home/you/.openclaw/workspace"
      }
    }
  }
}
```

### After (mind-mem)

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/home/you/mind-mem/mcp_server.py"],
      "env": {
        "MIND_MEM_WORKSPACE": "/home/you/.openclaw/workspace"
      }
    }
  }
}
```

Note: The environment variable changed from `MEM_OS_WORKSPACE` to `MIND_MEM_WORKSPACE`.

For HTTP transport with token auth:

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/home/you/mind-mem/mcp_server.py", "--transport", "http", "--port", "8765"],
      "env": {
        "MIND_MEM_WORKSPACE": "/home/you/.openclaw/workspace",
        "MIND_MEM_TOKEN": "your-secret-token"
      }
    }
  }
}
```

---

## 4. Workspace Schema Migration

Run the migration tool to upgrade your workspace schema from 1.0.0 through to 2.1.0:

```bash
mind-mem-migrate /path/to/workspace
```

Or if not installed as a package:

```bash
python3 /path/to/mind-mem/scripts/schema_version.py /path/to/workspace
```

The migration performs three steps (all idempotent, safe to re-run):

| Step         | Changes                                                                    |
|--------------|----------------------------------------------------------------------------|
| 1.0.0 -> 2.0.0 | Creates `intelligence/proposed/` and `shared/` directories; adds `schema_version` to config |
| 2.0.0 -> 2.1.0 | Renames `self_correcting_mode` to `governance_mode` in both `mind-mem.json` and `memory/intel-state.json` |

After migration, the workspace also supports new directories used by mind-mem features:

- `categories/` -- Auto-generated thematic summaries (created on first category distillation)
- `mind/` -- MIND kernel configs (optional, for custom kernels)

These directories are created automatically on first use; the migration tool does not create them.

---

## 5. HTTP Auth Header Rename

If you use HTTP transport with token authentication, update client code to use the new header name:

```
# Old
X-MemOS-Token: your-secret-token

# New
X-MindMem-Token: your-secret-token
```

The `Authorization: Bearer <token>` header continues to work unchanged.

---

## 6. MCP Resource URIs

All resource URIs changed from the `mem-os://` scheme to `mind-mem://`:

| Old (mem-os)                   | New (mind-mem)                    |
|--------------------------------|-----------------------------------|
| `mem-os://decisions`           | `mind-mem://decisions`            |
| `mem-os://tasks`               | `mind-mem://tasks`                |
| `mem-os://entities/{type}`     | `mind-mem://entities/{type}`      |
| `mem-os://signals`             | `mind-mem://signals`              |
| `mem-os://contradictions`      | `mind-mem://contradictions`       |
| `mem-os://health`              | `mind-mem://health`               |
| `mem-os://recall/{query}`      | `mind-mem://recall/{query}`       |
| `mem-os://ledger`              | `mind-mem://ledger`               |

Update any client code or tool configurations that reference these URIs.

---

## 7. New Features in mind-mem v1.0.2

These features are new in mind-mem and were not available in mem-os:

- **Category distillation** -- Auto-generated thematic summaries from block tags and keywords, with `_manifest.json` tracking
- **Prefetch context** -- Anticipatory pre-assembly of likely-needed memory blocks using intent routing and category signals
- **14 MIND kernels** -- Native C99 computation kernels (BM25, RRF, ranking, reranking, temporal, adversarial, etc.) with FFI bridge and pure Python fallback
- **16 MCP tools** (was 6 in mem-os) -- Added `hybrid_search`, `find_similar`, `intent_classify`, `index_stats`, `reindex`, `memory_evolution`, `list_mind_kernels`, `get_mind_kernel`, `category_summary`, `prefetch`
- **Hybrid BM25+Vector search** with Reciprocal Rank Fusion (RRF)
- **RM3 pseudo-relevance feedback** for query expansion
- **A-MEM block metadata evolution** -- Tracks block creation, access, and mutation history
- **Intent router** -- Classifies queries into 9 types for optimized retrieval strategy
- **Cross-encoder reranking** (optional, requires `sentence-transformers`)
- **Configurable prompts** -- Override LLM prompts via `mind-mem.json`

---

## 8. Breaking Changes

| Change | Details |
|--------|---------|
| Package name | `mem-os` -> `mind-mem` (PyPI, imports) |
| Config file | `mem-os.json` -> `mind-mem.json` |
| Console scripts | All 10 scripts renamed (`mem-os-*` -> `mind-mem-*`) |
| MCP server key | `"mem-os"` -> `"mind-mem"` in MCP config |
| Environment variable | `MEM_OS_WORKSPACE` -> `MIND_MEM_WORKSPACE` |
| Resource URIs | `mem-os://` -> `mind-mem://` |
| Auth header | `X-MemOS-Token` -> `X-MindMem-Token` |
| Config field | `self_correcting_mode` -> `governance_mode` |
| Schema version | Now targets `2.1.0` (was `1.0.0` in early mem-os) |
| MCP tool count | 6 -> 16 (additive, no tools removed) |

No existing workspace data is deleted or reformatted. All memory blocks, decisions, tasks, entities, and signals are preserved as-is.

---

## 9. Hook Migration (OpenClaw / naestro-bot)

If you use mem-os hooks with OpenClaw, update the hook path:

```bash
# Old location
~/.openclaw/hooks/mem-os/

# New location
~/.openclaw/hooks/mind-mem/
```

Update `~/.openclaw/openclaw.json` to reference the new hook directory:

```json
{
  "hooks": {
    "internal": {
      "entries": {
        "mind-mem": {
          "path": "~/.openclaw/hooks/mind-mem/"
        }
      }
    }
  }
}
```

The hook `handler.js` supports both `self_correcting_mode` and `governance_mode` fields for backward compatibility, but new workspaces should use `governance_mode` exclusively.

---

## Troubleshooting

**`mind-mem-migrate` reports "No migration needed"**
Your workspace schema is already at v2.1.0. Verify with:
```bash
python3 -c "import json; print(json.load(open('mind-mem.json')).get('schema_version', 'missing'))"
```

**MCP server fails to start after migration**
Confirm that `~/.claude/mcp.json` points to the correct `mcp_server.py` path and uses `MIND_MEM_WORKSPACE` (not `MEM_OS_WORKSPACE`).

**Old `mem-os-*` commands still found on PATH**
Uninstall the old package: `pip uninstall mem-os`

**`ImportError: No module named 'mind_mem'`**
Re-install: `pip install -e '.[all]'` from the mind-mem repository root.

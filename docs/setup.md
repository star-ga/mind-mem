# Setup

Full installation + configuration guide for mind-mem. All features
work with pure Python; the optional MIND native runtime is a drop-in
accelerator with automatic fallback.

## 1. Install

```bash
# Basic install — CPU-only, pure stdlib core
pip install mind-mem

# With MCP server support
pip install mind-mem[mcp]

# With optional local embeddings (ONNX)
pip install mind-mem[embeddings]

# With cross-encoder reranking
pip install mind-mem[cross-encoder]

# Everything
pip install mind-mem[all]
```

Python 3.10–3.14 supported. No required native dependencies — every
feature has a pure-Python path.

## 2. Initialise a workspace

```bash
# Create a new workspace in ./demo
mind-mem-init ./demo

# Or use an existing directory
cd my-project
mind-mem-init .
```

That command creates the standard directory layout:

```
workspace/
├── mind-mem.json         # config (see §3)
├── decisions/            # decision blocks
├── tasks/                # task blocks
├── entities/             # entity blocks
├── intelligence/         # signal capture
├── memory/               # daily logs + WAL + cores
└── .sqlite_index/        # FTS5 + block_meta index (auto-built)
```

## 3. Configuration (`mind-mem.json`)

Minimal config:

```json
{
  "version": "2.8.0",
  "workspace_path": ".",
  "auto_capture": true,
  "auto_recall": true,
  "governance_mode": "detect_only"
}
```

### Full config with every knob

```json
{
  "version": "2.8.0",
  "workspace_path": ".",
  "governance_mode": "detect_only",
  "recall": {
    "backend": "hybrid",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0,
    "query_expansion": { "enabled": false },
    "dedup": { "enabled": false }
  },
  "cognitive_forget": {
    "importance_threshold": 0.25,
    "stale_days": 14,
    "archive_after_days": 60,
    "grace_days": 30
  },
  "tiered_memory": {
    "working_ttl_days": 30,
    "promotion_threshold_sessions": 3,
    "half_life_days": 30
  },
  "context_core": {
    "cores_dir": "memory/cores"
  },
  "vault": {
    "path": "/path/to/obsidian/vault",
    "sync_dirs": ["decisions", "entities", "projects", "daily"],
    "exclude": [".obsidian", ".trash", "templates"],
    "reverse_sync": true,
    "conflict_policy": "vault_wins",
    "sync_interval_minutes": 5
  },
  "memory_mesh": {
    "enabled": false,
    "peers": []
  },
  "extraction": {
    "enabled": true,
    "model": "mind-mem:4b",
    "backend": "ollama"
  }
}
```

### LLM backend selection (v3.1.0)

`extraction.backend` accepts `ollama` (default), `vllm`,
`openai-compatible`, `llama-cpp`, `transformers`, or `auto`. The
`auto` mode probes each in order until one responds.

For vLLM or any OpenAI-compatible endpoint, set environment overrides:

```bash
# vLLM (default is http://127.0.0.1:8000/v1)
export MIND_MEM_VLLM_URL="http://gpu-host:8000/v1"

# Any OpenAI-compatible endpoint (LM Studio, llama-server, TGI, OpenAI)
export MIND_MEM_LLM_BASE_URL="http://lmstudio:1234/v1"
export MIND_MEM_LLM_API_KEY="optional-key"
```

See [Configuration Reference](configuration.md#extraction-llm-backend) for full details.

## 4. MCP server (for Claude Code / OpenClaw / any MCP-compatible agent)

```jsonc
// ~/.claude/mcp.json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["-m", "mind_mem.mcp_entry"],
      "env": {
        "MIND_MEM_WORKSPACE": "/absolute/path/to/workspace"
      }
    }
  }
}
```

Restart Claude Code. All 57 MCP tools (recall, recall_with_axis,
verify_merkle, observe_signal, graph_query, build_core,
agent_inject, etc.) become available.

## 5. All clients at once (v3.1.0 recommended)

Since v3.1.0, the fastest path for any setup is:

```bash
mm detect        # list every client mind-mem recognises on this machine
mm install-all   # write hook + native MCP config for every detected client
```

This installs **both** the text-hook (visibility + auto-capture) **and**
the native MCP server entry (full 57-tool surface) for all 8
MCP-aware clients: Codex, Gemini, Cursor, Windsurf, Continue, Cline,
Roo, Zed. Clients without MCP support (Claude Code hook-mode, Aider,
OpenClaw variants) get the hook only.

Pass `--no-mcp` to skip native MCP registration:

```bash
mm install-all --no-mcp   # hook config only
```

## 5a. Single-client install

Use `mm install` for a single client, or the legacy `mm hook install`:

```bash
mm install codex                              # hook + MCP
mm install gemini                             # hook + MCP
mm hook install --agent aider --workspace .   # hook only (legacy path)
```

`mm install` writes the right config file (AGENTS.md, .cursorrules,
.windsurfrules, .aider.conf.yml, GEMINI.md, …) for whichever agent
you target. See [Client Integrations](client-integrations.md) for
per-client MCP config paths and formats.

## 6. Optional: MIND native kernels

mind-mem ships four **hot-path Python kernels** (BM25F scoring,
SHA3-512 chain verify, vector similarity, RRF fusion) with a loader
that swaps in the STARGA proprietary MIND-compiled native library
when present. The native library is **not bundled** with this public
repo. To opt in:

```bash
export MIND_MEM_KERNELS_SO=/path/to/libmindmem_kernels.so
```

When the env var is unset or points at a missing file, mind-mem
transparently falls back to the pure-Python kernels in
`mind_mem.mind_kernels`. There is **no behaviour change** other than
throughput — everything works without the native library.

## 7. Optional: ledger anchoring

Set a chain endpoint + signer if you want Merkle roots published to
an external ledger:

```json
{
  "ledger_anchor": {
    "enabled": true,
    "chain": "sepolia",
    "endpoint_url": "https://rpc.sepolia.org",
    "anchor_every_n_blocks": 100
  }
}
```

When the poster clears a transaction, call
`mind_mem.ledger_anchor.anchor_root(history, root, tx_hash=...)`
so the local audit log records the confirmation. Without an
external poster, local records stay in `status="pending"` — still
fully auditable, just not on-chain.

## 8. Environment variables

| Var | Default | Purpose |
|---|---|---|
| `MIND_MEM_WORKSPACE` | `$(pwd)` | Active workspace path |
| `MIND_MEM_KERNELS_SO` | unset | Path to MIND native library (opt-in) |
| `MIND_MEM_ADMIN_TOKEN` | unset | Required for admin-scoped MCP tools |
| `MIND_MEM_SCOPE` | `user` | Override MCP scope when stdio |
| `PYPI_API_TOKEN` | unset | Only for maintainers (local publish) |

## 9. Verifying an install

```bash
mind-mem-verify ./demo --json
# → prints JSON with exit_code=0 on a clean workspace
```

The verifier opens ledgers read-only (`mode=ro`) so running it on a
production workspace is safe.

## 10. Upgrading from 1.9.x

v2.0.0 was the first stable 2.x; no breaking config changes. Run:

```bash
pip install --upgrade mind-mem
```

Re-run `mind-mem-init` with no arguments to add any new directories
from the 2.x layout (it's idempotent — existing data is untouched).

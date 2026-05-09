# Installation guide — every step + every option

End-to-end setup for `mind-mem` from `pip install` to running local
LLM, wired into every supported CLI. Three tiers: **Quick** (working in
3 commands), **Recommended** (working with local LLM in 5 commands),
**Full** (Postgres backend + every option).

---

## TL;DR — three commands

```bash
pip install mind-mem
mm install-all --force      # auto-wires every detected AI CLI
mm install-model            # downloads mind-mem-4b GGUF + imports to Ollama
```

That's it for most users. Reload your CLI and `mind-mem` MCP tools are
live. The rest of this doc explains what these commands do, every flag
they support, and how to upgrade to the Postgres backend.

---

## Tier 1 — Quick (SQLite + remote LLM)

**What you get**: full mind-mem stack on SQLite. Recall, hybrid
search, governance, MCP server. No local LLM.

```bash
pip install mind-mem
mm install-all --force
```

**Verify**:

```bash
mm status      # workspace + config sanity check
mm recall "anything"
```

**Workspace location** (where blocks + index live):

- Default: `~/.openclaw/workspace/` (compatible with OpenClaw users)
- Override: `export MIND_MEM_WORKSPACE=/path/to/your/workspace`

---

## Tier 2 — Recommended (SQLite + Ollama local LLM)

Everything in Tier 1 **plus** a 4 B-parameter LLM (`mind-mem-4b`)
running locally for extraction + governance.

**Prerequisite**: install Ollama from <https://ollama.com/download>.

```bash
pip install mind-mem
mm install-all --force
mm install-model
```

That third command:

1. Downloads `mind-mem-4b-Q4_K_M.gguf` (~2.5 GB) from HuggingFace into
   `~/.cache/mind-mem/`.
2. Writes a Modelfile and runs `ollama create mind-mem:4b`.
3. Sets `OLLAMA_KEEP_ALIVE=-1` so the model stays in VRAM (~5 GB).
4. Smoke-tests the model.

**Hardware**: fits in 6 GB VRAM (consumer RTX 3060 and up). CPU-only
also works, just slower.

**Verify**:

```bash
ollama list | grep mind-mem
mm install-model --dry-run   # safe re-run shows the plan, makes no changes
```

---

## Tier 3 — Full (Postgres backend + everything)

Use this when you want:

- Concurrent reads from many CLIs without SQLite WAL contention.
- pgvector HNSW indexing for sub-100 ms vector search at scale.
- Replicated read/write routing.

### 3a. Install + start Postgres

Ubuntu / Debian:

```bash
sudo apt install -y postgresql-16 postgresql-16-pgvector
sudo systemctl enable --now postgresql
```

macOS (Homebrew):

```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
```

### 3b. Create the `mindmem` role + database

```bash
sudo -u postgres psql <<'SQL'
CREATE USER mindmem WITH PASSWORD 'change-me-to-a-real-password';
CREATE DATABASE mindmem OWNER mindmem;
GRANT ALL PRIVILEGES ON DATABASE mindmem TO mindmem;
\c mindmem
CREATE EXTENSION IF NOT EXISTS vector;
SQL
```

### 3c. Allow password auth from localhost

Add to `/etc/postgresql/16/main/pg_hba.conf` (path varies on macOS):

```
local   mindmem   mindmem                          md5
host    mindmem   mindmem   127.0.0.1/32           md5
```

Then reload: `sudo systemctl reload postgresql`.

### 3d. Point mind-mem at Postgres

Edit your `mind-mem.json`:

```json
{
  "block_store": {
    "backend": "postgres",
    "dsn": "postgresql://mindmem:YOUR_PASSWORD@127.0.0.1:5432/mindmem",
    "schema": "mind_mem"
  },
  "semantic_search": { "enabled": true, "provider": "postgres" },
  "hybrid_search": { "enabled": true, "rrf_k": 60 }
}
```

### 3e. Migrate existing SQLite data (if any)

```bash
mm migrate-store --to postgres --dsn "postgresql://mindmem:YOUR_PASSWORD@127.0.0.1:5432/mindmem"
```

### 3f. Add the HNSW index (required for fast vector search at scale)

```sql
PGPASSWORD=YOUR_PASSWORD psql -h 127.0.0.1 -U mindmem -d mindmem <<'SQL'
CREATE INDEX IF NOT EXISTS blocks_embedding_hnsw_cos
ON mind_mem.blocks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE active;
VACUUM ANALYZE mind_mem.blocks;
SQL
```

### 3g. Verify

```bash
mm status   # should show backend=postgres
mm recall "test"
```

---

## Command reference — every flag

### `mm install-all`

Auto-detects every installed AI CLI on your machine and writes the
right config + system-prompt for each. Ten clients supported:
Claude Code, Codex, Gemini, Cursor, Windsurf, Roo, Continue, OpenClaw,
Vibe, Copilot.

```bash
mm install-all [--agent NAME]... [--dry-run] [--force] [--no-mcp]
```

| Flag | Meaning |
|---|---|
| `--agent NAME` | Restrict to specific agents. Repeat for multiple. Default = auto-detect every installed client. |
| `--dry-run` | Show what would change, write nothing. |
| `--force` | Overwrite existing config files. Without `--force`, existing files are left alone. |
| `--no-mcp` | Skip native MCP server registration. Default = write both the text hook AND the MCP entry for every MCP-aware client. |

**What it writes per client**:

- Text/yaml system-prompt: `AGENTS.md`, `GEMINI.md`, `.cursorrules`,
  `.windsurfrules`, `.clinerules`, `.roo/system-prompt.md`,
  `.aider.conf.yml` — each gets the canonical Memory Protocol snippet
  (recall, propose_update, hybrid_search, hallucination guardrail).
- MCP entry pointing at `python3 /…/mcp_server.py` with
  `MIND_MEM_WORKSPACE` set. Goes into the right format for each client
  (Codex TOML, Gemini/Cursor/Windsurf JSON, etc.).

Re-run anytime with `--force` to upgrade the snippet (e.g. after a
mind-mem version bump that improves the protocol text).

### `mm install-model`

Downloads `mind-mem-4b-Q4_K_M.gguf` from HuggingFace and imports it
into Ollama. Idempotent — safe to re-run.

```bash
mm install-model [--model NAME] [--name TAG] [--dest PATH] [--keep-alive VAL] [--dry-run]
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `mind-mem-4b-Q4_K_M.gguf` | Which GGUF file on HF to fetch. |
| `--name` | `mind-mem:4b` | Ollama tag to register the model under. |
| `--dest` | `~/.cache/mind-mem/mind-mem-4b-Q4_K_M.gguf` | Local download path. |
| `--keep-alive` | `-1` | Ollama keep-alive value. `-1` = forever, `30m` = 30 minutes, etc. |
| `--dry-run` | — | Print the plan, change nothing. |

**Other model formats** (advanced users):

- **Full bf16 safetensors** (8.4 GB) — for fine-tuning or
  high-performance serving via vLLM / exllamav2:

  ```bash
  huggingface-cli download star-ga/mind-mem-4b
  ```

- **Other quantizations** — only Q4_K_M is published. Build others
  yourself with `llama.cpp`'s `convert_hf_to_gguf.py` + `quantize`.

### `mm status`

Workspace + configuration sanity check. Use this first when something
seems wrong.

```bash
mm status
```

Returns JSON with `workspace`, `exists`, `config_exists`,
`decisions_dir_exists`, `backend`, `index_size`, recent activity.

### `mm recall` / `mm context` / `mm inject`

Direct CLI for memory queries.

- `mm recall "$query"` — JSON results, scored.
- `mm context "$query"` — formatted text, ready to paste into any
  prompt.
- `mm inject --agent claude "$query"` — writes the context block to
  stdout in the agent's expected snippet format.

### `mm serve` / `mm http-serve`

- `mm serve` — runs the MCP server in stdio mode (what each client
  invokes via its MCP entry).
- `mm http-serve --port 8765` — runs the REST API. Requires
  bearer-token auth via `X-MindMem-Token` header.

---

## Optional environment variables

| Var | Default | Use |
|---|---|---|
| `MIND_MEM_WORKSPACE` | `~/.openclaw/workspace` | workspace dir override |
| `MIND_MEM_TOKEN` | unset | bearer token for `mm http-serve` |
| `MIND_MEM_ENCRYPTION_PASSPHRASE` | unset | enables at-rest encryption (SQLite backend only) |
| `OLLAMA_KEEP_ALIVE` | `-1` after `mm install-model` | how long Ollama keeps `mind-mem:4b` in VRAM |
| `OLLAMA_HOST` | `http://localhost:11434` | use a remote Ollama box |
| `MIND_MEM_LLM_BACKEND` | `ollama` | switch to `openai-compatible` for vLLM / OpenAI-style endpoint |

---

## Multi-CLI setup notes

After `mm install-all --force`, **all detected CLIs share one
workspace by default** so a fact captured in one tool is immediately
visible to every other.

| CLI | Wiring path |
|---|---|
| Claude Code | hooks (`SessionStart` + `Stop` → `mm status`) |
| Codex | TOML `[mcp_servers.mind-mem]` |
| Gemini | JSON `mcpServers.mind-mem` in `~/.gemini/settings.json` |
| Cursor | JSON `mcpServers.mind-mem` in `~/.cursor/mcp.json` |
| Windsurf | JSON `mcpServers.mind-mem` in `~/.codeium/windsurf/mcp_config.json` |
| Roo | JSON `mcpServers.mind-mem` |
| Continue | JSON `mcpServers.mind-mem` in `~/.continue/config.json` |
| Vibe (Mistral) | TOML inline-table in `~/.vibe/config.toml` `mcp_servers` array |
| OpenClaw | hooks at `~/.openclaw/hooks/mind-mem/` |
| Copilot | instructions file `~/.github/copilot-instructions.md` (no native MCP — instructions only) |

To override the workspace per-CLI, edit the `MIND_MEM_WORKSPACE` env
in that CLI's config block.

---

## Troubleshooting

### `mm install-all` skipped a CLI I have installed

`mm install-all` checks for the binary on PATH **and** the standard
config dir (`~/.codex`, `~/.gemini`, etc.). If your CLI is installed
under a non-standard path, force it:

```bash
mm install-all --agent codex --force
```

### `mm install-model` says `ollama not found on PATH`

Install Ollama from <https://ollama.com/download>, then re-run.

### `mm install-model` succeeded but `ollama run mind-mem:4b` is slow on first call

Expected. The model loads into VRAM on first inference (~10 s).
After that it stays loaded indefinitely (`OLLAMA_KEEP_ALIVE=-1`).
Subsequent calls are < 100 ms first-token.

### Postgres recall returns 0 results but `mm status` shows blocks present

You're likely missing the HNSW index. Run section §3f above.

### Recall logs show `propagate_scores_failed: no such column: intent_type`

You upgraded mind-mem across a schema migration. Run:

```bash
mm migrate-store --check
```

If still failing, the SQLite retrieval-log auto-migrate is fine for
fresh DBs but pre-2026-04 DBs can miss it. Force the migration:

```python
import sqlite3, os
ws = os.path.expanduser("~/.openclaw/workspace")
db = os.path.join(ws, ".mind-mem-index/recall.db")
c = sqlite3.connect(db)
try: c.execute("ALTER TABLE retrieval_log ADD COLUMN intent_type TEXT DEFAULT ''")
except: pass
try: c.execute("ALTER TABLE retrieval_log ADD COLUMN stage_counts TEXT DEFAULT '{}'")
except: pass
c.commit(); c.close()
```

(Will be packaged as `mm doctor --migrate-recall-log` in v3.10.4.)

### MCP server not visible in CLI after `mm install-all`

Restart the CLI. Most clients only re-read MCP config at startup.

---

## Versions

- mind-mem: 3.10.3 (PyPI)
- mind-mem-4b model: v3.10.2-fullft, 6/6 eval (HF)
- Postgres: 16+ recommended, 14+ supported
- Postgres pgvector: 0.6.0+
- Ollama: 0.3.0+
- Python: 3.10+

---

## Next steps

- **Add memory to your daily flow**: see `docs/getting-started.md`
- **Use mind-mem-4b for extraction**: `docs/mind-mem-4b-setup.md`
- **Run the MCP server over HTTP** (multi-host setup): `docs/rest-api.md`
- **Operate the governance pipeline**: `docs/governance.md`

If anything in this guide is wrong on your system, please open an
issue at <https://github.com/star-ga/mind-mem/issues> with the output
of `mm status` + the failing command.

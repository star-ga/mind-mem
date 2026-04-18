# mind-mem — Persistent AI Memory System

## Overview
BM25F + vector hybrid search memory system for AI agents.
Published on PyPI: `pip install mind-mem`
v3.1.1 — 3610 tests, native MCP integration for 16 AI clients, 57 MCP
tools, full-fine-tune local model (`mind-mem-4b` on Qwen3.5-4B),
at-rest encryption (SQLCipher + BlockStore), tier decay, governance
alerting, audit-integrity patterns (Q16.16 fixed-point, NUL-separated
hash preimages).

## Architecture
```
src/mind_mem/           — Main package (src layout)
  core/                 — BlockStore, ConnectionManager, retrieval
  governance/           — Contradiction detection, drift, proposals,
                          audit chain, alerting hooks
  mcp_server.py         — MCP server (57 tools, 8 resources)
  ingestion/            — Auto-ingestion pipeline
  skill_opt/            — Skill optimization
  hook_installer/       — Client hook installation (v3.1.1+)
tests/                  — pytest suite (3610 tests, 177 files)
kernels/                — MIND scoring kernels (.mind)
docs/                   — User + integration docs (35+ files)
```

### Key Components
- **BM25F retrieval** with Porter stemming + RM3 query expansion
- **Hybrid search**: BM25 + sqlite-vec vector search with RRF fusion
- **A-MEM blocks**: metadata evolution (importance, access, keywords)
- **9-type intent router** with adaptive confidence weights
- **Cross-encoder reranking** (opt-in, config-gated)
- **Governance engine**: contradiction detection, drift, proposal queue,
  alerting hooks (webhook / Slack)
- **ConnectionManager**: Thread-safe SQLite pool with WAL read/write
  separation
- **Delta-based snapshot rollback**: MANIFEST.json for O(manifest)
  restore
- **At-rest encryption** (v3.0.0+): SQLCipher + BlockStore ciphertext
- **Tier decay** (v3.0.0+): TTL/LRU aging for lower tiers
- **Audit-integrity patterns** (v2.10.0+): Q16.16 fixed-point scoring in
  audit hash preimages, TAG_v1 NUL-separated composition for collision
  resistance
- **Local model** (v3.0.0+): `star-ga/mind-mem-4b` — full fine-tune of
  Qwen3.5-4B on the mind-mem domain (57 tools, 14 block schemas,
  governance workflows). See `docs/mind-mem-4b-setup.md`.
- **Native MCP integration** (v3.1.0+): 16 AI clients auto-wired via
  `mm install-all` (Claude Code, Claude Desktop, Codex CLI, Gemini CLI,
  Cursor, Windsurf, Zed, OpenClaw, and 8 more). See
  `docs/client-integrations.md`.

### MCP Tools (57)
Grouped surfaces (full list in `docs/api-reference.md` and
`src/mind_mem/mcp_server.py`):
recall, hybrid_search, prefetch, propose_update, approve_apply,
rollback_proposal, scan, list_contradictions, reindex, index_stats,
create_snapshot, list_snapshots, restore_snapshot, briefing,
category_summary, cross_encoder_rerank, find_similar, memory_evolution,
delete_memory, export_memory, import_memory, intent_classify,
retrieval_diagnostics, get_mind_kernel, list_mind_kernels,
verify_chain, audit_replay, tier_decay_apply, encrypt_status,
alerts_subscribe, and more.

## Config
- Config file: `mind-mem.json` (NOT `mem-os.json` — renamed)
- Auth header: `X-MindMem-Token` (NOT `X-MemOS-Token` — renamed)
- `governance_mode` (NOT `self_correcting_mode` — renamed)
- LLM backend: default `ollama`; switch to `openai-compatible` to point
  at a vLLM/exllamav2 endpoint running `mind-mem-4b`.

## Testing
```bash
pytest                           # full suite (3610 tests)
pytest tests/test_retrieval.py   # specific module
pytest -x --tb=short             # stop on first failure
pytest --collect-only -q | tail  # verify test count
```

## Conventions
- Python 3.10+ with type hints everywhere
- src layout: `src/mind_mem/`
- All public functions have docstrings
- No `print()` in library code — use `logging`
- SQLite WAL mode for concurrent reads

## Benchmarks (LoCoMo, Mistral Large)
Mean: 77.9, Adversarial: 82.3, Temporal: 88.5

## Git
- Remote: star-ga/mind-mem (public)
- Author: `STARGA Inc <noreply@star.ga>`
- CI: GitHub Actions (enabled); release workflow on tag push with
  trusted PyPI publishing via OIDC (`environment: pypi`)

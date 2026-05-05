# mind-mem — Persistent AI Memory System

## Overview
BM25F + vector hybrid search memory system for AI agents.
Published on PyPI: `pip install mind-mem`

**v3.9.0** (released 2026-05-04) — 4000+ tests, native MCP for 17 AI
clients, **81 tools** (58 legacy + 7 consolidated dispatchers + 12
v3.7→v3.8 additions + 4 v3.9 wrappers: `compile_truth_walkthrough`,
`recall_with_persona`, `pipeline_status`, `reindex_dirty`), Postgres
backend with pgvector + HNSW + correct GIN, full-fine-tune local
model (`mind-mem-4b` v3.9.0 on Qwen3.5-4B), at-rest encryption, tier
decay, governance alerting, MIC/MAP wire format (mic@2 text + mic-b
binary), and the v3.9 transport/runtime surface (HTTP REST adapter,
background daemon, inbox folder ingestion, hash-of-code pipeline
invalidation, persona-aware projection, dependency-ordered Kahn-topo
walkthrough, replicated Postgres routing).

## Architecture
```
src/mind_mem/           — Main package (src layout)
  core/                 — BlockStore, ConnectionManager, retrieval
  governance/           — Contradiction detection, drift, proposals,
                          audit chain, alerting hooks
  mcp_server.py         — MCP server (81 tools, 8 resources)
  ingestion/            — Auto-ingestion pipeline
  skill_opt/            — Skill optimization
  hook_installer/       — Client hook installation (v3.1.1+)
  http_transport.py     — Stdlib HTTP REST adapter (v3.9.0)
  daemon.py             — Background dream-cycle/intel-scan loop (v3.9.0)
  inbox.py              — File-drop folder ingestion (v3.9.0)
  pipeline_hash.py      — Hash-of-code invalidation (v3.9.0)
  personas.py           — brief/detailed/technical projection (v3.9.0)
  walkthrough.py        — Dependency-ordered learning sequence (v3.9.0)
tests/                  — pytest suite (4000+ tests)
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
- **Local model** (v3.9.0): `star-ga/mind-mem-4b` — full fine-tune of
  Qwen3.5-4B on the v3.9.0 mind-mem domain (81 tools, block schemas
  including `TransformHash`, governance + transport workflows). See
  `docs/mind-mem-4b-setup.md`. Prior v3.0.0 QLoRA fine-tune kept at
  HF revision `v3.0.0`.
- **Native MCP integration** (v3.1.0+): 16 AI clients auto-wired via
  `mm install-all` (Claude Code, Claude Desktop, Codex CLI, Gemini CLI,
  Cursor, Windsurf, Zed, OpenClaw, and 8 more). See
  `docs/client-integrations.md`.

### MCP Tools (81)
Grouped surfaces (full list in `docs/api-reference.md` and
`src/mind_mem/mcp_server.py`):
recall, hybrid_search, prefetch, propose_update, approve_apply,
rollback_proposal, scan, list_contradictions, reindex, index_stats,
create_snapshot, list_snapshots, restore_snapshot, briefing,
category_summary, cross_encoder_rerank, find_similar, memory_evolution,
delete_memory, export_memory, import_memory, intent_classify,
retrieval_diagnostics, get_mind_kernel, list_mind_kernels,
verify_chain, audit_replay, tier_decay_apply, encrypt_status,
alerts_subscribe, mic_convert_tool, mic_inspect_tool,
compile_truth_walkthrough, recall_with_persona, pipeline_status,
reindex_dirty, and more.

## Config
- Config file: `mind-mem.json` (NOT `mem-os.json` — renamed)
- Auth header: `X-MindMem-Token` (NOT `X-MemOS-Token` — renamed)
- `governance_mode` (NOT `self_correcting_mode` — renamed)
- LLM backend: default `ollama`; switch to `openai-compatible` to point
  at a vLLM/exllamav2 endpoint running `mind-mem-4b`.

## Testing
```bash
pytest                           # full suite (4000+ tests)
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

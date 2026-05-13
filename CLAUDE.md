# MIND-Mem — Persistent AI Memory System

## Overview
BM25F + vector hybrid search memory system for AI agents.
Published on PyPI: `pip install mind-mem`

**v4.0.0** (released 2026-05-10) — Cognitive kernel, knowledge graph,
resilience suite, and observability layer. All surfaces flag-gated under
`v4.<flag>` in `mind-mem.json`. No breaking changes. `mind-mem-4b`
retrained at **109/109 = 100%** on the un-softened harness (14 new
`V4_SURFACES` probes; `qg.escape_hatch` and `lin.cites=0.8` gaps from
v3.12.1 confirmed fixed). 376 v4 unit + 38 concurrency + 22 held-out
paraphrase tests. Architecture audited at unanimous **10/10** across 4
LLMs (Grok 4.3, DeepSeek v4-pro, Mistral large, GLM-5).
Builds on **v3.12.1** (2026-05-10) — patched eval, model card honesty.
Builds on **v3.12.0** (2026-05-09) — strict quality gate, lineage→staleness.
Builds on **v3.11.0** (2026-05-08) — typed lineage edges, recall explainability.
Builds on **v3.9.0** — 4000+ tests, native MCP for 17 AI clients, **84 tools**,
Postgres+pgvector, full-fine-tune local model, at-rest encryption, tier
decay, governance alerting, MIC/MAP wire format, HTTP transport, daemon,
inbox ingestion, pipeline hash, persona projection, Kahn walkthrough.

## Architecture
```
src/mind_mem/           — Main package (src layout)
  core/                 — BlockStore, ConnectionManager, retrieval
  governance/           — Contradiction detection, drift, proposals,
                          audit chain, alerting hooks
  mcp_server.py         — MCP server (84 tools, 8 resources)
  ingestion/            — Auto-ingestion pipeline
  skill_opt/            — Skill optimization
  hook_installer/       — Client hook installation (v3.1.1+)
  http_transport.py     — Stdlib HTTP REST adapter (v3.9.0)
  daemon.py             — Background dream-cycle/intel-scan loop (v3.9.0)
  inbox.py              — File-drop folder ingestion (v3.9.0)
  pipeline_hash.py      — Hash-of-code invalidation (v3.9.0)
  personas.py           — brief/detailed/technical projection (v3.9.0)
  walkthrough.py        — Dependency-ordered learning sequence (v3.9.0)
  quality_gate.py       — Deterministic quality validation (v3.11.0)
  block_lineage.py      — Typed relationship edges (v3.11.0)
  lineage_staleness.py  — BFS staleness propagation (v3.12.0)
  — v4.0.0 surfaces (all flag-gated) —
  tier_memory.py        — Hot/warm/cold block tiers + CAS (v4.0.0)
  cognitive_kernel.py   — KernelKind enum + mind_recall (v4.0.0)
  surprise_retrieval.py — compute_surprise + FallbackPolicy (v4.0.0)
  block_kinds.py        — Multi-label block_kind_tags table (v4.0.0)
  block_metadata.py     — Tag + TTL + schema validators (v4.0.0)
  kind_summaries.py     — Per-kind global summaries (v4.0.0)
  embedding_pipeline.py — Pluggable embedder + 3-gram default (v4.0.0)
  consolidation_worker.py — plan_consolidation pure fn (v4.0.0)
  eviction.py           — LRU/LOW_SURPRISE/AGE/COMPOSITE (v4.0.0)
  federation.py         — VClock + conflict log + MergeStrategy (v4.0.0)
  self_editing.py       — block_edits + propose/approve/reject (v4.0.0)
  pq.py                 — Product Quantization M=32 K=256 (v4.0.0)
  hnsw_kind_index.py    — sqlite-vec HNSW + brute-force fallback (v4.0.0)
  circuit_breaker.py    — CircuitBreaker + @circuit_breaker (v4.0.0)
  backpressure.py       — BackpressureController + hysteresis (v4.0.0)
  health.py             — health_check + 7 probes + register (v4.0.0)
  observability.py      — counter/gauge/histogram + @timed (v4.0.0)
  logging_context.py    — contextvar stack + StructuredLogFilter (v4.0.0)
  feature_flags.py      — 35 flags + is_enabled/require_enabled (v4.0.0)
tests/                  — pytest suite (4000+ tests + 376 v4 unit
                          + 38 concurrency + 22 paraphrase probes)
mind/                   — MIND scoring kernels (.mind)
docs/                   — User + integration docs (35+ files)
```

### Key Components
- **BM25F retrieval** with Porter stemming + RM3 query expansion
- **Hybrid search**: BM25 + vector search with RRF fusion (sqlite-vec
  on the SQLite backend; pgvector + HNSW on the Postgres backend)
- **Backends**: SQLite (default, zero-deps) or Postgres (pgvector +
  HNSW + GIN, with replicated read/write routing in v3.9)
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
- **Local model** (v4.0.0): `star-ga/mind-mem-4b` — full fine-tune of
  Qwen3.5-4B retrained for v4.0.0 (v4 weights revision). Knows all 84
  tools, v4 surfaces (cognitive kernel, block kinds, tier memory,
  self-editing), and the corrected `KIND_DECAY['cites']=0.8` value.
  Prior v3.12.0-fullft weights pinned at `v3.12.0` HF revision. Prior
  v3.0.0 QLoRA at `v3.0.0`. See `docs/mind-mem-4b-setup.md`.
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

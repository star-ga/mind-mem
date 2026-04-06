# mind-mem — Persistent AI Memory System

## Overview
BM25F + vector hybrid search memory system for AI agents.
Published on PyPI: `pip install mind-mem`
v1.9.1 — 2180 tests, CI green on 3 OS × 4 Python versions.

## Architecture
```
src/mind_mem/           — Main package (src layout)
  core/                 — BlockStore, ConnectionManager, retrieval
  governance/           — Contradiction detection, drift analysis, proposals
  mcp_server.py         — MCP server (19 tools, 8 resources)
  ingestion/            — Auto-ingestion pipeline
tests/                  — pytest test suite
kernels/                — 17 MIND kernel files (.mind)
```

### Key Components
- **BM25F retrieval** with Porter stemming + RM3 query expansion
- **Hybrid search**: BM25 + sqlite-vec vector search with RRF fusion
- **A-MEM blocks**: metadata evolution (importance, access tracking, keywords)
- **9-type intent router** with adaptive confidence weights
- **Cross-encoder reranking** (opt-in, config-gated)
- **Governance engine**: contradiction detection, drift analysis, proposal queue
- **ConnectionManager**: Thread-safe SQLite pool with WAL read/write separation
- **Delta-based snapshot rollback**: MANIFEST.json for O(manifest) restore

### MCP Tools (19)
recall, propose_update, approve_apply, rollback_proposal, scan,
list_contradictions, reindex, index_stats, create_snapshot, list_snapshots,
restore_snapshot, briefing, category_summary, prefetch, hybrid_search,
cross_encoder_rerank, find_similar, memory_evolution, delete_memory

## Config
- Config file: `mind-mem.json` (NOT `mem-os.json` — renamed)
- Auth header: `X-MindMem-Token` (NOT `X-MemOS-Token` — renamed)
- `governance_mode` (NOT `self_correcting_mode` — renamed)

## Testing
```bash
pytest                           # full suite
pytest tests/test_retrieval.py   # specific module
pytest -x --tb=short             # stop on first failure
```

## Conventions
- Python 3.9+ with type hints everywhere
- src layout: `src/mind_mem/`
- All public functions have docstrings
- No `print()` in library code — use `logging`
- SQLite WAL mode for concurrent reads

## Benchmarks (LoCoMo, Mistral Large)
Mean: 77.9, Adversarial: 82.3, Temporal: 88.5
Beats: full-context (72.9), Mem0 (66.9), Zep (66.0), LangMem (58.1)

## Git
- Remote: star-ga/mind-mem (public)
- Author: `STARGA Inc <noreply@star.ga>`
- CI: GitHub Actions, 16 matrix jobs (3 OS × 4 Python + extras)

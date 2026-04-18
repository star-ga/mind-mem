---
name: mind-mem-development
description: mind-mem Python development guide
tags: [python, memory, mind-mem, pypi]
---

# mind-mem Development

## Package
- PyPI: `pip install mind-mem`
- Source layout: `src/mind_mem/`
- Tests: pytest, 3610 passing
- CI: GitHub Actions, 3 OS × 4 Python versions (16 matrix jobs)

## Architecture
- BM25F retrieval with Porter stemming + RM3 query expansion
- Hybrid BM25+Vector search with RRF fusion (sqlite-vec)
- A-MEM block metadata evolution
- 9-type intent router with adaptive confidence weights
- ConnectionManager: thread-safe SQLite pool with WAL
- BlockStore protocol: decoupled block access
- Delta-based snapshot rollback

## MCP Server (57 tools)
Grouped surfaces (full list in `docs/api-reference.md` and
`src/mind_mem/mcp_server.py`):
recall, hybrid_search, prefetch, propose_update, approve_apply,
rollback_proposal, scan, list_contradictions, reindex, index_stats,
create_snapshot, list_snapshots, restore_snapshot, briefing,
category_summary, cross_encoder_rerank, find_similar,
memory_evolution, delete_memory, export_memory, import_memory,
intent_classify, retrieval_diagnostics, get_mind_kernel,
list_mind_kernels, verify_chain, audit_replay, tier_decay_apply,
encrypt_status, alerts_subscribe, and more (57 total).

## Key Files
- `src/mind_mem/mcp_server.py` — MCP server (57 @mcp.tool entries)
- `src/mind_mem/retrieval/` — search engine (bm25, vector, hybrid)
- `src/mind_mem/governance/` — contradiction detection, drift analysis
- `src/mind_mem/blocks/` — block store, parser, evolution
- `tests/` — comprehensive test suite

## Conventions
- Python 3.10+, type hints everywhere
- Docstrings: Google style
- No dynamic allocation in hot paths
- All SQL queries parameterized
- Config: mind-mem.json (not mem-os.json)
- Auth header: X-MindMem-Token

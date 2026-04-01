---
name: mind-mem-development
description: mind-mem Python development guide
tags: [python, memory, mind-mem, pypi]
---

# mind-mem Development

## Package
- PyPI: `pip install mind-mem`
- Source layout: `src/mind_mem/`
- Tests: pytest, 2180+ passing
- CI: 3 OS × 4 Python versions (16 matrix jobs)

## Architecture
- BM25F retrieval with Porter stemming + RM3 query expansion
- Hybrid BM25+Vector search with RRF fusion (sqlite-vec)
- A-MEM block metadata evolution
- 9-type intent router with adaptive confidence weights
- ConnectionManager: thread-safe SQLite pool with WAL
- BlockStore protocol: decoupled block access
- Delta-based snapshot rollback

## MCP Server (19 tools)
recall, propose_update, approve_apply, rollback_proposal, scan,
list_contradictions, reindex, index_stats, create_snapshot,
list_snapshots, restore_snapshot, briefing, category_summary,
prefetch, hybrid_search, cross_encoder_rerank, find_similar,
memory_evolution, delete_memory

## Key Files
- `src/mind_mem/server.py` — MCP server
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

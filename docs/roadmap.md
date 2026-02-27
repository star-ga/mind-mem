# Roadmap

## v1.8.0 (Current)

- [x] BM25F scoring with field weights
- [x] Co-retrieval graph boost
- [x] Fact card sub-block indexing
- [x] Knee score cutoff
- [x] Hard negative mining
- [x] 19 MCP tools
- [x] LoCoMo benchmark suite
- [x] Cross-platform CI (Ubuntu/macOS/Windows)
- [x] Baseline snapshot with chi-squared drift detection
- [x] Contradiction detection at governance gate

## v1.9.0 (Planned)

- [ ] Incremental reindexing
- [ ] Block versioning with diff tracking
- [ ] Enhanced vector search with HNSW
- [ ] Query result caching
- [ ] Workspace federation (multi-workspace queries)

## v2.0.0 (Future)

- [ ] Persistent FTS5 index
- [ ] Real-time workspace watching (inotify/FSEvents)
- [ ] Plugin system for custom scoring
- [ ] Web UI for memory browsing
- [ ] REST API server mode
- [ ] Distributed workspace sync

## Design Principles

1. **Zero dependencies** — No external services required
2. **Auditable** — Full proposal system with history
3. **Fast** — Sub-500ms recall on commodity hardware
4. **Portable** — Plain markdown files, any filesystem
5. **Extensible** — MIND kernels for custom scoring

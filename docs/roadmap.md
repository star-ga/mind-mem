# Roadmap

> This is the short-form roadmap. The canonical, detailed version lives in
> [`ROADMAP.md`](../ROADMAP.md) at the repo root and includes the full
> v2.0 → v2.7 milestone breakdown.

## v2.0.0b1 (Current — released 2026-04-13)

Inference acceleration — Python subset. LLM prefix cache + speculative
prefetch predictor. MIND-compiled hot paths deferred until the `mindc`
toolchain is available.

## v2.0.0a3 (Released 2026-04-13)

Observer-Dependent Cognition (ODC) axis-aware retrieval.

## v2.0.0a2 (Released 2026-04-13)

Cryptographic governance layer + GBrain enrichment.

## v1.9.1 (Previous stable)

- [x] BM25F scoring with field weights
- [x] Co-retrieval graph boost
- [x] Fact card sub-block indexing
- [x] Knee score cutoff
- [x] Hard negative mining
- [x] 57 MCP tools (expanded from 19)
- [x] LoCoMo benchmark suite
- [x] Cross-platform CI (Ubuntu/macOS/Windows)
- [x] Baseline snapshot with chi-squared drift detection
- [x] Contradiction detection at governance gate
- [x] Hash-chain mutation audit log
- [x] Per-field mutation tracking
- [x] Semantic belief drift detection
- [x] Temporal causal dependency graph
- [x] Coding-native memory schemas (ADR/CODE/PERF/ALGO/BUG)
- [x] Auto contradiction resolution with preference learning
- [x] Governance benchmark suite
- [x] AES-256 encryption at rest
- [x] LLM-free multi-query expansion with RRF fusion
- [x] 4-layer search deduplication (per-source, cosine, type diversity, chunk cap)
- [x] Semantic-aware smart chunking (markdown/code/prose boundaries)
- [x] Compiled truth pages (per-entity knowledge compilation with evidence trails)
- [x] Dream cycle (autonomous memory enrichment, repair, consolidation)
- [x] Calibration feedback loop with Bayesian weight computation
- [x] Graph traversal tool (walk cross-references by depth/direction)
- [x] Block compaction and stale block detection

## v2.0.0 → v2.9.0 (Shipped)

- [x] Incremental reindexing
- [x] Block versioning with diff tracking (delta snapshots + WAL)
- [x] Enhanced vector search with HNSW-compatible sqlite-vec
- [x] Query result caching (prefix cache + speculative prefetch)
- [x] Workspace federation (multi-workspace queries via namespaces)
- [x] Persistent FTS5 index
- [x] Plugin system for custom scoring (MIND kernels)

## v3.0.0 (Future)

- [ ] Real-time workspace watching (inotify/FSEvents)
- [ ] Web UI for memory browsing
- [ ] REST API server mode (distinct from MCP)
- [ ] Distributed workspace sync (mDNS-discovered mesh)
- [ ] Transparent at-rest encryption (read/write hooks in block_parser)
- [ ] LoRA retrain loop wired into production pipeline

## Design Principles

1. **Zero dependencies** — No external services required
2. **Auditable** — Full proposal system with history
3. **Fast** — Sub-500ms recall on commodity hardware
4. **Portable** — Plain markdown files, any filesystem
5. **Extensible** — MIND kernels for custom scoring

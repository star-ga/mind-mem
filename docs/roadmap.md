# Roadmap

> This is the short-form roadmap. The canonical, detailed version lives
> in [`../ROADMAP.md`](../ROADMAP.md) at the repo root and includes the
> full milestone breakdown.

## v3.1.1 (Current — released 2026-04-15)

Patch release. Claude Code hook-installer fix: `install claude-code`
now writes the required nested hook shape and migrates legacy flat
entries in-place. `mm inject` / `mm vault status` hooks replaced with
`mm status` where the old commands pointed at unshipped subcommands.

## v3.1.0 (Released 2026-04-15)

Native MCP integration for 8 additional AI clients plus a
multi-backend LLM extractor. `mm install-all` auto-wires 16 clients
total.

## v3.0.0 (Released 2026-04-14)

Governance alerting hooks (webhook / Slack), transparent at-rest
encryption (SQLCipher + BlockStore), TTL/LRU tier decay, and
full-fine-tune local model `star-ga/mind-mem-4b` (Qwen3.5-4B base).
Adversarial-memory corpus tests and Jepsen-style concurrency stress
tests merged. MCP tool surface now 57.

## v2.10.0 (Released 2026-04-14)

Audit-integrity patch series. TAG_v1 NUL-separated hash preimages for
collision resistance, Q16.16 fixed-point scores in audit hash
preimages.

## v2.9.0 and earlier

See `CHANGELOG.md` for the full history. Highlights across v2.x:
incremental reindexing, delta-snapshot block versioning with WAL,
sqlite-vec HNSW-compatible vector search, prefix cache + speculative
prefetch, workspace federation via namespaces, FTS5 index,
MIND-kernel plugin system for custom scoring, ODC axis-aware
retrieval, cryptographic governance layer, GBrain enrichment,
inference acceleration for the Python subset.

## v1.9.x (Earlier stable line, superseded)

Foundational retrieval and governance work:

- BM25F scoring with field weights
- Co-retrieval graph boost
- Fact card sub-block indexing
- Knee score cutoff, hard negative mining
- LoCoMo benchmark suite
- Cross-platform CI (Ubuntu / macOS / Windows)
- Baseline snapshot with chi-squared drift detection
- Contradiction detection at governance gate
- Hash-chain mutation audit log
- Per-field mutation tracking, semantic belief drift detection
- Temporal causal dependency graph
- Coding-native memory schemas (ADR / CODE / PERF / ALGO / BUG)
- Auto contradiction resolution with preference learning
- Governance benchmark suite, AES-256 encryption at rest
- LLM-free multi-query expansion with RRF fusion
- 4-layer search deduplication
- Semantic-aware smart chunking
- Compiled truth pages with evidence trails
- Dream cycle (autonomous memory enrichment / repair)
- Calibration feedback loop with Bayesian weight computation
- Graph traversal tool
- Block compaction and stale block detection

## Upcoming (v3.2 / v3.3 / v4.0)

Tracked in open GitHub issues and in the root `ROADMAP.md`.
Direction:

- MCP tool-surface reduction (57 → ~20 as a stable public surface, with
  the rest moving behind a `*/advanced` namespace)
- Multi-tenancy foundation (orgs / users / RBAC)
- Real-time workspace watching (inotify / FSEvents)
- Web UI for memory browsing
- REST API server mode (distinct from MCP)
- Distributed workspace sync (mDNS-discovered mesh)
- LoRA retrain loop wired into production pipeline

## Design Principles

1. **Zero dependencies** — No external services required for the core
2. **Auditable** — Every mutation goes through the proposal system
3. **Fast** — Sub-500ms recall on commodity hardware
4. **Portable** — Plain Markdown files, any filesystem
5. **Extensible** — MIND kernels for custom scoring

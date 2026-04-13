# Mind Mem — Memory + Immune System for AI agents
# Package: mind_mem (src/mind_mem/ via pyproject.toml package-dir)

"""mind-mem: governance-aware memory layer for AI agents.

Core modules:
    recall          — BM25 + graph recall engine
    recall_vector   — Optional vector (embedding) backend
    apply_engine    — Proposal apply with WAL, locking, and rollback
    block_parser    — Typed Markdown block parser
    capture         — Auto-capture engine (26 patterns)
    intel_scan      — Integrity scanner (contradictions, drift, impact)
    namespaces      — Multi-agent namespace & ACL engine
    mind_filelock   — Cross-platform advisory file locking
    backup_restore  — WAL + backup/restore + JSONL export
    compaction      — Compaction/GC/archival
    observability   — Structured JSON logging + metrics
    schema_version  — Schema migration tooling
    conflict_resolver — Automated conflict resolution
    transcript_capture — Transcript JSONL signal extraction
    connection_manager — SQLite connection pooling with read/write separation
    corpus_registry — Central source of truth for corpus directory paths
    block_store     — BlockStore protocol and MarkdownBlockStore implementation
    audit_chain     — Hash-chained append-only mutation ledger
    field_audit     — Per-field mutation tracking with attribution
    drift_detector  — Semantic belief drift detection
    causal_graph    — Temporal causal dependency graph
    coding_schemas  — Coding-native memory schemas (ADR, CODE, PERF, ALGO, BUG)
    auto_resolver   — Automatic contradiction resolution with preference learning
    governance_bench — Governance-specific benchmark suite
    encryption      — Optional AES-256 encryption at rest
    calibration     — Retrieval quality feedback loop with per-block calibration weights
    evidence_objects — Structured tamper-evident Evidence Objects for governance decisions
"""

__version__ = "2.8.0"

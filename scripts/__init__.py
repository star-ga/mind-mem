# Mind Mem — Memory + Immune System for AI agents
# Package: mind_mem (maps to scripts/ via pyproject.toml package-dir)

"""mind-mem: governance-aware memory layer for AI agents.

Core modules:
    recall          — BM25 + graph recall engine
    recall_vector   — Optional vector (embedding) backend
    apply_engine    — Proposal apply with WAL, locking, and rollback
    block_parser    — Typed Markdown block parser
    capture         — Auto-capture engine (26 patterns)
    intel_scan      — Integrity scanner (contradictions, drift, impact)
    namespaces      — Multi-agent namespace & ACL engine
    filelock        — Cross-platform advisory file locking
    backup_restore  — WAL + backup/restore + JSONL export
    compaction      — Compaction/GC/archival
    observability   — Structured JSON logging + metrics
    schema_version  — Schema migration tooling
    conflict_resolver — Automated conflict resolution
    transcript_capture — Transcript JSONL signal extraction
"""

__version__ = "1.0.7"

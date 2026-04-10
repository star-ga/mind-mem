# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind-mem`
**Files:** 373 | **Est. tokens:** ~768,455
**Generated:** 2026-04-10 09:47 UTC

## Token Budget Guide

| Size | Tokens | Read strategy |
|------|--------|---------------|
| tiny | <50 | Always safe to read |
| small | 50-200 | Read freely |
| medium | 200-500 | Read if relevant |
| large | 500-1500 | Use summary first, read specific sections |
| huge | >1500 | Avoid full read — use grep or read specific lines |

## Directory Overview

| Directory | Files | Est. tokens |
|-----------|-------|-------------|
| `./` | 27 | ~65,216 |
| `.agents/skills/mind-mem-development/` | 1 | ~371 |
| `benchmarks/` | 11 | ~39,472 |
| `docs/` | 28 | ~23,412 |
| `docs/adr/` | 2 | ~521 |
| `examples/` | 2 | ~466 |
| `.github/` | 7 | ~4,109 |
| `.github/ISSUE_TEMPLATE/` | 2 | ~179 |
| `.github/workflows/` | 9 | ~3,127 |
| `hooks/` | 3 | ~801 |
| `hooks/openclaw/mind-mem/` | 2 | ~1,211 |
| `lib/` | 1 | ~2,176 |
| `mind/` | 19 | ~5,518 |
| `scripts/` | 2 | ~2,268 |
| `skills/apply-proposal/` | 1 | ~345 |
| `skills/integrity-scan/` | 1 | ~376 |
| `skills/memory-recall/` | 1 | ~549 |
| `src/` | 1 | ~455 |
| `src/mind_mem/` | 81 | ~316,715 |
| `templates/` | 19 | ~1,041 |
| `tests/` | 151 | ~298,691 |
| `tests/integration/` | 2 | ~1,436 |

## Files

### `./`

- `CHANGELOG.md` (~10459 tok, huge) — Changelog
- `CLAUDE.md` (~611 tok, large) — mind-mem — Persistent AI Memory System
- `conftest.py` (~121 tok, small) — Shared pytest fixtures for mind-mem test suite."""
- `CONTRIBUTING.md` (~309 tok, medium) — Contributing to mind-mem
- `demo-setup.sh` (~323 tok, medium) — Pre-seed a demo workspace for VHS recording
- `demo.tape` (~93 tok, small) — # mind-mem demo — terminal recording for README
- `Dockerfile` (~54 tok, small) — FROM python:3.12-slim
- `.dockerignore` (~37 tok, tiny) — .git
- `.editorconfig` (~107 tok, small) — # EditorConfig — https://editorconfig.org
- `generate_mind7b_training.py` (~5398 tok, huge) — Generate training data for Mind7B — a purpose-trained 7B model for mind-mem.
- `.gitattributes` (~96 tok, small) — # Auto-detect text files and normalize line endings
- `.gitignore` (~89 tok, small) — *.pyc
- `install.sh` (~3337 tok, huge) — mind-mem installer — sets up MCP server + hooks for all supported clients
- `LICENSE` (~2695 tok, huge)
- `Makefile` (~532 tok, large) — .PHONY: test lint bench install dev clean smoke help
- `mcp_server.py` (~448 tok, medium) — Source-checkout entrypoint for the packaged Mind-Mem MCP server."""
- `mind-mem.example.json` (~174 tok, small) — Keys: recall, prompts, categories, extraction, limits
- `.pre-commit-config.yaml` (~131 tok, small) — repos:
- `pyproject.toml` (~768 tok, large) — [project]
- `.python-version` (~2 tok, tiny) — 3.12
- `README.md` (~21350 tok, huge) — Shared Memory Across All Your AI Agents
- `requirements-optional.txt` (~714 tok, large) — # mind-mem optional dependencies — pinned with SHA256 integrity hashes.
- `ROADMAP.md` (~8514 tok, huge) — mind-mem Roadmap
- `SECURITY.md` (~1414 tok, large) — Security Policy
- `SPEC.md` (~4880 tok, huge) — Mind Mem Formal Specification v1.0
- `train_mind7b_runpod.py` (~1652 tok, huge)
- `uninstall.sh` (~908 tok, large) — mind-mem uninstaller — removes MCP server entries from all configured clients
### `.agents/skills/mind-mem-development/`

- `SKILL.md` (~371 tok, medium) — mind-mem Development
### `benchmarks/`

- `bench_kernels.py` (~4031 tok, huge) — Benchmark: MIND kernels vs pure Python scoring.
- `compare_runs.py` (~857 tok, large) — Compare two LoCoMo benchmark runs side-by-side.
- `crossencoder_ab.py` (~3205 tok, huge) — Cross-Encoder A/B Test — retrieval-level comparison.
- `grid_search.py` (~2862 tok, huge) — BM25F Field Weight Grid Search for mind-mem Recall Engine.
- `__init__.py` (~0 tok, tiny)
- `locomo_harness.py` (~4159 tok, huge) — LoCoMo Benchmark Harness for mind-mem Recall Engine.
- `locomo_judge.py` (~10759 tok, huge) — LoCoMo LLM-as-Judge Evaluation for Mind-Mem.
- `longmemeval_harness.py` (~2973 tok, huge) — LongMemEval Benchmark Harness for mind-mem recall engine.
- `niah_full_results.txt` (~5140 tok, huge) — ============================= test session starts ==============================
- `NIAH.md` (~1513 tok, huge) — Needle In A Haystack (NIAH) Benchmark
- `REPORT.md` (~3973 tok, huge) — mind-mem Benchmark Report
### `docs/adr/`

- `001-zero-dependencies.md` (~316 tok, medium) — ADR-001: Zero External Dependencies in Core
- `002-bm25f-scoring.md` (~205 tok, medium) — ADR-002: BM25F as Primary Scoring Algorithm
### `docs/`

- `api-reference.md` (~957 tok, large) — API Reference
- `architecture.md` (~825 tok, large) — Architecture
- `benchmarks.md` (~579 tok, large) — Benchmarks
- `block-format.md` (~431 tok, medium) — Block Format
- `changelog-format.md` (~217 tok, medium) — Changelog Format Guide
- `ci-workflows.md` (~254 tok, medium) — CI Workflows
- `claude-desktop-setup.md` (~752 tok, large) — Claude Desktop Setup Guide
- `comparison.md` (~313 tok, medium) — Comparison with Alternatives
- `competitive-analysis-persistent-memory-2026.md` (~4089 tok, huge) — Comprehensive Competitive Analysis: Persistent Memory Systems for AI Coding Agents (2025–2026)
- `configuration.md` (~4599 tok, huge) — Configuration Reference
- `development.md` (~358 tok, medium) — Development Guide
- `faq.md` (~369 tok, medium) — FAQ
- `getting-started.md` (~405 tok, medium) — Getting Started
- `glossary.md` (~263 tok, medium) — Glossary
- `mcp-integration.md` (~474 tok, medium) — MCP Integration Guide
- `mcp-tool-examples.md` (~902 tok, large) — MCP Tool Examples
- `migration-guide.md` (~421 tok, medium) — Migration Guide
- `migration.md` (~2746 tok, huge) — Migration Guide: mem-os to mind-mem
- `mind-kernels.md` (~339 tok, medium) — MIND Kernels
- `odc-retrieval.md` (~320 tok, medium) — Observer-Dependent Cognition in mind-mem
- `performance-tuning.md` (~560 tok, large) — Performance Tuning
- `quickstart.md` (~601 tok, large) — mind-mem Quickstart
- `roadmap.md` (~369 tok, medium) — Roadmap
- `scoring.md` (~517 tok, large) — Scoring System
- `security-model.md` (~350 tok, medium) — Security Model
- `testing-guide.md` (~369 tok, medium) — Testing Guide
- `troubleshooting.md` (~681 tok, large) — Troubleshooting
- `workspace-structure.md` (~352 tok, medium) — Workspace Structure
### `examples/`

- `basic_usage.py` (~394 tok, medium) — Basic mind-mem usage example.
- `README.md` (~72 tok, small) — mind-mem Examples
### `.github/`

- `CODEOWNERS` (~25 tok, tiny) — # Default owners
- `FUNDING.yml` (~4 tok, tiny) — github: star-ga
### `.github/ISSUE_TEMPLATE/`

- `bug_report.md` (~78 tok, small) — Description
- `feature_request.md` (~101 tok, small) — Description
### `.github/`

- `labels.yml` (~216 tok, medium)
- `mlc_config.json` (~55 tok, small) — Keys: ignorePatterns, timeout, retryOn429, aliveStatusCodes
- `pilot-issues.md` (~3595 tok, huge) — Pilot Week Issues (Feb 19-25)
- `pull_request_template.md` (~90 tok, small) — Summary
- `SECURITY_CONTACTS.md` (~124 tok, small) — Security Contacts
### `.github/workflows/`

- `benchmark.yml` (~651 tok, large) — name: Benchmark
- `ci.yml` (~751 tok, large) — name: CI
- `codeql.yml` (~225 tok, medium) — name: CodeQL
- `dependency-review.yml` (~114 tok, small) — name: Dependency Review
- `docs.yml` (~262 tok, medium) — name: Docs
- `label-sync.yml` (~112 tok, small) — name: Label Sync
- `release.yml` (~531 tok, large) — name: Release
- `security-review.yml` (~240 tok, medium) — name: Security Review
- `stale.yml` (~241 tok, medium) — name: Stale Issues
### `hooks/`

- `hooks.json` (~79 tok, small) — Keys: hooks
### `hooks/openclaw/mind-mem/`

- `handler.js` (~941 tok, large) — Resolve MIND_MEM_WORKSPACE from hook config env, process env, or default
- `HOOK.md` (~270 tok, medium) — Mind Mem
### `hooks/`

- `session-end.sh` (~493 tok, medium) — mind-mem Stop hook — runs auto-capture if enabled
- `session-start.sh` (~229 tok, medium) — mind-mem SessionStart hook — prints health summary for context injection
### `lib/`

- `kernels.c` (~2176 tok, huge)
### `mind/`

- `abstention.mind` (~215 tok, medium) — Confidence gating: decide whether to abstain from answering
- `adversarial.mind` (~156 tok, small)
- `bm25.mind` (~477 tok, medium) — BM25F scoring kernel with field boosts and length normalization
- `category.mind` (~395 tok, medium) — Category distillation scoring kernel
- `cognitive.mind` (~439 tok, medium)
- `cross_encoder.mind` (~174 tok, small)
- `hybrid.mind` (~169 tok, small)
- `importance.mind` (~246 tok, medium) — A-MEM: auto-maintained importance scores for memory blocks
- `intent.mind` (~149 tok, small)
- `prefetch.mind` (~256 tok, medium) — Prefetch context scoring kernel
- `ranking.mind` (~227 tok, medium) — Evidence ranking: combine multiple scoring signals for final ranking
- `README.md` (~911 tok, large) — MIND Kernels
- `recall.mind` (~207 tok, medium)
- `reranker.mind` (~412 tok, medium) — Deterministic reranking features (no model needed)
- `rerank.mind` (~146 tok, small)
- `rm3.mind` (~189 tok, small)
- `rrf.mind` (~197 tok, small) — RRF: fuse ranked lists from multiple retrievers
- `temporal.mind` (~113 tok, small)
- `trajectory.mind` (~440 tok, medium)
### `scripts/`

- `anatomy-hook.sh` (~258 tok, medium) — anatomy-hook.sh — Git pre-commit hook to refresh ANATOMY.md
- `anatomy.sh` (~2010 tok, huge) — anatomy — Generate ANATOMY.md for any repo
### `skills/apply-proposal/`

- `SKILL.md` (~345 tok, medium) — /apply — Apply Proposals
### `skills/integrity-scan/`

- `SKILL.md` (~376 tok, medium) — /scan — Memory Integrity Scan
### `skills/memory-recall/`

- `SKILL.md` (~549 tok, large) — /recall — Memory Search
### `src/`

- `mcp_server.py` (~455 tok, medium) — Compatibility wrapper for the packaged Mind-Mem MCP server."""
### `src/mind_mem/`

- `abstention_classifier.py` (~3261 tok, huge) — Deterministic adversarial abstention classifier for Mind-Mem.
- `apply_engine.py` (~15091 tok, huge) — Mind Mem Apply Engine v1.0 — Atomic proposal application with rollback.
- `audit_chain.py` (~3633 tok, huge) — mind-mem Hash-Chain Mutation Log — tamper-evident append-only ledger.
- `auto_resolver.py` (~3256 tok, huge) — mind-mem Automatic Contradiction Resolution Suggestions.
- `backup_restore.py` (~3821 tok, huge) — mind-mem Backup & Restore CLI. Zero external deps.
- `baseline_snapshot.py` (~4176 tok, huge) — Baseline snapshot for intent drift detection.
- `block_metadata.py` (~2226 tok, huge) — mind-mem A-MEM — auto-evolving block metadata.
- `block_parser.py` (~7030 tok, huge) — Mind Mem Block Parser v1.0 — Self-hosted, zero external dependencies.
- `block_store.py` (~988 tok, large) — BlockStore abstraction — decouples block access from storage format.
- `bootstrap_corpus.py` (~2161 tok, huge) — mind-mem Bootstrap Corpus — one-time backfill from existing knowledge sources.
- `calibration.py` (~4819 tok, huge) — Calibration feedback loop — track retrieval quality and adjust block ranking.
- `capture.py` (~3493 tok, huge) — mind-mem Auto-Capture Engine with Structured Extraction. Zero external deps.
- `category_distiller.py` (~6264 tok, huge) — mind-mem Category Distiller — auto-generates thematic summary files from memory blocks.
- `causal_graph.py` (~3679 tok, huge) — mind-mem Temporal Causal Graph — directed dependency tracking with staleness.
- `check_version.py` (~622 tok, large) — Version consistency checker for mind-mem.
- `coding_schemas.py` (~2127 tok, huge) — mind-mem Coding-Native Memory Schemas.
- `compaction.py` (~2889 tok, huge) — mind-mem Compaction & GC Engine. Zero external deps.
- `compiled_truth.py` (~6277 tok, huge) — mind-mem Compiled Truth — synthesized entity pages with append-only evidence.
- `conflict_resolver.py` (~2947 tok, huge) — mind-mem Automated Conflict Resolution Pipeline. Zero external deps.
- `connection_manager.py` (~1059 tok, large) — SQLite connection manager with read/write separation and WAL mode.
- `contradiction_detector.py` (~4904 tok, huge) — mind-mem Contradiction Detector — Surface conflicts at the governance gate.
- `corpus_registry.py` (~262 tok, medium) — Central corpus path registry for mind-mem.
- `cron_runner.py` (~1846 tok, huge) — mind-mem Cron Runner — single entry point for all periodic jobs. Zero external deps.
- `cross_encoder_reranker.py` (~749 tok, large) — mind-mem Optional Cross-Encoder Reranker.
- `dedup.py` (~4473 tok, huge) — mind-mem 4-layer deduplication filter for search results.
- `dream_cycle.py` (~6432 tok, huge) — mind-mem Dream Cycle — autonomous memory enrichment. Zero external deps.
- `drift_detector.py` (~4216 tok, huge) — mind-mem Semantic Belief Drift Detection.
- `encryption.py` (~2527 tok, huge) — mind-mem Encryption at Rest — optional AES-256 encryption for memory blocks.
- `entity_ingest.py` (~3225 tok, huge) — mind-mem Entity Ingestion — regex-based entity extraction. Zero external deps.
- `error_codes.py` (~1751 tok, huge) — mind-mem Error Codes — structured error classification.
- `evidence_objects.py` (~4607 tok, huge) — # Copyright 2026 STARGA, Inc.
- `evidence_packer.py` (~3267 tok, huge) — Deterministic evidence packer for Mind-Mem.
- `extraction_feedback.py` (~1160 tok, large) — mind-mem Extraction Quality Feedback Tracker.
- `extractor.py` (~6597 tok, huge) — mind-mem Entity & Fact Extractor (Regex NER-lite). Zero external deps.
- `field_audit.py` (~3086 tok, huge) — mind-mem Per-Field Mutation Audit — tracks individual field changes.
- `governance_bench.py` (~1855 tok, huge) — mind-mem Governance Benchmark Suite.
- `governance_gate.py` (~1745 tok, huge) — # Copyright 2026 STARGA, Inc.
- `hash_chain_v2.py` (~4080 tok, huge) — # Copyright 2026 STARGA, Inc.
- `hybrid_recall.py` (~4850 tok, huge) — mind-mem Hybrid Recall -- BM25 + Vector + RRF fusion.
- `__init__.py` (~499 tok, medium) — # Mind Mem — Memory + Immune System for AI agents
- `init_workspace.py` (~2062 tok, huge) — mind-mem workspace initializer. Zero external deps.
- `intel_scan.py` (~12099 tok, huge) — Mind Mem Intelligence Scanner v2.0 — Self-hosted, zero external dependencies.
- `intent_router.py` (~3106 tok, huge) — mind-mem Intent Router — 9-type adaptive query intent classification.
- `kalman_belief.py` (~4252 tok, huge) — # Copyright 2026 STARGA, Inc.
- `llm_extractor.py` (~3009 tok, huge) — mind-mem LLM Entity & Fact Extractor (Optional, config-gated).
- `llm_noise_profile.py` (~2382 tok, huge) — # Copyright 2026 STARGA, Inc.
- `mcp_entry.py` (~217 tok, medium) — Thin entry point for mind-mem-mcp console script."""
- `mcp_server.py` (~24805 tok, huge) — Mind-Mem MCP Server — persistent memory for paranoid/safety-first coding agents.
- `memory_tiers.py` (~3534 tok, huge) — # Copyright 2026 STARGA, Inc.
- `merkle_tree.py` (~2953 tok, huge) — # Copyright 2026 STARGA, Inc.
- `mind_ffi.py` (~5127 tok, huge) — mind-mem FFI bridge — loads compiled MIND .so and exposes scoring functions.
- `mind_filelock.py` (~1844 tok, huge) — mind-mem file locking — cross-platform advisory locks. Zero external deps.
- `namespaces.py` (~3569 tok, huge) — mind-mem Multi-Agent Namespace & ACL Engine. Zero external deps.
- `observability.py` (~1416 tok, large) — mind-mem Observability Module. Zero external deps.
- `observation_compress.py` (~1353 tok, large) — Observation Compression Layer for Mind-Mem.
- `py.typed` (~0 tok, tiny)
- `query_expansion.py` (~4339 tok, huge) — Multi-query expansion for improved recall.
- `_recall_constants.py` (~2420 tok, huge) — Recall engine constants — search fields, BM25 params, regex patterns, limits."""
- `_recall_context.py` (~2601 tok, huge) — Recall engine context packing — post-retrieval augmentation rules."""
- `_recall_core.py` (~14213 tok, huge) — Recall engine core — RecallBackend, main BM25 pipeline, backend loading, prefetch, CLI."""
- `_recall_detection.py` (~5168 tok, huge) — Recall engine detection — query type classification, text extraction, block utilities."""
- `_recall_expansion.py` (~3267 tok, huge) — Recall engine query expansion — domain synonyms, month normalization, RM3."""
- `recall.py` (~1049 tok, large) — mind-mem Recall Engine (BM25 + TF-IDF + Graph + Stemming). Zero external deps.
- `_recall_reranking.py` (~3247 tok, huge) — Recall engine reranking — deterministic feature-based re-scoring of BM25 hits."""
- `_recall_scoring.py` (~3112 tok, huge) — Recall engine scoring — BM25F helper, date scores, graph boosting, negation, date proximity, categories."""
- `_recall_temporal.py` (~2044 tok, huge) — Recall engine temporal filtering — resolve relative time references and filter blocks."""
- `_recall_tokenization.py` (~784 tok, large) — Recall engine tokenization — Porter stemmer and tokenizer."""
- `recall_vector.py` (~13837 tok, huge) — mind-mem Vector Recall Backend (Semantic Search with Embeddings).
- `retrieval_graph.py` (~4996 tok, huge) — Retrieval logger + co-retrieval graph for usage-based score propagation.
- `schema_version.py` (~1897 tok, huge) — Mind-Mem Schema Version Migration. Zero external deps.
- `session_summarizer.py` (~2885 tok, huge) — mind-mem Session Summarizer. Zero external deps.
- `smart_chunker.py` (~6752 tok, huge) — mind-mem Smart Chunker — Semantic-boundary document chunking.
- `smoke_test.sh` (~633 tok, large) — mind-mem Smoke Test — end-to-end verification
- `spec_binding.py` (~2544 tok, huge) — # Copyright 2026 STARGA, Inc.
- `sqlite_index.py` (~10531 tok, huge) — Mind Mem SQLite FTS5 Index — incremental lexical indexing. Zero external deps.
- `trajectory.py` (~2233 tok, huge) — Trajectory Memory — task execution trace storage and recall.
- `transcript_capture.py` (~2333 tok, huge) — mind-mem Transcript JSONL Capture. Zero external deps.
- `uncertainty_propagation.py` (~1278 tok, large) — # Copyright 2026 STARGA, Inc.
- `validate_py.py` (~3335 tok, huge) — Mind Mem Integrity Validator (Python, cross-platform).
- `validate.sh` (~6653 tok, huge) — mind-mem Integrity Validator v1.1
- `watcher.py` (~886 tok, large) — Mind-Mem File Watcher — auto-reindex on workspace changes. Zero external deps.
### `templates/`

- `AUDIT.md` (~31 tok, tiny) — AUDIT — mind-mem v1.0
- `BRIEFINGS.md` (~47 tok, tiny) — BRIEFINGS — mind-mem v1.0
- `CONTRADICTIONS.md` (~47 tok, tiny) — CONTRADICTIONS — mind-mem v1.0
- `DECISIONS.md` (~77 tok, small) — DECISIONS — mind-mem v1.0
- `DECISIONS_PROPOSED.md` (~50 tok, small) — DECISIONS_PROPOSED — mind-mem v1.0
- `DRIFT.md` (~45 tok, tiny) — DRIFT — mind-mem v1.0
- `EDITS_PROPOSED.md` (~34 tok, tiny) — EDITS_PROPOSED — mind-mem v1.0
- `IMPACT.md` (~43 tok, tiny) — IMPACT — mind-mem v1.0
- `incidents.md` (~38 tok, tiny) — INCIDENTS — mind-mem v1.0
- `intel-state.json` (~197 tok, small) — Keys: governance_mode, version, auto_apply_low_risk, flip_gate_week1_clean, last_scan
- `maint-state.json` (~12 tok, tiny) — Keys: last_run, last_weekly
- `MEMORY.md` (~70 tok, small) — Memory Protocol v1.0
- `people.md` (~31 tok, tiny) — PEOPLE — mind-mem v1.0
- `projects.md` (~39 tok, tiny) — PROJECTS — mind-mem v1.0
- `SCAN_LOG.md` (~80 tok, small) — SCAN_LOG — mind-mem v1.0
- `SIGNALS.md` (~51 tok, small) — SIGNALS — mind-mem v1.0
- `TASKS.md` (~83 tok, small) — TASKS — mind-mem v1.0
- `TASKS_PROPOSED.md` (~33 tok, tiny) — TASKS_PROPOSED — mind-mem v1.0
- `tools.md` (~33 tok, tiny) — TOOLS — mind-mem v1.0
### `tests/integration/`

- `__init__.py` (~0 tok, tiny)
- `test_full_pipeline.py` (~1436 tok, large) — Integration test: full mind-mem pipeline.
### `tests/`

- `test_abstention_classifier.py` (~3963 tok, huge) — Tests for the adversarial abstention classifier."""
- `test_active_only_filter.py` (~307 tok, medium) — Tests for active_only recall filter."""
- `test_agent_id_filter.py` (~324 tok, medium) — Tests for agent_id namespace filtering."""
- `test_allow_decompose.py` (~300 tok, medium) — Tests for _allow_decompose recall parameter."""
- `test_apply_engine.py` (~11084 tok, huge) — Tests for apply_engine.py — focus on security, validation, and rollback."""
- `test_audit_chain.py` (~2371 tok, huge) — Tests for mind-mem hash-chain mutation log (audit_chain.py)."""
- `test_auto_resolver.py` (~1185 tok, large) — Tests for mind-mem auto contradiction resolution (auto_resolver.py)."""
- `test_backup_restore.py` (~3237 tok, huge) — Tests for backup_restore.py — zero external deps (stdlib unittest)."""
- `test_baseline_snapshot.py` (~2997 tok, huge) — Tests for baseline snapshot and drift detection (#431)."""
- `test_bigrams.py` (~168 tok, small) — Tests for bigram extraction."""
- `test_block_id_format.py` (~327 tok, medium) — Tests for block ID format validation."""
- `test_block_metadata.py` (~945 tok, large) — Tests for A-MEM block metadata evolution."""
- `test_block_parser_chunks.py` (~1658 tok, huge) — Tests for block_parser.py — overlapping chunk splitting + dedup."""
- `test_block_parser_edge.py` (~620 tok, large) — Extended block parser tests."""
- `test_block_parser_fields.py` (~372 tok, medium) — Tests for block parser field extraction."""
- `test_block_parser_multifile.py` (~304 tok, medium) — Tests for parsing multiple files."""
- `test_block_parser.py` (~3116 tok, huge) — Tests for block_parser.py — zero external deps (stdlib unittest)."""
- `test_block_store.py` (~1957 tok, huge) — Tests for block_store.py — BlockStore protocol and MarkdownBlockStore."""
- `test_block_types.py` (~426 tok, medium) — Tests for different block types in recall."""
- `test_bootstrap_corpus.py` (~1734 tok, huge) — Tests for bootstrap_corpus.py — backfill pipeline module."""
- `test_calibration.py` (~3286 tok, huge) — Tests for calibration feedback loop.
- `test_capture.py` (~2071 tok, huge) — Tests for capture.py — zero external deps (stdlib unittest)."""
- `test_category_distiller.py` (~2639 tok, huge) — Tests for category_distiller.py — CategoryDistiller class."""
- `test_causal_graph.py` (~1566 tok, huge) — Tests for mind-mem temporal causal graph (causal_graph.py)."""
- `test_check_version.py` (~271 tok, medium) — Tests for version consistency checker."""
- `test_chunk_text.py` (~231 tok, medium) — Tests for text chunking."""
- `test_coding_schemas.py` (~1294 tok, large) — Tests for mind-mem coding-native memory schemas."""
- `test_compaction.py` (~1796 tok, huge) — Tests for compaction.py — GC and archival engine."""
- `test_compiled_truth.py` (~3912 tok, huge) — Tests for mind-mem compiled truth pages (compiled_truth.py)."""
- `test_concurrency_stress.py` (~4065 tok, huge) — Concurrency and performance stress tests for recall engine.
- `test_concurrent_integration.py` (~10810 tok, huge) — Integration tests for concurrent access and partial failure in mind-mem.
- `test_conflict_resolver.py` (~2362 tok, huge) — Tests for conflict_resolver.py — zero external deps (stdlib unittest)."""
- `test_connection_manager.py` (~2536 tok, huge) — Tests for ConnectionManager — SQLite connection pooling with read/write separation (#466)."""
- `test_constants.py` (~371 tok, medium) — Tests for recall constants module."""
- `test_context_pack.py` (~2584 tok, huge) — Tests for context_pack rules: adjacency, diversity, pronoun rescue."""
- `test_context_pack_scripts.py` (~673 tok, large) — Tests for context packing via scripts._recall_context."""
- `test_contradiction_detector.py` (~5833 tok, huge) — Tests for contradiction_detector.py — Contradiction detection at governance gate (#432).
- `test_core_v140.py` (~2708 tok, huge) — Tests for v1.4.0 core hardening: issues #28, #30, #32, #34."""
- `test_cron_runner.py` (~2696 tok, huge) — Tests for cron_runner.py — periodic job orchestration, config loading, subprocess dispatch."""
- `test_cross_encoder.py` (~1324 tok, large) — Tests for optional cross-encoder reranker."""
- `test_date_score.py` (~174 tok, small) — Tests for date scoring function."""
- `test_decompose_query.py` (~223 tok, medium) — Tests for query decomposition."""
- `test_dedup.py` (~5632 tok, huge) — Tests for dedup.py -- 4-layer deduplication filter."""
- `test_dedup_vector.py` (~1129 tok, large) — Tests for vector-enhanced cosine dedup (Layer 2b)."""
- `test_delete_memory.py` (~329 tok, medium) — Tests for memory deletion functionality."""
- `test_detection.py` (~326 tok, medium) — Tests for query detection module."""
- `test_dream_cycle.py` (~4671 tok, huge) — Tests for dream_cycle.py — autonomous memory enrichment passes."""
- `test_drift_detector.py` (~1613 tok, huge) — Tests for mind-mem semantic belief drift detection (drift_detector.py)."""
- `test_edge_cases.py` (~3888 tok, huge) — Edge-case and stress tests for mind-mem — block_parser, recall, and MCP server."""
- `test_encryption.py` (~1701 tok, huge) — Tests for mind-mem encryption at rest."""
- `test_entity_ingest.py` (~4091 tok, huge) — Tests for the entity_ingest module — extraction, filtering, signal generation."""
- `test_error_codes.py` (~2394 tok, huge) — Tests for mind-mem Error Codes module."""
- `test_error_paths.py` (~5711 tok, huge) — Error path and edge-case tests for mind-mem — malformed inputs, missing files, bad configs."""
- `test_evidence_objects.py` (~3985 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_evidence_packer.py` (~5180 tok, huge) — Tests for the evidence packer module."""
- `test_excerpt.py` (~248 tok, medium) — Tests for excerpt generation."""
- `test_expand_query.py` (~265 tok, medium) — Tests for query expansion module."""
- `test_export_memory.py` (~335 tok, medium) — Tests for memory export functionality."""
- `test_extractor.py` (~3392 tok, huge) — Tests for the regex NER-lite entity/fact extractor."""
- `test_fact_indexing.py` (~3096 tok, huge) — Tests for Feature 2 (fact card indexing) and Feature 4 (metadata-augmented embeddings)."""
- `test_field_audit.py` (~1399 tok, large) — Tests for mind-mem per-field mutation audit (field_audit.py)."""
- `test_field_extraction.py` (~201 tok, medium) — Tests for field token extraction."""
- `test_filelock.py` (~975 tok, large) — Tests for filelock.py — cross-platform advisory locking."""
- `test_filelock_stress.py` (~1106 tok, large) — Stress tests for mind-mem file locking under contention."""
- `test_fts_fallback.py` (~4429 tok, huge) — Tests for FTS fallback behavior, recall envelope structure, block size cap,
- `test_governance_bench.py` (~802 tok, large) — Tests for mind-mem governance benchmark suite."""
- `test_graph_boost.py` (~6034 tok, huge) — Tests for graph boost, context packing, config validation, and block cap.
- `test_graph_boost_recall.py` (~310 tok, medium) — Tests for graph_boost recall parameter."""
- `test_grid_search.py` (~1199 tok, large) — Tests for benchmarks/grid_search.py — grid generation and utility functions."""
- `test_hash_chain_v2.py` (~3472 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_hybrid_recall.py` (~2855 tok, huge) — Tests for hybrid_recall.py -- HybridBackend + RRF fusion."""
- `test_hybrid_search.py` (~588 tok, large) — Tests for hybrid search functionality."""
- `test_index_stats.py` (~298 tok, medium) — Tests for index statistics."""
- `test_init_workspace.py` (~2243 tok, huge) — Tests for init_workspace — config validation and workspace scaffolding."""
- `test_install_script.py` (~376 tok, medium)
- `test_integration.py` (~1368 tok, large) — Integration test: full mind-mem lifecycle init → capture → scan → recall."""
- `test_intel_scan.py` (~5873 tok, huge) — Tests for intel_scan.py — contradiction detection, drift analysis, impact graph."""
- `test_intent_classify.py` (~328 tok, medium) — Tests for intent classification."""
- `test_intent_router_adaptive.py` (~3613 tok, huge) — Tests for adaptive intent routing (#470).
- `test_intent_router.py` (~1176 tok, large) — Tests for 9-type intent router."""
- `test_kalman_belief.py` (~3756 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_llm_extractor.py` (~1773 tok, huge) — Tests for the optional LLM entity/fact extractor module."""
- `test_llm_noise_profile.py` (~2358 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mcp_integration.py` (~5110 tok, huge) — MCP transport and auth integration tests (#474).
- `test_mcp_server.py` (~4897 tok, huge) — Tests for mcp_server.py — tests the MCP server resources and tool logic.
- `test_mcp_tools.py` (~277 tok, medium) — Tests for MCP server tool definitions."""
- `test_mcp_v140.py` (~5237 tok, huge) — Tests for MCP v1.4.0 features — issues #29, #31, #35, #36.
- `test_memory_evolution.py` (~329 tok, medium) — Tests for memory evolution tracking."""
- `test_memory_tiers.py` (~3509 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_merkle_tree.py` (~3328 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mind_ffi.py` (~291 tok, medium) — Tests for MIND FFI module."""
- `test_multi_file_recall.py` (~325 tok, medium) — Tests for recall across multiple files."""
- `test_namespaces.py` (~2393 tok, huge) — Tests for namespaces.py — zero external deps (stdlib unittest)."""
- `test_niah.py` (~4894 tok, huge) — Needle In A Haystack (NIAH) benchmark for mind-mem recall.
- `test_observability.py` (~791 tok, large) — Tests for observability.py — structured logging and metrics."""
- `test_observation_compress.py` (~2754 tok, huge) — Tests for observation_compress module.
- `test_prefetch_context.py` (~1482 tok, large) — Tests for prefetch_context() in recall.py."""
- `test_prefetch.py` (~315 tok, medium) — Tests for prefetch functionality."""
- `test_query_decomposition.py` (~1609 tok, huge) — Tests for multi-hop query decomposition (#6)."""
- `test_query_expansion_multi_provider.py` (~1055 tok, large) — Tests for multi-provider LLM query expansion (OpenAI-compatible endpoints)."""
- `test_query_expansion.py` (~3781 tok, huge) — Tests for query_expansion.py -- multi-query expansion for improved recall."""
- `test_recall_concurrent.py` (~333 tok, medium) — Tests for concurrent recall queries."""
- `test_recall_context_field.py` (~252 tok, medium) — Tests for context field in blocks."""
- `test_recall_cross_encoder.py` (~1372 tok, large) — Tests for cross-encoder reranker integration in recall pipeline."""
- `test_recall_date_field.py` (~304 tok, medium) — Tests for date field in recall results."""
- `test_recall_detection.py` (~1533 tok, huge) — Tests for _recall_detection.py — query type classification and text extraction."""
- `test_recall_edge_cases.py` (~566 tok, large) — Edge case tests for recall engine."""
- `test_recall_empty_query_types.py` (~311 tok, medium) — Tests for various empty/minimal query types."""
- `test_recall_empty_workspace.py` (~134 tok, small) — Tests for recall on empty workspaces."""
- `test_recall_intent_router.py` (~1234 tok, large) — Tests for IntentRouter integration in recall pipeline."""
- `test_recall_large_workspace.py` (~332 tok, medium) — Tests for recall with large workspaces."""
- `test_recall_limit.py` (~386 tok, medium) — Tests for recall limit parameter behavior."""
- `test_recall_metadata.py` (~1340 tok, large) — Tests for A-MEM block metadata integration in recall pipeline."""
- `test_recall_priority.py` (~410 tok, medium) — Tests for priority boost in recall."""
- `test_recall.py` (~3825 tok, huge) — Tests for recall.py — zero external deps (stdlib unittest)."""
- `test_recall_references.py` (~259 tok, medium) — Tests for reference-based recall."""
- `test_recall_reranking.py` (~2740 tok, huge) — Tests for _recall_reranking.py — deterministic reranker + LLM rerank."""
- `test_recall_scoring_order.py` (~435 tok, medium) — Tests for recall result scoring order."""
- `test_recall_source_field.py` (~268 tok, medium) — Tests for source field in recall results."""
- `test_recall_speaker.py` (~252 tok, medium) — Tests for speaker-based recall."""
- `test_recall_status_boost.py` (~394 tok, medium) — Tests for status boost in recall."""
- `test_recall_supersedes.py` (~211 tok, medium) — Tests for supersedes field in recall."""
- `test_recall_tags.py` (~309 tok, medium) — Tests for tag-based recall."""
- `test_recall_temporal.py` (~2800 tok, huge) — Tests for _recall_temporal.py — time-aware hard filters for temporal queries."""
- `test_recall_vector.py` (~4901 tok, huge) — Tests for recall_vector.py — VectorBackend semantic search."""
- `test_rerank_debug.py` (~331 tok, medium) — Tests for rerank debug mode."""
- `test_reranking.py` (~246 tok, medium) — Tests for reranking module."""
- `test_retrieval_diagnostics.py` (~2426 tok, huge) — Tests for retrieval diagnostics (#428), corpus isolation (#429), and intent instrumentation (#430)."""
- `test_retrieval_graph.py` (~2248 tok, huge) — Tests for retrieval_graph.py — retrieval logging, co-retrieval graph, hard negatives."""
- `test_rm3_expand.py` (~321 tok, medium) — Tests for RM3 query expansion."""
- `test_scan_engine.py` (~316 tok, medium) — Tests for integrity scan engine."""
- `test_schema_version.py` (~1726 tok, huge) — Tests for schema_version.py — zero external deps (stdlib unittest)."""
- `test_scoring.py` (~337 tok, medium) — Tests for BM25 scoring functions."""
- `test_session_summarizer.py` (~3933 tok, huge) — Comprehensive tests for mind_mem/session_summarizer.py.
- `test_skeptical_query.py` (~194 tok, small) — Tests for skeptical query detection."""
- `test_smart_chunker_code.py` (~1240 tok, large) — Tests for code-aware chunking in smart_chunker.py."""
- `test_smart_chunker.py` (~7767 tok, huge) — Tests for smart_chunker.py — semantic-boundary document chunking."""
- `test_spec_binding.py` (~3195 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_sqlite_index.py` (~4727 tok, huge) — Tests for sqlite_index.py — SQLite FTS5 index for mind-mem recall."""
- `test_stopwords.py` (~247 tok, medium) — Tests for stopword handling."""
- `test_temporal.py` (~223 tok, medium) — Tests for temporal filtering module."""
- `test_tokenization.py` (~436 tok, medium) — Tests for tokenization module."""
- `test_trajectory.py` (~2385 tok, huge) — Tests for trajectory.py — trajectory memory block operations."""
- `test_transcript_capture.py` (~3230 tok, huge) — Tests for transcript_capture.py — zero external deps (stdlib unittest)."""
- `test_uncertainty_propagation.py` (~2158 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_unicode_edge_cases.py` (~2440 tok, huge) — Tests for Unicode and edge case handling across mind-mem modules."""
- `test_validate_py.py` (~3441 tok, huge) — Tests for validate_py.py — workspace integrity validator."""
- `test_watcher.py` (~1203 tok, large) — Tests for watcher.py — file change detection for auto-reindex."""
- `test_wide_retrieval.py` (~335 tok, medium) — Tests for wide retrieval parameter."""
- `test_workspace_init.py` (~461 tok, medium) — Tests for workspace initialization."""
- `test_workspace_structure.py` (~509 tok, large) — Tests for workspace directory structure."""

---
*Generated by `anatomy 1.0.0`. Edit descriptions manually — re-run preserves structure.*

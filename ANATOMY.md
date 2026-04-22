# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind-mem`
**Files:** 667 | **Est. tokens:** ~1,304,090
**Generated:** 2026-04-22 04:57 UTC

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
| `./` | 30 | ~73,533 |
| `.agents/skills/mind-mem-development/` | 1 | ~456 |
| `benchmarks/` | 26 | ~59,234 |
| `deploy/` | 2 | ~690 |
| `deploy/docker/` | 1 | ~495 |
| `deploy/edge/` | 2 | ~1,149 |
| `deploy/grafana/` | 1 | ~1,145 |
| `docs/` | 57 | ~82,742 |
| `docs/adr/` | 2 | ~521 |
| `docs/design/` | 2 | ~2,416 |
| `docs/security-baselines/` | 1 | ~18,974 |
| `examples/` | 2 | ~466 |
| `.github/` | 7 | ~4,109 |
| `.github/ISSUE_TEMPLATE/` | 2 | ~179 |
| `.github/workflows/` | 10 | ~5,938 |
| `hooks/` | 3 | ~801 |
| `hooks/openclaw/mind-mem/` | 2 | ~1,211 |
| `intelligence/` | 1 | ~113 |
| `intelligence/state/snapshots/` | 1 | ~114 |
| `lib/` | 1 | ~2,176 |
| `mind/` | 26 | ~7,230 |
| `scripts/` | 4 | ~3,329 |
| `sdk/go/` | 9 | ~6,773 |
| `sdk/js/` | 5 | ~1,388 |
| `sdk/js/src/` | 4 | ~2,320 |
| `sdk/js/test/` | 1 | ~2,191 |
| `skills/apply-proposal/` | 1 | ~345 |
| `skills/integrity-scan/` | 1 | ~376 |
| `skills/memory-recall/` | 1 | ~549 |
| `src/` | 1 | ~259 |
| `src/mind_mem/` | 142 | ~470,443 |
| `src/mind_mem/api/` | 5 | ~14,584 |
| `src/mind_mem/mcp/` | 3 | ~3,225 |
| `src/mind_mem/mcp/infra/` | 8 | ~6,210 |
| `src/mind_mem/mcp/tools/` | 17 | ~37,534 |
| `src/mind_mem/skill_opt/` | 11 | ~13,558 |
| `src/mind_mem/storage/` | 2 | ~3,878 |
| `templates/` | 19 | ~1,041 |
| `tests/` | 228 | ~442,702 |
| `tests/integration/` | 2 | ~1,436 |
| `train/` | 10 | ~21,806 |
| `web/` | 5 | ~927 |
| `web/app/` | 2 | ~1,204 |
| `web/app/console/` | 1 | ~1,169 |
| `web/components/` | 4 | ~2,482 |
| `web/lib/` | 1 | ~669 |

## Files

### `./`

- `AUDIT_FINDINGS_FOR_CLAUDE.md` (~995 tok, large) — Comprehensive Architectural Audit: mind-mem (Commit 30d8b71)
- `CLAUDE.md` (~1028 tok, large) — mind-mem — Persistent AI Memory System
- `conftest.py` (~1010 tok, large) — Shared pytest fixtures for mind-mem test suite."""
- `CONTRIBUTING.md` (~309 tok, medium) — Contributing to mind-mem
- `demo-setup.sh` (~323 tok, medium) — Pre-seed a demo workspace for VHS recording
- `demo.tape` (~93 tok, small) — # mind-mem demo — terminal recording for README
- `Dockerfile` (~54 tok, small) — FROM python:3.12-slim
- `.dockerignore` (~37 tok, tiny) — .git
- `.editorconfig` (~107 tok, small) — # EditorConfig — https://editorconfig.org
- `generate_mind7b_training.py` (~5558 tok, huge) — Generate training data for Mind7B — a purpose-trained 7B model for mind-mem.
- `.gitattributes` (~96 tok, small) — # Auto-detect text files and normalize line endings
- `.gitignore` (~120 tok, small) — *.pyc
- `.gitleaks.toml` (~314 tok, medium) — title = "mind-mem gitleaks config"
- `install-bootstrap.sh` (~1756 tok, huge) — mind-mem one-command bootstrap installer
- `install.sh` (~3337 tok, huge) — mind-mem installer — sets up MCP server + hooks for all supported clients
- `LICENSE` (~2695 tok, huge)
- `Makefile` (~569 tok, large) — .PHONY: test lint bench install dev clean smoke help regen-bash-literals
- `mcp_server.py` (~662 tok, large) — Source-checkout entrypoint for the packaged Mind-Mem MCP server.
- `mind-mem.example.json` (~174 tok, small) — Keys: recall, prompts, categories, extraction, limits
- `.pre-commit-config.yaml` (~366 tok, medium) — repos:
- `pyproject.toml` (~1712 tok, huge) — [project]
- `.python-version` (~2 tok, tiny) — 3.12
- `README.md` (~22305 tok, huge) — Shared Memory Across All Your AI Agents
- `requirements-optional.txt` (~714 tok, large) — # mind-mem optional dependencies — pinned with SHA256 integrity hashes.
- `ROADMAP.md` (~17296 tok, huge) — mind-mem Roadmap
- `SECURITY_AUDIT_2026-04.md` (~2403 tok, huge) — Security Audit — mind-mem v3.1.9 (April 2026)
- `SECURITY.md` (~1752 tok, huge) — Security Policy
- `SPEC.md` (~5184 tok, huge) — Mind Mem Formal Specification v1.0
- `train_mind7b_runpod.py` (~1654 tok, huge)
- `uninstall.sh` (~908 tok, large) — mind-mem uninstaller — removes MCP server entries from all configured clients
### `.agents/skills/mind-mem-development/`

- `SKILL.md` (~456 tok, medium) — mind-mem Development
### `benchmarks/`

- `bench_kernels.py` (~4027 tok, huge) — Benchmark: MIND kernels vs pure Python scoring.
- `cache_effectiveness.py` (~2717 tok, huge) — Cache-effectiveness benchmark — Redis L2 vs LRU-only vs no-cache.
- `cache_effectiveness_v3.2.1.json` (~227 tok, medium) — Keys: n_blocks, n_queries, pool_size, repeat_pct, runs
- `CACHE.md` (~1028 tok, large) — Recall Cache Effectiveness Benchmark
- `compare_runs.py` (~857 tok, large) — Compare two LoCoMo benchmark runs side-by-side.
- `crossencoder_ab.py` (~3205 tok, huge) — Cross-Encoder A/B Test — retrieval-level comparison.
- `generate_dispatcher_examples.py` (~2346 tok, huge) — Generate synthetic training examples for the v3.2.x 7-dispatcher MCP surface.
- `generate_retrieval_examples.py` (~1686 tok, huge) — Generate training examples for v3.3.0 retrieval shapes.
- `grid_search.py` (~2849 tok, huge) — BM25F Field Weight Grid Search for mind-mem Recall Engine.
- `__init__.py` (~0 tok, tiny)
- `local_stack_audit.py` (~1802 tok, huge) — Single-shot audit of the local mind-mem stack before a bench run.
- `locomo_harness.py` (~4147 tok, huge) — LoCoMo Benchmark Harness for mind-mem Recall Engine.
- `locomo_judge.py` (~11545 tok, huge) — LoCoMo LLM-as-Judge Evaluation for Mind-Mem.
- `locomo_v3.3.0_benchmark_config.json` (~450 tok, medium) — Keys: _comment, version, recall, cache, cross_encoder
- `longmemeval_harness.py` (~2973 tok, huge) — LongMemEval Benchmark Harness for mind-mem recall engine.
- `niah_full_results.txt` (~5140 tok, huge) — ============================= test session starts ==============================
- `NIAH.md` (~1620 tok, huge) — Needle In A Haystack (NIAH) Benchmark
- `niah_v3.2.1_redis_results.txt` (~113 tok, small) — ============================= test session starts ==============================
- `niah_v3.2.1_results.txt` (~205 tok, medium) — ============================= test session starts ==============================
- `README_benchmark_mode.md` (~1089 tok, large) — Full-capability benchmark mode (v3.3.0)
- `REPORT.md` (~3973 tok, huge) — mind-mem Benchmark Report
- `runpod_kickoff.sh` (~1779 tok, huge) — mind-mem-4b v2 — Runpod one-shot kickoff.
- `tier_weight_search.py` (~1615 tok, huge) — Grid-search per-tier weights against LoCoMo judge scores (v3.3.0 T4 #10).
- `train_config_a100.yaml` (~347 tok, medium) — base_model: star-ga/mind-mem-4b
- `train_config.yaml` (~208 tok, medium) — base_model: star-ga/mind-mem-4b
- `train_mind_mem_4b.py` (~3286 tok, huge) — mind-mem-4b v2 training script — Runpod H200 full-fine-tune.
### `deploy/`

- `docker-compose.yml` (~608 tok, large) — name: mind-mem
### `deploy/docker/`

- `Dockerfile` (~495 tok, medium) — # Stage 1: build — install all deps and produce a pruned site-packages
### `deploy/edge/`

- `pyoxidizer.bzl` (~605 tok, large) — # mind-mem-edge — PyOxidizer build spec (v4.0 prep).
- `README.md` (~544 tok, large) — mind-mem-edge — single-binary distribution (v4.0 prep)
### `deploy/grafana/`

- `mind-mem-dashboard.json` (~1145 tok, large) — Keys: __inputs, __requires, annotations, description, editable
### `deploy/`

- `Makefile` (~82 tok, small) — .PHONY: up down logs shell status build pull
### `docs/adr/`

- `001-zero-dependencies.md` (~316 tok, medium) — ADR-001: Zero External Dependencies in Core
- `002-bm25f-scoring.md` (~205 tok, medium) — ADR-002: BM25F as Primary Scoring Algorithm
### `docs/`

- `api-reference.md` (~1477 tok, large) — API Reference
- `architecture.md` (~1849 tok, huge) — Architecture
- `benchmarks.md` (~759 tok, large) — Benchmarks
- `block-format.md` (~431 tok, medium) — Block Format
- `changelog-format.md` (~217 tok, medium) — Changelog Format Guide
- `ci-workflows.md` (~254 tok, medium) — CI Workflows
- `claude-desktop-setup.md` (~752 tok, large) — Claude Desktop Setup Guide
- `client-integrations.md` (~2533 tok, huge) — Client Integrations
- `cli-reference.md` (~1595 tok, huge) — CLI Reference
- `comparison.md` (~313 tok, medium) — Comparison with Alternatives
- `competitive-analysis-persistent-memory-2026.md` (~4089 tok, huge) — Comprehensive Competitive Analysis: Persistent Memory Systems for AI Coding Agents (2025–2026)
- `configuration.md` (~7946 tok, huge) — Configuration Reference
### `docs/design/`

- `v3-mcp-surface-reduction.md` (~1080 tok, large) — v3.0 Design: MCP Tool Surface Reduction
- `v3-multi-tenancy.md` (~1336 tok, large) — v3.0 Design: Multi-Tenancy Foundation
### `docs/`

- `development.md` (~358 tok, medium) — Development Guide
- `docker-deployment.md` (~446 tok, medium) — Docker Deployment
- `faq.md` (~374 tok, medium) — FAQ
- `getting-started.md` (~405 tok, medium) — Getting Started
- `glossary.md` (~263 tok, medium) — Glossary
- `hf-mind-mem-4b-v2-README.md` (~2154 tok, huge) — mind-mem-4b v2 (2026-04-21)
- `maintenance-namespaces.md` (~1601 tok, huge) — `maintenance/` namespaces
- `mcp-integration.md` (~1538 tok, huge) — MCP Integration Guide
- `mcp-tool-examples.md` (~902 tok, large) — MCP Tool Examples
- `migration-guide.md` (~421 tok, medium) — Migration Guide
- `migration.md` (~2754 tok, huge) — Migration Guide: mem-os to mind-mem
- `mind-kernels.md` (~339 tok, medium) — MIND Kernels
- `mind-mem-4b-setup.md` (~2338 tok, huge) — Setting up the mind-mem-4b model
- `mind-mem-4b-v2-training-recipe.md` (~1683 tok, huge) — mind-mem-4b v2 training recipe — Runpod H200
- `odc-retrieval.md` (~834 tok, large) — Observer-Dependent Cognition in mind-mem
- `performance-tuning.md` (~560 tok, large) — Performance Tuning
- `protection.md` (~1443 tok, large) — mind-mem Library Protection
- `quickstart.md` (~601 tok, large) — mind-mem Quickstart
- `rest-api.md` (~1137 tok, large) — mind-mem REST API
- `review-architecture-v3.2.0.md` (~1919 tok, huge) — Architecture Review — mind-mem v3.2.0 (Release Candidate)
- `review-database-v3.2.0.md` (~3171 tok, huge) — Database Review — PostgresBlockStore v3.2.0
- `review-docs-v3.2.0.md` (~1957 tok, huge) — Documentation Review — mind-mem v3.2.0
- `review-tests-v3.2.0.md` (~1300 tok, large) — Test Review — mind-mem v3.2.0
- `roadmap.md` (~1294 tok, large) — Roadmap
- `scoring.md` (~517 tok, large) — Scoring System
- `SECURITY_AUDIT_SELF_2026_04.md` (~2257 tok, huge) — mind-mem v3.2.0 — Self-Audit Plan (Post-Release Deliverable)
- `security-audit-sow.md` (~3336 tok, huge) — mind-mem — External Security Audit Statement of Work (SoW)
### `docs/security-baselines/`

- `bandit-v3.2.0-baseline.json` (~18974 tok, huge) — Keys: errors, generated_at, metrics, results
### `docs/`

- `security-model.md` (~350 tok, medium) — Security Model
- `setup.md` (~1741 tok, huge) — Setup
- `storage-backends.md` (~1091 tok, large) — Storage Backends
- `storage-migration.md` (~2391 tok, huge) — Storage Backend Migration Guide
- `supply-chain-security.md` (~1051 tok, large) — Supply-Chain Security
- `testing-guide.md` (~369 tok, medium) — Testing Guide
- `troubleshooting.md` (~681 tok, large) — Troubleshooting
- `usage.md` (~2011 tok, huge) — Usage
- `v3.1.9-self-audit.md` (~1396 tok, large) — Self-audit after v3.1.9
- `v3.2.0-atomicity-scope-plan.md` (~1681 tok, huge) — v3.2.0 — Atomicity scope plan (§2.2)
- `v3.2.0-blockstore-routing-plan.md` (~2116 tok, huge) — v3.2.0 — Apply engine → BlockStore routing plan
- `v3.2.0-mcp-decomposition-plan.md` (~2575 tok, huge) — v3.2.0 — MCP server decomposition plan
- `v3.2.0-release-notes.md` (~1881 tok, huge) — mind-mem v3.2.0 — Production Deployment Release
- `v3.2.1-release-notes.md` (~1302 tok, large) — mind-mem v3.2.1 release notes
- `v3.3.0-release-notes.md` (~1157 tok, large) — mind-mem v3.3.0 release notes
- `v3.4.0-release-notes.md` (~1209 tok, large) — mind-mem v3.4.0 release notes
- `v3.4.0-roadmap-llm-consensus.md` (~1271 tok, large) — v3.4.0 roadmap — path to 90+ on LoCoMo
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

- `benchmark.yml` (~735 tok, large) — name: Benchmark
- `ci.yml` (~813 tok, large) — name: CI
- `codeql.yml` (~225 tok, medium) — name: CodeQL
- `dependency-review.yml` (~114 tok, small) — name: Dependency Review
- `docs.yml` (~262 tok, medium) — name: Docs
- `label-sync.yml` (~112 tok, small) — name: Label Sync
- `release.yml` (~1432 tok, large) — name: Release
- `security-review.yml` (~240 tok, medium) — name: Security Review
- `security.yml` (~1764 tok, huge) — name: Supply-Chain Security
- `stale.yml` (~241 tok, medium) — name: Stale Issues
### `hooks/`

- `hooks.json` (~79 tok, small) — Keys: hooks
### `hooks/openclaw/mind-mem/`

- `handler.js` (~941 tok, large) — Resolve MIND_MEM_WORKSPACE from hook config env, process env, or default
- `HOOK.md` (~270 tok, medium) — Mind Mem
### `hooks/`

- `session-end.sh` (~493 tok, medium) — mind-mem Stop hook — runs auto-capture if enabled
- `session-start.sh` (~229 tok, medium) — mind-mem SessionStart hook — prints health summary for context injection
### `intelligence/`

- `BRIEFINGS.md` (~113 tok, small) — Intelligence Briefings
### `intelligence/state/snapshots/`

- `S-2026-04-13.json` (~114 tok, small) — Keys: date, generated_at, decisions, tasks, projects
### `lib/`

- `kernels.c` (~2176 tok, huge)
### `mind/`

- `abstention.mind` (~215 tok, medium) — Confidence gating: decide whether to abstain from answering
- `adversarial.mind` (~156 tok, small)
- `answer.mind` (~413 tok, medium)
- `bm25.mind` (~477 tok, medium) — BM25F scoring kernel with field boosts and length normalization
- `category.mind` (~395 tok, medium) — Category distillation scoring kernel
- `cognitive.mind` (~437 tok, medium)
- `cross_encoder.mind` (~174 tok, small)
- `ensemble.mind` (~230 tok, medium)
- `evidence.mind` (~225 tok, medium)
- `graph.mind` (~228 tok, medium)
- `hybrid.mind` (~169 tok, small)
- `importance.mind` (~246 tok, medium) — A-MEM: auto-maintained importance scores for memory blocks
- `intent.mind` (~149 tok, small)
- `prefetch.mind` (~256 tok, medium) — Prefetch context scoring kernel
- `query_plan.mind` (~259 tok, medium)
- `ranking.mind` (~227 tok, medium) — Evidence ranking: combine multiple scoring signals for final ranking
- `README.md` (~911 tok, large) — MIND Kernels
- `recall.mind` (~207 tok, medium)
- `reranker.mind` (~412 tok, medium) — Deterministic reranking features (no model needed)
- `rerank.mind` (~146 tok, small)
- `rm3.mind` (~189 tok, small)
- `rrf.mind` (~197 tok, small) — RRF: fuse ranked lists from multiple retrievers
- `session.mind` (~148 tok, small)
- `temporal.mind` (~113 tok, small)
- `trajectory.mind` (~440 tok, medium)
- `truth.mind` (~211 tok, medium)
### `scripts/`

- `anatomy-hook.sh` (~258 tok, medium) — anatomy-hook.sh — Git pre-commit hook to refresh ANATOMY.md
- `anatomy.sh` (~2010 tok, huge) — anatomy — Generate ANATOMY.md for any repo
- `build_integrity_manifest.py` (~637 tok, large) — Bake ``_integrity_manifest.json`` into the package before wheel build.
- `regen_bash_literals.py` (~424 tok, medium) — Regenerate src/mind_mem/_task_status_literals.sh from enums.py.
### `sdk/go/`

- `client.go` (~1008 tok, large) — Option is a functional option for NewClient.
- `client_test.go` (~2905 tok, huge) — Helpers
- `doc.go` (~334 tok, medium) — Package mindmem is the official Go SDK for the mind-mem REST API.
- `errors.go` (~640 tok, large) — APIError is returned for any non-2xx response from the mind-mem server.
- `.gitignore` (~5 tok, tiny) — *.test
- `go.mod` (~13 tok, tiny) — module github.com/star-ga/mind-mem/sdk/go
- `methods.go` (~500 tok, large) — Recall queries the memory store using full-text and semantic search.
- `README.md` (~520 tok, large) — mind-mem Go SDK
- `types.go` (~848 tok, large) — BlockTier represents the storage tier of a memory block.
### `sdk/js/`

- `.gitignore` (~5 tok, tiny) — node_modules/
- `package.json` (~275 tok, medium) — Keys: name, version, description, license, type
- `package-lock.json` (~381 tok, medium) — Keys: name, version, lockfileVersion, requires, packages
- `README.md` (~580 tok, large) — @mind-mem/sdk
### `sdk/js/src/`

- `client.ts` (~1250 tok, large) — Normalise: strip trailing slash so path joining is consistent.
- `errors.ts` (~438 tok, medium) — Restore prototype chain (required when extending built-ins in TS)
- `index.ts` (~99 tok, small)
- `types.ts` (~533 tok, large) — Shared domain types
### `sdk/js/test/`

- `client.test.ts` (~2191 tok, huge) — Minimal fetch mock helpers
### `sdk/js/`

- `tsconfig.json` (~147 tok, small) — Keys: compilerOptions, include, exclude
### `skills/apply-proposal/`

- `SKILL.md` (~345 tok, medium) — /apply — Apply Proposals
### `skills/integrity-scan/`

- `SKILL.md` (~376 tok, medium) — /scan — Memory Integrity Scan
### `skills/memory-recall/`

- `SKILL.md` (~549 tok, large) — /recall — Memory Search
### `src/`

- `mcp_server.py` (~259 tok, medium) — Wheel-level compatibility module for `mind_mem.mcp_server`.
### `src/mind_mem/`

- `abstention_classifier.py` (~3261 tok, huge) — Deterministic adversarial abstention classifier for Mind-Mem.
- `agent_bridge.py` (~4158 tok, huge) — # Copyright 2026 STARGA, Inc.
- `alerting.py` (~2411 tok, huge) — # Copyright 2026 STARGA, Inc.
- `answer_quality.py` (~2396 tok, huge) — Answer-quality layer: verification + self-consistency + per-category spec.
### `src/mind_mem/api/`

- `api_keys.py` (~2717 tok, huge) — Per-agent API key store for the mind-mem REST API.
- `auth.py` (~2476 tok, huge) — OIDC/SSO authentication for the mind-mem REST API.
- `grpc_server.py` (~1723 tok, huge) — gRPC wire protocol for mind-mem (v4.0 prep).
- `__init__.py` (~20 tok, tiny)
- `rest.py` (~7648 tok, huge) — REST API layer for mind-mem (v3.2.0, v3.2.1 hardening).
### `src/mind_mem/`

- `apply_engine.py` (~14536 tok, huge) — Mind Mem Apply Engine v1.0 — Atomic proposal application with rollback.
- `audit_chain.py` (~4167 tok, huge) — mind-mem Hash-Chain Mutation Log — tamper-evident append-only ledger.
- `auto_resolver.py` (~3194 tok, huge) — mind-mem Automatic Contradiction Resolution Suggestions.
- `axis_recall.py` (~4217 tok, huge) — # Copyright 2026 STARGA, Inc.
- `backup_restore.py` (~3821 tok, huge) — mind-mem Backup & Restore CLI. Zero external deps.
- `baseline_snapshot.py` (~4176 tok, huge) — Baseline snapshot for intent drift detection.
- `block_metadata.py` (~2223 tok, huge) — mind-mem A-MEM — auto-evolving block metadata.
- `block_parser.py` (~7111 tok, huge) — Mind Mem Block Parser v1.0 — Self-hosted, zero external dependencies.
- `block_store_encrypted.py` (~2313 tok, huge) — # Copyright 2026 STARGA, Inc.
- `block_store_postgres.py` (~7858 tok, huge) — PostgresBlockStore — PostgreSQL-backed BlockStore for mind-mem v3.2.0.
- `block_store_postgres_replica.py` (~2095 tok, huge) — v3.2.0 — read-replica routing for PostgresBlockStore.
- `block_store.py` (~8923 tok, huge) — BlockStore abstraction — decouples block access from storage format.
- `bootstrap_corpus.py` (~2158 tok, huge) — mind-mem Bootstrap Corpus — one-time backfill from existing knowledge sources.
- `calibration.py` (~4811 tok, huge) — Calibration feedback loop — track retrieval quality and adjust block ranking.
- `capture.py` (~3698 tok, huge) — mind-mem Auto-Capture Engine with Structured Extraction. Zero external deps.
- `category_distiller.py` (~6264 tok, huge) — mind-mem Category Distiller — auto-generates thematic summary files from memory blocks.
- `causal_graph.py` (~3956 tok, huge) — mind-mem Temporal Causal Graph — directed dependency tracking with staleness.
- `chain_of_note.py` (~1435 tok, large) — Chain-of-note evidence packing (v3.4.0).
- `change_stream.py` (~1553 tok, huge) — # Copyright 2026 STARGA, Inc.
- `check_version.py` (~622 tok, large) — Version consistency checker for mind-mem.
- `coding_schemas.py` (~2127 tok, huge) — mind-mem Coding-Native Memory Schemas.
- `cognitive_forget.py` (~2667 tok, huge) — # Copyright 2026 STARGA, Inc.
- `compaction.py` (~3270 tok, huge) — mind-mem Compaction & GC Engine. Zero external deps.
- `compiled_truth.py` (~6414 tok, huge) — mind-mem Compiled Truth — synthesized entity pages with append-only evidence.
- `conflict_resolver.py` (~3119 tok, huge) — mind-mem Automated Conflict Resolution Pipeline. Zero external deps.
- `connection_manager.py` (~1059 tok, large) — SQLite connection manager with read/write separation and WAL mode.
- `consensus_vote.py` (~1871 tok, huge) — Quorum-based consensus voting on contradictions (v3.3.0).
- `context_core.py` (~4313 tok, huge) — # Copyright 2026 STARGA, Inc.
- `contradiction_detector.py` (~4888 tok, huge) — mind-mem Contradiction Detector — Surface conflicts at the governance gate.
- `core_export.py` (~1689 tok, huge) — # Copyright 2026 STARGA, Inc.
- `corpus_registry.py` (~471 tok, medium) — Central corpus path registry for mind-mem.
- `cron_runner.py` (~1846 tok, huge) — mind-mem Cron Runner — single entry point for all periodic jobs. Zero external deps.
- `cross_encoder_reranker.py` (~749 tok, large) — mind-mem Optional Cross-Encoder Reranker.
- `dedup.py` (~4593 tok, huge) — mind-mem 4-layer deduplication filter for search results.
- `dream_cycle.py` (~8852 tok, huge) — mind-mem Dream Cycle — autonomous memory enrichment. Zero external deps.
- `drift_detector.py` (~4365 tok, huge) — mind-mem Semantic Belief Drift Detection.
- `encryption.py` (~3115 tok, huge) — mind-mem Encryption at Rest — optional AES-256 encryption for memory blocks.
- `entity_ingest.py` (~3220 tok, huge) — mind-mem Entity Ingestion — regex-based entity extraction. Zero external deps.
- `entity_prefetch.py` (~2867 tok, huge) — Entity-graph prefetch for recall (v3.3.0 Tier 3 #8).
- `enums.py` (~471 tok, medium) — Centralised enum definitions for mind-mem.
- `error_codes.py` (~1751 tok, huge) — mind-mem Error Codes — structured error classification.
- `event_fanout.py` (~2126 tok, huge) — Governance event fan-out (v4.0 prep).
- `evidence_bundle.py` (~2205 tok, huge) — Structured evidence bundle for answerer co-design (v3.3.0 Tier 3 #7).
- `evidence_objects.py` (~5859 tok, huge) — # Copyright 2026 STARGA, Inc.
- `evidence_packer.py` (~3267 tok, huge) — Deterministic evidence packer for Mind-Mem.
- `extraction_feedback.py` (~1177 tok, large) — mind-mem Extraction Quality Feedback Tracker.
- `extractor.py` (~6597 tok, huge) — mind-mem Entity & Fact Extractor (Regex NER-lite). Zero external deps.
- `feature_gate.py` (~1377 tok, large) — Shared config-resolver for retrieval features (architect audit item #6).
- `field_audit.py` (~3103 tok, huge) — mind-mem Per-Field Mutation Audit — tracks individual field changes.
- `governance_bench.py` (~1855 tok, huge) — mind-mem Governance Benchmark Suite.
- `governance_gate.py` (~2212 tok, huge) — # Copyright 2026 STARGA, Inc.
- `governance_raft.py` (~2208 tok, huge) — Raft-style consensus wrapper for governance writes (v4.0 prep).
- `graph_recall.py` (~1907 tok, huge) — Multi-hop graph traversal for recall (v3.3.0 Tier 1 #2).
- `hash_chain_v2.py` (~5512 tok, huge) — # Copyright 2026 STARGA, Inc.
- `hook_installer.py` (~9442 tok, huge) — # Copyright 2026 STARGA, Inc.
- `hybrid_recall.py` (~8806 tok, huge) — mind-mem Hybrid Recall -- BM25 + Vector + RRF fusion.
- `ingestion_pipeline.py` (~1752 tok, huge) — # Copyright 2026 STARGA, Inc.
- `__init__.py` (~667 tok, large) — # Mind Mem — Memory + Immune System for AI agents
- `init_workspace.py` (~2062 tok, huge) — mind-mem workspace initializer. Zero external deps.
- `intel_scan.py` (~12579 tok, huge) — Mind Mem Intelligence Scanner v2.0 — Self-hosted, zero external dependencies.
- `intent_router.py` (~3106 tok, huge) — mind-mem Intent Router — 9-type adaptive query intent classification.
- `interaction_signals.py` (~4278 tok, huge) — # Copyright 2026 STARGA, Inc.
- `iterative_recall.py` (~2807 tok, huge) — Iterative chain-of-retrieval for multi-hop evidence (v3.4.0).
- `kalman_belief.py` (~4219 tok, huge) — # Copyright 2026 STARGA, Inc.
- `knowledge_graph.py` (~5308 tok, huge) — # Copyright 2026 STARGA, Inc.
- `ledger_anchor.py` (~1183 tok, large) — # Copyright 2026 STARGA, Inc.
- `llm_extractor.py` (~4538 tok, huge) — mind-mem LLM Entity & Fact Extractor (Optional, config-gated).
- `llm_noise_profile.py` (~2339 tok, huge) — # Copyright 2026 STARGA, Inc.
- `maintenance_migrate.py` (~1243 tok, large) — v3.2.0 §2.2 — one-shot migration helper for ``maintenance/`` subdivision.
- `mcp_entry.py` (~217 tok, medium) — Thin entry point for mind-mem-mcp console script."""
### `src/mind_mem/mcp/infra/`

- `acl.py` (~895 tok, large) — Per-tool ACL — scope enforcement for the MCP surface.
- `config.py` (~817 tok, large) — ``mind-mem.json`` config loading + configurable limits.
- `constants.py` (~98 tok, small) — MCP-surface-wide constants shared by the infra submodules.
- `http_auth.py` (~1005 tok, large) — HTTP bearer-token authentication helpers for the MCP surface.
- `__init__.py` (~449 tok, medium) — Cross-cutting infra helpers extracted from mcp_server.py (v3.2.0 §1.2 PR-1).
- `observability.py` (~1175 tok, large) — Observability + DB-busy helpers for the MCP surface.
- `rate_limit.py` (~928 tok, large) — Per-client sliding-window rate limiter for the MCP surface.
- `workspace.py` (~843 tok, large) — Workspace resolution + path-safety helpers.
### `src/mind_mem/mcp/`

- `__init__.py` (~215 tok, medium) — v3.2.0 §1.2 decomposition namespace — subpackage for MCP server modules.
- `resources.py` (~1342 tok, large) — MCP ``@mcp.resource`` declarations.
- `server.py` (~1668 tok, huge) — FastMCP instance + ``main()`` entry point for the Mind-Mem MCP server.
### `src/mind_mem/`

- `mcp_server.py` (~1780 tok, huge) — Mind-Mem MCP Server — public facade (v3.2.0 §1.2 PR-final shim).
### `src/mind_mem/mcp/tools/`

- `agent.py` (~1767 tok, huge) — Agent-bridge + vault MCP tools.
- `audit.py` (~2187 tok, huge) — Audit MCP tools — Merkle proofs, hash chain + evidence chain verification.
- `benchmark.py` (~1019 tok, large) — Benchmark + category-summary MCP tools.
- `calibration.py` (~1162 tok, large) — Calibration feedback MCP tools — ``calibration_feedback`` + ``calibration_stats``.
- `consolidation.py` (~2361 tok, huge) — Memory-consolidation MCP tools.
- `core.py` (~1508 tok, huge) — Context-core MCP tools — ``.mmcore`` bundle lifecycle.
- `encryption.py` (~1144 tok, large) — At-rest encryption MCP tools — ``encrypt_file`` / ``decrypt_file``.
- `governance.py` (~3409 tok, huge) — Governance MCP tools — propose / apply / rollback / scan / contradictions / memory_evolution.
- `graph.py` (~2204 tok, huge) — Knowledge-graph + causal-graph MCP tools.
- `_helpers.py` (~596 tok, large) — Shared tool-internal helpers — workspace paths + lazy-init singletons.
- `__init__.py` (~107 tok, small) — Per-domain ``@mcp.tool`` modules (v3.2.0 §1.2 PR-3+).
- `kernels.py` (~1905 tok, huge) — MIND kernel + compiled-truth MCP tools.
- `memory_ops.py` (~7082 tok, huge) — Memory operations MCP tools — index / lifecycle / health / export.
- `ontology.py` (~969 tok, large) — Ontology MCP tools — ``ontology_load`` + ``ontology_validate``.
- `public.py` (~4238 tok, huge) — # mypy: disable-error-code="no-any-return"
- `recall.py` (~4950 tok, huge) — Recall surface — the retrieval core of the MCP API.
- `signal.py` (~926 tok, large) — Interaction-signal MCP tools — ``observe_signal`` + ``signal_stats``.
### `src/mind_mem/`

- `memory_mesh.py` (~1903 tok, huge) — # Copyright 2026 STARGA, Inc.
- `memory_tiers.py` (~4934 tok, huge) — # Copyright 2026 STARGA, Inc.
- `merkle_tree.py` (~3354 tok, huge) — # Copyright 2026 STARGA, Inc.
- `mind_ffi.py` (~5094 tok, huge) — mind-mem FFI bridge — loads compiled MIND .so and exposes scoring functions.
- `mind_filelock.py` (~1844 tok, huge) — mind-mem file locking — cross-platform advisory locks. Zero external deps.
- `mind_kernels.py` (~1706 tok, huge) — # Copyright 2026 STARGA, Inc.
- `mm_cli.py` (~8350 tok, huge) — # Copyright 2026 STARGA, Inc.
- `mrs.py` (~1604 tok, huge) — # Copyright 2026 STARGA, Inc.
- `multi_modal.py` (~1659 tok, huge) — # Copyright 2026 STARGA, Inc.
- `namespaces.py` (~3560 tok, huge) — mind-mem Multi-Agent Namespace & ACL Engine. Zero external deps.
- `observability.py` (~1416 tok, large) — mind-mem Observability Module. Zero external deps.
- `observation_axis.py` (~3925 tok, huge) — # Copyright 2026 STARGA, Inc.
- `observation_compress.py` (~1353 tok, large) — Observation Compression Layer for Mind-Mem.
- `online_trainer.py` (~2751 tok, huge) — # Copyright 2026 STARGA, Inc.
- `ontology.py` (~2843 tok, huge) — # Copyright 2026 STARGA, Inc.
- `prefix_cache.py` (~3043 tok, huge) — # Copyright 2026 STARGA, Inc.
- `preimage.py` (~1102 tok, large) — # Copyright 2026 STARGA, Inc.
- `project_profile.py` (~1681 tok, huge) — # Copyright 2026 STARGA, Inc.
- `protection.py` (~1563 tok, huge) — Runtime protection layer for mind-mem (v3.3.0+).
- `py.typed` (~0 tok, tiny)
- `q1616.py` (~562 tok, large) — # Copyright 2026 STARGA, Inc.
- `query_expansion.py` (~4600 tok, huge) — Multi-query expansion for improved recall.
- `query_planner.py` (~2836 tok, huge) — Query decomposition for multi-hop questions (v3.3.0 Tier 1 #1).
- `recall_cache.py` (~2938 tok, huge) — v3.2.0 — distributed recall result cache (Redis + in-process LRU fallback).
- `_recall_constants.py` (~2420 tok, huge) — Recall engine constants — search fields, BM25 params, regex patterns, limits."""
- `_recall_context.py` (~2601 tok, huge) — Recall engine context packing — post-retrieval augmentation rules."""
- `_recall_core.py` (~14244 tok, huge) — Recall engine core — RecallBackend, main BM25 pipeline, backend loading, prefetch, CLI."""
- `_recall_detection.py` (~5162 tok, huge) — Recall engine detection — query type classification, text extraction, block utilities."""
- `_recall_expansion.py` (~3267 tok, huge) — Recall engine query expansion — domain synonyms, month normalization, RM3."""
- `recall.py` (~1049 tok, large) — mind-mem Recall Engine (BM25 + TF-IDF + Graph + Stemming). Zero external deps.
- `_recall_reranking.py` (~3247 tok, huge) — Recall engine reranking — deterministic feature-based re-scoring of BM25 hits."""
- `_recall_scoring.py` (~3715 tok, huge) — Recall engine scoring — BM25F helper, date scores, graph boosting, negation, date proximity, categories."""
- `_recall_temporal.py` (~2044 tok, huge) — Recall engine temporal filtering — resolve relative time references and filter blocks."""
- `_recall_tokenization.py` (~784 tok, large) — Recall engine tokenization — Porter stemmer and tokenizer."""
- `recall_vector.py` (~13960 tok, huge) — mind-mem Vector Recall Backend (Semantic Search with Embeddings).
- `rerank_ensemble.py` (~3310 tok, huge) — Reranker ensemble via Borda count (v3.3.0 Tier 4 #9).
- `retrieval_graph.py` (~4984 tok, huge) — Retrieval logger + co-retrieval graph for usage-based score propagation.
- `retrieval_trace.py` (~1252 tok, large) — Per-feature retrieval attribution (v3.3.0 architect audit item #7).
- `schema_version.py` (~1897 tok, huge) — Mind-Mem Schema Version Migration. Zero external deps.
- `session_boost.py` (~1533 tok, huge) — Session-boundary preservation for recall (v3.3.0 Tier 2 #5).
- `session_summarizer.py` (~2885 tok, huge) — mind-mem Session Summarizer. Zero external deps.
### `src/mind_mem/skill_opt/`

- `adapters.py` (~2237 tok, huge) — # Copyright 2026 STARGA, Inc.
- `analyzer.py` (~1145 tok, large) — # Copyright 2026 STARGA, Inc.
- `config.py` (~925 tok, large) — # Copyright 2026 STARGA, Inc.
- `fleet_bridge.py` (~1364 tok, large) — # Copyright 2026 STARGA, Inc.
- `history.py` (~1698 tok, huge) — # Copyright 2026 STARGA, Inc.
- `__init__.py` (~89 tok, small) — # Copyright 2026 STARGA, Inc.
- `mutator.py` (~907 tok, large) — # Copyright 2026 STARGA, Inc.
- `scorer.py` (~1406 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_runner.py` (~1375 tok, large) — # Copyright 2026 STARGA, Inc.
- `_types.py` (~1428 tok, large) — # Copyright 2026 STARGA, Inc.
- `validator.py` (~984 tok, large) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/`

- `smart_chunker.py` (~6814 tok, huge) — mind-mem Smart Chunker — Semantic-boundary document chunking.
- `smoke_test.sh` (~633 tok, large) — mind-mem Smoke Test — end-to-end verification
- `spec_binding.py` (~2883 tok, huge) — # Copyright 2026 STARGA, Inc.
- `speculative_prefetch.py` (~3195 tok, huge) — # Copyright 2026 STARGA, Inc.
- `sqlite_index.py` (~10959 tok, huge) — Mind Mem SQLite FTS5 Index — incremental lexical indexing. Zero external deps.
- `staleness.py` (~1179 tok, large) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/storage/`

- `__init__.py` (~980 tok, large) — Storage factory for mind-mem block stores (v3.2.0).
- `sharded_pg.py` (~2898 tok, huge) — Sharded Postgres / Citus routing (v4.0 prep).
### `src/mind_mem/`

- `streaming.py` (~1655 tok, huge) — Back-pressure-aware streaming ingest (v3.3.0).
- `_task_status_literals.sh` (~118 tok, small) — AUTO-GENERATED — do not edit by hand.
- `telemetry.py` (~2401 tok, huge) — mind-mem Telemetry — OpenTelemetry traces + Prometheus metrics.
- `temporal_metadata.py` (~1451 tok, large) — Temporal metadata injection for retrieved blocks (v3.4.0).
- `tenant_audit.py` (~1798 tok, huge) — Per-tenant audit chain isolation (v4.0 prep).
- `tenant_kms.py` (~2181 tok, huge) — Per-tenant key management + envelope encryption (v4.0 prep).
- `tiered_memory.py` (~1102 tok, large) — # Copyright 2026 STARGA, Inc.
- `tier_recall.py` (~2096 tok, huge) — Tier-aware recall score boosting (v3.2.0 hot/cold tier wire-up).
- `tracking.py` (~1918 tok, huge) — # Copyright 2026 STARGA, Inc.
- `trajectory.py` (~2233 tok, huge) — Trajectory Memory — task execution trace storage and recall.
- `transcript_capture.py` (~2333 tok, huge) — mind-mem Transcript JSONL Capture. Zero external deps.
- `truth_score.py` (~1678 tok, huge) — Probabilistic truth score for memory blocks (v3.3.0).
- `turbo_quant.py` (~1078 tok, large) — # Copyright 2026 STARGA, Inc.
- `uncertainty_propagation.py` (~1262 tok, large) — # Copyright 2026 STARGA, Inc.
- `union_recall.py` (~1310 tok, large) — Union-style retrieval for decomposed queries (v3.4.0).
- `validate_py.py` (~4830 tok, huge) — Mind Mem Integrity Validator — canonical engine.
- `validate.sh` (~352 tok, medium) — src/mind_mem/validate.sh — thin forwarder to the Python validator.
- `validate.sh.pre-forwarder` (~7140 tok, huge) — #!/usr/bin/env bash
- `verify_cli.py` (~3178 tok, huge) — # Copyright 2026 STARGA, Inc.
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
- `test_active_only_filter.py` (~312 tok, medium) — Tests for active_only recall filter."""
- `test_adversarial_corpus.py` (~2321 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_agent_bridge.py` (~2324 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_agent_id_filter.py` (~335 tok, medium) — Tests for agent_id namespace filtering."""
- `test_alerting.py` (~1777 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_allow_decompose.py` (~311 tok, medium) — Tests for _allow_decompose recall parameter."""
- `test_answer_quality.py` (~1288 tok, large) — Tests for the v3.3.0 answer-quality shims."""
- `test_api_keys.py` (~2133 tok, huge) — Tests for APIKeyStore in src/mind_mem/api/api_keys.py."""
- `test_apply_engine_backend_routing.py` (~1066 tok, large) — v3.2.0 §1.4 PR-6 — apply_engine routes through configured BlockStore."""
- `test_apply_engine_op_routing.py` (~1830 tok, huge) — v3.2.2 — execute_op routes block-level ops through BlockStore.
- `test_apply_engine.py` (~11568 tok, huge) — Tests for apply_engine.py — focus on security, validation, and rollback."""
- `test_atomicity_maintenance_scope.py` (~1283 tok, large) — v3.2.0 §2.2 — regression test for the ``maintenance/`` atomicity fix.
- `test_audit_chain.py` (~2398 tok, huge) — Tests for mind-mem hash-chain mutation log (audit_chain.py)."""
- `test_auto_resolver.py` (~1185 tok, large) — Tests for mind-mem auto contradiction resolution (auto_resolver.py)."""
- `test_axis_recall_mcp.py` (~1381 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_axis_recall.py` (~3683 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_backup_restore.py` (~3256 tok, huge) — Tests for backup_restore.py — zero external deps (stdlib unittest)."""
- `test_baseline_snapshot.py` (~2997 tok, huge) — Tests for baseline snapshot and drift detection (#431)."""
- `test_bigrams.py` (~168 tok, small) — Tests for bigram extraction."""
- `test_block_id_format.py` (~327 tok, medium) — Tests for block ID format validation."""
- `test_block_metadata.py` (~945 tok, large) — Tests for A-MEM block metadata evolution."""
- `test_block_parser_chunks.py` (~1658 tok, huge) — Tests for block_parser.py — overlapping chunk splitting + dedup."""
- `test_block_parser_edge.py` (~620 tok, large) — Extended block parser tests."""
- `test_block_parser_fields.py` (~372 tok, medium) — Tests for block parser field extraction."""
- `test_block_parser_multifile.py` (~337 tok, medium) — Tests for parsing multiple files."""
- `test_block_parser.py` (~3093 tok, huge) — Tests for block_parser.py — zero external deps (stdlib unittest)."""
- `test_block_store_encrypted.py` (~999 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_block_store_lock.py` (~928 tok, large) — v3.2.0 §1.4 PR-4 — MarkdownBlockStore.lock() tests."""
- `test_block_store.py` (~2202 tok, huge) — Tests for block_store.py — BlockStore protocol and MarkdownBlockStore."""
- `test_block_store_snapshot.py` (~746 tok, large) — v3.2.0 §1.4 PR-3 — MarkdownBlockStore.snapshot / restore / diff tests."""
- `test_block_store_write.py` (~2346 tok, huge) — v3.2.0 §1.4 PR-2 — MarkdownBlockStore.write_block + delete_block tests."""
- `test_block_types.py` (~437 tok, medium) — Tests for different block types in recall."""
- `test_bootstrap_corpus.py` (~1798 tok, huge) — Tests for bootstrap_corpus.py — backfill pipeline module."""
- `test_calibration.py` (~3269 tok, huge) — Tests for calibration feedback loop.
- `test_capture.py` (~2180 tok, huge) — Tests for capture.py — zero external deps (stdlib unittest)."""
- `test_category_distiller.py` (~2656 tok, huge) — Tests for category_distiller.py — CategoryDistiller class."""
- `test_causal_graph.py` (~1566 tok, huge) — Tests for mind-mem temporal causal graph (causal_graph.py)."""
- `test_check_version.py` (~271 tok, medium) — Tests for version consistency checker."""
- `test_chunk_text.py` (~231 tok, medium) — Tests for text chunking."""
- `test_coding_schemas.py` (~1284 tok, large) — Tests for mind-mem coding-native memory schemas."""
- `test_cognitive_forget.py` (~2315 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compaction.py` (~1869 tok, huge) — Tests for compaction.py — GC and archival engine."""
- `test_competitive_intel.py` (~1881 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compiled_truth.py` (~3929 tok, huge) — Tests for mind-mem compiled truth pages (compiled_truth.py)."""
- `test_concurrency_stress.py` (~4095 tok, huge) — Concurrency and performance stress tests for recall engine.
- `test_concurrent_integration.py` (~10786 tok, huge) — Integration tests for concurrent access and partial failure in mind-mem.
- `test_conflict_resolver.py` (~2340 tok, huge) — Tests for conflict_resolver.py — zero external deps (stdlib unittest)."""
- `test_connection_manager.py` (~2536 tok, huge) — Tests for ConnectionManager — SQLite connection pooling with read/write separation (#466)."""
- `test_consensus_vote.py` (~1137 tok, large) — v3.3.0 — quorum-based consensus voting on contradictions."""
- `test_constants.py` (~371 tok, medium) — Tests for recall constants module."""
- `test_context_core.py` (~3175 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_context_pack.py` (~2584 tok, huge) — Tests for context_pack rules: adjacency, diversity, pronoun rescue."""
- `test_context_pack_scripts.py` (~673 tok, large) — Tests for context packing via scripts._recall_context."""
- `test_contradiction_detector.py` (~5832 tok, huge) — Tests for contradiction_detector.py — Contradiction detection at governance gate (#432).
- `test_core_v140.py` (~2707 tok, huge) — Tests for v1.4.0 core hardening: issues #28, #30, #32, #34."""
- `test_cron_runner.py` (~2716 tok, huge) — Tests for cron_runner.py — periodic job orchestration, config loading, subprocess dispatch."""
- `test_cross_encoder_auto_enable.py` (~1489 tok, large) — v3.3.0 Tier 2 #6 — cross-encoder rerank auto-enables on ambiguous queries.
- `test_cross_encoder.py` (~1324 tok, large) — Tests for optional cross-encoder reranker."""
- `test_date_score.py` (~174 tok, small) — Tests for date scoring function."""
- `test_decompose_query.py` (~223 tok, medium) — Tests for query decomposition."""
- `test_dedup.py` (~5670 tok, huge) — Tests for dedup.py -- 4-layer deduplication filter."""
- `test_dedup_vector.py` (~1087 tok, large) — Tests for vector-enhanced cosine dedup (Layer 2b)."""
- `test_delete_memory.py` (~340 tok, medium) — Tests for memory deletion functionality."""
- `test_detection.py` (~326 tok, medium) — Tests for query detection module."""
- `test_downgrade_mitigation.py` (~1339 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_dream_cycle.py` (~4567 tok, huge) — Tests for dream_cycle.py — autonomous memory enrichment passes."""
- `test_drift_detector.py` (~1617 tok, huge) — Tests for mind-mem semantic belief drift detection (drift_detector.py)."""
- `test_edge_cases.py` (~3943 tok, huge) — Edge-case and stress tests for mind-mem — block_parser, recall, and MCP server."""
- `test_encryption.py` (~1732 tok, huge) — Tests for mind-mem encryption at rest."""
- `test_entity_ingest.py` (~4091 tok, huge) — Tests for the entity_ingest module — extraction, filtering, signal generation."""
- `test_entity_prefetch.py` (~1674 tok, huge) — v3.3.0 Tier 3 #8 — entity-graph prefetch.
- `test_enums.py` (~534 tok, large) — Tests for centralised enums (mind_mem.enums)."""
- `test_error_codes.py` (~2394 tok, huge) — Tests for mind-mem Error Codes module."""
- `test_error_paths.py` (~5892 tok, huge) — Error path and edge-case tests for mind-mem — malformed inputs, missing files, bad configs."""
- `test_event_fanout.py` (~1153 tok, large) — v4.0 prep — governance event fan-out."""
- `test_evidence_bundle.py` (~1562 tok, huge) — v3.3.0 Tier 3 #7 — structured evidence bundle.
- `test_evidence_objects.py` (~4031 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_evidence_packer.py` (~5180 tok, huge) — Tests for the evidence packer module."""
- `test_excerpt.py` (~248 tok, medium) — Tests for excerpt generation."""
- `test_expand_query.py` (~265 tok, medium) — Tests for query expansion module."""
- `test_export_memory.py` (~346 tok, medium) — Tests for memory export functionality."""
- `test_extractor.py` (~3387 tok, huge) — Tests for the regex NER-lite entity/fact extractor."""
- `test_fact_indexing.py` (~3101 tok, huge) — Tests for Feature 2 (fact card indexing) and Feature 4 (metadata-augmented embeddings)."""
- `test_feature_gate.py` (~1031 tok, large) — Tests for FeatureGate — the shared config-resolver for retrieval features."""
- `test_field_audit.py` (~1399 tok, large) — Tests for mind-mem per-field mutation audit (field_audit.py)."""
- `test_field_extraction.py` (~201 tok, medium) — Tests for field token extraction."""
- `test_filelock.py` (~979 tok, large) — Tests for filelock.py — cross-platform advisory locking."""
- `test_filelock_stress.py` (~1124 tok, large) — Stress tests for mind-mem file locking under contention."""
- `test_fts_fallback.py` (~4436 tok, huge) — Tests for FTS fallback behavior, recall envelope structure, block size cap,
- `test_governance_bench.py` (~811 tok, large) — Tests for mind-mem governance benchmark suite."""
- `test_governance_concurrency.py` (~1363 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_governance_raft.py` (~1398 tok, large) — v4.0 prep — Raft-style consensus wrapper for governance writes."""
- `test_graph_boost.py` (~6050 tok, huge) — Tests for graph boost, context packing, config validation, and block cap.
- `test_graph_boost_recall.py` (~315 tok, medium) — Tests for graph_boost recall parameter."""
- `test_graph_recall.py` (~1498 tok, large) — v3.3.0 Tier 1 #2 — multi-hop graph traversal on recall results.
- `test_grid_search.py` (~1199 tok, large) — Tests for benchmarks/grid_search.py — grid generation and utility functions."""
- `test_grpc_server.py` (~731 tok, large) — v4.0 prep — gRPC wire protocol (tests for the grpcio-free handlers)."""
- `test_hash_chain_v2.py` (~3462 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_hook_installer_force_preserves_siblings.py` (~703 tok, large) — Regression test for the --force clobber bug in hook_installer."""
- `test_hook_installer_registry.py` (~3841 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_hybrid_recall.py` (~2855 tok, huge) — Tests for hybrid_recall.py -- HybridBackend + RRF fusion."""
- `test_hybrid_search.py` (~599 tok, large) — Tests for hybrid search functionality."""
- `test_index_stats_b1.py` (~523 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_index_stats.py` (~316 tok, medium) — Tests for index statistics."""
- `test_init_workspace.py` (~2280 tok, huge) — Tests for init_workspace — config validation and workspace scaffolding."""
- `test_install_script.py` (~376 tok, medium)
- `test_integration.py` (~1381 tok, large) — Integration test: full mind-mem lifecycle init → capture → scan → recall."""
- `test_intel_scan.py` (~5905 tok, huge) — Tests for intel_scan.py — contradiction detection, drift analysis, impact graph."""
- `test_intent_classify.py` (~328 tok, medium) — Tests for intent classification."""
- `test_intent_router_adaptive.py` (~3618 tok, huge) — Tests for adaptive intent routing (#470).
- `test_intent_router.py` (~1176 tok, large) — Tests for 9-type intent router."""
- `test_interaction_signals.py` (~3177 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_kalman_belief.py` (~3728 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_knowledge_graph.py` (~3437 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_llm_extractor.py` (~1820 tok, huge) — Tests for the optional LLM entity/fact extractor module."""
- `test_llm_noise_profile.py` (~2354 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_maintenance_migrate.py` (~710 tok, large) — v3.2.0 §2.2 — tests for maintenance/ subdivision migration."""
- `test_mcp_integration.py` (~5177 tok, huge) — MCP transport and auth integration tests (#474).
- `test_mcp_server.py` (~4897 tok, huge) — Tests for mcp_server.py — tests the MCP server resources and tool logic.
- `test_mcp_tools.py` (~277 tok, medium) — Tests for MCP server tool definitions."""
- `test_mcp_tool_surface_v3_2.py` (~1733 tok, huge) — v3.2.0 — consolidated MCP public dispatcher tests."""
- `test_mcp_v140.py` (~5456 tok, huge) — Tests for MCP v1.4.0 features — issues #29, #31, #35, #36.
- `test_memory_evolution.py` (~340 tok, medium) — Tests for memory evolution tracking."""
- `test_memory_practical_e2e.py` (~2389 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_memory_tiers.py` (~3479 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_merkle_tree.py` (~3185 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mind_ffi.py` (~291 tok, medium) — Tests for MIND FFI module."""
- `test_mind_kernels_v3_3.py` (~1002 tok, large) — Kernel-loading tests for v3.3.0 features.
- `test_mm_cli_debug.py` (~3339 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_multi_file_recall.py` (~329 tok, medium) — Tests for recall across multiple files."""
- `test_namespaces.py` (~2411 tok, huge) — Tests for namespaces.py — zero external deps (stdlib unittest)."""
- `test_niah.py` (~4987 tok, huge) — Needle In A Haystack (NIAH) benchmark for mind-mem recall.
- `test_observability.py` (~791 tok, large) — Tests for observability.py — structured logging and metrics."""
- `test_observation_axis.py` (~3330 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_observation_compress.py` (~2754 tok, huge) — Tests for observation_compress module.
- `test_oidc_admin_enforcement.py` (~1716 tok, huge) — v3.2.1 — OIDC JWTs must pass through ``_require_admin`` checks.
- `test_oidc_auth.py` (~2754 tok, huge) — Tests for OIDCProvider / OIDCConfig in src/mind_mem/api/auth.py."""
- `test_ontology.py` (~2306 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_postgres_block_store.py` (~3136 tok, huge) — v3.2.0 §1.4 PR-5 — PostgresBlockStore integration tests.
- `test_postgres_replica_routing.py` (~1777 tok, huge) — v3.2.0 — tests for read-replica routing in ReplicatedPostgresBlockStore."""
- `test_prefetch_context.py` (~1487 tok, large) — Tests for prefetch_context() in recall.py."""
- `test_prefetch.py` (~326 tok, medium) — Tests for prefetch functionality."""
- `test_prefix_cache.py` (~3140 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_protection.py` (~1419 tok, large) — Tests for mind_mem.protection (v3.3.0+)."""
- `test_q1616_preimage.py` (~1496 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_query_decomposition.py` (~1604 tok, huge) — Tests for multi-hop query decomposition (#6)."""
- `test_query_expansion_auto_enable.py` (~1091 tok, large) — v3.3.0 Tier 2 #4 — query expansion auto-enables on ambiguous queries.
- `test_query_expansion_multi_provider.py` (~1237 tok, large) — Tests for multi-provider LLM query expansion (OpenAI-compatible endpoints)."""
- `test_query_expansion.py` (~3809 tok, huge) — Tests for query_expansion.py -- multi-query expansion for improved recall."""
- `test_query_planner.py` (~1348 tok, large) — v3.3.0 Tier 1 #1 — query decomposition for multi-hop questions.
- `test_recall_cache.py` (~1916 tok, huge) — Tests for v3.2.0 distributed recall cache (LRU + Redis)."""
- `test_recall_concurrent.py` (~344 tok, medium) — Tests for concurrent recall queries."""
- `test_recall_context_field.py` (~263 tok, medium) — Tests for context field in blocks."""
- `test_recall_cross_encoder.py` (~1345 tok, large) — Tests for cross-encoder reranker integration in recall pipeline."""
- `test_recall_date_field.py` (~315 tok, medium) — Tests for date field in recall results."""
- `test_recall_detection.py` (~1523 tok, huge) — Tests for _recall_detection.py — query type classification and text extraction."""
- `test_recall_edge_cases.py` (~570 tok, large) — Edge case tests for recall engine."""
- `test_recall_empty_query_types.py` (~322 tok, medium) — Tests for various empty/minimal query types."""
- `test_recall_empty_workspace.py` (~134 tok, small) — Tests for recall on empty workspaces."""
- `test_recall_intent_router.py` (~1207 tok, large) — Tests for IntentRouter integration in recall pipeline."""
- `test_recall_large_workspace.py` (~343 tok, medium) — Tests for recall with large workspaces."""
- `test_recall_limit.py` (~395 tok, medium) — Tests for recall limit parameter behavior."""
- `test_recall_metadata.py` (~1340 tok, large) — Tests for A-MEM block metadata integration in recall pipeline."""
- `test_recall_priority.py` (~410 tok, medium) — Tests for priority boost in recall."""
- `test_recall.py` (~3880 tok, huge) — Tests for recall.py — zero external deps (stdlib unittest)."""
- `test_recall_references.py` (~270 tok, medium) — Tests for reference-based recall."""
- `test_recall_reranking.py` (~2740 tok, huge) — Tests for _recall_reranking.py — deterministic reranker + LLM rerank."""
- `test_recall_scoring_order.py` (~440 tok, medium) — Tests for recall result scoring order."""
- `test_recall_source_field.py` (~279 tok, medium) — Tests for source field in recall results."""
- `test_recall_speaker.py` (~263 tok, medium) — Tests for speaker-based recall."""
- `test_recall_status_boost.py` (~394 tok, medium) — Tests for status boost in recall."""
- `test_recall_supersedes.py` (~216 tok, medium) — Tests for supersedes field in recall."""
- `test_recall_tags.py` (~320 tok, medium) — Tests for tag-based recall."""
- `test_recall_temporal.py` (~2800 tok, huge) — Tests for _recall_temporal.py — time-aware hard filters for temporal queries."""
- `test_recall_vector.py` (~4901 tok, huge) — Tests for recall_vector.py — VectorBackend semantic search."""
- `test_rerank_debug.py` (~342 tok, medium) — Tests for rerank debug mode."""
- `test_rerank_ensemble.py` (~1531 tok, huge) — v3.3.0 Tier 4 #9 — reranker ensemble via Borda count.
- `test_reranking.py` (~246 tok, medium) — Tests for reranking module."""
- `test_rest_api_oidc.py` (~2686 tok, huge) — Tests for OIDC callback + admin API key endpoints (v3.2.0)."""
- `test_rest_api.py` (~3505 tok, huge) — Tests for the mind-mem REST API layer (v3.2.0).
- `test_retrieval_diagnostics.py` (~2419 tok, huge) — Tests for retrieval diagnostics (#428), corpus isolation (#429), and intent instrumentation (#430)."""
- `test_retrieval_graph.py` (~2242 tok, huge) — Tests for retrieval_graph.py — retrieval logging, co-retrieval graph, hard negatives."""
- `test_retrieval_trace.py` (~904 tok, large) — Tests for v3.3.0 per-feature retrieval attribution."""
- `test_rm3_expand.py` (~321 tok, medium) — Tests for RM3 query expansion."""
- `test_scan_engine.py` (~333 tok, medium) — Tests for integrity scan engine."""
- `test_schema_version.py` (~1758 tok, huge) — Tests for schema_version.py — zero external deps (stdlib unittest)."""
- `test_scoring.py` (~337 tok, medium) — Tests for BM25 scoring functions."""
- `test_session_boost.py` (~1488 tok, large) — v3.3.0 Tier 2 #5 — session-boundary preservation via recall-side boost.
- `test_session_summarizer.py` (~3973 tok, huge) — Comprehensive tests for mind_mem/session_summarizer.py.
- `test_sharded_pg.py` (~1222 tok, large) — v4.0 prep — sharded Postgres routing tests (mock underlying stores)."""
- `test_skeptical_query.py` (~194 tok, small) — Tests for skeptical query detection."""
- `test_skill_opt.py` (~3356 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_smart_chunker_code.py` (~1135 tok, large) — Tests for code-aware chunking in smart_chunker.py."""
- `test_smart_chunker.py` (~7744 tok, huge) — Tests for smart_chunker.py — semantic-boundary document chunking."""
- `test_spec_binding.py` (~3156 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_speculative_prefetch.py` (~3071 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_sqlite_index.py` (~4726 tok, huge) — Tests for sqlite_index.py — SQLite FTS5 index for mind-mem recall."""
- `test_stopwords.py` (~247 tok, medium) — Tests for stopword handling."""
- `test_storage_factory.py` (~1066 tok, large) — Tests for mind_mem.storage.get_block_store factory (v3.2.0)."""
- `test_streaming.py` (~1446 tok, large) — v3.3.0 — back-pressure-aware streaming ingest queue."""
- `test_telemetry.py` (~2829 tok, huge) — Tests for src/mind_mem/telemetry.py.
- `test_temporal_decay_scoring.py` (~863 tok, large) — v3.3.0 Tier 1 #3 — half-life decay on block ``Created``/``Date`` field.
- `test_temporal.py` (~223 tok, medium) — Tests for temporal filtering module."""
- `test_tenant_audit.py` (~1462 tok, large) — v4.0 prep — per-tenant audit chain façade."""
- `test_tenant_kms.py` (~955 tok, large) — v4.0 prep — per-tenant KMS envelope encryption."""
- `test_tier_decay.py` (~924 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_tier_recall.py` (~1418 tok, large) — Tests for tier-aware recall boosting (v3.2.0 hot/cold tier wire-up)."""
- `test_tier_weights_config.py` (~784 tok, large) — v3.3.0 Tier 4 #10 — per-tier learned weights override.
- `test_tokenization.py` (~436 tok, medium) — Tests for tokenization module."""
- `test_train_mind_mem_4b.py` (~962 tok, large) — Smoke tests for benchmarks/train_mind_mem_4b.py.
- `test_trajectory.py` (~2392 tok, huge) — Tests for trajectory.py — trajectory memory block operations."""
- `test_transcript_capture.py` (~3235 tok, huge) — Tests for transcript_capture.py — zero external deps (stdlib unittest)."""
- `test_truth_score.py` (~1520 tok, huge) — v3.3.0 — probabilistic truth score.
- `test_uncertainty_propagation.py` (~2158 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_unicode_edge_cases.py` (~2440 tok, huge) — Tests for Unicode and edge case handling across mind-mem modules."""
- `test_v28_completion.py` (~4565 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v320_gaps.py` (~3257 tok, huge) — v3.2.0 gap tests — regression and edge-case coverage for new modules.
- `test_v34_features.py` (~2861 tok, huge) — Tests for v3.4.0 retrieval features.
- `test_validate_py.py` (~3438 tok, huge) — Tests for validate_py.py — workspace integrity validator."""
- `test_validate_sh_deprecation.py` (~547 tok, large) — Pin the runtime deprecation warning on validate.sh.
- `test_vault_wikilinks.py` (~1783 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_verify_cli.py` (~3202 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_watcher.py` (~1217 tok, large) — Tests for watcher.py — file change detection for auto-reindex."""
- `test_wide_retrieval.py` (~346 tok, medium) — Tests for wide retrieval parameter."""
- `test_workspace_contextvar.py` (~1047 tok, large) — v3.2.1 — regression test for per-request workspace ContextVar scoping.
- `test_workspace_init.py` (~498 tok, medium) — Tests for workspace initialization."""
- `test_workspace_structure.py` (~546 tok, large) — Tests for workspace directory structure."""
### `train/`

- `backport_sweep.py` (~1658 tok, huge) — Backport v2.9.0 audit fixes to every prior v2.x release as .post1.
- `build_corpus.py` (~8110 tok, huge) — Harvest a training corpus for the mind-mem-4b model.
- `build_model_card.py` (~1876 tok, huge) — Generate the HuggingFace model-card README for mind-mem-4b v3.0.0."""
- `eval_harness.py` (~2137 tok, huge) — Eval harness for mind-mem-4b.
- `export_gguf.py` (~1047 tok, large) — Merge the LoRA adapter into the base weights, then export to GGUF.
- `README.md` (~577 tok, large) — mind-mem-4b training pipeline
- `runpod_deploy.py` (~3095 tok, huge) — End-to-end RunPod driver for full-FT on Qwen3.5-4B.
- `runpod_full_ft.py` (~1246 tok, large) — Full fine-tune of Qwen3.5-4B on RunPod (A100/H100) for mind-mem-4b.
- `train_qlora.py` (~1195 tok, large) — QLoRA fine-tune for mind-mem-4b on the harvested corpus.
- `upload_to_hf.py` (~865 tok, large) — Push the retrained adapter + model card to star-ga/mind-mem-4b.
### `web/app/console/`

- `page.tsx` (~1169 tok, large) — Tolerate missing endpoint — show single-tenant UI.
### `web/app/`

- `layout.tsx` (~168 tok, small)
- `page.tsx` (~1036 tok, large)
### `web/components/`

- `FactList.tsx` (~281 tok, medium)
- `GraphView.tsx` (~1063 tok, large)
- `TenantSwitcher.tsx` (~839 tok, large) — HeadersInit can be a Headers, a [string, string][], or a Record.
- `TimelineView.tsx` (~299 tok, medium)
### `web/`

- `.gitignore` (~17 tok, tiny) — node_modules/
### `web/lib/`

- `api.ts` (~669 tok, large)
### `web/`

- `next.config.ts` (~104 tok, small) — mind-mem-web is a thin client — the REST API lives on the
- `package.json` (~193 tok, small) — Keys: name, version, private, description, license
- `README.md` (~464 tok, medium) — mind-mem web console
- `tsconfig.json` (~149 tok, small) — Keys: compilerOptions, include, exclude

---
*Generated by `anatomy 1.0.0`. Edit descriptions manually — re-run preserves structure.*

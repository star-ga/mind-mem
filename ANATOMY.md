# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind-mem`
**Files:** 883 | **Est. tokens:** ~1,911,023
**Generated:** 2026-06-22 21:07 UTC

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
| `./` | 35 | ~68,562 |
| `.agents/skills/mind-mem-development/` | 1 | ~456 |
| `.arch-mind/` | 4 | ~1,313 |
| `audits/` | 5 | ~24,039 |
| `benchmarks/` | 40 | ~103,186 |
| `deploy/` | 2 | ~772 |
| `deploy/docker/` | 1 | ~495 |
| `deploy/edge/` | 2 | ~1,149 |
| `deploy/grafana/` | 1 | ~1,145 |
| `docs/` | 80 | ~144,941 |
| `docs/adr/` | 2 | ~521 |
| `docs/design/` | 3 | ~4,055 |
| `docs/security-baselines/` | 1 | ~18,974 |
| `examples/` | 3 | ~1,201 |
| `.gemini/` | 1 | ~28 |
| `.githooks/` | 1 | ~98 |
| `.github/` | 8 | ~4,180 |
| `.github/ISSUE_TEMPLATE/` | 2 | ~179 |
| `.github/workflows/` | 12 | ~9,936 |
| `hooks/` | 3 | ~801 |
| `hooks/openclaw/mind-mem/` | 2 | ~1,211 |
| `intelligence/` | 1 | ~113 |
| `intelligence/state/snapshots/` | 1 | ~114 |
| `lib/` | 1 | ~2,170 |
| `mind/` | 27 | ~9,687 |
| `.roo/` | 1 | ~22 |
| `scripts/` | 6 | ~4,675 |
| `sdk/go/` | 9 | ~6,773 |
| `sdk/js/` | 5 | ~1,388 |
| `sdk/js/src/` | 4 | ~2,320 |
| `sdk/js/test/` | 1 | ~2,191 |
| `security/` | 3 | ~11,009 |
| `skills/apply-proposal/` | 1 | ~345 |
| `skills/integrity-scan/` | 1 | ~376 |
| `skills/memory-recall/` | 1 | ~549 |
| `src/` | 1 | ~280 |
| `src/mind_mem/` | 161 | ~592,867 |
| `src/mind_mem/api/` | 5 | ~16,595 |
| `src/mind_mem/mcp/` | 3 | ~4,128 |
| `src/mind_mem/mcp/infra/` | 8 | ~9,991 |
| `src/mind_mem/mcp/tools/` | 24 | ~58,292 |
| `src/mind_mem/skill_opt/` | 11 | ~13,702 |
| `src/mind_mem/storage/` | 2 | ~5,519 |
| `src/mind_mem/v4/` | 22 | ~58,472 |
| `templates/` | 19 | ~1,041 |
| `tests/` | 306 | ~656,279 |
| `tests/integration/` | 2 | ~1,575 |
| `tests/red_team/` | 3 | ~806 |
| `tests/red_team/transcripts/` | 1 | ~0 |
| `train/` | 31 | ~56,055 |
| `web/` | 5 | ~927 |
| `web/app/` | 2 | ~1,204 |
| `web/app/console/` | 1 | ~1,169 |
| `web/components/` | 4 | ~2,482 |
| `web/lib/` | 1 | ~665 |

## Files

### `./`

- `AGENTS.md` (~995 tok, large) — mind-mem: agent instructions (auto-written)
- `AUDIT_FINDINGS_FOR_CLAUDE.md` (~995 tok, large) — Comprehensive Architectural Audit: MIND-Mem (Commit 30d8b71)
- `CLAUDE.md` (~4084 tok, huge) — MIND-Mem — Persistent AI Memory System
- `conftest.py` (~1010 tok, large) — Shared pytest fixtures for mind-mem test suite."""
- `CONTRIBUTING.md` (~751 tok, large) — Contributing to MIND-Mem
- `.cursorrules` (~23 tok, tiny) — # mind-mem
- `demo-setup.sh` (~323 tok, medium) — Pre-seed a demo workspace for VHS recording
- `demo.tape` (~93 tok, small) — # mind-mem demo — terminal recording for README
- `Dockerfile` (~306 tok, medium) — FROM python:3.12-slim
- `.dockerignore` (~37 tok, tiny) — .git
- `.editorconfig` (~107 tok, small) — # EditorConfig — https://editorconfig.org
- `EVIDENCE.md` (~1206 tok, large) — Evidence Matrix
- `generate_mind7b_training.py` (~5558 tok, huge) — Generate training data for Mind7B — a purpose-trained 7B model for mind-mem.
- `.gitattributes` (~96 tok, small) — # Auto-detect text files and normalize line endings
- `.gitignore` (~237 tok, medium) — *.pyc
- `.gitleaks.toml` (~314 tok, medium) — title = "mind-mem gitleaks config"
- `install-bootstrap.sh` (~1756 tok, huge) — mind-mem one-command bootstrap installer
- `install.sh` (~4935 tok, huge) — mind-mem installer — installs the package + wires MCP config for AI clients
- `LICENSE` (~2695 tok, huge)
- `Makefile` (~671 tok, large) — .PHONY: test lint bench install dev clean smoke help regen-bash-literals
- `mcp_server.py` (~683 tok, large) — Source-checkout entrypoint for the packaged Mind-Mem MCP server.
- `mind-mem.example.json` (~174 tok, small) — Keys: recall, prompts, categories, extraction, limits
- `.pre-commit-config.yaml` (~366 tok, medium) — repos:
- `pyproject.toml` (~2257 tok, huge) — [project]
- `.python-version` (~2 tok, tiny) — 3.12
- `README.md` (~25018 tok, huge) — 30-Second Demo
- `requirements-optional.txt` (~768 tok, large) — # mind-mem optional ML stack — pinned with SHA256 integrity hashes for
- `.run-ledger.jsonl` (~154 tok, small) — {"ended_at": "2026-05-11T03:10:20+00:00", "eval_summary": "127/131 (109 main + 1
- `SECURITY_AUDIT_2026-04.md` (~2403 tok, huge) — Security Audit — MIND-Mem v3.1.9 (April 2026)
- `SECURITY.md` (~1752 tok, huge) — Security Policy
- `setup.py` (~397 tok, medium) — Conditional setup hook for the optional Cython accelerator.
- `SPEC.md` (~5816 tok, huge) — Mind Mem Formal Specification v1.5.1
- `train_mind7b_runpod.py` (~1654 tok, huge)
- `uninstall.sh` (~908 tok, large) — mind-mem uninstaller — removes MCP server entries from all configured clients
- `.windsurfrules` (~18 tok, tiny) — # mind-mem
### `.agents/skills/mind-mem-development/`

- `SKILL.md` (~456 tok, medium) — MIND-Mem Development
### `.arch-mind/`

- `last_summary.json` (~154 tok, small) — Keys: _aggregated_for_phase_a, _comment, _languages, _repo_root, edges
- `rules.mind` (~921 tok, large) — mind-mem architectural-governance rules
- `scan.json` (~152 tok, small) — Keys: _aggregated_for_phase_a, _comment, _languages, _repo_root, edges
- `scan_v3813.json` (~86 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
### `audits/`

- `2026-05-18-copilot-audit.md` (~1332 tok, large) — Copilot CLI audit — mind-mem (2026-05-18)
- `v3.11.0-integration-consensus-2026-05-08.json` (~4878 tok, huge) — Keys: audit_id, generated_at_utc, models_queried, models_parsed, fleet
- `v3.11-v3.12-corpus-final-audit-2026-05-09.json` (~6742 tok, huge) — Keys: audit_id, commit, audited_at, models_run, models_succeeded
- `v3.12-corpus-final-audit-2026-05-09.json` (~3802 tok, huge) — Keys: audit_id, generated_at_utc, models_queried, models_parsed, fleet
- `v4.0.1-claude-2026-05-12.md` (~7285 tok, huge) — mind-mem v4.0.1 — Multi-Source Audit (cross-model review + arch-mind + agents)
### `benchmarks/`

- `bench_kernels.py` (~4027 tok, huge) — Benchmark: MIND kernels vs pure Python scoring.
- `cache_effectiveness.py` (~2717 tok, huge) — Cache-effectiveness benchmark — Redis L2 vs LRU-only vs no-cache.
- `cache_effectiveness_v3.2.1.json` (~227 tok, medium) — Keys: n_blocks, n_queries, pool_size, repeat_pct, runs
- `CACHE.md` (~1028 tok, large) — Recall Cache Effectiveness Benchmark
- `_capfix_probe.py` (~1023 tok, large) — Isolation probe: quantify the recall.dedup type_cap=3 ceiling.
- `_ch_minilm.py` (~2032 tok, huge) — LongMemEval-S — per-turn chunking + hybrid (BM25F+mxbai RRF) + 4b expansion.
- `_ch_minilm_spawn.py` (~6417 tok, huge) — LongMemEval-S — mind-mem FULL POTENTIAL harness (same-equipment, best-vs-best).
- `compare_runs.py` (~857 tok, large) — Compare two LoCoMo benchmark runs side-by-side.
- `crossencoder_ab.py` (~3205 tok, huge) — Cross-Encoder A/B Test — retrieval-level comparison.
- `generate_dispatcher_examples.py` (~2346 tok, huge) — Generate synthetic training examples for the v3.2.x 7-dispatcher MCP surface.
- `generate_retrieval_examples.py` (~1686 tok, huge) — Generate training examples for v3.3.0 retrieval shapes.
- `grid_search.py` (~2849 tok, huge) — BM25F Field Weight Grid Search for mind-mem Recall Engine.
- `__init__.py` (~0 tok, tiny)
- `local_stack_audit.py` (~1871 tok, huge) — Single-shot audit of the local mind-mem stack before a bench run.
- `locomo_harness.py` (~4147 tok, huge) — LoCoMo Benchmark Harness for mind-mem Recall Engine.
- `locomo_judge.py` (~17178 tok, huge) — LoCoMo LLM-as-Judge Evaluation for Mind-Mem.
- `locomo_v3.3.0_benchmark_config.json` (~450 tok, medium) — Keys: _comment, version, recall, cache, cross_encoder
- `longmemeval_apples.py` (~1660 tok, huge) — LongMemEval-S apples-to-apples harness (mind-mem hybrid BM25+vector).
- `longmemeval_chunk_bm25_4b.py` (~1605 tok, huge) — Per-turn chunking + BM25 + mind-mem:4b multi-query expansion.
- `longmemeval_chunked_hybrid.py` (~2055 tok, huge) — LongMemEval-S — per-turn chunking + hybrid (BM25F+mxbai RRF) + 4b expansion.
- `longmemeval_chunked_minilm.py` (~6570 tok, huge) — LongMemEval-S — mind-mem FULL POTENTIAL harness (same-equipment, best-vs-best).
- `longmemeval_chunked.py` (~1222 tok, large) — LongMemEval-S — per-turn passage chunking (architectural gap closure).
- `longmemeval_clean.py` (~1397 tok, large) — LongMemEval-S — clean reproducible mind-mem retrieval benchmark.
- `LONGMEMEVAL_FINDINGS_2026-05-19.md` (~2123 tok, huge) — LongMemEval-S Benchmark Audit — Findings (2026-05-19)
- `longmemeval_fullpotential.py` (~6296 tok, huge) — LongMemEval-S — mind-mem FULL POTENTIAL harness (same-equipment, best-vs-best).
- `longmemeval_harness.py` (~2973 tok, huge) — LongMemEval Benchmark Harness for mind-mem recall engine.
- `longmemeval_hybrid4b.py` (~1809 tok, huge) — LongMemEval-S — mind-mem hybrid (BM25F+mxbai RRF) + 4b query expansion.
- `longmemeval_real_harness.py` (~1642 tok, huge) — LongMemEval-S harness — real product pipeline (Phase A: own best honest number).
- `niah_full_results.txt` (~5134 tok, huge) — ============================= test session starts ==============================
- `NIAH.md` (~1620 tok, huge) — Needle In A Haystack (NIAH) Benchmark
- `niah_v3.2.1_redis_results.txt` (~111 tok, small) — ============================= test session starts ==============================
- `niah_v3.2.1_results.txt` (~203 tok, medium) — ============================= test session starts ==============================
- `README_benchmark_mode.md` (~1106 tok, large) — Full-capability benchmark mode (v3.3.0)
- `REPORT.md` (~3980 tok, huge) — MIND-Mem Benchmark Report
- `repro_niah.py` (~2385 tok, huge) — Reproducible NIAH benchmark harness — emits independently-verifiable evidence.
- `runpod_kickoff.sh` (~1779 tok, huge) — mind-mem-4b v2 — Runpod one-shot kickoff.
- `tier_weight_search.py` (~1615 tok, huge) — Grid-search per-tier weights against LoCoMo judge scores (v3.3.0 T4 #10).
- `train_config_a100.yaml` (~347 tok, medium) — base_model: star-ga/mind-mem-4b
- `train_config.yaml` (~208 tok, medium) — base_model: star-ga/mind-mem-4b
- `train_mind_mem_4b.py` (~3286 tok, huge) — mind-mem-4b v2 training script — Runpod H200 full-fine-tune.
### `deploy/`

- `docker-compose.yml` (~690 tok, large) — name: mind-mem
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

- `agent-memory-protocol.md` (~597 tok, large) — Agent Memory Protocol — canonical system-prompt snippet
- `api-reference.md` (~1683 tok, huge) — API Reference
- `append-only-audit-logs.md` (~1424 tok, large) — Append-Only Audit Logs — Operator Runbook
- `architecture.md` (~1936 tok, huge) — Architecture
- `audit_response.md` (~950 tok, large) — MIND-Mem — response to the 2026-05-02 ecosystem audit
- `benchmarks.md` (~757 tok, large) — Benchmarks
- `block-format.md` (~431 tok, medium) — Block Format
- `block-type-taxonomy-roadmap.md` (~911 tok, large) — Block-Type Taxonomy Enhancement — Roadmap Note
- `changelog-format.md` (~217 tok, medium) — Changelog Format Guide
- `ci-workflows.md` (~254 tok, medium) — CI Workflows
- `claude-desktop-setup.md` (~752 tok, large) — Claude Desktop Setup Guide
- `client-integrations.md` (~2684 tok, huge) — Client Integrations
- `cli-reference.md` (~1857 tok, huge) — CLI Reference
- `companion-tools.md` (~1113 tok, large) — Companion Tools
- `comparison.md` (~313 tok, medium) — Comparison with Alternatives
- `competitive-analysis-persistent-memory-2026.md` (~4089 tok, huge) — Comprehensive Competitive Analysis: Persistent Memory Systems for AI Coding Agents (2025–2026)
- `configuration.md` (~8174 tok, huge) — Configuration Reference
### `docs/design/`

- `recall-harness.md` (~1639 tok, huge) — Design: the recall harness — deterministic working memory for search agents
- `v3-mcp-surface-reduction.md` (~1080 tok, large) — v3.0 Design: MCP Tool Surface Reduction
- `v3-multi-tenancy.md` (~1336 tok, large) — v3.0 Design: Multi-Tenancy Foundation
### `docs/`

- `development.md` (~358 tok, medium) — Development Guide
- `docker-deployment.md` (~571 tok, large) — Docker Deployment
- `faq.md` (~374 tok, medium) — FAQ
- `federation-setup.md` (~930 tok, large) — mind-mem federation & multi-machine setup
- `getting-started.md` (~493 tok, medium) — Getting Started
- `glossary.md` (~263 tok, medium) — Glossary
- `governance.md` (~1353 tok, large) — MIND-Mem — governance design (5 layers)
- `hf-mind-mem-4b-v2-README.md` (~2152 tok, huge) — mind-mem-4b v2 (2026-04-21)
- `HYPEREDGE_DESIGN_2026-06-17.md` (~1350 tok, large) — Hyperedge + temporal-anchor design (Hyper-Extract steal)
- `install-guide.md` (~2855 tok, huge) — Installation guide — every step + every option
- `integrations.md` (~1540 tok, huge) — Integrations
- `locomo-v3.4-conv0-results.md` (~475 tok, medium) — LoCoMo v3.4.0 conv-0 results (2026-04-22)
- `maintenance-namespaces.md` (~1601 tok, huge) — `maintenance/` namespaces
- `mcp-integration.md` (~1691 tok, huge) — MCP Integration Guide
- `mcp-tool-examples.md` (~902 tok, large) — MCP Tool Examples
- `mic-map.md` (~1686 tok, huge) — MIC/MAP — MIND IR Graph Serialization
- `migration-guide.md` (~421 tok, medium) — Migration Guide
- `migration.md` (~2754 tok, huge) — Migration Guide: mem-os to MIND-Mem
- `MIND_CONFIG_VS_MIND_LANG.md` (~2275 tok, huge) — MIND configuration vs MIND language — clarifying the .mind extension
- `mind-kernels.md` (~339 tok, medium) — MIND Kernels
- `mind-mem-4b-setup.md` (~2615 tok, huge) — Setting up the mind-mem-4b model
- `mind-mem-4b-training-runbook.md` (~3586 tok, huge) — mind-mem-4b training runbook (post-v3.10.2 lessons)
- `mind-mem-4b-v2-training-recipe.md` (~1683 tok, huge) — mind-mem-4b v2 training recipe — Runpod H200
- `odc-retrieval.md` (~834 tok, large) — Observer-Dependent Cognition in MIND-Mem
- `performance-tuning.md` (~560 tok, large) — Performance Tuning
- `postgres-parity-audit-2026-06-14.md` (~5468 tok, huge) — mind-mem — Postgres/SQLite backend parity audit (2026-06-14)
- `protection.md` (~1443 tok, large) — MIND-Mem Library Protection
- `quality-gate.md` (~1267 tok, large) — Quality Gate — Operator Runbook
- `quickstart.md` (~601 tok, large) — MIND-Mem Quickstart
- `red-team-audit.md` (~1164 tok, large) — Behavioral Audit — Operator Runbook
- `rest-api.md` (~1135 tok, large) — MIND-Mem REST API
- `review-architecture-v3.2.0.md` (~1919 tok, huge) — Architecture Review — MIND-Mem v3.2.0 (Release Candidate)
- `review-database-v3.2.0.md` (~3171 tok, huge) — Database Review — PostgresBlockStore v3.2.0
- `review-docs-v3.2.0.md` (~1957 tok, huge) — Documentation Review — MIND-Mem v3.2.0
- `review-tests-v3.2.0.md` (~1300 tok, large) — Test Review — MIND-Mem v3.2.0
- `roadmap.md` (~13371 tok, huge) — Roadmap
- `roadmap-v4.md` (~10304 tok, huge) — mind-mem v4.0 — Design Rationale
- `scoring.md` (~517 tok, large) — Scoring System
- `SECURITY_AUDIT_SELF_2026_04.md` (~2267 tok, huge) — MIND-Mem v3.2.0 — Self-Audit Plan (Post-Release Deliverable)
- `security-audit-sow.md` (~3353 tok, huge) — MIND-Mem — External Security Audit Statement of Work (SoW)
### `docs/security-baselines/`

- `bandit-v3.2.0-baseline.json` (~18974 tok, huge) — Keys: errors, generated_at, metrics, results
### `docs/`

- `security-model.md` (~350 tok, medium) — Security Model
- `setup.md` (~1741 tok, huge) — Setup
- `status.md` (~1125 tok, large) — MIND-Mem — implementation status (alignment companion)
- `storage-backends.md` (~1264 tok, large) — Storage Backends
- `storage-migration.md` (~2391 tok, huge) — Storage Backend Migration Guide
- `supply-chain-security.md` (~1051 tok, large) — Supply-Chain Security
- `testing-guide.md` (~369 tok, medium) — Testing Guide
- `troubleshooting.md` (~681 tok, large) — Troubleshooting
- `usage.md` (~2002 tok, huge) — Usage
- `v3.11.0-implementation-plan.md` (~1570 tok, huge) — v3.11.0 Implementation Plan — synthesis from cross-model review
- `v3.11.0-mind-mem-4b-retrain-plan.md` (~1529 tok, huge) — mind-mem-4b v3.11.0 Retrain Plan
- `v3.1.9-self-audit.md` (~1396 tok, large) — Self-audit after v3.1.9
- `v3.2.0-atomicity-scope-plan.md` (~1681 tok, huge) — v3.2.0 — Atomicity scope plan (§2.2)
- `v3.2.0-blockstore-routing-plan.md` (~2116 tok, huge) — v3.2.0 — Apply engine → BlockStore routing plan
- `v3.2.0-mcp-decomposition-plan.md` (~2575 tok, huge) — v3.2.0 — MCP server decomposition plan
- `v3.2.0-release-notes.md` (~1883 tok, huge) — MIND-Mem v3.2.0 — Production Deployment Release
- `v3.2.1-release-notes.md` (~1302 tok, large) — MIND-Mem v3.2.1 release notes
- `v3.3.0-release-notes.md` (~1129 tok, large) — MIND-Mem v3.3.0 release notes
- `v3.4.0-release-notes.md` (~1189 tok, large) — MIND-Mem v3.4.0 release notes
- `v3.4.0-roadmap-llm-consensus.md` (~1269 tok, large) — v3.4.0 roadmap — path to 90+ on LoCoMo
- `v4-audit-2026-05-10.md` (~994 tok, large) — v4 architecture audit — 2026-05-10
- `v4-release.md` (~3942 tok, huge) — v4.0.0 Release Notes
- `workspace-structure.md` (~352 tok, medium) — Workspace Structure
### `examples/`

- `basic_usage.py` (~394 tok, medium) — Basic mind-mem usage example.
- `mic_map_quickstart.py` (~735 tok, large) — MIC/MAP quickstart — emit, parse, round-trip, stream.
- `README.md` (~72 tok, small) — MIND-Mem Examples
### `.gemini/`

- `settings.json` (~28 tok, tiny) — Keys: system_instruction
### `.githooks/`

- `pre-commit` (~98 tok, small) — #!/usr/bin/env bash
### `.github/`

- `CODEOWNERS` (~25 tok, tiny) — # Default owners
- `copilot-instructions.md` (~71 tok, small) — mind-mem: GitHub Copilot workspace instructions
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

- `audit-pinned.yml` (~412 tok, medium) — name: Audit Pinned Models
- `benchmark.yml` (~761 tok, large) — name: Benchmark
- `ci.yml` (~3046 tok, huge) — name: CI
- `codeql.yml` (~225 tok, medium) — name: CodeQL
- `dependency-review.yml` (~114 tok, small) — name: Dependency Review
- `docs.yml` (~262 tok, medium) — name: Docs
- `label-sync.yml` (~112 tok, small) — name: Label Sync
- `red-team.yml` (~385 tok, medium) — name: Red Team Audit
- `release.yml` (~2244 tok, huge) — name: Release
- `security-review.yml` (~240 tok, medium) — name: Security Review
- `security.yml` (~1894 tok, huge) — name: Supply-Chain Security
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

- `kernels.c` (~2170 tok, huge)
### `mind/`

- `abstention.mind` (~215 tok, medium) — Confidence gating: decide whether to abstain from answering
- `adversarial.mind` (~156 tok, small)
- `answer.mind` (~1294 tok, large)
- `bm25.mind` (~477 tok, medium) — BM25F scoring kernel with field boosts and length normalization
- `category.mind` (~395 tok, medium) — Category distillation scoring kernel
- `cognitive.mind` (~434 tok, medium)
- `cross_encoder.mind` (~174 tok, small)
- `ensemble.mind` (~237 tok, medium)
- `evidence.mind` (~232 tok, medium)
- `governance.mind` (~1537 tok, huge)
- `graph.mind` (~235 tok, medium)
- `hybrid.mind` (~169 tok, small)
- `importance.mind` (~246 tok, medium) — A-MEM: auto-maintained importance scores for memory blocks
- `intent.mind` (~149 tok, small)
- `prefetch.mind` (~256 tok, medium) — Prefetch context scoring kernel
- `query_plan.mind` (~266 tok, medium)
- `ranking.mind` (~227 tok, medium) — Evidence ranking: combine multiple scoring signals for final ranking
- `README.md` (~911 tok, large) — MIND Kernels
- `recall.mind` (~207 tok, medium)
- `reranker.mind` (~412 tok, medium) — Deterministic reranking features (no model needed)
- `rerank.mind` (~146 tok, small)
- `rm3.mind` (~189 tok, small)
- `rrf.mind` (~197 tok, small) — RRF: fuse ranked lists from multiple retrievers
- `session.mind` (~155 tok, small)
- `temporal.mind` (~113 tok, small)
- `trajectory.mind` (~440 tok, medium)
- `truth.mind` (~218 tok, medium)
### `.roo/`

- `system-prompt.md` (~22 tok, tiny) — mind-mem
### `scripts/`

- `anatomy-hook.sh` (~258 tok, medium) — anatomy-hook.sh — Git pre-commit hook to refresh ANATOMY.md
- `anatomy.sh` (~2010 tok, huge) — anatomy — Generate ANATOMY.md for any repo
- `build_integrity_manifest.py` (~634 tok, large) — Bake ``_integrity_manifest.json`` into the package before wheel build.
- `check_claims.sh` (~385 tok, medium) — Cross-repo docs-claim regression gate (mind-mem side).
- `count_mcp_tools.py` (~964 tok, large) — Count registered MCP tools and assert the count matches CLAUDE.md.
- `regen_bash_literals.py` (~424 tok, medium) — Regenerate src/mind_mem/_task_status_literals.sh from enums.py.
### `sdk/go/`

- `client.go` (~1008 tok, large) — Option is a functional option for NewClient.
- `client_test.go` (~2905 tok, huge) — Helpers
- `doc.go` (~334 tok, medium) — Package mindmem is the official Go SDK for the mind-mem REST API.
- `errors.go` (~640 tok, large) — APIError is returned for any non-2xx response from the mind-mem server.
- `.gitignore` (~5 tok, tiny) — *.test
- `go.mod` (~13 tok, tiny) — module github.com/star-ga/mind-mem/sdk/go
- `methods.go` (~500 tok, large) — Recall queries the memory store using full-text and semantic search.
- `README.md` (~520 tok, large) — MIND-Mem Go SDK
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
### `security/`

- `api-security-2026-04-28.md` (~5929 tok, huge) — MIND-Mem v3.1.8 — API / MCP Surface Security Audit
- `api-security-review-2026-04-28.md` (~3563 tok, huge) — MIND-Mem API Security Review — 2026-04-28
- `threat-model-2026-04-28.md` (~1517 tok, huge) — MIND-Mem Threat Model — 2026-04-28
### `skills/apply-proposal/`

- `SKILL.md` (~345 tok, medium) — /apply — Apply Proposals
### `skills/integrity-scan/`

- `SKILL.md` (~376 tok, medium) — /scan — Memory Integrity Scan
### `skills/memory-recall/`

- `SKILL.md` (~549 tok, large) — /recall — Memory Search
### `src/`

- `mcp_server.py` (~280 tok, medium) — Wheel-level compatibility module for `mind_mem.mcp_server`.
### `src/mind_mem/`

- `abstention_classifier.py` (~3261 tok, huge) — Deterministic adversarial abstention classifier for Mind-Mem.
- `agent_bridge.py` (~4158 tok, huge) — # Copyright 2026 STARGA, Inc.
- `alerting.py` (~2476 tok, huge) — # Copyright 2026 STARGA, Inc.
- `answer_quality.py` (~2681 tok, huge) — Answer-quality layer: verification + self-consistency + per-category spec.
### `src/mind_mem/api/`

- `api_keys.py` (~2717 tok, huge) — Per-agent API key store for the mind-mem REST API.
- `auth.py` (~2476 tok, huge) — OIDC/SSO authentication for the mind-mem REST API.
- `grpc_server.py` (~1850 tok, huge) — gRPC wire protocol for mind-mem (v4.0 prep).
- `__init__.py` (~20 tok, tiny)
- `rest.py` (~9532 tok, huge) — REST API layer for mind-mem (v3.2.0, v3.2.1 hardening).
### `src/mind_mem/`

- `apply_engine.py` (~16524 tok, huge) — Mind Mem Apply Engine v1.0 — Atomic proposal application with rollback.
- `audit_chain.py` (~4547 tok, huge) — mind-mem Hash-Chain Mutation Log — tamper-evident append-only ledger.
- `audit_pinned.py` (~3078 tok, huge) — Pinned-model audit pipeline — release-CI gate for ``mind-mem.json``.
- `auto_resolver.py` (~3194 tok, huge) — mind-mem Automatic Contradiction Resolution Suggestions.
- `axis_recall.py` (~4217 tok, huge) — # Copyright 2026 STARGA, Inc.
- `backup_restore.py` (~3821 tok, huge) — mind-mem Backup & Restore CLI. Zero external deps.
- `baseline_snapshot.py` (~4176 tok, huge) — Baseline snapshot for intent drift detection.
- `block_lineage.py` (~3590 tok, huge) — Typed block-lineage edges + bounded BFS reader (v3.11.0+, Pattern 3).
- `block_maturity.py` (~1977 tok, huge) — Block maturity metric — consolidation gate (Group H, v4.0.x).
- `block_metadata.py` (~2223 tok, huge) — mind-mem A-MEM — auto-evolving block metadata.
- `block_parser.py` (~7555 tok, huge) — Mind Mem Block Parser v1.0 — Self-hosted, zero external dependencies.
- `block_store_encrypted.py` (~2313 tok, huge) — # Copyright 2026 STARGA, Inc.
- `block_store_postgres.py` (~14850 tok, huge) — PostgresBlockStore — PostgreSQL-backed BlockStore for mind-mem v3.2.0.
- `block_store_postgres_replica.py` (~2530 tok, huge) — v3.2.0 — read-replica routing for PostgresBlockStore.
- `block_store.py` (~10467 tok, huge) — BlockStore abstraction — decouples block access from storage format.
- `bootstrap_corpus.py` (~2158 tok, huge) — mind-mem Bootstrap Corpus — one-time backfill from existing knowledge sources.
- `calibration.py` (~4838 tok, huge) — Calibration feedback loop — track retrieval quality and adjust block ranking.
- `capture.py` (~3722 tok, huge) — mind-mem Auto-Capture Engine with Structured Extraction. Zero external deps.
- `category_distiller.py` (~6284 tok, huge) — mind-mem Category Distiller — auto-generates thematic summary files from memory blocks.
- `causal_graph.py` (~3956 tok, huge) — mind-mem Temporal Causal Graph — directed dependency tracking with staleness.
- `chain_of_note.py` (~1435 tok, large) — Chain-of-note evidence packing (v3.4.0).
- `change_stream.py` (~1553 tok, huge) — # Copyright 2026 STARGA, Inc.
- `check_version.py` (~622 tok, large) — Version consistency checker for mind-mem.
- `coding_schemas.py` (~2127 tok, huge) — mind-mem Coding-Native Memory Schemas.
- `cognitive_forget.py` (~2667 tok, huge) — # Copyright 2026 STARGA, Inc.
- `compaction.py` (~3270 tok, huge) — mind-mem Compaction & GC Engine. Zero external deps.
- `compiled_truth.py` (~6410 tok, huge) — mind-mem Compiled Truth — synthesized entity pages with append-only evidence.
- `conflict_resolver.py` (~3377 tok, huge) — mind-mem Automated Conflict Resolution Pipeline. Zero external deps.
- `connection_manager.py` (~1165 tok, large) — SQLite connection manager with read/write separation and WAL mode.
- `consensus_vote.py` (~1871 tok, huge) — Quorum-based consensus voting on contradictions (v3.3.0).
- `context_core.py` (~4313 tok, huge) — # Copyright 2026 STARGA, Inc.
- `contradiction_detector.py` (~4888 tok, huge) — mind-mem Contradiction Detector — Surface conflicts at the governance gate.
- `core_export.py` (~5295 tok, huge) — # Copyright 2026 STARGA, Inc.
- `corpus_registry.py` (~471 tok, medium) — Central corpus path registry for mind-mem.
- `cron_runner.py` (~2305 tok, huge) — mind-mem Cron Runner — single entry point for all periodic jobs. Zero external deps.
- `cross_encoder_reranker.py` (~749 tok, large) — mind-mem Optional Cross-Encoder Reranker.
- `daemon.py` (~2914 tok, huge) — Background daemon — `mm daemon` (v3.9.0 candidate).
- `dedup.py` (~4593 tok, huge) — mind-mem 4-layer deduplication filter for search results.
- `dream_cycle.py` (~11560 tok, huge) — mind-mem Dream Cycle — autonomous memory enrichment. Zero external deps.
- `drift_detector.py` (~5106 tok, huge) — mind-mem Semantic Belief Drift Detection.
- `encryption.py` (~3889 tok, huge) — mind-mem Encryption at Rest — optional authenticated encryption for blocks.
- `entity_ingest.py` (~3220 tok, huge) — mind-mem Entity Ingestion — regex-based entity extraction. Zero external deps.
- `entity_prefetch.py` (~2913 tok, huge) — Entity-graph prefetch for recall (v3.3.0 Tier 3 #8).
- `enums.py` (~471 tok, medium) — Centralised enum definitions for mind-mem.
- `error_codes.py` (~1751 tok, huge) — mind-mem Error Codes — structured error classification.
- `event_fanout.py` (~2150 tok, huge) — Governance event fan-out (v4.0 prep).
- `evidence_bundle.py` (~2205 tok, huge) — Structured evidence bundle for answerer co-design (v3.3.0 Tier 3 #7).
- `evidence_objects.py` (~5859 tok, huge) — # Copyright 2026 STARGA, Inc.
- `evidence_packer.py` (~3265 tok, huge) — Deterministic evidence packer for Mind-Mem.
- `extraction_feedback.py` (~1177 tok, large) — mind-mem Extraction Quality Feedback Tracker.
- `extractor.py` (~7670 tok, huge) — mind-mem Entity & Fact Extractor (Regex NER-lite). Zero external deps.
- `feature_gate.py` (~1377 tok, large) — Shared config-resolver for retrieval features (architect audit item #6).
- `field_audit.py` (~3103 tok, huge) — mind-mem Per-Field Mutation Audit — tracks individual field changes.
- `governance_bench.py` (~1855 tok, huge) — mind-mem Governance Benchmark Suite.
- `governance_gate.py` (~2212 tok, huge) — # Copyright 2026 STARGA, Inc.
- `governance_raft.py` (~2208 tok, huge) — Raft-style consensus wrapper for governance writes (v4.0 prep).
- `granularity_align.py` (~3714 tok, huge) — Granularity / abstraction alignment — named merge operation (Group H, v4.0.x).
- `graph_recall.py` (~1907 tok, huge) — Multi-hop graph traversal for recall (v3.3.0 Tier 1 #2).
- `hash_chain_v2.py` (~5512 tok, huge) — # Copyright 2026 STARGA, Inc.
- `hook_installer.py` (~10495 tok, huge) — # Copyright 2026 STARGA, Inc.
- `http_transport.py` (~11242 tok, huge) — HTTP transport adapter for mind-mem (v3.9.0 candidate).
- `hybrid_recall.py` (~10274 tok, huge) — mind-mem Hybrid Recall -- BM25 + Vector + RRF fusion.
- `inbox.py` (~3595 tok, huge) — Inbox folder ingestion — `mm inbox-watch` (v3.9.0 candidate).
- `ingestion_pipeline.py` (~1752 tok, huge) — # Copyright 2026 STARGA, Inc.
- `__init__.py` (~714 tok, large) — # Mind Mem — Memory + Immune System for AI agents
- `init_workspace.py` (~4222 tok, huge) — mind-mem workspace initializer. Zero external deps (Postgres optional).
- `intel_scan.py` (~12607 tok, huge) — Mind Mem Intelligence Scanner v2.0 — Self-hosted, zero external dependencies.
- `intent_router.py` (~3134 tok, huge) — mind-mem Intent Router — 9-type adaptive query intent classification.
- `interaction_signals.py` (~4278 tok, huge) — # Copyright 2026 STARGA, Inc.
- `iterative_recall.py` (~2808 tok, huge) — Iterative chain-of-retrieval for multi-hop evidence (v3.4.0).
- `kalman_belief.py` (~4219 tok, huge) — # Copyright 2026 STARGA, Inc.
- `knowledge_graph.py` (~5948 tok, huge) — # Copyright 2026 STARGA, Inc.
- `ledger_anchor.py` (~1183 tok, large) — # Copyright 2026 STARGA, Inc.
- `lineage_staleness.py` (~1916 tok, huge) — Lineage→staleness propagation (v3.12.0, Theme C).
- `llm_extractor.py` (~5749 tok, huge) — mind-mem LLM Entity & Fact Extractor (Optional, config-gated).
- `llm_noise_profile.py` (~2339 tok, huge) — # Copyright 2026 STARGA, Inc.
- `maintenance_migrate.py` (~1243 tok, large) — v3.2.0 §2.2 — one-shot migration helper for ``maintenance/`` subdivision.
- `mcp_entry.py` (~495 tok, medium) — Thin entry point for the ``mind-mem-mcp`` console script.
### `src/mind_mem/mcp/infra/`

- `acl.py` (~1570 tok, huge) — Per-tool ACL — scope enforcement for the MCP surface.
- `config.py` (~1070 tok, large) — ``mind-mem.json`` config loading + configurable limits.
- `constants.py` (~98 tok, small) — MCP-surface-wide constants shared by the infra submodules.
- `http_auth.py` (~1454 tok, large) — HTTP bearer-token authentication helpers for the MCP surface.
- `__init__.py` (~449 tok, medium) — Cross-cutting infra helpers extracted from mcp_server.py (v3.2.0 §1.2 PR-1).
- `observability.py` (~2078 tok, huge) — Observability + DB-busy helpers for the MCP surface.
- `rate_limit.py` (~1035 tok, large) — Per-client sliding-window rate limiter for the MCP surface.
- `workspace.py` (~2237 tok, huge) — Workspace resolution + path-safety helpers.
### `src/mind_mem/mcp/`

- `__init__.py` (~215 tok, medium) — v3.2.0 §1.2 decomposition namespace — subpackage for MCP server modules.
- `resources.py` (~1342 tok, large) — MCP ``@mcp.resource`` declarations.
- `server.py` (~2571 tok, huge) — FastMCP instance + ``main()`` entry point for the Mind-Mem MCP server.
### `src/mind_mem/`

- `mcp_server.py` (~1824 tok, huge) — Mind-Mem MCP Server — public facade (v3.2.0 §1.2 PR-final shim).
### `src/mind_mem/mcp/tools/`

- `agent.py` (~1909 tok, huge) — Agent-bridge + vault MCP tools.
- `arch_mind.py` (~3425 tok, huge) — arch-mind MCP tools — wraps the ``arch-mind`` binary as 7 MCP tools.
- `audit.py` (~2183 tok, huge) — Audit MCP tools — Merkle proofs, hash chain + evidence chain verification.
- `benchmark.py` (~1016 tok, large) — Benchmark + category-summary MCP tools.
- `calibration.py` (~1158 tok, large) — Calibration feedback MCP tools — ``calibration_feedback`` + ``calibration_stats``.
- `consolidation.py` (~2358 tok, huge) — Memory-consolidation MCP tools.
- `core.py` (~1508 tok, huge) — Context-core MCP tools — ``.mmcore`` bundle lifecycle.
- `encryption.py` (~2626 tok, huge) — At-rest encryption MCP tools — ``encrypt_file`` / ``decrypt_file``.
- `governance.py` (~8144 tok, huge) — Governance MCP tools — propose / apply / rollback / scan / contradictions / memory_evolution.
- `graph.py` (~2196 tok, huge) — Knowledge-graph + causal-graph MCP tools.
- `_helpers.py` (~816 tok, large) — Shared tool-internal helpers — workspace paths + lazy-init singletons.
- `__init__.py` (~107 tok, small) — Per-domain ``@mcp.tool`` modules (v3.2.0 §1.2 PR-3+).
- `kernels.py` (~1902 tok, huge) — MIND kernel + compiled-truth MCP tools.
- `lineage.py` (~717 tok, large) — MCP wrapping for the v3.11.0 typed block-lineage graph (Pattern 3).
- `memory_ops.py` (~8362 tok, huge) — Memory operations MCP tools — index / lifecycle / health / export.
- `mic_map.py` (~2436 tok, huge) — MIC/MAP serialization MCP tools — wraps ``mind_mem.mic_map``.
- `model.py` (~2586 tok, huge) — Model audit / signing MCP tools — wraps ``mind_mem.model_audit``,
- `ontology.py` (~969 tok, large) — Ontology MCP tools — ``ontology_load`` + ``ontology_validate``.
- `pipeline.py` (~916 tok, large) — MCP wrapping for pipeline-hash inspection + dirty-block re-extraction.
- `public.py` (~4557 tok, huge) — # mypy: disable-error-code="no-any-return"
- `quality.py` (~534 tok, large) — MCP wrapping for the v3.11.0 deterministic quality gate.
- `recall.py` (~5448 tok, huge) — Recall surface — the retrieval core of the MCP API.
- `signal.py` (~926 tok, large) — Interaction-signal MCP tools — ``observe_signal`` + ``signal_stats``.
- `walkthrough_persona.py` (~1493 tok, large) — MCP wrapping for v3.9 walkthrough + persona projection.
### `src/mind_mem/`

- `memory_mesh.py` (~1903 tok, huge) — # Copyright 2026 STARGA, Inc.
- `memory_tiers.py` (~4934 tok, huge) — # Copyright 2026 STARGA, Inc.
- `merkle_tree.py` (~3354 tok, huge) — # Copyright 2026 STARGA, Inc.
- `_mic_map_accel.pyx` (~1136 tok, large) — # cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
- `mic_map.py` (~8384 tok, huge) — MIC/MAP — STARGA-native serialization for MIND IR graphs.
- `mind_ffi.py` (~5479 tok, huge) — mind-mem FFI bridge — loads compiled MIND .so and exposes scoring functions.
- `mind_filelock.py` (~1844 tok, huge) — mind-mem file locking — cross-platform advisory locks. Zero external deps.
- `mind_kernels.py` (~1706 tok, huge) — # Copyright 2026 STARGA, Inc.
- `mm_cli.py` (~25031 tok, huge) — # Copyright 2026 STARGA, Inc.
- `model_audit.py` (~4370 tok, huge) — Model checkpoint audit — scan for remote-code hooks, unsafe pickle, tokenizer injection.
- `model_gate.py` (~2549 tok, huge) — Load-gate registry for ``mm audit-model`` checkpoints.
- `model_provenance.py` (~1751 tok, huge) — Provenance allowlist check for ``mm audit-model`` checkpoints.
- `model_signing.py` (~2990 tok, huge) — Ed25519 manifest signing for ``mm audit-model`` checkpoints.
- `mrs.py` (~1604 tok, huge) — # Copyright 2026 STARGA, Inc.
- `multi_modal.py` (~1659 tok, huge) — # Copyright 2026 STARGA, Inc.
- `namespaces.py` (~3824 tok, huge) — mind-mem Multi-Agent Namespace & ACL Engine. Zero external deps.
- `observability.py` (~2246 tok, huge) — mind-mem Observability Module. Zero external deps.
- `observation_axis.py` (~3925 tok, huge) — # Copyright 2026 STARGA, Inc.
- `observation_compress.py` (~1353 tok, large) — Observation Compression Layer for Mind-Mem.
- `online_trainer.py` (~2751 tok, huge) — # Copyright 2026 STARGA, Inc.
- `ontology.py` (~2843 tok, huge) — # Copyright 2026 STARGA, Inc.
- `personas.py` (~1256 tok, large) — Persona-aware recall projection (v3.9.0 candidate).
- `pipeline_hash.py` (~3145 tok, huge) — Hash-of-code pipeline invalidation (v3.9.0 candidate).
- `prefix_cache.py` (~3043 tok, huge) — # Copyright 2026 STARGA, Inc.
- `preimage.py` (~1329 tok, large) — # Copyright 2026 STARGA, Inc.
- `project_profile.py` (~1681 tok, huge) — # Copyright 2026 STARGA, Inc.
- `protection.py` (~1545 tok, huge) — Runtime protection layer for mind-mem (v3.3.0+).
- `py.typed` (~0 tok, tiny)
- `q1616.py` (~562 tok, large) — # Copyright 2026 STARGA, Inc.
- `quality_gate.py` (~2253 tok, huge) — Deterministic block quality gate (v3.11.0, Pattern 2).
- `query_expansion.py` (~4680 tok, huge) — Multi-query expansion for improved recall.
- `query_planner.py` (~2865 tok, huge) — Query decomposition for multi-hop questions (v3.3.0 Tier 1 #1).
- `recall_cache.py` (~2938 tok, huge) — v3.2.0 — distributed recall result cache (Redis + in-process LRU fallback).
- `_recall_constants.py` (~2420 tok, huge) — Recall engine constants — search fields, BM25 params, regex patterns, limits."""
- `_recall_context.py` (~2601 tok, huge) — Recall engine context packing — post-retrieval augmentation rules."""
- `_recall_core.py` (~19077 tok, huge) — Recall engine core — RecallBackend, main BM25 pipeline, backend loading, prefetch, CLI."""
- `_recall_detection.py` (~5477 tok, huge) — Recall engine detection — query type classification, text extraction, block utilities."""
- `_recall_expansion.py` (~3249 tok, huge) — Recall engine query expansion — domain synonyms, month normalization, RM3."""
- `_recall_explain.py` (~1331 tok, large) — Score decomposition record for explainable recall (v3.11.0, Pattern 1).
- `recall.py` (~1049 tok, large) — mind-mem Recall Engine (BM25 + TF-IDF + Graph + Stemming). Zero external deps.
- `_recall_reranking.py` (~3554 tok, huge) — Recall engine reranking — deterministic feature-based re-scoring of BM25 hits."""
- `_recall_scoring.py` (~3715 tok, huge) — Recall engine scoring — BM25F helper, date scores, graph boosting, negation, date proximity, categories."""
- `_recall_temporal.py` (~2044 tok, huge) — Recall engine temporal filtering — resolve relative time references and filter blocks."""
- `_recall_tokenization.py` (~784 tok, large) — Recall engine tokenization — Porter stemmer and tokenizer."""
- `recall_vector.py` (~14688 tok, huge) — mind-mem Vector Recall Backend (Semantic Search with Embeddings).
- `rerank_ensemble.py` (~3364 tok, huge) — Reranker ensemble via Borda count (v3.3.0 Tier 4 #9).
- `retrieval_graph.py` (~5099 tok, huge) — Retrieval logger + co-retrieval graph for usage-based score propagation.
- `retrieval_trace.py` (~1252 tok, large) — Per-feature retrieval attribution (v3.3.0 architect audit item #7).
- `schema_version.py` (~1897 tok, huge) — Mind-Mem Schema Version Migration. Zero external deps.
- `session_boost.py` (~1533 tok, huge) — Session-boundary preservation for recall (v3.3.0 Tier 2 #5).
- `session_summarizer.py` (~2885 tok, huge) — mind-mem Session Summarizer. Zero external deps.
### `src/mind_mem/skill_opt/`

- `adapters.py` (~2237 tok, huge) — # Copyright 2026 STARGA, Inc.
- `analyzer.py` (~1145 tok, large) — # Copyright 2026 STARGA, Inc.
- `config.py` (~973 tok, large) — # Copyright 2026 STARGA, Inc.
- `fleet_bridge.py` (~1427 tok, large) — # Copyright 2026 STARGA, Inc.
- `history.py` (~1731 tok, huge) — # Copyright 2026 STARGA, Inc.
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
- `sqlite_index.py` (~14504 tok, huge) — Mind Mem SQLite FTS5 Index — incremental lexical indexing. Zero external deps.
- `staleness.py` (~1179 tok, large) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/storage/`

- `__init__.py` (~2531 tok, huge) — Storage factory for mind-mem block stores (v3.2.0).
- `sharded_pg.py` (~2988 tok, huge) — Sharded Postgres / Citus routing (v4.0 prep).
### `src/mind_mem/`

- `streaming.py` (~1655 tok, huge) — Back-pressure-aware streaming ingest (v3.3.0).
- `_task_status_literals.sh` (~118 tok, small) — AUTO-GENERATED — do not edit by hand.
- `telemetry.py` (~2558 tok, huge) — mind-mem Telemetry — OpenTelemetry traces + Prometheus metrics.
- `temporal_metadata.py` (~1831 tok, huge) — Temporal metadata injection for retrieved blocks (v3.4.0).
- `tenant_audit.py` (~1798 tok, huge) — Per-tenant audit chain isolation (v4.0 prep).
- `tenant_kms.py` (~2960 tok, huge) — Per-tenant key management + envelope encryption (v4.0 prep).
- `tiered_memory.py` (~1102 tok, large) — # Copyright 2026 STARGA, Inc.
- `tier_recall.py` (~2096 tok, huge) — Tier-aware recall score boosting (v3.2.0 hot/cold tier wire-up).
- `tracking.py` (~2018 tok, huge) — # Copyright 2026 STARGA, Inc.
- `trajectory.py` (~2233 tok, huge) — Trajectory Memory — task execution trace storage and recall.
- `transcript_capture.py` (~2333 tok, huge) — mind-mem Transcript JSONL Capture. Zero external deps.
- `truth_score.py` (~1678 tok, huge) — Probabilistic truth score for memory blocks (v3.3.0).
- `turbo_quant.py` (~1078 tok, large) — # Copyright 2026 STARGA, Inc.
- `uncertainty_propagation.py` (~1262 tok, large) — # Copyright 2026 STARGA, Inc.
- `union_recall.py` (~1310 tok, large) — Union-style retrieval for decomposed queries (v3.4.0).
### `src/mind_mem/v4/`

- `backpressure.py` (~2058 tok, huge) — v4 backpressure controller (round 4 audit, DeepSeek 9.75→10 gap).
- `block_kinds.py` (~3123 tok, huge) — v4 block-kind taxonomy (Group B: knowledge graph).
- `block_metadata.py` (~2713 tok, huge) — v4 block metadata + schema-validation hooks.
- `circuit_breaker.py` (~3390 tok, huge) — v4 circuit breaker (round 5 audit, Mistral + GLM 9.9→10 gap).
- `cognitive_kernel.py` (~2375 tok, huge) — v4 Cognitive Mind Kernel — composable retrieval strategies (Group A).
- `consolidation_worker.py` (~2427 tok, huge) — v4 consolidation worker (Group A — MemGPT-pattern).
- `embedding_pipeline.py` (~1500 tok, huge) — v4 embedding auto-derivation pipeline (Group A — closes the
- `eviction.py` (~3008 tok, huge) — v4 eviction policies (Group A — long-term decay).
- `feature_flags.py` (~1533 tok, huge) — v4.0 feature-flag registry.
- `federation_client.py` (~3468 tok, huge) — Federation wire-transport client for mind-mem v4.
- `federation.py` (~6059 tok, huge) — v4 federated cross-agent consistency (Group D).
- `health.py` (~1884 tok, huge) — v4 health-check surface (round 4 audit, DeepSeek 9.75→10 gap).
- `hnsw_kind_index.py` (~2122 tok, huge) — v4 HNSW kind-filtered ANN index (Group D).
- `__init__.py` (~824 tok, large) — mind-mem v4.0 surface — side-by-side scaffolding, default OFF.
- `kernels.py` (~3674 tok, huge) — v4 kernel strategy implementations (Group A).
- `kind_summaries.py` (~1878 tok, huge) — v4 per-kind global summaries (Group B — GraphRAG-style).
- `logging_context.py` (~1232 tok, large) — v4 structured logging context (round 4 audit, DeepSeek 9.75→10 gap).
- `observability.py` (~2633 tok, huge) — v4 observability — counters, timers, histograms, exporters.
- `pq.py` (~4343 tok, huge) — v4 product-quantization (PQ) encoding for embedding storage (Group D).
- `self_editing.py` (~2420 tok, huge) — v4 self-editing on recall (Group A — MemGPT pattern).
- `surprise_retrieval.py` (~2575 tok, huge) — v4 surprise-weighted retrieval term (Group A: cognition / model layer).
- `tier_memory.py` (~3233 tok, huge) — v4 recall-tier memory (Group A: cognition / model layer).
### `src/mind_mem/`

- `validate_py.py` (~4890 tok, huge) — Mind Mem Integrity Validator — canonical engine.
- `validate.sh` (~352 tok, medium) — src/mind_mem/validate.sh — thin forwarder to the Python validator.
- `validate.sh.pre-forwarder` (~7140 tok, huge) — #!/usr/bin/env bash
- `verify_cli.py` (~3178 tok, huge) — # Copyright 2026 STARGA, Inc.
- `walkthrough.py` (~2449 tok, huge) — Dependency-ordered walkthrough — `compile_walkthrough` (v3.9.0 candidate).
- `watcher.py` (~886 tok, large) — Mind-Mem File Watcher — auto-reindex on workspace changes. Zero external deps.
### `templates/`

- `AUDIT.md` (~31 tok, tiny) — AUDIT — MIND-Mem v1.0
- `BRIEFINGS.md` (~47 tok, tiny) — BRIEFINGS — MIND-Mem v1.0
- `CONTRADICTIONS.md` (~47 tok, tiny) — CONTRADICTIONS — MIND-Mem v1.0
- `DECISIONS.md` (~77 tok, small) — DECISIONS — MIND-Mem v1.0
- `DECISIONS_PROPOSED.md` (~50 tok, small) — DECISIONS_PROPOSED — MIND-Mem v1.0
- `DRIFT.md` (~45 tok, tiny) — DRIFT — MIND-Mem v1.0
- `EDITS_PROPOSED.md` (~34 tok, tiny) — EDITS_PROPOSED — MIND-Mem v1.0
- `IMPACT.md` (~43 tok, tiny) — IMPACT — MIND-Mem v1.0
- `incidents.md` (~38 tok, tiny) — INCIDENTS — MIND-Mem v1.0
- `intel-state.json` (~197 tok, small) — Keys: governance_mode, version, auto_apply_low_risk, flip_gate_week1_clean, last_scan
- `maint-state.json` (~12 tok, tiny) — Keys: last_run, last_weekly
- `MEMORY.md` (~70 tok, small) — Memory Protocol v1.0
- `people.md` (~31 tok, tiny) — PEOPLE — MIND-Mem v1.0
- `projects.md` (~39 tok, tiny) — PROJECTS — MIND-Mem v1.0
- `SCAN_LOG.md` (~80 tok, small) — SCAN_LOG — MIND-Mem v1.0
- `SIGNALS.md` (~51 tok, small) — SIGNALS — MIND-Mem v1.0
- `TASKS.md` (~83 tok, small) — TASKS — MIND-Mem v1.0
- `TASKS_PROPOSED.md` (~33 tok, tiny) — TASKS_PROPOSED — MIND-Mem v1.0
- `tools.md` (~33 tok, tiny) — TOOLS — MIND-Mem v1.0
### `tests/integration/`

- `__init__.py` (~0 tok, tiny)
- `test_full_pipeline.py` (~1575 tok, huge) — Integration test: full mind-mem pipeline.
### `tests/red_team/`

- `behavioral_audit.py` (~636 tok, large) — Behavioral audit scaffold for the mind-mem MCP surface.
- `conftest.py` (~170 tok, small) — pytest configuration for the red_team test package.
- `__init__.py` (~0 tok, tiny)
### `tests/red_team/transcripts/`

- `.gitkeep` (~0 tok, tiny)
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
- `test_apply_double_apply_guard.py` (~421 tok, medium) — apply must re-validate proposal status UNDER the workspace lock.
- `test_apply_engine_backend_routing.py` (~1066 tok, large) — v3.2.0 §1.4 PR-6 — apply_engine routes through configured BlockStore."""
- `test_apply_engine_op_routing.py` (~1830 tok, huge) — v3.2.2 — execute_op routes block-level ops through BlockStore.
- `test_apply_engine.py` (~12501 tok, huge) — Tests for apply_engine.py — focus on security, validation, and rollback."""
- `test_atomicity_maintenance_scope.py` (~1283 tok, large) — v3.2.0 §2.2 — regression test for the ``maintenance/`` atomicity fix.
- `test_audit_chain.py` (~2398 tok, huge) — Tests for mind-mem hash-chain mutation log (audit_chain.py)."""
- `test_audit_pinned.py` (~2880 tok, huge) — Pinned-model audit pipeline — release-CI gate.
- `test_auto_resolver.py` (~1185 tok, large) — Tests for mind-mem auto contradiction resolution (auto_resolver.py)."""
- `test_axis_recall_mcp.py` (~1381 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_axis_recall.py` (~3683 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_backup_restore.py` (~3256 tok, huge) — Tests for backup_restore.py — zero external deps (stdlib unittest)."""
- `test_baseline_snapshot.py` (~2997 tok, huge) — Tests for baseline snapshot and drift detection (#431)."""
- `test_bigrams.py` (~168 tok, small) — Tests for bigram extraction."""
- `test_block_id_format.py` (~327 tok, medium) — Tests for block ID format validation."""
- `test_block_lineage.py` (~2122 tok, huge) — Tests for the v3.11.0 typed block-lineage graph (Pattern 3)."""
- `test_block_maturity_group_h.py` (~4175 tok, huge) — Tests for Group H maturity metric — consolidation gate.
- `test_block_metadata.py` (~945 tok, large) — Tests for A-MEM block metadata evolution."""
- `test_block_parser_chunks.py` (~1658 tok, huge) — Tests for block_parser.py — overlapping chunk splitting + dedup."""
- `test_block_parser_edge.py` (~620 tok, large) — Extended block parser tests."""
- `test_block_parser_fields.py` (~372 tok, medium) — Tests for block parser field extraction."""
- `test_block_parser_multifile.py` (~337 tok, medium) — Tests for parsing multiple files."""
- `test_block_parser_no_silent_truncation.py` (~638 tok, large) — Regression: block_parser must not silently drop corpus past a size cap.
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
- `test_check_workspace_backend.py` (~3028 tok, huge) — Backend-aware workspace validation — ``mcp.infra.workspace._check_workspace``.
- `test_chunk_text.py` (~231 tok, medium) — Tests for text chunking."""
- `test_coding_schemas.py` (~1284 tok, large) — Tests for mind-mem coding-native memory schemas."""
- `test_cognitive_forget.py` (~2315 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compaction.py` (~1869 tok, huge) — Tests for compaction.py — GC and archival engine."""
- `test_competitive_intel.py` (~1881 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compiled_truth.py` (~3929 tok, huge) — Tests for mind-mem compiled truth pages (compiled_truth.py)."""
- `test_concurrency_stress.py` (~4160 tok, huge) — Concurrency and performance stress tests for recall engine.
- `test_concurrent_integration.py` (~10855 tok, huge) — Integration tests for concurrent access and partial failure in mind-mem.
- `test_conflict_resolver_hash_mapping.py` (~316 tok, medium) — Audit trail must print the WINNER's hash next to Winner, not block_a's.
- `test_conflict_resolver.py` (~2340 tok, huge) — Tests for conflict_resolver.py — zero external deps (stdlib unittest)."""
- `test_connection_manager.py` (~2536 tok, huge) — Tests for ConnectionManager — SQLite connection pooling with read/write separation (#466)."""
- `test_consensus_vote.py` (~1137 tok, large) — v3.3.0 — quorum-based consensus voting on contradictions."""
- `test_constants.py` (~371 tok, medium) — Tests for recall constants module."""
- `test_context_core.py` (~3175 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_context_pack.py` (~2584 tok, huge) — Tests for context_pack rules: adjacency, diversity, pronoun rescue."""
- `test_context_pack_scripts.py` (~673 tok, large) — Tests for context packing via scripts._recall_context."""
- `test_contradiction_detector.py` (~5832 tok, huge) — Tests for contradiction_detector.py — Contradiction detection at governance gate (#432).
- `test_core_v140.py` (~2707 tok, huge) — Tests for v1.4.0 core hardening: issues #28, #30, #32, #34."""
- `test_cron_runner.py` (~4306 tok, huge) — Tests for cron_runner.py — periodic job orchestration, config loading, subprocess dispatch."""
- `test_cross_encoder_auto_enable.py` (~1801 tok, huge) — v3.3.0 Tier 2 #6 — cross-encoder rerank auto-enables on ambiguous queries.
- `test_cross_encoder.py` (~1324 tok, large) — Tests for optional cross-encoder reranker."""
- `test_daemon.py` (~1886 tok, huge) — Tests for the v3.9 background daemon (`mm daemon`)."""
- `test_date_score.py` (~174 tok, small) — Tests for date scoring function."""
- `test_decompose_query.py` (~223 tok, medium) — Tests for query decomposition."""
- `test_decrypt_file_audit_trail.py` (~1262 tok, large) — Regression test for the `decrypt_file` forensic audit trail
- `test_dedup.py` (~5670 tok, huge) — Tests for dedup.py -- 4-layer deduplication filter."""
- `test_dedup_vector.py` (~1087 tok, large) — Tests for vector-enhanced cosine dedup (Layer 2b)."""
- `test_detection.py` (~326 tok, medium) — Tests for query detection module."""
- `test_downgrade_mitigation.py` (~1339 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_dream_cycle_backends.py` (~2497 tok, huge) — Backend-aware dream-cycle maintenance passes (audit bug 11).
- `test_dream_cycle.py` (~4567 tok, huge) — Tests for dream_cycle.py — autonomous memory enrichment passes."""
- `test_drift_detector.py` (~3697 tok, huge) — Tests for mind-mem semantic belief drift detection (drift_detector.py)."""
- `test_dsn_redaction.py` (~542 tok, large) — Tests for DSN password redaction in mm_cli.
- `test_edge_cases.py` (~4055 tok, huge) — Edge-case and stress tests for mind-mem — block_parser, recall, and MCP server."""
- `test_encryption.py` (~1732 tok, huge) — Tests for mind-mem encryption at rest."""
- `test_entity_ingest.py` (~4091 tok, huge) — Tests for the entity_ingest module — extraction, filtering, signal generation."""
- `test_entity_prefetch.py` (~1674 tok, huge) — v3.3.0 Tier 3 #8 — entity-graph prefetch.
- `test_enums.py` (~534 tok, large) — Tests for centralised enums (mind_mem.enums)."""
- `test_error_codes.py` (~2394 tok, huge) — Tests for mind-mem Error Codes module."""
- `test_error_paths.py` (~5892 tok, huge) — Error path and edge-case tests for mind-mem — malformed inputs, missing files, bad configs."""
- `test_event_fanout.py` (~1153 tok, large) — v4.0 prep — governance event fan-out."""
- `test_event_id_filter.py` (~1776 tok, huge) — Unit tests for the event_id recall post-filter.
- `test_evidence_bundle.py` (~1562 tok, huge) — v3.3.0 Tier 3 #7 — structured evidence bundle.
- `test_evidence_objects.py` (~4031 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_evidence_packer.py` (~5180 tok, huge) — Tests for the evidence packer module."""
- `test_excerpt.py` (~248 tok, medium) — Tests for excerpt generation."""
- `test_expand_query.py` (~265 tok, medium) — Tests for query expansion module."""
- `test_extractor.py` (~3387 tok, huge) — Tests for the regex NER-lite entity/fact extractor."""
- `test_extractor_windowed_scan.py` (~1838 tok, huge) — Regression tests for the windowed extract_facts scan (issue #530).
- `test_fact_indexing.py` (~3101 tok, huge) — Tests for Feature 2 (fact card indexing) and Feature 4 (metadata-augmented embeddings)."""
- `test_feature_gate.py` (~1031 tok, large) — Tests for FeatureGate — the shared config-resolver for retrieval features."""
- `test_federation_peer_allowlist.py` (~1606 tok, huge) — Regression tests for MIND_MEM_FED_PEERS operator-side peer allowlist
- `test_federation_resolve_race.py` (~548 tok, large) — resolve_conflict must NOT run vclock upserts when its UPDATE was a no-op.
- `test_field_audit.py` (~1399 tok, large) — Tests for mind-mem per-field mutation audit (field_audit.py)."""
- `test_field_extraction.py` (~201 tok, medium) — Tests for field token extraction."""
- `test_filelock.py` (~979 tok, large) — Tests for filelock.py — cross-platform advisory locking."""
- `test_filelock_stress.py` (~1182 tok, large) — Stress tests for mind-mem file locking under contention."""
- `test_fts5_bm25_weights.py` (~391 tok, medium) — bm25() weights must align 1:1 with the indexed blocks_fts columns.
- `test_fts_fallback.py` (~4436 tok, huge) — Tests for FTS fallback behavior, recall envelope structure, block size cap,
- `test_governance_bench.py` (~811 tok, large) — Tests for mind-mem governance benchmark suite."""
- `test_governance_concurrency.py` (~1363 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_governance_raft.py` (~1398 tok, large) — v4.0 prep — Raft-style consensus wrapper for governance writes."""
- `test_governance_scan_backends.py` (~2651 tok, huge) — Backend-aware governance ``scan`` — audit bugs #3 / #10.
- `test_granularity_align.py` (~3310 tok, huge) — Tests for granularity_align — named merge operation (Group H, v4.0.x).
- `test_graph_boost.py` (~6050 tok, huge) — Tests for graph boost, context packing, config validation, and block cap.
- `test_graph_boost_recall.py` (~315 tok, medium) — Tests for graph_boost recall parameter."""
- `test_graph_recall.py` (~1498 tok, large) — v3.3.0 Tier 1 #2 — multi-hop graph traversal on recall results.
- `test_grid_search.py` (~1199 tok, large) — Tests for benchmarks/grid_search.py — grid generation and utility functions."""
- `test_group_h_robustness.py` (~10224 tok, huge) — Robustness tests for Group-H modules: edge-cases, error-paths, boundary values.
- `test_grpc_server.py` (~731 tok, large) — v4.0 prep — gRPC wire protocol (tests for the grpcio-free handlers)."""
- `test_hash_chain_v2.py` (~3462 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_hook_installer_force_preserves_siblings.py` (~703 tok, large) — Regression test for the --force clobber bug in hook_installer."""
- `test_hook_installer_registry.py` (~3841 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_http_auth_fail_closed.py` (~1724 tok, huge) — v3.7.0 H4: HTTP / REST auth must fail CLOSED by default.
- `test_http_transport.py` (~4326 tok, huge) — Tests for the v3.9 HTTP transport adapter.
- `test_hybrid_recall.py` (~3107 tok, huge) — Tests for hybrid_recall.py -- HybridBackend + RRF fusion."""
- `test_hybrid_search.py` (~599 tok, large) — Tests for hybrid search functionality."""
- `test_inbox.py` (~2352 tok, huge) — Tests for the v3.9 inbox folder ingestion."""
- `test_index_stats_b1.py` (~523 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_init_workspace_postgres.py` (~1640 tok, huge) — Postgres regression tests for ``init_workspace`` (audit bug #8).
- `test_init_workspace.py` (~4065 tok, huge) — Tests for init_workspace — config validation and workspace scaffolding."""
- `test_install_script.py` (~481 tok, medium) — # Force the pip installer. The pipx path is exercised independently
- `test_integration.py` (~1381 tok, large) — Integration test: full mind-mem lifecycle init → capture → scan → recall."""
- `test_intel_scan.py` (~5905 tok, huge) — Tests for intel_scan.py — contradiction detection, drift analysis, impact graph."""
- `test_intent_classify.py` (~328 tok, medium) — Tests for intent classification."""
- `test_intent_router_adaptive.py` (~3618 tok, huge) — Tests for adaptive intent routing (#470).
- `test_intent_router.py` (~1176 tok, large) — Tests for 9-type intent router."""
- `test_interaction_signals.py` (~3177 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_issue_526_acl_fail_closed.py` (~682 tok, large) — Regression for issue #526: ACL `_get_request_scope` must fail-closed.
- `test_issue_527_three_way_merge_vclock.py` (~1018 tok, large) — Regression for issue #527: THREE_WAY_MERGE must bump the vclock.
- `test_issue_529_federation_client_hardening.py` (~1202 tok, large) — Regression for issue #529: FederationClient hardening.
- `test_iter_active_blocks.py` (~1940 tok, huge) — Backend-aware active-block enumeration — ``storage.iter_active_blocks``.
- `test_kalman_belief.py` (~3728 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_knowledge_graph.py` (~3437 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_lifecycle_filter.py` (~1777 tok, huge) — Unit tests for the optional lifecycle block field and recall filter.
- `test_lineage_staleness.py` (~2161 tok, huge) — End-to-end tests for the v3.12 lineage→staleness wiring (Theme C).
- `test_llm_extractor_gate.py` (~2214 tok, huge) — Backend wiring — :func:`mind_mem.llm_extractor._gate_check_local`.
- `test_llm_extractor.py` (~1820 tok, huge) — Tests for the optional LLM entity/fact extractor module."""
- `test_llm_noise_profile.py` (~2354 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_maintenance_migrate.py` (~710 tok, large) — v3.2.0 §2.2 — tests for maintenance/ subdivision migration."""
- `test_mcp_arch_mind_tools.py` (~2312 tok, huge) — Tests for the arch-mind MCP tool wrapper.
- `test_mcp_db_error_backstop.py` (~699 tok, large) — The MCP tool decorator must not let a backend DB error crash the server.
- `test_mcp_integration.py` (~5478 tok, huge) — MCP transport and auth integration tests (#474).
- `test_mcp_pipeline.py` (~1465 tok, large) — Tests for the v3.9.0 pipeline-hash MCP tools."""
- `test_mcp_server.py` (~5277 tok, huge) — Tests for mcp_server.py — tests the MCP server resources and tool logic.
- `test_mcp_tools_model.py` (~2249 tok, huge) — Tests for ``mind_mem.mcp.tools.model`` — MCP wrappers for audit / sign / verify."""
- `test_mcp_tools.py` (~277 tok, medium) — Tests for MCP server tool definitions."""
- `test_mcp_tool_surface_v3_2.py` (~2002 tok, huge) — v3.2.0 — consolidated MCP public dispatcher tests."""
- `test_mcp_v140.py` (~5966 tok, huge) — Tests for MCP v1.4.0 features — issues #29, #31, #35, #36.
- `test_mcp_walkthrough_persona.py` (~1942 tok, huge) — Tests for the v3.9.0 MCP walkthrough + persona wrapper tools."""
- `test_memory_ops_postgres_backend.py` (~3037 tok, huge) — Backend-aware memory_ops tools — Postgres parity (audit bug 5).
- `test_memory_practical_e2e.py` (~2389 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_memory_tiers.py` (~3479 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_merkle_tree.py` (~3185 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mic_map_accel.py` (~1656 tok, huge) — Regression tests for the optional Cython accelerator at
- `test_mic_map_adversarial.py` (~3397 tok, huge) — Adversarial corpus for ``mind_mem.mic_map`` parsers.
- `test_mic_map_bench.py` (~2544 tok, huge) — pytest-benchmark suite for ``mind_mem.mic_map``.
- `test_mic_map_cli.py` (~1613 tok, huge) — Integration tests for the ``mm mic`` CLI subcommand.
- `test_mic_map_fuzz.py` (~2243 tok, huge) — Property-based fuzz tests for ``mind_mem.mic_map``.
- `test_mic_map_mcp.py` (~1812 tok, huge) — Integration tests for the MIC/MAP MCP tools (``mic_convert_tool``,
- `test_mic_map.py` (~3230 tok, huge) — Tests for ``mind_mem.mic_map`` — STARGA mic@2 / mic-b serialization.
- `test_mic_map_stream.py` (~2770 tok, huge) — Streaming-parser tests for ``mind_mem.mic_map.parse_micb_stream``.
- `test_mind_ffi.py` (~291 tok, medium) — Tests for MIND FFI module."""
- `test_mind_kernels_v3_3.py` (~998 tok, large) — Kernel-loading tests for v3.3.0 features.
- `test_mindllm_backend.py` (~1324 tok, large) — Regression tests for the MindLLM backend (roadmap v4.0.15).
- `test_mm_cli_debug.py` (~3339 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mm_doctor_postgres_hint.py` (~1124 tok, large) — Regression test: mm doctor must emit a clear hint when backend=postgres
- `test_model_audit.py` (~3383 tok, huge) — Tests for ``mind_mem.model_audit`` — checkpoint static-security audit.
- `test_model_gate.py` (~2026 tok, huge) — Tests for ``mind_mem.model_gate`` — load-gate registry."""
- `test_model_provenance.py` (~1977 tok, huge) — Tests for ``mind_mem.model_provenance`` — base_model allowlist."""
- `test_model_signing.py` (~2287 tok, huge) — Tests for ``mind_mem.model_signing`` — Ed25519 manifest signing."""
- `test_multi_file_recall.py` (~329 tok, medium) — Tests for recall across multiple files."""
- `test_namespaces.py` (~2827 tok, huge) — Tests for namespaces.py — zero external deps (stdlib unittest)."""
- `test_niah.py` (~4987 tok, huge) — Needle In A Haystack (NIAH) benchmark for mind-mem recall.
- `test_observability.py` (~791 tok, large) — Tests for observability.py — structured logging and metrics."""
- `test_observation_axis.py` (~3330 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_observation_compress.py` (~2754 tok, huge) — Tests for observation_compress module.
- `test_oidc_admin_enforcement.py` (~1716 tok, huge) — v3.2.1 — OIDC JWTs must pass through ``_require_admin`` checks.
- `test_oidc_auth.py` (~2970 tok, huge) — Tests for OIDCProvider / OIDCConfig in src/mind_mem/api/auth.py."""
- `test_okf_export.py` (~2008 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_ontology.py` (~2306 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_personas.py` (~1336 tok, large) — Tests for the v3.9 persona-aware recall projection."""
- `test_pg_block_store_ping.py` (~738 tok, large) — Tests for ``PostgresBlockStore.ping()`` — active backend health probe.
- `test_pg_restore_file_path.py` (~468 tok, medium) — restore() must preserve each block's file_path (routing metadata).
- `test_pipeline_hash.py` (~3309 tok, huge) — Tests for v3.9 hash-of-code pipeline invalidation."""
- `test_postgres_block_store.py` (~7397 tok, huge) — v3.2.0 §1.4 PR-5 — PostgresBlockStore integration tests.
- `test_postgres_replica_routing.py` (~1965 tok, huge) — v3.2.0 — tests for read-replica routing in ReplicatedPostgresBlockStore."""
- `test_prefetch_context.py` (~1487 tok, large) — Tests for prefetch_context() in recall.py."""
- `test_prefix_cache.py` (~3140 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_protection.py` (~1408 tok, large) — Tests for mind_mem.protection (v3.3.0+)."""
- `test_q1616_preimage.py` (~1496 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_quality_gate.py` (~1971 tok, huge) — Tests for the v3.11.0 deterministic block quality gate.
- `test_quality_gate_strict_mode.py` (~2937 tok, huge) — Tests for v3.12.0 Theme B: quality-gate config plumbing + propose_update wiring.
- `test_query_decomposition.py` (~1604 tok, huge) — Tests for multi-hop query decomposition (#6)."""
- `test_query_expansion_auto_enable.py` (~1270 tok, large) — v3.3.0 Tier 2 #4 — query expansion auto-enables on ambiguous queries.
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
- `test_recall_expansion_no_overbroad_synonyms.py` (~811 tok, large) — Regression tests for over-broad synonym entries in _QUERY_EXPANSIONS.
- `test_recall_explain.py` (~3501 tok, huge) — Tests for the explain=True flag on recall and hybrid_search MCP tools.
- `test_recall_intent_router.py` (~1207 tok, large) — Tests for IntentRouter integration in recall pipeline."""
- `test_recall_large_workspace.py` (~343 tok, medium) — Tests for recall with large workspaces."""
- `test_recall_limit.py` (~395 tok, medium) — Tests for recall limit parameter behavior."""
- `test_recall_metadata.py` (~1340 tok, large) — Tests for A-MEM block metadata integration in recall pipeline."""
- `test_recall_post_filters.py` (~310 tok, medium) — recall() applies lifecycle/event_id/min_maturity on EVERY dispatch path.
- `test_recall_postgres_backend.py` (~2696 tok, huge) — Backend-aware recall dispatch — Postgres parity (audit bug 1).
- `test_recall_priority.py` (~410 tok, medium) — Tests for priority boost in recall."""
- `test_recall.py` (~3880 tok, huge) — Tests for recall.py — zero external deps (stdlib unittest)."""
- `test_recall_quality_locomo.py` (~2610 tok, huge) — LoCoMo recall-quality regression gate.
- `test_recall_recursion_fix.py` (~2069 tok, huge) — Regression tests for the recall ↔ query_index mutual recursion bug.
- `test_recall_references.py` (~270 tok, medium) — Tests for reference-based recall."""
- `test_recall_reranking.py` (~2740 tok, huge) — Tests for _recall_reranking.py — deterministic reranker + LLM rerank."""
- `test_recall_scoring_order.py` (~440 tok, medium) — Tests for recall result scoring order."""
- `test_recall_source_field.py` (~279 tok, medium) — Tests for source field in recall results."""
- `test_recall_speaker.py` (~263 tok, medium) — Tests for speaker-based recall."""
- `test_recall_status_boost.py` (~394 tok, medium) — Tests for status boost in recall."""
- `test_recall_supersedes.py` (~216 tok, medium) — Tests for supersedes field in recall."""
- `test_recall_tags.py` (~320 tok, medium) — Tests for tag-based recall."""
- `test_recall_temporal.py` (~2800 tok, huge) — Tests for _recall_temporal.py — time-aware hard filters for temporal queries."""
- `test_recall_time_bounded.py` (~1965 tok, huge) — Regression tests for time-bounded recall (roadmap v4.0.0 Group E).
- `test_recall_vector.py` (~4901 tok, huge) — Tests for recall_vector.py — VectorBackend semantic search."""
- `test_rerank_debug.py` (~342 tok, medium) — Tests for rerank debug mode."""
- `test_rerank_ensemble.py` (~1531 tok, huge) — v3.3.0 Tier 4 #9 — reranker ensemble via Borda count.
- `test_reranking.py` (~246 tok, medium) — Tests for reranking module."""
- `test_rest_api_oidc.py` (~2686 tok, huge) — Tests for OIDC callback + admin API key endpoints (v3.2.0)."""
- `test_rest_api.py` (~3683 tok, huge) — Tests for the mind-mem REST API layer (v3.2.0).
- `test_rest_audit_headers.py` (~1517 tok, huge) — Regression tests for the audit-header middleware (roadmap v4.0.0 Group D).
- `test_retrieval_diagnostics.py` (~2419 tok, huge) — Tests for retrieval diagnostics (#428), corpus isolation (#429), and intent instrumentation (#430)."""
- `test_retrieval_graph.py` (~2242 tok, huge) — Tests for retrieval_graph.py — retrieval logging, co-retrieval graph, hard negatives."""
- `test_retrieval_trace.py` (~978 tok, large) — Tests for v3.3.0 per-feature retrieval attribution."""
- `test_rm3_expand.py` (~321 tok, medium) — Tests for RM3 query expansion."""
- `test_schema_version.py` (~1758 tok, huge) — Tests for schema_version.py — zero external deps (stdlib unittest)."""
- `test_scoring.py` (~337 tok, medium) — Tests for BM25 scoring functions."""
- `test_security_scanning_alerts.py` (~1984 tok, huge) — Regression tests for code-scanning alerts #181–#189.
- `test_session_boost.py` (~1488 tok, large) — v3.3.0 Tier 2 #5 — session-boundary preservation via recall-side boost.
- `test_session_summarizer.py` (~3972 tok, huge) — Comprehensive tests for mind_mem/session_summarizer.py.
- `test_sharded_pg.py` (~1222 tok, large) — v4.0 prep — sharded Postgres routing tests (mock underlying stores)."""
- `test_skeptical_query.py` (~194 tok, small) — Tests for skeptical query detection."""
- `test_skill_opt.py` (~3356 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_smart_chunker_code.py` (~1135 tok, large) — Tests for code-aware chunking in smart_chunker.py."""
- `test_smart_chunker.py` (~7744 tok, huge) — Tests for smart_chunker.py — semantic-boundary document chunking."""
- `test_spec_binding.py` (~3156 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_speculative_prefetch.py` (~3071 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_sqlite_index_backends.py` (~2539 tok, huge) — Backend-parity regression tests for ``sqlite_index`` (audit bugs 4, 9, 13, 14).
- `test_sqlite_index.py` (~5479 tok, huge) — Tests for sqlite_index.py — SQLite FTS5 index for mind-mem recall."""
- `test_stopwords.py` (~247 tok, medium) — Tests for stopword handling."""
- `test_storage_factory.py` (~1458 tok, large) — Tests for mind_mem.storage.get_block_store factory (v3.2.0)."""
- `test_streaming.py` (~1446 tok, large) — v3.3.0 — back-pressure-aware streaming ingest queue."""
- `test_telemetry.py` (~2829 tok, huge) — Tests for src/mind_mem/telemetry.py.
- `test_temporal_decay_scoring.py` (~863 tok, large) — v3.3.0 Tier 1 #3 — half-life decay on block ``Created``/``Date`` field.
- `test_temporal.py` (~223 tok, medium) — Tests for temporal filtering module."""
- `test_tenant_audit.py` (~1462 tok, large) — v4.0 prep — per-tenant audit chain façade."""
- `test_tenant_kms.py` (~1450 tok, large) — v4.0 prep — per-tenant KMS envelope encryption."""
- `test_tier_decay.py` (~924 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_tier_recall.py` (~1418 tok, large) — Tests for tier-aware recall boosting (v3.2.0 hot/cold tier wire-up)."""
- `test_tier_weights_config.py` (~784 tok, large) — v3.3.0 Tier 4 #10 — per-tier learned weights override.
- `test_tokenization.py` (~436 tok, medium) — Tests for tokenization module."""
- `test_token_rotation.py` (~1766 tok, huge) — Regression tests for the token rotation primitive (roadmap v4.0.x).
- `test_train_mind_mem_4b.py` (~962 tok, large) — Smoke tests for benchmarks/train_mind_mem_4b.py.
- `test_trajectory.py` (~2392 tok, huge) — Tests for trajectory.py — trajectory memory block operations."""
- `test_transcript_capture.py` (~3235 tok, huge) — Tests for transcript_capture.py — zero external deps (stdlib unittest)."""
- `test_truth_score.py` (~1520 tok, huge) — v3.3.0 — probabilistic truth score.
- `test_typed_edges_group_h.py` (~2870 tok, huge) — Tests for Group H typed-edge additions: supports, derived_from, edge_aware_boost.
- `test_uncertainty_propagation.py` (~2158 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_unicode_edge_cases.py` (~2440 tok, huge) — Tests for Unicode and edge case handling across mind-mem modules."""
- `test_v28_completion.py` (~4622 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v320_gaps.py` (~3257 tok, huge) — v3.2.0 gap tests — regression and edge-case coverage for new modules.
- `test_v34_features.py` (~2993 tok, huge) — Tests for v3.4.0 retrieval features.
- `test_v4_block_kinds.py` (~4112 tok, huge) — Tests for the v4 block-kind taxonomy module."""
- `test_v4_circuit_breaker.py` (~3770 tok, huge) — Tests for v4 circuit breaker (round 5 audit, Mistral + GLM 9.9→10)."""
- `test_v4_cognitive_kernel.py` (~2770 tok, huge) — Tests for the v4 Cognitive Mind Kernel registry + dispatcher."""
- `test_v4_concurrency.py` (~2325 tok, huge) — v4 concurrency / fuzz tests.
- `test_v4_federation_wire.py` (~2585 tok, huge) — Wire-transport tests for v4 federation.
- `test_v4_hnsw_and_consolidation.py` (~3243 tok, huge) — Tests for HNSW kind-filtered ANN + consolidation worker."""
- `test_v4_kernels.py` (~3855 tok, huge) — Tests for the v4 kernel strategy implementations.
- `test_v4_pq.py` (~2800 tok, huge) — Tests for v4 product-quantization (PQ) encoding."""
- `test_v4_round2_extensions.py` (~4983 tok, huge) — Tests for round-2 audit extensions: federation, embedding_pipeline,
- `test_v4_round3_extensions.py` (~2836 tok, huge) — Tests for round-3 audit extensions: observability + eviction."""
- `test_v4_round4_concurrency.py` (~8595 tok, huge) — Concurrency + adversarial-input tests for round-4 v4 modules.
- `test_v4_round4_extensions.py` (~5138 tok, huge) — Tests for round-4 audit extensions.
- `test_v4_surprise_retrieval.py` (~2281 tok, huge) — Tests for the v4 surprise-weighted retrieval scoring module."""
- `test_v4_tier_memory.py` (~4065 tok, huge) — Tests for the v4 recall-tier module.
- `test_validate_py.py` (~3438 tok, huge) — Tests for validate_py.py — workspace integrity validator."""
- `test_validate_sh_deprecation.py` (~547 tok, large) — Pin the runtime deprecation warning on validate.sh.
- `test_vault_wikilinks.py` (~1783 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_verify_cli.py` (~3202 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_walkthrough.py` (~2432 tok, huge) — Tests for the v3.9 dependency-ordered walkthrough."""
- `test_watcher.py` (~1217 tok, large) — Tests for watcher.py — file change detection for auto-reindex."""
- `test_wide_retrieval.py` (~346 tok, medium) — Tests for wide retrieval parameter."""
- `test_workspace_contextvar.py` (~1047 tok, large) — v3.2.1 — regression test for per-request workspace ContextVar scoping.
- `test_workspace_init.py` (~498 tok, medium) — Tests for workspace initialization."""
- `test_workspace_structure.py` (~546 tok, large) — Tests for workspace directory structure."""
### `train/`

- `audit_canonical_coverage.json` (~5300 tok, huge) — Keys: threshold, total_probes, total_weak, by_group, weak
- `audit_canonical_coverage.py` (~1479 tok, large) — Audit canonical-answer coverage on every eval probe.
- `audit_semantic_correctness.json` (~5 tok, tiny) — Keys: findings
- `audit_semantic_correctness.py` (~1963 tok, huge) — Cross-check every 'file X ships Y' claim in the corpus against src/.
- `backport_sweep.py` (~1661 tok, huge) — Backport v2.9.0 audit fixes to every prior v2.x release as .post1.
- `build_model_card.py` (~3690 tok, huge) — Generate the HuggingFace model-card README for mind-mem-4b.
- `CORPUS_HASH_v3.11.0` (~21 tok, tiny) — 02b3ba6a1433e25bdbefe3cebf992ca961734850d1e3550e9496905abbadb3b7  build_corpus.p
- `CORPUS_HASH_v3.12.0-fullft` (~21 tok, tiny) — 568d1559631a590e44eeec6716081b4534a40ab5f3047feb622cc225ead9ad01  build_corpus.p
- `eval_harness.py` (~8643 tok, huge) — Eval harness for mind-mem-4b.
- `eval_holdout.py` (~2182 tok, huge) — Held-out paraphrase eval set — runs AFTER training, BEFORE ship.
- `export_gguf.py` (~1274 tok, large) — Export the trained model to GGUF for Ollama / LM Studio / llama.cpp.
- `HF_MODEL_CARD_v4.md` (~2788 tok, huge) — mind-mem-4b v4.1.1
- `merge_and_eval_v4.1.0.py` (~1471 tok, large) — Post-Kaggle: pull LoRA adapter, merge with v4.0.0-base, eval 131 probes.
- `Modelfile.v3.9.0` (~389 tok, medium) — FROM /data/checkpoints/mm-workspace/train-output/mind-mem-4b-Q4_K_M.gguf
- `Modelfile.v4.0.0` (~576 tok, large) — FROM /data/checkpoints/mm-workspace/train-output/mind-mem-4b-Q4_K_M.gguf
- `Modelfile.v4.1.0` (~395 tok, medium) — FROM /data/checkpoints/mm-workspace/gguf-v4.1.0/mind-mem-4b-v4.1.0-Q4_K_M.gguf
- `Modelfile.v4.1.1` (~440 tok, medium) — FROM /data/checkpoints/mm-workspace/gguf-v4.1.1/mind-mem-4b-v4.1.1-Q4_K_M.gguf
- `post_train_chain.sh` (~632 tok, large) — Post-training chain: wait for deploy → verify scp + SHA256 + pod-destroy markers → run eval.
- `post_train_pipeline.sh` (~592 tok, large) — Post-training pipeline for mind-mem-4b v3.9.2 (augmented-corpus retrain).
- `qlora_local_3080.py` (~1182 tok, large) — Local QLoRA fallback on RTX 3080 (10GB VRAM).
- `README.md` (~569 tok, large) — mind-mem-4b training pipeline
- `resume_pod_train.sh` (~1138 tok, large) — Recovery: pod uz2uajluzskmm2 was preempted mid-run. Wake it up,
- `RETRAIN_v3.9.0.md` (~1405 tok, large) — mind-mem-4b — v3.9.0 retrain plan
- `runpod_deploy.py` (~5323 tok, huge) — End-to-end RunPod driver for full-FT on Qwen3.5-4B.
- `runpod_full_ft.py` (~2551 tok, huge) — Full fine-tune of Qwen3.5-4B on RunPod (A100/H100) for mind-mem-4b.
- `ship_gguf_ollama_v4.1.0.py` (~1741 tok, huge) — GGUF + Ollama shipper for mind-mem-4b v4.1.0.
- `ship_gguf_ollama_v4.1.1.py` (~1791 tok, huge) — GGUF + Ollama shipper for mind-mem-4b v4.1.1.
- `spend_guard.py` (~2036 tok, huge) — spend_guard — mechanical interlock on cloud spend.
- `train_qlora.py` (~1314 tok, large) — QLoRA fine-tune for mind-mem-4b on the harvested corpus.
- `upload_to_hf.py` (~1122 tok, large) — Push the retrained adapter + model card to star-ga/mind-mem-4b.
- `V4_RETRAIN_TODO.md` (~2361 tok, huge) — v4 Retrain — Probe Honesty TODO
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

- `api.ts` (~665 tok, large)
### `web/`

- `next.config.ts` (~104 tok, small) — mind-mem-web is a thin client — the REST API lives on the
- `package.json` (~193 tok, small) — Keys: name, version, private, description, license
- `README.md` (~464 tok, medium) — MIND-Mem web console
- `tsconfig.json` (~149 tok, small) — Keys: compilerOptions, include, exclude

---
*Generated by `anatomy 1.0.0`. Edit descriptions manually — re-run preserves structure.*

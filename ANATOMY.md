# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind-mem`
**Files:** 1536 | **Est. tokens:** ~4,558,436
**Generated:** 2026-09-11 23:23 UTC

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
| `./` | 36 | ~50,424 |
| `.agents/skills/mind-mem-development/` | 1 | ~456 |
| `.arch-mind/` | 7 | ~5,887 |
| `audits/` | 5 | ~24,039 |
| `benchmarks/` | 66 | ~181,940 |
| `benchmarks/repro/` | 1 | ~689 |
| `benchmarks/repro/niah/` | 4 | ~2,557 |
| `benchmarks/repro/niah-smoke/` | 5 | ~4,684 |
| `bin/` | 1 | ~526 |
| `deploy/` | 2 | ~772 |
| `deploy/docker/` | 1 | ~592 |
| `deploy/edge/` | 2 | ~1,149 |
| `deploy/grafana/` | 1 | ~1,145 |
| `docs/` | 93 | ~206,831 |
| `docs/adr/` | 2 | ~521 |
| `docs/advisories/` | 1 | ~834 |
| `docs/audit/` | 1 | ~4,973 |
| `docs/benchmarks/` | 25 | ~42,861 |
| `docs/benchmarks/ablation/` | 2 | ~12,181 |
| `docs/benchmarks/head-20260907/` | 8 | ~7,240 |
| `docs/benchmarks/memory-ab-420s-runs/` | 26 | ~84,421 |
| `docs/decisions/` | 1 | ~3,174 |
| `docs/design/` | 9 | ~16,095 |
| `docs/evidence/5.0.2-f1/` | 12 | ~188,585 |
| `docs/plans/` | 1 | ~6,134 |
| `docs/security-baselines/` | 1 | ~18,974 |
| `examples/` | 3 | ~1,203 |
| `.gemini/` | 1 | ~28 |
| `.githooks/` | 1 | ~98 |
| `.github/` | 9 | ~4,466 |
| `.github/ISSUE_TEMPLATE/` | 2 | ~179 |
| `.github/workflows/` | 11 | ~19,164 |
| `hooks/` | 3 | ~1,026 |
| `hooks/openclaw/mind-mem/` | 2 | ~1,211 |
| `intelligence/` | 1 | ~113 |
| `intelligence/state/snapshots/` | 1 | ~114 |
| `lib/` | 1 | ~2,170 |
| `mind/` | 27 | ~9,687 |
| `.roo/` | 1 | ~22 |
| `scripts/` | 22 | ~66,997 |
| `sdk/go/` | 10 | ~9,098 |
| `sdk/js/` | 6 | ~4,864 |
| `sdk/js/src/` | 5 | ~3,178 |
| `sdk/js/test/` | 1 | ~3,096 |
| `sdk/release/` | 3 | ~4,922 |
| `sdk/spec/` | 2 | ~5,702 |
| `security/` | 5 | ~24,696 |
| `skills/apply-proposal/` | 1 | ~345 |
| `skills/integrity-scan/` | 1 | ~376 |
| `skills/memory-recall/` | 1 | ~549 |
| `src/` | 1 | ~280 |
| `src/mind_mem/` | 232 | ~1,079,435 |
| `src/mind_mem/api/` | 5 | ~26,782 |
| `src/mind_mem/bench/` | 17 | ~48,632 |
| `src/mind_mem/compliance/` | 7 | ~13,934 |
| `src/mind_mem/importers/` | 9 | ~26,660 |
| `src/mind_mem/mcp/` | 3 | ~6,417 |
| `src/mind_mem/mcp/infra/` | 8 | ~12,671 |
| `src/mind_mem/mcp/tools/` | 29 | ~110,616 |
| `src/mind_mem/skill_opt/` | 11 | ~20,899 |
| `src/mind_mem/spec/` | 2 | ~2,005 |
| `src/mind_mem/storage/` | 2 | ~11,368 |
| `src/mind_mem/templates/` | 19 | ~1,041 |
| `src/mind_mem/tool_output/` | 3 | ~5,895 |
| `src/mind_mem/v4/` | 24 | ~92,798 |
| `tests/` | 657 | ~1,986,856 |
| `tests/fixtures/` | 7 | ~11,912 |
| `tests/fixtures/importers/` | 5 | ~1,218 |
| `tests/fixtures/importers/agent_memory/` | 4 | ~383 |
| `tests/fixtures/importers/agent_memory/memory/` | 1 | ~73 |
| `tests/fixtures/importers/vault/daily/` | 1 | ~31 |
| `tests/fixtures/importers/vault/notes/` | 2 | ~100 |
| `tests/fixtures/importers/vault/notes/incidents/` | 1 | ~75 |
| `tests/fixtures/importers/vault/.obsidian/` | 1 | ~7 |
| `tests/fixtures/importers/vault/templates/` | 1 | ~18 |
| `tests/integration/` | 2 | ~1,982 |
| `tests/red_team/` | 3 | ~806 |
| `tests/red_team/transcripts/` | 1 | ~0 |
| `train/` | 31 | ~58,107 |
| `web/` | 5 | ~927 |
| `web/app/` | 2 | ~1,204 |
| `web/app/console/` | 1 | ~1,169 |
| `web/components/` | 4 | ~2,482 |
| `web/lib/` | 1 | ~665 |

## Files

### `./`

- `AGENTS.md` (~995 tok, large) — mind-mem: agent instructions (auto-written)
- `AUDIT_FINDINGS_FOR_CLAUDE.md` (~995 tok, large) — Comprehensive Architectural Audit: MIND-Mem (Commit 30d8b71)
- `CLAUDE.md` (~4413 tok, huge) — MIND-Mem — Persistent AI Memory System
- `conftest.py` (~1010 tok, large) — Shared pytest fixtures for mind-mem test suite."""
- `conftest_trace.py` (~466 tok, medium) — Opt-in pytest plugin: find which test/product code leaks live sqlite3 handles.
- `CONTRIBUTING.md` (~753 tok, large) — Contributing to MIND-Mem
- `.cursorrules` (~23 tok, tiny) — # mind-mem
- `demo-setup.sh` (~323 tok, medium) — Pre-seed a demo workspace for VHS recording
- `demo.tape` (~93 tok, small) — # mind-mem demo — terminal recording for README
- `Dockerfile` (~521 tok, large) — FROM python:3.12-slim
- `.dockerignore` (~37 tok, tiny) — .git
- `.editorconfig` (~107 tok, small) — # EditorConfig — https://editorconfig.org
- `EVIDENCE.md` (~2166 tok, huge) — Evidence Matrix
- `generate_mind7b_training.py` (~5567 tok, huge) — Generate training data for Mind7B — a purpose-trained 7B model for mind-mem.
- `.gitattributes` (~243 tok, medium) — # Auto-detect text files and normalize line endings
- `.gitignore` (~525 tok, large) — *.pyc
- `.gitleaks.toml` (~314 tok, medium) — title = "mind-mem gitleaks config"
- `install-bootstrap.sh` (~1756 tok, huge) — mind-mem one-command bootstrap installer
- `install.sh` (~5444 tok, huge) — mind-mem installer — installs the package + wires MCP config for AI clients
- `LICENSE` (~2695 tok, huge)
- `Makefile` (~1097 tok, large) — .PHONY: test lint bench install dev clean smoke help regen-bash-literals
- `mcp_server.py` (~683 tok, large) — Source-checkout entrypoint for the packaged Mind-Mem MCP server.
- `mind-mem.example.json` (~203 tok, medium) — Keys: recall, prompts, categories, extraction, limits
- `.pre-commit-config.yaml` (~366 tok, medium) — repos:
- `pyproject.toml` (~3733 tok, huge) — [project]
- `.python-version` (~2 tok, tiny) — 3.12
- `requirements-optional.txt` (~1129 tok, large) — # mind-mem optional ML stack — pinned with SHA256 integrity hashes for
- `.run-ledger.jsonl` (~154 tok, small) — {"ended_at": "2026-05-11T03:10:20+00:00", "eval_summary": "127/131 (109 main + 1
- `SECURITY_AUDIT_2026-04.md` (~2403 tok, huge) — Security Audit — MIND-Mem v3.1.9 (April 2026)
- `SECURITY.md` (~2459 tok, huge) — Security Policy
- `setup.py` (~397 tok, medium) — Conditional setup hook for the optional Cython accelerator.
- `SPEC.md` (~6429 tok, huge) — Mind Mem Formal Specification v1.5.1
- `train_mind7b_runpod.py` (~1659 tok, huge)
- `.trivyignore` (~338 tok, medium) — # Trivy ignore file — DOCUMENTED, un-actionable pip-vendored findings only.
- `uninstall.sh` (~908 tok, large) — mind-mem uninstaller — removes MCP server entries from all configured clients
- `.windsurfrules` (~18 tok, tiny) — # mind-mem
### `.agents/skills/mind-mem-development/`

- `SKILL.md` (~456 tok, medium) — MIND-Mem Development
### `.arch-mind/`

- `baseline_2026-08-29.json` (~159 tok, small) — Keys: _aggregated_for_phase_a, _comment, _languages, _repo_root, edges
- `fixture.json` (~265 tok, medium) — Keys: _aggregated_for_phase_a, _comment, _languages, _repo_root, edges
- `last_summary.json` (~265 tok, medium) — Keys: _aggregated_for_phase_a, _comment, _languages, _repo_root, edges
- `rescan.py` (~2916 tok, huge) — # Copyright 2026 STARGA, Inc. — Apache-2.0 (see ../LICENSE).
- `rules.mind` (~1931 tok, huge) — mind-mem architectural-governance rules
- `scan.json` (~265 tok, medium) — Keys: _aggregated_for_phase_a, _comment, _languages, _repo_root, edges
- `scan_v3813.json` (~86 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
### `audits/`

- `2026-05-18-copilot-audit.md` (~1332 tok, large) — Copilot CLI audit — mind-mem (2026-05-18)
- `v3.11.0-integration-consensus-2026-05-08.json` (~4878 tok, huge) — Keys: audit_id, generated_at_utc, models_queried, models_parsed, fleet
- `v3.11-v3.12-corpus-final-audit-2026-05-09.json` (~6742 tok, huge) — Keys: audit_id, commit, audited_at, models_run, models_succeeded
- `v3.12-corpus-final-audit-2026-05-09.json` (~3802 tok, huge) — Keys: audit_id, generated_at_utc, models_queried, models_parsed, fleet
- `v4.0.1-claude-2026-05-12.md` (~7285 tok, huge) — mind-mem v4.0.1 — Multi-Source Audit (cross-model review + arch-mind + agents)
### `benchmarks/`

- `ablation_mask.py` (~3414 tok, huge) — # Copyright 2026 STARGA, Inc.
- `bench_kernels.py` (~4027 tok, huge) — Benchmark: MIND kernels vs pure Python scoring.
- `cache_effectiveness.py` (~2717 tok, huge) — Cache-effectiveness benchmark — Redis L2 vs LRU-only vs no-cache.
- `cache_effectiveness_v3.2.1.json` (~227 tok, medium) — Keys: n_blocks, n_queries, pool_size, repeat_pct, runs
- `CACHE.md` (~1028 tok, large) — Recall Cache Effectiveness Benchmark
- `_capfix_probe.py` (~1032 tok, large) — Isolation probe: quantify the recall.dedup type_cap=3 ceiling.
- `_ch_minilm.py` (~2045 tok, huge) — LongMemEval-S — per-turn chunking + hybrid (BM25F+mxbai RRF) + 4b expansion.
- `_ch_minilm_spawn.py` (~6904 tok, huge) — LongMemEval-S — mind-mem FULL POTENTIAL harness (same-equipment, best-vs-best).
- `compare_runs.py` (~3022 tok, huge) — Compare two benchmark runs -- unpaired for LoCoMo, paired for ranking.
- `crossencoder_ab.py` (~3214 tok, huge) — Cross-Encoder A/B Test — retrieval-level comparison.
- `embed_augmentation_ab.py` (~1416 tok, large) — M1 — does `_augment_for_embedding` help or hurt? Measure it.
- `expansion_reentrancy_identity.py` (~2677 tok, huge) — # Copyright 2026 STARGA, Inc.
- `f1_evidence_report.py` (~2271 tok, huge) — # Copyright 2026 STARGA, Inc.
- `f1_score_contract_probe.py` (~2954 tok, huge) — # Copyright 2026 STARGA, Inc.
- `f1_session_scorecard_probe.py` (~2982 tok, huge) — # Copyright 2026 STARGA, Inc.
- `feedback_success_bench.py` (~4449 tok, huge) — Feedback-quality -> downstream-success bench (Group I item 3).
- `feedback_success_results.json` (~230 tok, medium) — Keys: accuracy, f1, mean_sufficiency_failure, mean_sufficiency_success, n_episodes
- `generate_dispatcher_examples.py` (~2346 tok, huge) — Generate synthetic training examples for the v3.2.x 7-dispatcher MCP surface.
- `generate_retrieval_examples.py` (~1686 tok, huge) — Generate training examples for v3.3.0 retrieval shapes.
- `grid_search.py` (~2849 tok, huge) — BM25F Field Weight Grid Search for mind-mem Recall Engine.
- `hard_timeout.py` (~2768 tok, huge) — Per-unit timeouts that actually preempt, for long benchmark runs.
- `__init__.py` (~0 tok, tiny)
- `integrity_benchmark.py` (~2833 tok, huge) — # Copyright 2026 STARGA, Inc.
- `lme_ablation_report.py` (~1300 tok, large) — # Copyright 2026 STARGA, Inc.
- `local_stack_audit.py` (~1822 tok, huge) — Single-shot audit of the local mind-mem stack before a bench run.
- `locomo_harness.py` (~4147 tok, huge) — LoCoMo Benchmark Harness for mind-mem Recall Engine.
- `locomo_judge.py` (~17205 tok, huge) — LoCoMo LLM-as-Judge Evaluation for Mind-Mem.
- `locomo_suite.py` (~4818 tok, huge) — # Relocated out of the wheel in 5.0.0: this is a benchmark entry-point
- `locomo_v3.3.0_benchmark_config.json` (~450 tok, medium) — Keys: _comment, version, recall, cache, cross_encoder
- `longmemeval_apples.py` (~1674 tok, huge) — LongMemEval-S apples-to-apples harness (mind-mem hybrid BM25+vector).
- `longmemeval_chunk_bm25_4b.py` (~1619 tok, huge) — Per-turn chunking + BM25 + mind-mem:4b multi-query expansion.
- `longmemeval_chunked_hybrid.py` (~2068 tok, huge) — LongMemEval-S — per-turn chunking + hybrid (BM25F+mxbai RRF) + 4b expansion.
- `longmemeval_chunked_minilm.py` (~6564 tok, huge) — LongMemEval-S — mind-mem FULL POTENTIAL harness (same-equipment, best-vs-best).
- `longmemeval_chunked.py` (~1235 tok, large) — LongMemEval-S — per-turn passage chunking (architectural gap closure).
- `longmemeval_clean.py` (~1411 tok, large) — LongMemEval-S — clean reproducible mind-mem retrieval benchmark.
- `LONGMEMEVAL_FINDINGS_2026-05-19.md` (~2123 tok, huge) — LongMemEval-S Benchmark Audit — Findings (2026-05-19)
- `longmemeval_fullpotential.py` (~6291 tok, huge) — LongMemEval-S — mind-mem FULL POTENTIAL harness (same-equipment, best-vs-best).
- `longmemeval_full_run.py` (~4953 tok, huge) — Full-corpus LongMemEval-S driver: subprocess-isolated, resumable.
- `longmemeval_harness.py` (~2973 tok, huge) — LongMemEval Benchmark Harness for mind-mem recall engine.
- `longmemeval_hybrid4b.py` (~1823 tok, huge) — LongMemEval-S — mind-mem hybrid (BM25F+mxbai RRF) + 4b query expansion.
- `longmemeval_real_harness.py` (~1656 tok, huge) — LongMemEval-S harness — real product pipeline (Phase A: own best honest number).
- `make_public_integrity_workspace.py` (~1084 tok, large) — # Copyright 2026 STARGA, Inc.
- `memory_ab_analysis.py` (~3471 tok, huge) — Reduce repeated memory-A/B runs to one paired number, plus the counter-metric.
- `memory_ab_bench.py` (~342 tok, medium) — With-memory versus without-memory, on this repository's own history.
- `memory_ab_placebo.py` (~1754 tok, huge) — The placebo arm: same shape, same length, same framing — wrong corpus.
- `memory_ab_stratum.py` (~1285 tok, large) — Run a memory A/B stratum one task at a time, resumably.
- `niah_full_results.txt` (~5134 tok, huge) — ============================= test session starts ==============================
- `NIAH.md` (~1995 tok, huge) — Needle In A Haystack (NIAH) Benchmark
- `niah_v3.2.1_redis_results.txt` (~111 tok, small) — ============================= test session starts ==============================
- `niah_v3.2.1_results.txt` (~203 tok, medium) — ============================= test session starts ==============================
- `paired_scorecard.py` (~5476 tok, huge) — # Copyright 2026 STARGA, Inc.
- `ranking_identity.py` (~2910 tok, huge) — # Copyright 2026 STARGA, Inc.
- `README_benchmark_mode.md` (~1106 tok, large) — Full-capability benchmark mode (v3.3.0)
- `recompaction_bench.py` (~6760 tok, huge) — # Relocated out of the wheel in 5.0.0: this is a benchmark entry-point
- `REPORT.md` (~5281 tok, huge) — MIND-Mem Benchmark Report
- `repro_manifest.py` (~2145 tok, huge) — Write a repro package: raw rows, recomputed metrics, and a manifest that pins
- `repro_metrics.py` (~2213 tok, huge) — Recompute every headline metric from RAW per-unit rows, and nothing else.
### `benchmarks/repro/niah/`

- `dataset.json` (~792 tok, large) — Keys: content_sha256, depth_percentages, generated_by, generator_sha256, haystack_sizes
- `environment.json` (~95 tok, small) — Keys: captured_utc, cpu_count, machine, mind_mem_version, packages
- `manifest.json` (~1429 tok, large) — Keys: artifacts, benchmark, commands, headline, headline_claim
- `metrics.json` (~241 tok, medium) — Keys: benchmark, breakdown, determinism, headline, integrity
### `benchmarks/`

- `repro_niah.py` (~3976 tok, huge) — Reproducible NIAH benchmark harness -- emits an independently-verifiable package.
### `benchmarks/repro/niah-smoke/`

- `dataset.json` (~792 tok, large) — Keys: content_sha256, depth_percentages, generated_by, generator_sha256, haystack_sizes
- `environment.json` (~95 tok, small) — Keys: captured_utc, cpu_count, machine, mind_mem_version, packages
- `manifest.json` (~1438 tok, large) — Keys: artifacts, benchmark, commands, headline, headline_claim
- `metrics.json` (~234 tok, medium) — Keys: benchmark, breakdown, determinism, headline, integrity
- `raw.ndjson` (~2125 tok, huge) — {"depth_pct":100,"expected_keywords":["42.7","Z-Prime"],"found":true,"haystack_s
### `benchmarks/repro/`

- `README.md` (~689 tok, large) — Repro packages
### `benchmarks/`

- `repro_verify.py` (~4137 tok, huge) — Verify that a published number can be recomputed from its committed evidence.
- `runpod_kickoff.sh` (~1779 tok, huge) — mind-mem-4b v2 — Runpod one-shot kickoff.
- `STATUS.md` (~3163 tok, huge) — Benchmark Status
- `strat_probe.py` (~936 tok, large) — Stratified LongMemEval probe.
- `tier_weight_search.py` (~1615 tok, huge) — Grid-search per-tier weights against LoCoMo judge scores (v3.3.0 T4 #10).
- `train_config_a100.yaml` (~347 tok, medium) — base_model: star-ga/mind-mem-4b
- `train_config.yaml` (~208 tok, medium) — base_model: star-ga/mind-mem-4b
- `train_mind_mem_4b.py` (~3286 tok, huge) — mind-mem-4b v2 training script — Runpod H200 full-fine-tune.
### `bin/`

- `mm-run` (~526 tok, large) — #!/usr/bin/env bash
### `deploy/`

- `docker-compose.yml` (~690 tok, large) — name: mind-mem
### `deploy/docker/`

- `Dockerfile` (~592 tok, large) — # Stage 1: build — install all deps and produce a pruned site-packages
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
### `docs/advisories/`

- `2026-09-05-recall-reentrant-fanout-dos.md` (~834 tok, large) — Advisory (DRAFT — not published): unbounded thread fan-out in auto-enabled recall
### `docs/`

- `agent-comm.md` (~1251 tok, large) — Agent-to-agent messaging (`mm send` / `mm inbox`)
- `AGENTIC-MEMORY-SOTA.md` (~1674 tok, huge) — mind-mem → SOTA Agentic Memory (v4.5.0+ design brief)
- `agent-memory-protocol.md` (~700 tok, large) — Agent Memory Protocol — canonical system-prompt snippet
- `api-reference.md` (~2673 tok, huge) — API Reference
- `append-only-audit-logs.md` (~1626 tok, huge) — Append-Only Audit Logs — Operator Runbook
- `architecture.md` (~3226 tok, huge) — Architecture
### `docs/audit/`

- `GROUP-R-AUDIT-2026-08-28.md` (~4973 tok, huge) — Group R — Independent Architecture Audit
### `docs/`

- `audit_response.md` (~950 tok, large) — MIND-Mem — response to the 2026-05-02 ecosystem audit
### `docs/benchmarks/`

- `2026-07-30-longmemeval-s-bm25_baseline.md` (~646 tok, large) — LongMemEval-S scorecard — `bm25_baseline` (2026-07-30)
- `2026-07-30-longmemeval-s-bm25_baseline.ndjson` (~3322 tok, huge) — {"adapter": "bm25_baseline", "first_gold_rank": 1, "hit": true, "latency_ms": 2.
- `2026-07-30-longmemeval-s-mind_mem.md` (~643 tok, large) — LongMemEval-S scorecard — `mind_mem` (2026-07-30)
- `2026-07-30-longmemeval-s-mind_mem.ndjson` (~3078 tok, huge) — {"adapter": "mind_mem", "first_gold_rank": 1, "hit": true, "latency_ms": 303.282
- `2026-07-31-locomo-baseline-bm25_baseline.md` (~663 tok, large) — LoCoMo recall scorecard — `bm25_baseline` (2026-07-31)
- `2026-07-31-locomo-baseline-mind_mem.md` (~643 tok, large) — LoCoMo recall scorecard — `mind_mem` (2026-07-31)
- `2026-09-03-longmemeval-s-full-bm25_baseline-rep1.md` (~838 tok, large) — LongMemEval-S scorecard — `bm25_baseline` (2026-09-03)
- `2026-09-03-longmemeval-s-full-bm25_baseline-rep2.md` (~839 tok, large) — LongMemEval-S scorecard — `bm25_baseline` (2026-09-03)
- `2026-09-03-longmemeval-s-full-mind_mem-rep1.md` (~817 tok, large) — LongMemEval-S scorecard — `mind_mem` (2026-09-03)
- `2026-09-03-longmemeval-s-full-mind_mem-rep2.md` (~817 tok, large) — LongMemEval-S scorecard — `mind_mem` (2026-09-03)
- `2026-09-04-lme-ablation-PREREGISTRATION.md` (~1114 tok, large) — F5 — SQLite ranking-stack ablation on LongMemEval-S: pre-registration
- `2026-09-04-memory-ab-420s-INTERIM.json` (~1695 tok, huge) — Keys: schema_version, inputs, harness, prompt_lengths, reduction
- `2026-09-05-lme-ablation-ADDENDUM-facts.md` (~850 tok, large) — F5 addendum — the FACT sub-block layer, added post-hoc, pre-committed
- `2026-09-05-lme-ablation-RESULTS.md` (~3527 tok, huge) — F5 results — ablating the SQLite ranking stack on LongMemEval-S
- `2026-09-06-dialogue-diversity-cap0.ndjson` (~3862 tok, huge) — {"all": true, "any": true, "distinct": 2, "n_gold": 2, "qid": "6a1eabeb", "qtype
- `2026-09-06-dialogue-diversity-cap1.ndjson` (~3666 tok, huge) — {"all": true, "any": true, "distinct": 5, "n_gold": 2, "qid": "6a1eabeb", "qtype
- `2026-09-06-dialogue-diversity-cap2.ndjson` (~3774 tok, huge) — {"all": true, "any": true, "distinct": 3, "n_gold": 2, "qid": "6a1eabeb", "qtype
- `2026-09-06-stemming-union-cap1.ndjson` (~3682 tok, huge) — {"all": true, "any": true, "distinct": 5, "n_gold": 2, "qid": "6a1eabeb", "qtype
- `2026-09-07-external-chroma.md` (~1035 tok, large) — An external system under the same contract — and what it actually shows
- `2026-09-07-hybrid-vector-on.md` (~1001 tok, large) — Vector-on hybrid on the full set — `all@5` beats the floor, and 36 runs did not get a dense leg
- `2026-09-07-longmemeval-s-HEAD.md` (~1088 tok, large) — LongMemEval-S on HEAD — the 09-03 deficit is gone
- `2026-09-07-no-expansion-gate.md` (~940 tok, large) — `no_expansion` on the full set — the ordering gap closes, at no recall cost
- `2026-09-08-class-a-public-integrity.md` (~701 tok, large) — Class A integrity — public synthetic-workspace evidence
### `docs/benchmarks/ablation/`

- `paired-scorecards.json` (~10689 tok, huge) — Keys: control_path, floor_by_type, floor_headline, floor_path, k
- `paired-scorecards-posthoc.json` (~1492 tok, large) — Keys: no_facts, no_facts_plain
### `docs/benchmarks/head-20260907/`

- `hybrid-config.json` (~71 tok, small) — Keys: recall
- `integrity-scorecard.json` (~324 tok, medium) — Keys: dimensions, workspace
- `lme-chroma.md` (~1357 tok, large) — LongMemEval-S scorecard — `chroma_baseline` (2026-09-07)
- `lme-floor-head.md` (~1039 tok, large) — LongMemEval-S scorecard — `bm25_baseline` (2026-09-07)
- `lme-hybrid-vecon.md` (~998 tok, large) — LongMemEval-S scorecard — `mind_mem` (2026-09-07)
- `lme-mind_mem-head.md` (~1016 tok, large) — LongMemEval-S scorecard — `mind_mem` (2026-09-07)
- `lme-no_expansion.md` (~1015 tok, large) — LongMemEval-S scorecard — `mind_mem` (2026-09-07)
- `report.json` (~1420 tok, large) — Keys: control_path, control_path_status, floor_by_type, floor_headline, floor_path
### `docs/`

- `benchmarks.md` (~1746 tok, huge) — Benchmarks
### `docs/benchmarks/memory-ab-420s-runs/`

- `MANIFEST.json` (~2906 tok, huge) — Keys: count, executable_binding, originals, receipts, what
- `README.md` (~464 tok, medium) — A/B run receipts — path-sanitized public copies
- `single_file__mm-00bb4eeb3eb0__rep1.json` (~3248 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-1ddc953ce424__rep1.json` (~3535 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-27c60f9b55ad__rep1.json` (~3450 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-28203bdf502e__rep1.json` (~3231 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-39b49f88e7ab__rep1.json` (~3583 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-3b811e71490d__rep1.json` (~2890 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-5d66147a3d1d__rep1.json` (~3159 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-605384718d16__rep1.json` (~2980 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-66b03be1ef24__rep1.json` (~3653 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-7571daeac51c__rep1.json` (~2758 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-78c09fbe2c2f__rep1.json` (~2843 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-8009be3fbd79__rep1.json` (~3202 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-81142206d8ac__rep1.json` (~2950 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-8ac223858f56__rep1.json` (~3262 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-9956b7aba6cf__rep1.json` (~3172 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-a63d572ddb0d__rep1.json` (~3314 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-d048b5565cd9__rep1.json` (~3726 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-d59f9832a452__rep1.json` (~3598 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-db4e90b3bf73__rep1.json` (~3117 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-e79c2b1f94b3__rep1.json` (~5667 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-e94acc7bb86c__rep1.json` (~3568 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-e98c1449bce7__rep1.json` (~3547 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-f64409645dfa__rep1.json` (~3226 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
- `single_file__mm-fa0b654ca566__rep1.json` (~3372 tok, huge) — Keys: schema_version, harness, budget, agent, task_set
### `docs/benchmarks/`

- `SOTA-GATE.md` (~1680 tok, huge) — The SOTA gate — what is required, and what is measured
- `SOTA-NORTH-STAR.md` (~1940 tok, huge) — 100% across every dimension — what that means per dimension, measured
### `docs/`

- `block-format.md` (~431 tok, medium) — Block Format
- `block-type-taxonomy-roadmap.md` (~911 tok, large) — Block-Type Taxonomy Enhancement — Roadmap Note
- `changelog-format.md` (~217 tok, medium) — Changelog Format Guide
- `ci-workflows.md` (~728 tok, large) — CI Workflows
- `claude-desktop-setup.md` (~764 tok, large) — Claude Desktop Setup Guide
- `client-integrations.md` (~3635 tok, huge) — Client Integrations
- `cli-reference.md` (~7063 tok, huge) — CLI Reference
- `companion-tools.md` (~1113 tok, large) — Companion Tools
- `comparison.md` (~599 tok, large) — Comparison with Alternatives
- `competitive-analysis-persistent-memory-2026.md` (~4089 tok, huge) — Comprehensive Competitive Analysis: Persistent Memory Systems for AI Coding Agents (2025–2026)
- `configuration.md` (~15987 tok, huge) — Configuration Reference
### `docs/decisions/`

- `author-identity.md` (~3174 tok, huge) — Commit author identity — policy, measured state, and the open decision
### `docs/design/`

- `eval-set-ground-truth.md` (~1740 tok, huge) — Design: the ground-truth eval set — the shared blocker for L1 and M7
- `m1-embedded-field-vocabulary.md` (~2143 tok, huge) — Design: M1 — the embedded field must speak the query's vocabulary
- `m2-m3-namespace-retrieval-properties.md` (~1835 tok, huge) — Design: M2/M3 — namespace retrieval properties, asserted and tuned
- `m4-closed-set-slots.md` (~2218 tok, huge) — Design: M4 — closed-set slots, structural contradiction prevention
- `m5-enforcement-in-code-audit.md` (~1933 tok, huge) — Design: M5 — enforcement in code, not in the prompt
- `m6-negative-results.md` (~2171 tok, huge) — Design: M6 — negative results as a first-class recorded outcome
- `recall-harness.md` (~1639 tok, huge) — Design: the recall harness — deterministic working memory for search agents
- `v3-mcp-surface-reduction.md` (~1080 tok, large) — v3.0 Design: MCP Tool Surface Reduction
- `v3-multi-tenancy.md` (~1336 tok, large) — v3.0 Design: Multi-Tenancy Foundation
### `docs/`

- `development.md` (~359 tok, medium) — Development Guide
- `docker-deployment.md` (~571 tok, large) — Docker Deployment
### `docs/evidence/5.0.2-f1/`

- `score-contract-after-5.0.2.json` (~21977 tok, huge) — Keys: cases, provenance
- `score-contract-before-6cd37e5.json` (~25044 tok, huge) — Keys: cases, provenance
- `score-contract-report.json` (~1400 tok, large) — Keys: after, before, cases, default_path, monotonicity_after
- `session-scorecard-after-5.0.2-sn000.ndjson` (~23048 tok, huge) — {"first_gold_rank": 1, "n_gold": 3, "n_served": 10, "question_id": "Q0000", "rec
- `session-scorecard-after-5.0.2-sn060.ndjson` (~23056 tok, huge) — {"first_gold_rank": 1, "n_gold": 3, "n_served": 10, "question_id": "Q0000", "rec
- `session-scorecard-after-5.0.2-sn085.ndjson` (~22953 tok, huge) — {"first_gold_rank": 1, "n_gold": 3, "n_served": 10, "question_id": "Q0000", "rec
- `session-scorecard-before-6cd37e5-sn000.ndjson` (~23054 tok, huge) — {"first_gold_rank": 1, "n_gold": 3, "n_served": 10, "question_id": "Q0000", "rec
- `session-scorecard-before-6cd37e5-sn060.ndjson` (~23067 tok, huge) — {"first_gold_rank": 1, "n_gold": 3, "n_served": 10, "question_id": "Q0000", "rec
- `session-scorecard-before-6cd37e5-sn085.ndjson` (~22965 tok, huge) — {"first_gold_rank": 1, "n_gold": 3, "n_served": 10, "question_id": "Q0000", "rec
- `session-scorecard-sn000.json` (~678 tok, large) — Keys: baseline_label, baseline_path, candidate_label, candidate_path, comparisons
- `session-scorecard-sn060.json` (~669 tok, large) — Keys: baseline_label, baseline_path, candidate_label, candidate_path, comparisons
- `session-scorecard-sn085.json` (~674 tok, large) — Keys: baseline_label, baseline_path, candidate_label, candidate_path, comparisons
### `docs/`

- `faq.md` (~383 tok, medium) — FAQ
- `federation-setup.md` (~2332 tok, huge) — mind-mem federation & multi-machine setup
- `getting-started.md` (~493 tok, medium) — Getting Started
- `glossary.md` (~263 tok, medium) — Glossary
- `governance.md` (~1374 tok, large) — MIND-Mem — governance design (5 layers)
- `GOVERNED_WRITES.md` (~2664 tok, huge) — Governed writes
- `guardrails.md` (~1920 tok, huge) — GUARDRAIL blocks
- `hf-mind-mem-4b-v2-README.md` (~2426 tok, huge) — mind-mem-4b v2 (2026-04-21)
- `HYPEREDGE_DESIGN_2026-06-17.md` (~1350 tok, large) — Hyperedge + temporal-anchor design (Hyper-Extract steal)
- `install-guide.md` (~2855 tok, huge) — Installation guide — every step + every option
- `integrations.md` (~1540 tok, huge) — Integrations
- `locomo-v3.4-conv0-results.md` (~475 tok, medium) — LoCoMo v3.4.0 conv-0 results (2026-04-22)
- `maintenance-namespaces.md` (~1625 tok, huge) — `maintenance/` namespaces
- `mcp-integration.md` (~1770 tok, huge) — MCP Integration Guide
- `mcp-tool-examples.md` (~902 tok, large) — MCP Tool Examples
- `MHS_DEVICE_MEMORY.md` (~524 tok, large) — MHS / Device Memory Boundary
- `mic-map.md` (~1686 tok, huge) — MIC/MAP — MIND IR Graph Serialization
- `migration-guide.md` (~421 tok, medium) — Migration Guide
- `migration.md` (~2761 tok, huge) — Migration Guide: mem-os to MIND-Mem
- `MIND_CONFIG_VS_MIND_LANG.md` (~2540 tok, huge) — MIND configuration vs MIND language — clarifying the .mind extension
- `mind-kernels.md` (~339 tok, medium) — MIND Kernels
- `mind-mem-4b-setup.md` (~3193 tok, huge) — Setting up the mind-mem-4b model
- `mind-mem-4b-training-runbook.md` (~3586 tok, huge) — mind-mem-4b training runbook (post-v3.10.2 lessons)
- `mind-mem-4b-v2-training-recipe.md` (~1683 tok, huge) — mind-mem-4b v2 training recipe — Runpod H200
- `odc-retrieval.md` (~834 tok, large) — Observer-Dependent Cognition in MIND-Mem
- `performance-tuning.md` (~3162 tok, huge) — Performance Tuning
### `docs/plans/`

- `RESTORE-44-WIRING-PLAN.md` (~6134 tok, huge) — Restore-44 Wiring Plan — from restored to reachable-and-working
### `docs/`

- `postgres-parity-audit-2026-06-14.md` (~5466 tok, huge) — mind-mem — Postgres/SQLite backend parity audit (2026-06-14)
- `POST-V4.4.0-ROADMAP-PLAN.md` (~1905 tok, huge) — mind-mem — Post-v4.4.0 Roadmap Plan (reference)
- `protection.md` (~1443 tok, large) — MIND-Mem Library Protection
- `quality-gate.md` (~1267 tok, large) — Quality Gate — Operator Runbook
- `quickstart.md` (~602 tok, large) — MIND-Mem Quickstart
- `recompaction.md` (~3132 tok, huge) — Iterative Re-Compression Engine (Recompaction)
- `red-team-audit.md` (~1164 tok, large) — Behavioral Audit — Operator Runbook
- `rest-api.md` (~1579 tok, huge) — MIND-Mem REST API
- `review-architecture-v3.2.0.md` (~1919 tok, huge) — Architecture Review — MIND-Mem v3.2.0 (Release Candidate)
- `review-database-v3.2.0.md` (~3171 tok, huge) — Database Review — PostgresBlockStore v3.2.0
- `review-docs-v3.2.0.md` (~1957 tok, huge) — Documentation Review — MIND-Mem v3.2.0
- `review.md` (~1647 tok, huge) — `mm review` — batch approval for the HITL queue
- `review-tests-v3.2.0.md` (~1300 tok, large) — Test Review — MIND-Mem v3.2.0
- `roadmap.md` (~14073 tok, huge) — Roadmap
- `ROADMAP-RETRIEVAL-ACCOUNTABILITY.md` (~2736 tok, huge) — Group R — Retrieval Accountability (memory as rent, not storage)
- `roadmap-v4.md` (~11235 tok, huge) — mind-mem v4.0 — Design Rationale
- `scoring.md` (~517 tok, large) — Scoring System
- `SECURITY_AUDIT_SELF_2026_04.md` (~2267 tok, huge) — MIND-Mem v3.2.0 — Self-Audit Plan (Post-Release Deliverable)
- `security-audit-sow.md` (~3353 tok, huge) — MIND-Mem — External Security Audit Statement of Work (SoW)
### `docs/security-baselines/`

- `bandit-v3.2.0-baseline.json` (~18974 tok, huge) — Keys: errors, generated_at, metrics, results
### `docs/`

- `security-model.md` (~823 tok, large) — Security Model
- `setup.md` (~1870 tok, huge) — Setup
- `SOTA_GAP_RULING_2026-09-03.md` (~8053 tok, huge) — mind-mem SOTA gap ruling — architecture seat, 2026-09-03
- `status.md` (~1335 tok, large) — MIND-Mem — implementation status (alignment companion)
- `storage-backends.md` (~1620 tok, huge) — Storage Backends
- `storage-migration.md` (~2391 tok, huge) — Storage Backend Migration Guide
- `supply-chain-security.md` (~1051 tok, large) — Supply-Chain Security
- `task-frames.md` (~2947 tok, huge) — Task Frames & the Dead-End Registry
- `testing-guide.md` (~382 tok, medium) — Testing Guide
- `tool-output-architecture.md` (~2067 tok, huge) — Tool-output offload — architecture
- `trajectory-memory.md` (~1346 tok, large) — Trajectory memory
- `troubleshooting.md` (~809 tok, large) — Troubleshooting
- `usage.md` (~2653 tok, huge) — Usage
- `v3.11.0-implementation-plan.md` (~1609 tok, huge) — v3.11.0 Implementation Plan — synthesis from cross-model review
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
- `v4-release.md` (~6313 tok, huge) — v4.0.0 Release Notes
- `workspace-structure.md` (~352 tok, medium) — Workspace Structure
### `examples/`

- `basic_usage.py` (~399 tok, medium) — Basic mind-mem usage example.
- `mic_map_quickstart.py` (~732 tok, large) — MIC/MAP quickstart — emit, parse, round-trip, stream.
- `README.md` (~72 tok, small) — MIND-Mem Examples
### `.gemini/`

- `settings.json` (~28 tok, tiny) — Keys: system_instruction
### `.githooks/`

- `pre-commit` (~98 tok, small) — #!/usr/bin/env bash
### `.github/`

- `CODEOWNERS` (~25 tok, tiny) — # Default owners
- `copilot-instructions.md` (~71 tok, small) — mind-mem: GitHub Copilot workspace instructions
- `dependabot.yml` (~289 tok, medium) — version: 2
- `FUNDING.yml` (~4 tok, tiny) — github: star-ga
### `.github/ISSUE_TEMPLATE/`

- `bug_report.md` (~78 tok, small) — Description
- `feature_request.md` (~101 tok, small) — Description
### `.github/`

- `labels.yml` (~216 tok, medium)
- `mlc_config.json` (~55 tok, small) — Keys: ignorePatterns, timeout, retryOn429, aliveStatusCodes
- `pilot-issues.md` (~3595 tok, huge) — Pilot Week Issues (Feb 19-25)
- `pull_request_template.md` (~87 tok, small) — Summary
- `SECURITY_CONTACTS.md` (~124 tok, small) — Security Contacts
### `.github/workflows/`

- `audit-pinned.yml` (~412 tok, medium) — name: Audit Pinned Models
- `benchmark.yml` (~761 tok, large) — name: Benchmark
- `ci.yml` (~7479 tok, huge) — name: CI
- `codeql.yml` (~225 tok, medium) — name: CodeQL
- `dependency-review.yml` (~114 tok, small) — name: Dependency Review
- `docs.yml` (~262 tok, medium) — name: Docs
- `label-sync.yml` (~112 tok, small) — name: Label Sync
- `red-team.yml` (~385 tok, medium) — name: Red Team Audit
- `release.yml` (~6318 tok, huge) — name: Release
- `security.yml` (~2854 tok, huge) — name: Supply-Chain Security
- `stale.yml` (~242 tok, medium) — name: Stale Issues
### `hooks/`

- `hooks.json` (~79 tok, small) — Keys: hooks
### `hooks/openclaw/mind-mem/`

- `handler.js` (~941 tok, large) — Resolve MIND_MEM_WORKSPACE from hook config env, process env, or default
- `HOOK.md` (~270 tok, medium) — Mind Mem
### `hooks/`

- `session-end.sh` (~493 tok, medium) — mind-mem Stop hook — runs auto-capture if enabled
- `session-start.sh` (~454 tok, medium) — mind-mem SessionStart hook — prints health summary for context injection
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

- `alignment_authorities.py` (~8062 tok, huge) — Where each counted doc claim gets its TRUE value from.
- `anatomy-hook.sh` (~237 tok, medium) — anatomy-hook.sh — Git pre-commit hook to refresh ANATOMY.md
- `anatomy.sh` (~2248 tok, huge) — anatomy — Generate ANATOMY.md for any repo
- `bandit_gate.py` (~3203 tok, huge) — # Copyright 2026 STARGA, Inc.
- `build_integrity_manifest.py` (~634 tok, large) — Bake ``_integrity_manifest.json`` into the package before wheel build.
- `check_author_identity.sh` (~1880 tok, huge) — check_author_identity.sh — enforce the single-author identity rule.
- `check_ci_green.py` (~5240 tok, huge) — Release gate: CI must have concluded success for the EXACT commit being released.
- `check_claims.sh` (~385 tok, medium) — Cross-repo docs-claim regression gate (mind-mem side).
- `check_code_scanning_alerts.py` (~5185 tok, huge) — Release gate: zero open code-scanning alerts, AND proof the scanner actually ran.
- `check_docs_alignment.py` (~15109 tok, huge) — Recompute every counted doc claim from its authority and fail on drift.
- `check_evidence_executes.py` (~2249 tok, huge) — Gate: every EVIDENCE.md claim's test must EXECUTE. A skip is a red build.
- `check_index_absence.py` (~2246 tok, huge) — Release gate: the version being released must not exist on the index in ANY state.
- `check_reachable_modules.py` (~5034 tok, huge) — # Copyright 2026 STARGA, Inc.
- `check_roadmap_ticks.py` (~4387 tok, huge) — # Copyright 2026 STARGA, Inc.
- `check_tool_surface.py` (~2214 tok, huge) — Reachability, applied to the MCP tool surface.
- `count_mcp_tools.py` (~5216 tok, huge) — Count registered MCP tools and assert the count matches CLAUDE.md.
- `docs-alignment-hook.sh` (~454 tok, medium) — docs-alignment-hook.sh — Git pre-commit step to refresh derived doc counts.
- `pre-commit-hook.sh` (~489 tok, medium) — STARGA author guard (chained first: a wrong-identity commit must never be created).
- `pre-push-hook.sh` (~460 tok, medium) — pre-push-hook.sh — the last LOCAL gate before an identity becomes public.
- `reachability_baseline.txt` (~366 tok, medium) — api.grpc_server  # waiting: a named client integration that requires gRPC (strea
- `regen_bash_literals.py` (~424 tok, medium) — Regenerate src/mind_mem/_task_status_literals.sh from enums.py.
- `require_named_controls.py` (~1275 tok, large) — # Copyright 2026 STARGA, Inc.
### `sdk/go/`

- `client.go` (~1194 tok, large) — Option is a functional option for NewClient.
- `client_test.go` (~3904 tok, huge) — Helpers
- `doc.go` (~334 tok, medium) — Package mindmem is the official Go SDK for the mind-mem REST API.
- `errors.go` (~640 tok, large) — APIError is returned for any non-2xx response from the mind-mem server.
- `.gitignore` (~5 tok, tiny) — *.test
- `go.mod` (~198 tok, small) — // The /v5 suffix is required, not cosmetic. This is a subdirectory module in
- `methods.go` (~603 tok, large) — recallRequest is the JSON body POST /v1/recall accepts. Field names and the
- `README.md` (~704 tok, large) — MIND-Mem Go SDK
- `routes.go` (~667 tok, large) — Route is one REST operation this client knows how to call, expressed in the
- `types.go` (~849 tok, large) — BlockTier represents the storage tier of a memory block.
### `sdk/js/`

- `.gitignore` (~7 tok, tiny) — node_modules/
- `package.json` (~289 tok, medium) — Keys: name, private, version, description, license
- `package-lock.json` (~3326 tok, huge) — Keys: name, version, lockfileVersion, requires, packages
- `README.md` (~917 tok, large) — @mind-mem/sdk
### `sdk/js/src/`

- `client.ts` (~1450 tok, large) — Normalise: strip trailing slash so path joining is consistent.
- `errors.ts` (~438 tok, medium) — Restore prototype chain (required when extending built-ins in TS)
- `index.ts` (~150 tok, small)
- `routes.ts` (~607 tok, large)
- `types.ts` (~533 tok, large) — Shared domain types
### `sdk/js/test/`

- `client.test.ts` (~3096 tok, huge) — Minimal fetch mock helpers
### `sdk/js/`

- `tsconfig.json` (~147 tok, small) — Keys: compilerOptions, include, exclude
- `tsconfig.test.json` (~178 tok, small) — Keys: //, extends, compilerOptions, include, exclude
### `sdk/release/`

- `pack_js.py` (~2235 tok, huge) — # Copyright 2026 STARGA, Inc.
- `README.md` (~985 tok, large) — SDK release path
- `version.py` (~1702 tok, huge) — # Copyright 2026 STARGA, Inc.
### `sdk/spec/`

- `openapi.json` (~5030 tok, huge) — Keys: components, info, openapi, paths
- `README.md` (~672 tok, large) — API specifications
### `security/`

- `api-security-2026-04-28.md` (~5929 tok, huge) — MIND-Mem v3.1.8 — API / MCP Surface Security Audit
- `api-security-review-2026-04-28.md` (~3563 tok, huge) — MIND-Mem API Security Review — 2026-04-28
- `code-scanning-triage-2026-09-04.md` (~6863 tok, huge) — Code-scanning triage — historical findings and follow-up
- `threat-model-2026-04-28.md` (~1517 tok, huge) — MIND-Mem Threat Model — 2026-04-28
- `threat-model-online-trainer.md` (~6824 tok, huge) — MIND-Mem Threat Model — `online_trainer.py` (T-009) — 2026-08-31
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
- `accountability_dashboard.py` (~6663 tok, huge) — # Copyright 2026 STARGA, Inc.
- `accountability_views.py` (~10302 tok, huge) — # Copyright 2026 STARGA, Inc.
- `admissibility.py` (~5838 tok, huge) — What recall is allowed to serve — the servability allow-list.
- `admission.py` (~11936 tok, huge) — # Copyright 2026 STARGA, Inc.
- `agent_bridge.py` (~5340 tok, huge) — # Copyright 2026 STARGA, Inc.
- `agent_messaging.py` (~2877 tok, huge) — # Copyright 2026 STARGA, Inc.
- `alerting.py` (~2738 tok, huge) — # Copyright 2026 STARGA, Inc.
- `alert_urls.py` (~1873 tok, huge) — # Copyright 2026 STARGA, Inc.
- `anchoring.py` (~4266 tok, huge) — # Copyright 2026 STARGA, Inc.
- `answer_quality.py` (~3059 tok, huge) — Answer-quality layer: verification + self-consistency + per-category spec.
### `src/mind_mem/api/`

- `api_keys.py` (~2717 tok, huge) — Per-agent API key store for the mind-mem REST API.
- `auth.py` (~4581 tok, huge) — OIDC/SSO authentication for the mind-mem REST API.
- `grpc_server.py` (~3956 tok, huge) — gRPC wire protocol for mind-mem (v4.0 prep).
- `__init__.py` (~20 tok, tiny)
- `rest.py` (~15508 tok, huge) — REST API layer for mind-mem (v3.2.0, v3.2.1 hardening).
### `src/mind_mem/`

- `append_only.py` (~3657 tok, huge) — # Copyright 2026 STARGA, Inc.
- `apply_engine.py` (~25378 tok, huge) — Mind Mem Apply Engine v1.0 — Atomic proposal application with rollback.
- `audit_chain.py` (~6665 tok, huge) — mind-mem field-level audit sidecar — tamper-evident append-only ledger.
- `audit_context.py` (~4198 tok, huge) — Request-scoped audit attribution for mind-mem's network transports.
- `audit_pinned.py` (~3194 tok, huge) — Pinned-model audit pipeline — release-CI gate for ``mind-mem.json``.
- `auto_resolver.py` (~3194 tok, huge) — mind-mem Automatic Contradiction Resolution Suggestions.
- `axis_recall.py` (~4688 tok, huge) — # Copyright 2026 STARGA, Inc.
- `backup_restore.py` (~7048 tok, huge) — mind-mem Backup & Restore CLI. Zero external deps.
- `baseline_snapshot.py` (~4176 tok, huge) — Baseline snapshot for intent drift detection.
### `src/mind_mem/bench/`

- `ab_agent.py` (~2139 tok, huge) — Agent adapters: the one component the harness does not own.
- `ab_arms.py` (~3609 tok, huge) — The two arms, and the invariants that make "memory is the only variable" true.
- `ab_cli.py` (~3334 tok, huge) — ``mind-mem-bench-ab`` -- run the with-memory versus without-memory comparison.
- `ab_grade.py` (~1294 tok, large) — Machine-checked grading. A task passes iff its named tests pass.
- `ab_harness.py` (~3755 tok, huge) — Drive both arms over a task set and produce one number with its uncertainty.
- `ab_report.py` (~2631 tok, huge) — Pool several A/B run artifacts into one delta, with its uncertainty.
- `ab_seed.py` (~2496 tok, huge) — Seed the memory arm from material that existed BEFORE the task's commit.
- `ab_stats.py` (~1524 tok, huge) — The delta, with its uncertainty attached.
- `ab_task.py` (~1210 tok, large) — The unit of the with-memory versus without-memory comparison.
- `eval_adapter.py` (~1250 tok, large) — Pluggable retrieval-eval adapter contract + pipeline self-assertion.
- `eval_adapters.py` (~8229 tok, huge) — Concrete retrieval-eval adapters.
- `eval_scorer.py` (~1122 tok, large) — Dual-protocol retrieval scoring for the eval harness.
- `__init__.py` (~696 tok, large) — mind-mem benchmark harnesses — scalar metrics over the live corpus.
- `longmemeval_suite.py` (~4270 tok, huge) — LongMemEval-S consolidation harness — one loop, any adapter, self-asserting.
- `repo_task_cli.py` (~3637 tok, huge) — Generate ``benchmarks/tasks/real_repo_tasks.json`` -- the A/B task set.
- `repo_task_mining.py` (~3145 tok, huge) — Mine this repository's own git history for machine-checkable agent tasks.
- `repo_task_validation.py` (~4291 tok, huge) — Execute a mined commit to prove it is a real red->green task.
### `src/mind_mem/`

- `block_lineage.py` (~5901 tok, huge) — Typed block-lineage edges + bounded BFS reader (v3.11.0+, Pattern 3).
- `block_maturity.py` (~3317 tok, huge) — Block maturity metric — consolidation gate (Group H, v4.0.x).
- `block_metadata.py` (~5685 tok, huge) — mind-mem A-MEM — auto-evolving block metadata.
- `block_parser.py` (~8138 tok, huge) — Mind Mem Block Parser v1.0 — Self-hosted, zero external dependencies.
- `block_provenance.py` (~2728 tok, huge) — Provenance-rich blocks — optional actor/session/tool/source metadata.
- `block_store_encrypted.py` (~5788 tok, huge) — # Copyright 2026 STARGA, Inc.
- `block_store_postgres.py` (~22445 tok, huge) — PostgresBlockStore — PostgreSQL-backed BlockStore for mind-mem v3.2.0.
- `block_store_postgres_replica.py` (~3116 tok, huge) — v3.2.0 — read-replica routing for PostgresBlockStore.
- `block_store.py` (~18068 tok, huge) — BlockStore abstraction — decouples block access from storage format.
- `bootstrap_corpus.py` (~4094 tok, huge) — mind-mem Bootstrap Corpus — one-time backfill from existing knowledge sources.
- `boundary_witness.py` (~3878 tok, huge) — # Copyright 2026 STARGA, Inc.
- `calibration.py` (~7485 tok, huge) — Calibration feedback loop — track retrieval quality and adjust block ranking.
- `capture.py` (~5255 tok, huge) — mind-mem Auto-Capture Engine with Structured Extraction. Zero external deps.
- `category_distiller.py` (~6359 tok, huge) — mind-mem Category Distiller — auto-generates thematic summary files from memory blocks.
- `causal_graph.py` (~4689 tok, huge) — mind-mem Temporal Causal Graph — directed dependency tracking with staleness.
- `chain_of_note.py` (~1512 tok, huge) — Chain-of-note evidence packing (v3.4.0).
- `change_stream.py` (~3164 tok, huge) — # Copyright 2026 STARGA, Inc.
- `chat_citations.py` (~2480 tok, huge) — Citation extraction + validation for the conversational chat layer.
- `chat_cli.py` (~989 tok, large) — ``mind-mem-chat`` — ask a workspace a question, get cited answers.
- `chat_generators.py` (~2047 tok, huge) — Pluggable answer generators for the conversational chat layer.
- `chat_memory.py` (~3952 tok, huge) — Conversational chat layer — grounded answers with ``[[block_id]]`` citations.
- `check_version.py` (~1189 tok, large) — Version consistency checker for mind-mem.
- `codepoint_sanitize.py` (~2007 tok, huge) — Invisible-Unicode codepoint sanitization for block ingestion (security).
- `coding_schemas.py` (~2127 tok, huge) — mind-mem Coding-Native Memory Schemas.
- `cognitive_forget.py` (~3179 tok, huge) — # Copyright 2026 STARGA, Inc.
- `compaction.py` (~6823 tok, huge) — mind-mem Compaction & GC Engine. Zero external deps.
- `compiled_truth.py` (~8420 tok, huge) — mind-mem Compiled Truth — synthesized entity pages with append-only evidence.
### `src/mind_mem/compliance/`

- `audit.py` (~995 tok, large) — # Copyright 2026 STARGA, Inc.
- `detectors.py` (~3136 tok, huge) — # Copyright 2026 STARGA, Inc.
- `export.py` (~3268 tok, huge) — # Copyright 2026 STARGA, Inc.
- `__init__.py` (~1041 tok, large) — # Copyright 2026 STARGA, Inc.
- `prewrite.py` (~1552 tok, huge) — # Copyright 2026 STARGA, Inc.
- `provenance_policy.py` (~1878 tok, huge) — # Copyright 2026 STARGA, Inc.
- `redaction.py` (~2064 tok, huge) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/`

- `compressors.py` (~2253 tok, huge) — Real `Compressor` implementations for mind_mem.recompaction. Zero new deps.
- `conflict_resolver.py` (~5592 tok, huge) — mind-mem Automated Conflict Resolution Pipeline. Zero external deps.
- `connection_manager.py` (~3185 tok, huge) — SQLite connection manager with read/write separation and WAL mode.
- `consensus_vote.py` (~2067 tok, huge) — Quorum-based consensus voting on contradictions (v3.3.0).
- `consolidation_maturity_gate.py` (~2589 tok, huge) — # Copyright 2026 STARGA, Inc.
- `context_core.py` (~4313 tok, huge) — # Copyright 2026 STARGA, Inc.
- `contradiction_detector.py` (~4893 tok, huge) — mind-mem Contradiction Detector — Surface conflicts at the governance gate.
- `core_export.py` (~8336 tok, huge) — # Copyright 2026 STARGA, Inc.
- `corpus_registry.py` (~5127 tok, huge) — Central corpus path registry for mind-mem.
- `cron_runner.py` (~3583 tok, huge) — mind-mem Cron Runner — single entry point for all periodic jobs. Zero external deps.
- `cross_encoder_reranker.py` (~1463 tok, large) — mind-mem Optional Cross-Encoder Reranker.
- `cross_ledger.py` (~7869 tok, huge) — # Copyright 2026 STARGA, Inc.
- `daemon.py` (~3623 tok, huge) — Background daemon — `mm daemon` (v3.9.0 candidate).
- `data_marking.py` (~1133 tok, large) — # Copyright 2026 STARGA, Inc.
- `dead_ends.py` (~3855 tok, huge) — # Copyright 2026 STARGA, Inc.
- `dedup.py` (~4593 tok, huge) — mind-mem 4-layer deduplication filter for search results.
- `dream_cycle.py` (~13342 tok, huge) — mind-mem Dream Cycle — autonomous memory enrichment. Zero external deps.
- `drift_detector.py` (~5752 tok, huge) — mind-mem Semantic Belief Drift Detection.
- `edge_grounded_answer.py` (~5493 tok, huge) — # Copyright 2026 STARGA, Inc.
- `encryption.py` (~8391 tok, huge) — mind-mem Encryption at Rest — optional authenticated encryption for blocks.
- `entity_ingest.py` (~3309 tok, huge) — mind-mem Entity Ingestion — regex-based entity extraction. Zero external deps.
- `entity_prefetch.py` (~3059 tok, huge) — Entity-graph prefetch for recall (v3.3.0 Tier 3 #8).
- `enums.py` (~3698 tok, huge) — Centralised enum definitions for mind-mem.
- `error_codes.py` (~1918 tok, huge) — mind-mem Error Codes — structured error classification.
- `event_fanout.py` (~4850 tok, huge) — Governance event fan-out (v4.0 prep).
- `evidence_bundle.py` (~2205 tok, huge) — Structured evidence bundle for answerer co-design (v3.3.0 Tier 3 #7).
- `evidence_objects.py` (~15297 tok, huge) — # Copyright 2026 STARGA, Inc.
- `evidence_packer.py` (~3313 tok, huge) — Deterministic evidence packer for Mind-Mem.
- `evidence_recovery.py` (~10620 tok, huge) — # Copyright 2026 STARGA, Inc.
- `extraction_feedback.py` (~1877 tok, huge) — mind-mem Extraction Quality Feedback Tracker.
- `extractor.py` (~7670 tok, huge) — mind-mem Entity & Fact Extractor (Regex NER-lite). Zero external deps.
- `feature_gate.py` (~2240 tok, huge) — Shared config-resolver for retrieval features (architect audit item #6).
- `federation_connect.py` (~3549 tok, huge) — # Copyright 2026 STARGA, Inc.
- `field_audit.py` (~3103 tok, huge) — mind-mem Per-Field Mutation Audit — tracks individual field changes.
- `frame_fields.py` (~2143 tok, huge) — # Copyright 2026 STARGA, Inc.
- `governance_bench.py` (~1855 tok, huge) — mind-mem Governance Benchmark Suite.
- `governance_gate.py` (~21316 tok, huge) — # Copyright 2026 STARGA, Inc.
- `governance_raft.py` (~2474 tok, huge) — Raft-style consensus wrapper for governance writes (v4.0 prep).
- `granularity_align.py` (~3714 tok, huge) — Granularity / abstraction alignment — named merge operation (Group H, v4.0.x).
- `graph_ingest.py` (~7386 tok, huge) — Corpus → typed knowledge-graph ingestion (HITL-gated).
- `graph_recall.py` (~4558 tok, huge) — Multi-hop graph traversal for recall (v3.3.0 Tier 1 #2).
- `graph_schema.py` (~2082 tok, huge) — # Copyright 2026 STARGA, Inc.
- `guardrail_patterns.py` (~1463 tok, large) — # Copyright 2026 STARGA, Inc.
- `guardrails.py` (~5942 tok, huge) — # Copyright 2026 STARGA, Inc.
- `guardrail_surface.py` (~1704 tok, huge) — # Copyright 2026 STARGA, Inc.
- `hash_chain_v2.py` (~11016 tok, huge) — # Copyright 2026 STARGA, Inc.
- `hook_installer.py` (~10904 tok, huge) — # Copyright 2026 STARGA, Inc.
- `http_transport.py` (~23483 tok, huge) — HTTP transport adapter for mind-mem (v3.9.0 candidate).
- `hybrid_recall.py` (~24311 tok, huge) — mind-mem Hybrid Recall -- BM25 + Vector + RRF fusion.
### `src/mind_mem/importers/`

- `engine.py` (~6090 tok, huge) — # Copyright 2026 STARGA, Inc.
- `fs_source.py` (~2582 tok, huge) — # Copyright 2026 STARGA, Inc.
- `__init__.py` (~1793 tok, huge) — # Copyright 2026 STARGA, Inc.
- `note_parsers.py` (~3667 tok, huge) — # Copyright 2026 STARGA, Inc.
- `okf_source.py` (~1885 tok, huge) — # Copyright 2026 STARGA, Inc.
- `parsers.py` (~3243 tok, huge) — # Copyright 2026 STARGA, Inc.
- `quarantine.py` (~5463 tok, huge) — # Copyright 2026 STARGA, Inc.
- `records.py` (~990 tok, large) — # Copyright 2026 STARGA, Inc.
- `_shared.py` (~947 tok, large) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/`

- `inbox.py` (~9199 tok, huge) — Inbox folder ingestion — `mm inbox-watch` (v3.9.0 candidate).
- `ingestion_pipeline.py` (~8045 tok, huge) — # Copyright 2026 STARGA, Inc.
- `__init__.py` (~982 tok, large) — # Mind Mem — Memory + Immune System for AI agents
- `init_workspace.py` (~6291 tok, huge) — mind-mem workspace initializer. Zero external deps (Postgres optional).
- `intel_scan.py` (~14746 tok, huge) — Mind Mem Intelligence Scanner v2.0 — Self-hosted, zero external dependencies.
- `intent_router.py` (~3143 tok, huge) — mind-mem Intent Router — 9-type adaptive query intent classification.
- `interaction_signals.py` (~4425 tok, huge) — # Copyright 2026 STARGA, Inc.
- `iterative_recall.py` (~2856 tok, huge) — Iterative chain-of-retrieval for multi-hop evidence (v3.4.0).
- `kalman_belief.py` (~4672 tok, huge) — # Copyright 2026 STARGA, Inc.
- `kg_fusion.py` (~2054 tok, huge) — Typed knowledge-graph fusion into recall (opt-in, default OFF).
- `knowledge_graph.py` (~17602 tok, huge) — # Copyright 2026 STARGA, Inc.
- `ledger_anchor.py` (~1944 tok, huge) — # Copyright 2026 STARGA, Inc.
- `lifecycle_evidence.py` (~5228 tok, huge) — # Copyright 2026 STARGA, Inc.
- `lineage_staleness.py` (~1916 tok, huge) — Lineage→staleness propagation (v3.12.0, Theme C).
- `lint_autofix.py` (~2148 tok, huge) — # Copyright 2026 STARGA, Inc.
- `lint.py` (~3856 tok, huge) — # Copyright 2026 STARGA, Inc.
- `llm_extractor.py` (~8811 tok, huge) — mind-mem LLM Entity & Fact Extractor (Optional, config-gated).
- `llm_noise_profile.py` (~5228 tok, huge) — # Copyright 2026 STARGA, Inc.
- `maintenance_migrate.py` (~2109 tok, huge) — v3.2.0 §2.2 — one-shot migration helper for ``maintenance/`` subdivision.
- `maturity_breadth_scan.py` (~3027 tok, huge) — # Copyright 2026 STARGA, Inc.
- `mcp_entry.py` (~495 tok, medium) — Thin entry point for the ``mind-mem-mcp`` console script.
### `src/mind_mem/mcp/infra/`

- `acl.py` (~3954 tok, huge) — Per-tool ACL — scope enforcement for the MCP surface.
- `config.py` (~1070 tok, large) — ``mind-mem.json`` config loading + configurable limits.
- `constants.py` (~98 tok, small) — MCP-surface-wide constants shared by the infra submodules.
- `http_auth.py` (~1454 tok, large) — HTTP bearer-token authentication helpers for the MCP surface.
- `__init__.py` (~449 tok, medium) — Cross-cutting infra helpers extracted from mcp_server.py (v3.2.0 §1.2 PR-1).
- `observability.py` (~2374 tok, huge) — Observability + DB-busy helpers for the MCP surface.
- `rate_limit.py` (~1035 tok, large) — Per-client sliding-window rate limiter for the MCP surface.
- `workspace.py` (~2237 tok, huge) — Workspace resolution + path-safety helpers.
### `src/mind_mem/mcp/`

- `__init__.py` (~215 tok, medium) — v3.2.0 §1.2 decomposition namespace — subpackage for MCP server modules.
- `resources.py` (~3199 tok, huge) — MCP ``@mcp.resource`` declarations.
- `server.py` (~3003 tok, huge) — FastMCP instance + ``main()`` entry point for the Mind-Mem MCP server.
### `src/mind_mem/`

- `mcp_server.py` (~1871 tok, huge) — Mind-Mem MCP Server — public facade (v3.2.0 §1.2 PR-final shim).
### `src/mind_mem/mcp/tools/`

- `agent.py` (~3001 tok, huge) — Agent-bridge + vault MCP tools.
- `arch_mind.py` (~3475 tok, huge) — arch-mind MCP tools — wraps the ``arch-mind`` binary as 7 MCP tools.
- `audit.py` (~4247 tok, huge) — Audit MCP tools — Merkle proofs, hash chain + evidence chain verification.
- `benchmark.py` (~2334 tok, huge) — Benchmark + category-summary MCP tools.
- `calibration.py` (~2845 tok, huge) — Calibration MCP tools — feedback (quality) + outcome attribution (utility).
- `chat.py` (~922 tok, large) — Chat surface — grounded question answering over the workspace.
- `consolidation.py` (~6920 tok, huge) — Memory-consolidation MCP tools.
- `core.py` (~2862 tok, huge) — Context-core MCP tools — ``.mmcore`` bundle lifecycle.
- `encryption.py` (~2670 tok, huge) — At-rest encryption MCP tools — ``encrypt_file`` / ``decrypt_file``.
- `frames.py` (~2034 tok, huge) — # Copyright 2026 STARGA, Inc.
- `governance.py` (~13751 tok, huge) — Governance MCP tools — propose / apply / rollback / scan / contradictions / memory_evolution.
- `graph.py` (~7091 tok, huge) — Knowledge-graph + causal-graph MCP tools.
- `guardrails.py` (~1811 tok, huge) — # Copyright 2026 STARGA, Inc.
- `_helpers.py` (~1412 tok, large) — Shared tool-internal helpers — workspace paths + lazy-init singletons.
- `__init__.py` (~107 tok, small) — Per-domain ``@mcp.tool`` modules (v3.2.0 §1.2 PR-3+).
- `kernels.py` (~1902 tok, huge) — MIND kernel + compiled-truth MCP tools.
- `lineage.py` (~717 tok, large) — MCP wrapping for the v3.11.0 typed block-lineage graph (Pattern 3).
- `lint.py` (~1511 tok, huge) — # Copyright 2026 STARGA, Inc.
- `memory_ops.py` (~16514 tok, huge) — Memory operations MCP tools — index / lifecycle / health / export.
- `mic_map.py` (~2436 tok, huge) — MIC/MAP serialization MCP tools — wraps ``mind_mem.mic_map``.
- `model.py` (~3185 tok, huge) — Model audit / signing MCP tools — wraps ``mind_mem.model_audit``,
- `ontology.py` (~969 tok, large) — Ontology MCP tools — ``ontology_load`` + ``ontology_validate``.
- `pipeline.py` (~916 tok, large) — MCP wrapping for pipeline-hash inspection + dirty-block re-extraction.
- `public.py` (~5248 tok, huge) — # mypy: disable-error-code="no-any-return"
- `quality.py` (~1009 tok, large) — MCP wrapping for the v3.11.0 deterministic quality gate.
- `recall.py` (~16756 tok, huge) — Recall surface — the retrieval core of the MCP API.
- `signal.py` (~926 tok, large) — Interaction-signal MCP tools — ``observe_signal`` + ``signal_stats``.
- `trajectory.py` (~1474 tok, large) — Trajectory-memory MCP tool — case-based recall over past task outcomes.
- `walkthrough_persona.py` (~1571 tok, huge) — MCP wrapping for v3.9 walkthrough + persona projection.
### `src/mind_mem/`

- `memory_index.py` (~2925 tok, huge) — Auto-generated hierarchical index — ``index.md`` + ``log.md`` (Group C).
- `memory_mesh.py` (~1903 tok, huge) — # Copyright 2026 STARGA, Inc.
- `memory_tiers.py` (~6427 tok, huge) — # Copyright 2026 STARGA, Inc.
- `merkle_tree.py` (~3673 tok, huge) — # Copyright 2026 STARGA, Inc.
- `_mic_map_accel.pyx` (~1136 tok, large) — # cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
- `mic_map.py` (~8384 tok, huge) — MIC/MAP — STARGA-native serialization for MIND IR graphs.
- `mind_ffi.py` (~7630 tok, huge) — mind-mem FFI bridge — loads compiled MIND .so and exposes scoring functions.
- `mind_filelock.py` (~10088 tok, huge) — mind-mem file locking — cross-platform advisory locks. Zero external deps.
- `mind_kernels.py` (~3107 tok, huge) — # Copyright 2026 STARGA, Inc.
- `model_audit.py` (~4920 tok, huge) — Model checkpoint audit — scan for remote-code hooks, unsafe pickle, tokenizer injection.
- `model_gate.py` (~7436 tok, huge) — Load-gate registry for ``mm audit-model`` checkpoints.
- `model_provenance.py` (~2127 tok, huge) — Provenance allowlist check for ``mm audit-model`` checkpoints.
- `model_signing.py` (~2998 tok, huge) — Ed25519 manifest signing for ``mm audit-model`` checkpoints.
- `mrs.py` (~5268 tok, huge) — # Copyright 2026 STARGA, Inc.
- `multi_modal.py` (~3184 tok, huge) — # Copyright 2026 STARGA, Inc.
- `namespaces.py` (~5119 tok, huge) — mind-mem Multi-Agent Namespace & ACL Engine. Zero external deps.
- `novel_term_gate.py` (~1729 tok, huge) — # Copyright 2026 STARGA, Inc.
- `observability.py` (~3828 tok, huge) — mind-mem Observability Module. Zero external deps.
- `observation_axis.py` (~3925 tok, huge) — # Copyright 2026 STARGA, Inc.
- `observation_compress.py` (~1401 tok, large) — Observation Compression Layer for Mind-Mem.
- `ollama_host.py` (~1022 tok, large) — Single source of truth for the ollama base URL.
- `online_trainer.py` (~6077 tok, huge) — # Copyright 2026 STARGA, Inc.
- `ontology.py` (~3933 tok, huge) — # Copyright 2026 STARGA, Inc.
- `outcome_attribution.py` (~4288 tok, huge) — Outcome attribution — did the recalled memory actually *help*?
- `outcome_store.py` (~4094 tok, huge) — Outcome-attribution persistence over the calibration store.
- `payload_admission.py` (~3614 tok, huge) — # Copyright 2026 STARGA, Inc.
- `personas.py` (~1256 tok, large) — Persona-aware recall projection (v3.9.0 candidate).
- `pipeline_hash.py` (~3314 tok, huge) — Hash-of-code pipeline invalidation (v3.9.0 candidate).
- `prefetch.py` (~6830 tok, huge) — # Copyright 2026 STARGA, Inc.
- `prefix_cache.py` (~3043 tok, huge) — # Copyright 2026 STARGA, Inc.
- `preimage.py` (~1329 tok, large) — # Copyright 2026 STARGA, Inc.
- `project_key.py` (~3151 tok, huge) — # Copyright 2026 STARGA, Inc.
- `project_profile.py` (~1681 tok, huge) — # Copyright 2026 STARGA, Inc.
- `protection.py` (~1959 tok, huge) — Runtime protection layer for mind-mem (v3.3.0+).
- `provenance_class.py` (~2682 tok, huge) — Provenance class — the FIFTH deterministic component of the validity gate.
- `py.typed` (~0 tok, tiny)
- `q1616.py` (~562 tok, large) — # Copyright 2026 STARGA, Inc.
- `quality_gate.py` (~2691 tok, huge) — Deterministic block quality gate (v3.11.0, Pattern 2).
- `query_expansion.py` (~5349 tok, huge) — Multi-query expansion for improved recall.
- `query_planner.py` (~2865 tok, huge) — Query decomposition for multi-hop questions (v3.3.0 Tier 1 #1).
- `recall_attestation.py` (~11755 tok, huge) — Per-run recall attestation — runtime evidence of *how* an answer was produced.
- `recall_cache.py` (~3659 tok, huge) — v3.2.0 — distributed recall result cache (Redis + in-process LRU fallback).
- `_recall_constants.py` (~3385 tok, huge) — Recall engine constants — search fields, BM25 params, regex patterns, limits."""
- `_recall_context.py` (~2609 tok, huge) — Recall engine context packing — post-retrieval augmentation rules."""
- `_recall_detection.py` (~6268 tok, huge) — Recall engine detection — query type classification, text extraction, block utilities."""
- `recall_digests.py` (~2101 tok, huge) — Canonical digests the recall attestation commits to.
- `_recall_expansion.py` (~3249 tok, huge) — Recall engine query expansion — domain synonyms, month normalization, RM3."""
- `_recall_explain.py` (~2233 tok, huge) — Score decomposition record for explainable recall (v3.11.0, Pattern 1).
- `recall.py` (~5539 tok, huge) — mind-mem Recall Engine (BM25 + TF-IDF + Graph + Stemming). Zero external deps.
- `_recall_reranking.py` (~4277 tok, huge) — Recall engine reranking — deterministic feature-based re-scoring of BM25 hits."""
- `_recall_scoring.py` (~4992 tok, huge) — Recall engine scoring — BM25F helper, date scores, graph boosting, negation, date proximity, categories.
- `recall_smart_chunk.py` (~2826 tok, huge) — Config seam wiring :mod:`smart_chunker` into the BM25 chunk-boost path.
- `_recall_temporal.py` (~2214 tok, huge) — Recall engine temporal filtering — resolve relative time references and filter blocks."""
- `_recall_tokenization.py` (~784 tok, large) — Recall engine tokenization — Porter stemmer and tokenizer."""
- `recall_vector.py` (~22639 tok, huge) — mind-mem Vector Recall Backend (Semantic Search with Embeddings).
- `_recall_workspace.py` (~1990 tok, huge) — # Copyright 2026 STARGA, Inc.
- `recompaction.py` (~2489 tok, huge) — mind-mem Iterative Re-Compression ("sleep") Engine. Zero external deps.
- `replay_check.py` (~4309 tok, huge) — # Copyright 2026 STARGA, Inc.
- `rerank_ensemble.py` (~4393 tok, huge) — Reranker ensemble via Borda count (v3.3.0 Tier 4 #9).
- `resume_brief.py` (~2720 tok, huge) — # Copyright 2026 STARGA, Inc.
- `retention_class.py` (~1840 tok, huge) — # Copyright 2026 STARGA, Inc.
- `retrieval_graph.py` (~8339 tok, huge) — Retrieval logger + co-retrieval graph for usage-based score propagation.
- `retrieval_trace.py` (~1252 tok, large) — Per-feature retrieval attribution (v3.3.0 architect audit item #7).
- `review_batch.py` (~3102 tok, huge) — # Copyright 2026 STARGA, Inc.
- `review_cli.py` (~2207 tok, huge) — # Copyright 2026 STARGA, Inc.
- `review_evidence.py` (~1722 tok, huge) — # Copyright 2026 STARGA, Inc.
- `review_metrics.py` (~1169 tok, large) — # Copyright 2026 STARGA, Inc.
- `review_preview.py` (~4537 tok, huge) — # Copyright 2026 STARGA, Inc.
- `review_queue.py` (~2612 tok, huge) — # Copyright 2026 STARGA, Inc.
- `review_render.py` (~1836 tok, huge) — # Copyright 2026 STARGA, Inc.
- `review_session.py` (~1739 tok, huge) — # Copyright 2026 STARGA, Inc.
- `schema_version.py` (~2452 tok, huge) — Mind-Mem Schema Version Migration. Zero external deps.
- `scopes.py` (~283 tok, medium) — # Copyright 2026 STARGA, Inc.
- `scoring_instant.py` (~1643 tok, huge) — The recency seam — one UTC date, resolved once, threaded everywhere.
- `self_update.py` (~5198 tok, huge) — # Copyright 2026 STARGA, Inc.
- `served_ledger.py` (~11663 tok, huge) — # Copyright 2026 STARGA, Inc.
- `session_boost.py` (~1511 tok, huge) — Session-boundary preservation for recall (v3.3.0 Tier 2 #5).
- `session_summarizer.py` (~3852 tok, huge) — mind-mem Session Summarizer. Zero external deps.
### `src/mind_mem/skill_opt/`

- `adapters.py` (~2790 tok, huge) — # Copyright 2026 STARGA, Inc.
- `analyzer.py` (~2231 tok, huge) — # Copyright 2026 STARGA, Inc.
- `config.py` (~1669 tok, huge) — # Copyright 2026 STARGA, Inc.
- `fleet_bridge.py` (~2244 tok, huge) — # Copyright 2026 STARGA, Inc.
- `history.py` (~2570 tok, huge) — # Copyright 2026 STARGA, Inc.
- `__init__.py` (~89 tok, small) — # Copyright 2026 STARGA, Inc.
- `mutator.py` (~907 tok, large) — # Copyright 2026 STARGA, Inc.
- `scorer.py` (~1406 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_runner.py` (~2420 tok, huge) — # Copyright 2026 STARGA, Inc.
- `_types.py` (~1612 tok, huge) — # Copyright 2026 STARGA, Inc.
- `validator.py` (~2961 tok, huge) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/`

- `smart_chunker.py` (~7592 tok, huge) — mind-mem Smart Chunker — Semantic-boundary document chunking.
- `smoke_test.sh` (~633 tok, large) — mind-mem Smoke Test — end-to-end verification
- `spec_binding.py` (~2883 tok, huge) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/spec/`

- `export_openapi.py` (~1790 tok, huge) — # Copyright 2026 STARGA, Inc.
- `__init__.py` (~215 tok, medium) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/`

- `speculative_prefetch.py` (~3195 tok, huge) — # Copyright 2026 STARGA, Inc.
- `staleness.py` (~1179 tok, large) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/storage/`

- `__init__.py` (~5255 tok, huge) — Storage factory for mind-mem block stores (v3.2.0).
- `sharded_pg.py` (~6113 tok, huge) — Sharded Postgres / Citus routing (v4.0 prep).
### `src/mind_mem/`

- `streaming.py` (~6278 tok, huge) — Rate-limited front gate for the ingest webhook (v3.3.0, wired 5.0.1).
- `task_frames.py` (~3571 tok, huge) — # Copyright 2026 STARGA, Inc.
- `_task_status_literals.sh` (~118 tok, small) — AUTO-GENERATED — do not edit by hand.
- `telemetry.py` (~3174 tok, huge) — mind-mem Telemetry — OpenTelemetry traces + Prometheus metrics.
### `src/mind_mem/templates/`

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
### `src/mind_mem/`

- `temporal_metadata.py` (~1938 tok, huge) — Temporal metadata injection for retrieved blocks (v3.4.0).
- `tenant_audit.py` (~2355 tok, huge) — Per-tenant audit chain isolation (v4.0 prep).
- `tenant_kms.py` (~3010 tok, huge) — Per-tenant key management + envelope encryption (v4.0 prep).
### `src/mind_mem/tool_output/`

- `__init__.py` (~200 tok, medium) — mind_mem.tool_output — context-offload for large command/tool output (§5).
- `store.py` (~3522 tok, huge) — Tool-output store — full text out-of-context, keyed by handle (mind-mem §5).
- `summarize.py` (~2173 tok, huge) — Deterministic tool-output summarizer (mind-mem §5 — context offload).
### `src/mind_mem/`

- `tracking.py` (~6283 tok, huge) — # Copyright 2026 STARGA, Inc.
- `trajectory.py` (~6318 tok, huge) — Trajectory Memory — task execution trace storage and recall.
- `transcript_capture.py` (~2839 tok, huge) — mind-mem Transcript JSONL Capture. Zero external deps.
- `trust_scores.py` (~2416 tok, huge) — Standalone trust surface — a thin façade over the validity gate.
- `trust_signals.py` (~2107 tok, huge) — Workspace signal loaders for the validity gate's provenance component.
- `truth_score.py` (~1975 tok, huge) — Probabilistic truth score for memory blocks (v3.3.0).
- `turbo_quant.py` (~1295 tok, large) — # Copyright 2026 STARGA, Inc.
- `uncertainty_propagation.py` (~1262 tok, large) — # Copyright 2026 STARGA, Inc.
- `union_recall.py` (~1310 tok, large) — Union-style retrieval for decomposed queries (v3.4.0).
- `upsert_slots.py` (~2471 tok, huge) — Enum-keyed upsert slots: make contradiction structurally impossible (M4).
- `usage_meter.py` (~5249 tok, huge) — # Copyright 2026 STARGA, Inc.
### `src/mind_mem/v4/`

- `backpressure.py` (~4318 tok, huge) — v4 backpressure controller (round 4 audit, DeepSeek 9.75→10 gap).
- `block_kinds.py` (~6493 tok, huge) — v4 block-kind taxonomy (Group B: knowledge graph).
- `block_metadata.py` (~3852 tok, huge) — v4 block metadata + schema-validation hooks.
- `block_versioning.py` (~1861 tok, huge) — Block versioning + time-travel — reconstruct what a block said, and when.
- `circuit_breaker.py` (~4577 tok, huge) — v4 circuit breaker (round 5 audit, Mistral + GLM 9.9→10 gap).
- `cognitive_kernel.py` (~2576 tok, huge) — v4 Cognitive Mind Kernel — composable retrieval strategies (Group A).
- `embedding_pipeline.py` (~2377 tok, huge) — v4 embedding auto-derivation pipeline (Group A — closes the
- `feature_flags.py` (~6166 tok, huge) — v4.0 feature-flag registry.
- `federation_client.py` (~5276 tok, huge) — Federation wire-transport client for mind-mem v4.
- `federation.py` (~7339 tok, huge) — v4 federated cross-agent consistency (Group D).
- `flag_registry.py` (~8516 tok, huge) — Three-state registry for every declared v4 feature flag.
- `health.py` (~2475 tok, huge) — v4 health-check surface (round 4 audit, DeepSeek 9.75→10 gap).
- `hnsw_kind_index.py` (~3545 tok, huge) — v4 HNSW kind-filtered ANN index (Group D).
- `__init__.py` (~1010 tok, large) — mind-mem v4.0 surface — side-by-side scaffolding, default OFF.
- `kernels.py` (~3819 tok, huge) — v4 kernel strategy implementations (Group A).
- `kind_backfill.py` (~3063 tok, huge) — The v4 kind-index build pass — ``mm kinds backfill``'s engine.
- `kind_summaries.py` (~2669 tok, huge) — v4 per-kind global summaries (Group B — GraphRAG-style).
- `logging_context.py` (~1424 tok, large) — v4 structured logging context (round 4 audit, DeepSeek 9.75→10 gap).
- `observability.py` (~2993 tok, huge) — v4 observability — counters, timers, histograms, exporters.
- `pq.py` (~5255 tok, huge) — v4 product-quantization (PQ) encoding for embedding storage (Group D).
- `self_editing.py` (~3295 tok, huge) — v4 self-editing on recall (Group A — MemGPT pattern).
- `surprise_retrieval.py` (~2619 tok, huge) — v4 surprise-weighted retrieval term (Group A: cognition / model layer).
- `tls_floor.py` (~4385 tok, huge) — TLS 1.3 floor and certificate pinning for mind-mem's own network surfaces.
- `vocabulary.py` (~2895 tok, huge) — v4 vocabulary-bound fields — per-workspace controlled vocabularies.
### `src/mind_mem/`

- `validate_py.py` (~5676 tok, huge) — Mind Mem Integrity Validator — canonical engine.
- `validate.sh` (~1350 tok, large) — src/mind_mem/validate.sh — thin forwarder to the Python validator.
- `validate.sh.pre-forwarder` (~7140 tok, huge) — #!/usr/bin/env bash
- `validity_gate.py` (~3906 tok, huge) — Phase-2 recall validity gate — flag-gated, deterministic demotion.
- `vector_inertness.py` (~3125 tok, huge) — # Copyright 2026 STARGA, Inc.
- `verify_cli.py` (~10537 tok, huge) — # Copyright 2026 STARGA, Inc.
- `walkthrough.py` (~2449 tok, huge) — Dependency-ordered walkthrough — `compile_walkthrough` (v3.9.0 candidate).
- `watcher.py` (~1453 tok, large) — Mind-Mem File Watcher — auto-reindex on workspace changes. Zero external deps.
- `world_anchors.py` (~2986 tok, huge) — # Copyright 2026 STARGA, Inc.
- `world_git_probe.py` (~1660 tok, huge) — # Copyright 2026 STARGA, Inc.
- `world_staleness_config.py` (~1676 tok, huge) — # Copyright 2026 STARGA, Inc.
- `world_staleness.py` (~3891 tok, huge) — # Copyright 2026 STARGA, Inc.
- `world_symbol_probe.py` (~1454 tok, large) — # Copyright 2026 STARGA, Inc.
### `tests/`

- `conftest.py` (~4070 tok, huge) — Shared test fixtures.
- `evidence_manifest.toml` (~790 tok, large) — [claim.governed_write]
### `tests/fixtures/`

- `ci_jobs_advisory_red.json` (~2657 tok, huge) — Keys: _provenance, workflow_runs, 90000000001
- `ci_jobs_by_run.json` (~7130 tok, huge) — Keys: _provenance, 33579619488, 33511997100, 33566252435
- `ci_runs_by_sha.json` (~471 tok, medium) — Keys: 1ec7f63e011443eca2e63299b08c51241fa09915, 313134b66d99b9ebaed8af77fb0092a6b5115c53, a0850815782277b17fbd1243fd0f084d51a0e651, ba4cd4e12de5091b38d971b0ee921f47effd2acd
- `code_scanning_alerts_open.json` (~248 tok, medium)
- `code_scanning_analyses.json` (~213 tok, medium)
### `tests/fixtures/importers/agent_memory/`

- `feedback_append_only_writes.md` (~125 tok, small)
- `MEMORY.md` (~59 tok, small) — Auto Memory
### `tests/fixtures/importers/agent_memory/memory/`

- `nested_retention_policy.md` (~73 tok, small)
### `tests/fixtures/importers/agent_memory/`

- `OPERATING-MANUAL.md` (~63 tok, small) — Operating manual
- `reference_pool_health_checks.md` (~136 tok, small)
### `tests/fixtures/importers/`

- `chat_session.json` (~293 tok, medium) — Keys: session_id, created_at, messages
- `chroma_export.json` (~245 tok, medium) — Keys: collection, ids, documents, metadatas, embeddings
- `chroma_near_duplicates.json` (~86 tok, small) — Keys: collection, ids, documents, metadatas
- `letta_agent.af.json` (~268 tok, medium) — Keys: agent_type, name, created_at, system, core_memory
- `mem0_export.json` (~326 tok, medium) — Keys: results
### `tests/fixtures/importers/vault/daily/`

- `2026-01-15.md` (~31 tok, tiny) — 2026-01-15
### `tests/fixtures/importers/vault/notes/`

- `architecture.md` (~93 tok, small)
- `empty-note.md` (~7 tok, tiny)
### `tests/fixtures/importers/vault/notes/incidents/`

- `connection-pool-outage.md` (~75 tok, small)
### `tests/fixtures/importers/vault/.obsidian/`

- `app.json` (~7 tok, tiny) — Keys: editorMode
### `tests/fixtures/importers/vault/templates/`

- `note.md` (~18 tok, tiny)
### `tests/fixtures/`

- `locomo_mini.json` (~710 tok, large)
- `pypi_mind_mem_releases.json` (~483 tok, medium) — Keys: info, releases
### `tests/integration/`

- `__init__.py` (~0 tok, tiny)
- `test_full_pipeline.py` (~1982 tok, huge) — Integration test: full mind-mem pipeline.
### `tests/`

- `_ledger_rows.py` (~1486 tok, large) — # Copyright 2026 STARGA, Inc.
- `_platform_compat.py` (~3662 tok, huge) — # Copyright 2026 STARGA, Inc.
- `_recall_clock_sentinel.py` (~3033 tok, huge) — Two independent instruments for proving the recall scoring path reads no clock.
### `tests/red_team/`

- `behavioral_audit.py` (~636 tok, large) — Behavioral audit scaffold for the mind-mem MCP surface.
- `conftest.py` (~170 tok, small) — pytest configuration for the red_team test package.
- `__init__.py` (~0 tok, tiny)
### `tests/red_team/transcripts/`

- `.gitkeep` (~0 tok, tiny)
### `tests/`

- `_restore_scope.py` (~541 tok, large) — # Copyright 2026 STARGA, Inc.
- `_review_autoapprove_scan.py` (~1369 tok, large) — # Copyright 2026 STARGA, Inc.
- `_review_fixtures.py` (~2055 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_ab_report_unstated_comparability.py` (~1031 tok, large) — An unstated comparability field is not agreement.
- `test_abstention_classifier.py` (~3963 tok, huge) — Tests for the adversarial abstention classifier."""
- `test_accountability_dashboard.py` (~6883 tok, huge) — RA.5 — the lifecycle-tier dashboard, and the four refusals it inherits.
- `test_accountability_views.py` (~6320 tok, huge) — RA.2 — precision and waste as derived views, and the four refusals that shape them.
- `test_acl_surface_complete.py` (~1020 tok, large) — Every registered MCP tool is ACL-classified. No tool is unreachable.
- `test_acl_tool_coverage.py` (~2384 tok, huge) — ACL coverage invariant for the MCP tool surface.
- `test_active_only_filter.py` (~312 tok, medium) — Tests for active_only recall filter."""
- `test_admissibility_unreadable_status.py` (~1104 tok, large) — An unreadable ``Status`` must be withheld on the stale-index path too.
- `test_admission_seam.py` (~12891 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_admit_proposal_openers.py` (~3596 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_adversarial_corpus.py` (~2329 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_agent_bridge.py` (~2335 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_agent_bridge_sync_dirs.py` (~1328 tok, large) — Regression tests for VaultBridge.scan's sync_dirs handling.
- `test_agent_id_filter.py` (~335 tok, medium) — Tests for agent_id namespace filtering."""
- `test_agent_messaging.py` (~2778 tok, huge) — Tests for v4.0.19 agent-to-agent messaging (`mm send` / `mm inbox`).
- `test_alerting.py` (~1964 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_alert_url_allowlist.py` (~1925 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_allow_decompose.py` (~311 tok, medium) — Tests for _allow_decompose recall parameter."""
- `test_answer_quality_confidence.py` (~761 tok, large) — ``self_consistency`` confidence must be votes over samples REQUESTED.
- `test_answer_quality.py` (~1288 tok, large) — Tests for the v3.3.0 answer-quality shims."""
- `test_anticipation_cache.py` (~8342 tok, huge) — The anticipation cache, proven to retire with the ledger head rather than a clock.
- `test_api_keys.py` (~2133 tok, huge) — Tests for APIKeyStore in src/mind_mem/api/api_keys.py."""
- `test_append_only_audit_log.py` (~3895 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_apply_abort_is_recorded.py` (~4908 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_apply_double_apply_guard.py` (~421 tok, medium) — apply must re-validate proposal status UNDER the workspace lock.
- `test_apply_engine_audit_gates.py` (~3031 tok, huge) — Apply-engine gates that were reporting a verdict they had not earned.
- `test_apply_engine_backend_routing.py` (~1075 tok, large) — v3.2.0 §1.4 PR-6 — apply_engine routes through configured BlockStore."""
- `test_apply_engine_op_routing.py` (~1849 tok, huge) — v3.2.2 — execute_op routes block-level ops through BlockStore.
- `test_apply_engine.py` (~13107 tok, huge) — Tests for apply_engine.py — focus on security, validation, and rollback."""
- `test_apply_engine_text_range_atomic.py` (~2033 tok, huge) — Text-range ops (``insert_after_block`` / ``replace_range``) must be atomic.
- `test_apply_engine_timestamp_utc.py` (~3319 tok, huge) — Apply/rollback audit timestamps must be genuinely UTC, not local-time-with-a-Z.
- `test_arch_mind_argument_guards.py` (~1530 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_arch_mind_fixture_provenance.py` (~3452 tok, huge) — Provenance of the arch-mind governance fixtures.
- `test_arch_mind_rules_gate.py` (~4959 tok, huge) — Mechanical enforcement of ``.arch-mind/rules.mind``.
- `test_atomicity_maintenance_scope.py` (~1326 tok, large) — v3.2.0 §2.2 — regression test for the ``maintenance/`` atomicity fix.
- `test_audit_chain.py` (~2552 tok, huge) — Tests for mind-mem hash-chain mutation log (audit_chain.py)."""
- `test_audit_pinned.py` (~3323 tok, huge) — Pinned-model audit pipeline — release-CI gate.
- `test_auto_resolver.py` (~1185 tok, large) — Tests for mind-mem auto contradiction resolution (auto_resolver.py)."""
- `test_axis_recall_mcp.py` (~1381 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_axis_recall.py` (~3683 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_backpressure_recovery.py` (~1081 tok, large) — Two defects the slice-3 verifier reproduced, pinned so they cannot return.
- `test_backpressure_wiring.py` (~5799 tok, huge) — ``v4/backpressure`` wired into the producer loops that can drown the store.
- `test_backup_restore.py` (~3306 tok, huge) — Tests for backup_restore.py — zero external deps (stdlib unittest)."""
- `test_bandit_gate.py` (~5280 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_baseline_snapshot.py` (~3033 tok, huge) — Tests for baseline snapshot and drift detection (#431)."""
- `test_bench_eval_harness.py` (~4522 tok, huge) — Tests for the LongMemEval consolidation harness (self-asserting adapters).
- `test_bench_hard_timeout.py` (~1824 tok, huge) — The per-question timeout must preempt a NATIVE hang, not just a Python one.
- `test_bench_hybrid_dispatch.py` (~5937 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_bigrams.py` (~168 tok, small) — Tests for bigram extraction."""
- `test_bitemporal_edge_validity.py` (~2524 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_block_id_format.py` (~345 tok, medium) — Tests for block ID format validation."""
- `test_block_lineage.py` (~2122 tok, huge) — Tests for the v3.11.0 typed block-lineage graph (Pattern 3)."""
- `test_block_maturity_group_h.py` (~4178 tok, huge) — Tests for Group H maturity metric — consolidation gate.
- `test_block_metadata.py` (~945 tok, large) — Tests for A-MEM block metadata evolution."""
- `test_block_metadata_wiring.py` (~3680 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_block_meta_store_wiring.py` (~1963 tok, huge) — The block-metadata store must exist, and one importance scale must reach the planner.
- `test_block_parser_chunks.py` (~1658 tok, huge) — Tests for block_parser.py — overlapping chunk splitting + dedup."""
- `test_block_parser_edge.py` (~643 tok, large) — Extended block parser tests."""
- `test_block_parser_fields.py` (~377 tok, medium) — Tests for block parser field extraction."""
- `test_block_parser_multifile.py` (~337 tok, medium) — Tests for parsing multiple files."""
- `test_block_parser_no_silent_truncation.py` (~638 tok, large) — Regression: block_parser must not silently drop corpus past a size cap.
- `test_block_parser.py` (~3419 tok, huge) — Tests for block_parser.py — zero external deps (stdlib unittest)."""
- `test_block_provenance.py` (~3753 tok, huge) — Tests for provenance-rich blocks (roadmap Group E).
- `test_block_store_encrypted.py` (~1004 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_block_store_encrypted_write_surface.py` (~4220 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_block_store_lock.py` (~928 tok, large) — v3.2.0 §1.4 PR-4 — MarkdownBlockStore.lock() tests."""
- `test_block_store.py` (~2220 tok, huge) — Tests for block_store.py — BlockStore protocol and MarkdownBlockStore."""
- `test_block_store_snapshot.py` (~1706 tok, huge) — v3.2.0 §1.4 PR-3 — MarkdownBlockStore.snapshot / restore / diff tests.
- `test_block_store_write.py` (~3903 tok, huge) — v3.2.0 §1.4 PR-2 — MarkdownBlockStore.write_block + delete_block tests."""
- `test_block_types.py` (~437 tok, medium) — Tests for different block types in recall."""
- `test_bootstrap_corpus.py` (~1798 tok, huge) — Tests for bootstrap_corpus.py — backfill pipeline module."""
- `test_bootstrap_corpus_wiring.py` (~5443 tok, huge) — Wiring + quarantine proof for the ``mind-mem-bootstrap`` ingest door.
- `test_boundary_witness_cli_e2e.py` (~3954 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_boundary_witness.py` (~2442 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_calibration.py` (~3269 tok, huge) — Tests for calibration feedback loop.
- `test_calibration_window_determinism.py` (~1159 tok, large) — The calibration window boundary is UTC-anchored and pinnable.
- `test_capture_governed_signals.py` (~2903 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_capture.py` (~2193 tok, huge) — Tests for capture.py — zero external deps (stdlib unittest)."""
- `test_category_distiller.py` (~2660 tok, huge) — Tests for category_distiller.py — CategoryDistiller class."""
- `test_causal_graph.py` (~1566 tok, huge) — Tests for mind-mem temporal causal graph (causal_graph.py)."""
- `test_chain_transaction_capability.py` (~1204 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_chain_truncation.py` (~5069 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_change_stream_backpressure.py` (~956 tok, large) — The change stream's backpressure counters, and the drain that gives
- `test_chat_with_memory.py` (~5985 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_check_version_gate.py` (~475 tok, medium) — Regression tests for the version-consistency gate.
- `test_check_version.py` (~271 tok, medium) — Tests for version consistency checker."""
- `test_check_workspace_backend.py` (~3248 tok, huge) — Backend-aware workspace validation — ``mcp.infra.workspace._check_workspace``.
- `test_chunk_text.py` (~231 tok, medium) — Tests for text chunking."""
- `test_ci_green_per_job_gate.py` (~4344 tok, huge) — The CI-green gate must read JOBS, not just the workflow run's conclusion.
- `test_codepoint_sanitize.py` (~3333 tok, huge) — Tests for invisible-Unicode ingest sanitization (security).
- `test_coding_schemas.py` (~1284 tok, large) — Tests for mind-mem coding-native memory schemas."""
- `test_cognitive_forget.py` (~2315 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compaction.py` (~1905 tok, huge) — Tests for compaction.py — GC and archival engine."""
- `test_companion_baseline_attestation.py` (~5756 tok, huge) — A legacy recovery anchor may be bound forward, never blessed.
- `test_competitive_intel.py` (~1881 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compiled_truth.py` (~4665 tok, huge) — Tests for mind-mem compiled truth pages (compiled_truth.py)."""
- `test_compliance_export.py` (~3734 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compliance_provenance.py` (~3329 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compliance_redaction.py` (~5255 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_compressors.py` (~2757 tok, huge) — Tests for compressors.py — real Compressor implementations.
- `test_concurrency_stress.py` (~4169 tok, huge) — Concurrency and performance stress tests for recall engine.
- `test_concurrent_integration.py` (~10941 tok, huge) — Integration tests for concurrent access and partial failure in mind-mem.
- `test_config_keys_have_readers.py` (~5175 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_conflict_resolver_hash_mapping.py` (~320 tok, medium) — Audit trail must print the WINNER's hash next to Winner, not block_a's.
- `test_conflict_resolver_id_counter.py` (~744 tok, large) — Proposal ids must never be re-minted over ids already in the file.
- `test_conflict_resolver.py` (~2363 tok, huge) — Tests for conflict_resolver.py — zero external deps (stdlib unittest)."""
- `test_connection_manager_close_all_threads.py` (~1926 tok, huge) — Regression tests: close() must reach read connections in every thread.
- `test_connection_manager.py` (~2536 tok, huge) — Tests for ConnectionManager — SQLite connection pooling with read/write separation (#466)."""
- `test_connection_release_belief_and_v4_probes.py` (~2634 tok, huge) — Every SQLite connection these modules open is CLOSED, not just committed.
- `test_conn_manager_cache_is_bounded.py` (~847 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_consensus_vote.py` (~1137 tok, large) — v3.3.0 — quorum-based consensus voting on contradictions."""
- `test_consensus_vote_trust.py` (~441 tok, medium) — Regression tests: configured namespace trust actually reaches the tally.
- `test_consensus_vote_wiring.py` (~4142 tok, huge) — Restore-44 slice 5 — ``consensus_vote`` wired into ``conflict_resolver``.
- `test_consolidation_index_path.py` (~2113 tok, huge) — Regression: consolidation tools must read the index the product writes.
- `test_consolidation_maturity_gate.py` (~3750 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_constants.py` (~371 tok, medium) — Tests for recall constants module."""
- `test_content_source_provenance.py` (~5229 tok, huge) — Tests for content-provenance tagging (roadmap T-001).
- `test_context_core.py` (~3175 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_context_pack.py` (~2584 tok, huge) — Tests for context_pack rules: adjacency, diversity, pronoun rescue."""
- `test_context_pack_scripts.py` (~673 tok, large) — Tests for context packing via scripts._recall_context."""
- `test_contradiction_detector.py` (~5871 tok, huge) — Tests for contradiction_detector.py — Contradiction detection at governance gate (#432).
- `test_core_export_wiring.py` (~4543 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_core_v140.py` (~2707 tok, huge) — Tests for v1.4.0 core hardening: issues #28, #30, #32, #34."""
- `test_cron_runner_config_fail_closed.py` (~1159 tok, large) — An unreadable toggle file must mean "run nothing", never "run everything".
- `test_cron_runner.py` (~4320 tok, huge) — Tests for cron_runner.py — periodic job orchestration, config loading, subprocess dispatch."""
- `test_cross_encoder_auto_enable.py` (~1801 tok, huge) — v3.3.0 Tier 2 #6 — cross-encoder rerank auto-enables on ambiguous queries.
- `test_cross_encoder_model_cache_key.py` (~1012 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_cross_encoder.py` (~1324 tok, large) — Tests for optional cross-encoder reranker."""
- `test_daemon.py` (~1895 tok, huge) — Tests for the v3.9 background daemon (`mm daemon`)."""
- `test_date_bound_normalisation.py` (~1547 tok, huge) — Date bounds must mean the same thing whatever a Date field looks like.
- `test_date_score.py` (~174 tok, small) — Tests for date scoring function."""
- `test_decompose_query.py` (~223 tok, medium) — Tests for query decomposition."""
- `test_decrypt_file_audit_trail.py` (~1282 tok, large) — Regression test for the `decrypt_file` forensic audit trail
- `test_dedup.py` (~5670 tok, huge) — Tests for dedup.py -- 4-layer deduplication filter."""
- `test_dedup_vector.py` (~1087 tok, large) — Tests for vector-enhanced cosine dedup (Layer 2b)."""
- `test_dependency_audit_closure.py` (~1464 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_detection.py` (~326 tok, medium) — Tests for query detection module."""
- `test_dialogue_diversity.py` (~1519 tok, huge) — One conversation must not be able to occupy the whole answer.
- `test_docs_alignment.py` (~24506 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_docs_claims.py` (~818 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_documented_surfaces_exist.py` (~2128 tok, huge) — Documentation that names a symbol, a backend or an installable extra has
- `test_downgrade_mitigation.py` (~1016 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_dream_cycle_backends.py` (~2824 tok, huge) — Backend-aware dream-cycle maintenance passes (audit bug 11).
- `test_dream_cycle_governed_entities.py` (~3249 tok, huge) — The dream cycle's auto-created entities are governed blocks (AUD-06).
- `test_dream_cycle.py` (~4665 tok, huge) — Tests for dream_cycle.py — autonomous memory enrichment passes."""
- `test_drift_detector_encrypted_backend.py` (~1531 tok, huge) — Drift detection over an ``encrypted`` block-store backend.
- `test_drift_detector.py` (~3914 tok, huge) — Tests for mind-mem semantic belief drift detection (drift_detector.py)."""
- `test_dsn_redaction.py` (~542 tok, large) — Tests for DSN password redaction in mm_cli.
- `test_edge_cases.py` (~4078 tok, huge) — Edge-case and stress tests for mind-mem — block_parser, recall, and MCP server."""
- `test_edge_corroboration.py` (~3178 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_edge_grounded_answer.py` (~4262 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_embedding_augmentation_probe.py` (~1685 tok, huge) — M1 — the embed-vs-store exposure, measured rather than assumed.
- `test_encryption.py` (~2848 tok, huge) — Tests for mind-mem encryption at rest."""
- `test_entity_ingest.py` (~4122 tok, huge) — Tests for the entity_ingest module — extraction, filtering, signal generation."""
- `test_entity_observations.py` (~1975 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_entity_prefetch.py` (~1674 tok, huge) — v3.3.0 Tier 3 #8 — entity-graph prefetch.
- `test_enums.py` (~534 tok, large) — Tests for centralised enums (mind_mem.enums)."""
- `test_error_codes.py` (~2394 tok, huge) — Tests for mind-mem Error Codes module."""
- `test_error_codes_wiring.py` (~823 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_error_paths.py` (~5924 tok, huge) — Error path and edge-case tests for mind-mem — malformed inputs, missing files, bad configs."""
- `test_event_fanout.py` (~1153 tok, large) — v4.0 prep — governance event fan-out."""
- `test_event_fanout_wiring.py` (~7068 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_event_id_filter.py` (~1779 tok, huge) — Unit tests for the event_id recall post-filter.
- `test_every_serving_surface_attests.py` (~10651 tok, huge) — Every door proves what it served — not one of them.
- `test_evidence_bundle.py` (~1562 tok, huge) — v3.3.0 Tier 3 #7 — structured evidence bundle.
- `test_evidence_chain_fork_refusal.py` (~2849 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_evidence_chain_recovery.py` (~9797 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_evidence_forward_compat.py` (~2819 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_evidence_gate_canary.py` (~144 tok, small) — A deliberately skipping test, used only by the evidence gate's self-test.
- `test_evidence_objects.py` (~4263 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_evidence_packer.py` (~5180 tok, huge) — Tests for the evidence packer module."""
- `test_excerpt.py` (~248 tok, medium) — Tests for excerpt generation."""
- `test_expand_query.py` (~265 tok, medium) — Tests for query expansion module."""
- `test_extraction_feedback_durability.py` (~1194 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_extraction_recall_gate.py` (~1749 tok, huge) — Read-path extraction gate + feedback anchoring + graph-edge ACL.
- `test_extractor.py` (~3387 tok, huge) — Tests for the regex NER-lite entity/fact extractor."""
- `test_extractor_windowed_scan.py` (~1838 tok, huge) — Regression tests for the windowed extract_facts scan (issue #530).
- `test_fact_card_context.py` (~1339 tok, large) — A fact card must stay SMALL. It must not become a copy of its parent.
- `test_fact_indexing.py` (~3305 tok, huge) — Tests for Feature 2 (fact card indexing) and Feature 4 (metadata-augmented embeddings)."""
- `test_feature_gate.py` (~7614 tok, huge) — Tests for FeatureGate — the shared config-resolver for retrieval features."""
- `test_federation_connect.py` (~3244 tok, huge) — ``mind-mem-connect`` — the join, and the three ways a join can go wrong.
- `test_federation_lww_vclock_convergence.py` (~1699 tok, huge) — LAST_WRITER_WINS must converge the version vector, not just the log.
- `test_federation_peer_allowlist.py` (~1610 tok, huge) — Regression tests for MIND_MEM_FED_PEERS operator-side peer allowlist
- `test_federation_resolve_race.py` (~553 tok, large) — resolve_conflict must NOT run vclock upserts when its UPDATE was a no-op.
- `test_feedback_credit.py` (~1375 tok, large) — Regression gate for Group I per-hit feedback-quality credit (Stage 3.1).
- `test_feedback_success_bench.py` (~1129 tok, large) — Regression gate for benchmarks/feedback_success_bench.py (Group I item 3).
- `test_field_audit.py` (~1399 tok, large) — Tests for mind-mem per-field mutation audit (field_audit.py)."""
- `test_field_extraction.py` (~201 tok, medium) — Tests for field token extraction."""
- `test_filelock.py` (~7529 tok, huge) — Tests for filelock.py — cross-platform advisory locking."""
- `test_filelock_stress.py` (~1196 tok, large) — Stress tests for mind-mem file locking under contention."""
- `test_flag_composition.py` (~6812 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_flag_registry.py` (~8442 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_frames_disclosure.py` (~3138 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_fts5_bm25_weights.py` (~391 tok, medium) — bm25() weights must align 1:1 with the indexed blocks_fts columns.
- `test_fts_fallback.py` (~4472 tok, huge) — Tests for FTS fallback behavior, recall envelope structure, block size cap,
- `test_governance_bench.py` (~815 tok, large) — Tests for mind-mem governance benchmark suite."""
- `test_governance_concurrency.py` (~1368 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_governance_raft.py` (~1398 tok, large) — v4.0 prep — Raft-style consensus wrapper for governance writes."""
- `test_governance_receipt_ts_anchor.py` (~484 tok, medium) — # Copyright 2026 STARGA, Inc.
- `test_governance_scan_backends.py` (~2722 tok, huge) — Backend-aware governance ``scan`` — audit bugs #3 / #10.
- `test_governed_artifact_writes.py` (~7353 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_delete_clear_enumeration.py` (~4272 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_delete_compaction.py` (~5793 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_delete_forward_compat.py` (~3018 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_delete_http.py` (~9789 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_delete_mcp_tool.py` (~6382 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_delete_stores.py` (~8439 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_delete_unmapped_prefix.py` (~2991 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_detector_writes.py` (~5345 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_edge_scope.py` (~3361 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_restore_seam.py` (~5688 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_signal_and_edge.py` (~9813 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_governed_write_is_screened.py` (~1252 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_governed_write_paths.py` (~10704 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_granularity_align.py` (~3310 tok, huge) — Tests for granularity_align — named merge operation (Group H, v4.0.x).
- `test_granularity_align_wiring.py` (~6427 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_graph_boost.py` (~6077 tok, huge) — Tests for graph boost, context packing, config validation, and block cap.
- `test_graph_boost_recall.py` (~315 tok, medium) — Tests for graph_boost recall parameter."""
- `test_graph_ingest.py` (~3700 tok, huge) — Corpus → typed knowledge-graph wiring (extraction → HITL signal → apply).
- `test_graph_recall.py` (~1498 tok, large) — v3.3.0 Tier 1 #2 — multi-hop graph traversal on recall results.
- `test_graph_schema_versioning.py` (~5941 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_grid_search.py` (~1199 tok, large) — Tests for benchmarks/grid_search.py — grid generation and utility functions."""
- `test_group_h_robustness.py` (~7219 tok, huge) — Robustness tests for Group-H modules: edge-cases, error-paths, boundary values.
- `test_group_s_corroboration_breadth.py` (~6857 tok, huge) — Group S — cross-project corroboration as a maturity component.
- `test_grpc_server.py` (~2246 tok, huge) — v4.0 prep — gRPC wire protocol (tests for the grpcio-free handlers)."""
- `test_guardrail_blocks.py` (~10742 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_guardrail_status_and_sources.py` (~1989 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_guardrail_surface_never_grows.py` (~1279 tok, large) — Guardrail surfacing must never return more hits than it was given.
- `test_hash_chain_v2.py` (~5528 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_health_embedding_coverage.py` (~3068 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_hook_installer_force_preserves_siblings.py` (~703 tok, large) — Regression test for the --force clobber bug in hook_installer."""
- `test_hook_installer_mcp_force_preserves_siblings.py` (~1490 tok, large) — Regression test: install_mcp_config(force=True) must merge, not clobber.
- `test_hook_installer_registry.py` (~4945 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_http_auth_fail_closed.py` (~1884 tok, huge) — v3.7.0 H4: HTTP / REST auth must fail CLOSED by default.
- `test_http_read_admission.py` (~10399 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_http_transport_audit_headers.py` (~6628 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_http_transport.py` (~5409 tok, huge) — Tests for the v3.9 HTTP transport adapter.
- `test_hybrid_degraded_marker.py` (~4151 tok, huge) — Tests for the in-band recall degradation marker (Task 2).
- `test_hybrid_expansion_reentrancy.py` (~4145 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_hybrid_recall_fusion_noise.py` (~2099 tok, huge) — Regression gate for the hybrid-recall NOISE bug (empty BM25 arm → 1/(k+1) floor).
- `test_hybrid_recall.py` (~3107 tok, huge) — Tests for hybrid_recall.py -- HybridBackend + RRF fusion."""
- `test_hybrid_search.py` (~832 tok, large) — Tests for hybrid search functionality."""
- `test_identity_seam_is_transport_neutral.py` (~5628 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_importers_field_injection.py` (~1609 tok, huge) — Regression gate: dump content cannot forge a second block field.
- `test_importers_notes.py` (~7184 tok, huge) — Tests for the note-tree and transcript importers.
- `test_importers.py` (~5662 tok, huge) — Tests for the roadmap Group G migration importers (file-based subset).
- `test_importers_quarantine.py` (~5278 tok, huge) — Acceptance gate for import quarantine (``mind_mem.importers.quarantine``).
- `test_inbox.py` (~2419 tok, huge) — Tests for the v3.9 inbox folder ingestion."""
- `test_index_stats_b1.py` (~523 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_ingestion_pipeline_wiring.py` (~6467 tok, huge) — The webhook ingest door — wiring proof for `mm ingest-serve` (5.0.1).
- `test_ingest_tiers.py` (~4614 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_init_workspace_postgres.py` (~1650 tok, huge) — Postgres regression tests for ``init_workspace`` (audit bug #8).
- `test_init_workspace.py` (~4105 tok, huge) — Tests for init_workspace — config validation and workspace scaffolding."""
- `test_injection_framing_e2e.py` (~4777 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_install_script.py` (~750 tok, large) — # pip --user honours PYTHONUSERBASE on every platform; without it a
- `test_integration.py` (~1386 tok, large) — Integration test: full mind-mem lifecycle init → capture → scan → recall."""
- `test_intel_scan.py` (~5959 tok, huge) — Tests for intel_scan.py — contradiction detection, drift analysis, impact graph."""
- `test_intent_classify.py` (~328 tok, medium) — Tests for intent classification."""
- `test_intent_router_adaptive.py` (~3622 tok, huge) — Tests for adaptive intent routing (#470).
- `test_intent_router.py` (~1176 tok, large) — Tests for 9-type intent router."""
- `test_interaction_signals.py` (~3185 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_issue_139_140_recall_infra.py` (~3170 tok, huge) — Regression tests for recall-infra issues #139 and #140.
- `test_issue_526_acl_fail_closed.py` (~682 tok, large) — Regression for issue #526: ACL `_get_request_scope` must fail-closed.
- `test_issue_527_three_way_merge_vclock.py` (~1469 tok, large) — Regression for issue #527: THREE_WAY_MERGE must bump the vclock.
- `test_issue_529_federation_client_hardening.py` (~1202 tok, large) — Regression for issue #529: FederationClient hardening.
- `test_iter_active_blocks.py` (~2267 tok, huge) — Backend-aware active-block enumeration — ``storage.iter_active_blocks``.
- `test_iter_blocks_encrypted_backend.py` (~1758 tok, huge) — Backend-aware enumeration on the ``encrypted`` block-store backend.
- `test_kalman_belief.py` (~3728 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_kg_fusion.py` (~2013 tok, huge) — Typed-knowledge-graph fusion into recall (opt-in, default OFF).
- `test_knowledge_graph.py` (~3753 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_ledger_hierarchy.py` (~8001 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_legacy_restore_symlink_confinement.py` (~1152 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_lifecycle_evidence.py` (~5244 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_lifecycle_filter.py` (~1781 tok, huge) — Unit tests for the optional lifecycle block field and recall filter.
- `test_lifecycle_retention_class.py` (~1751 tok, huge) — A lifecycle loss records the RETENTION CLASS of what was lost (RA.4 fold-in).
- `test_lineage_staleness.py` (~2161 tok, huge) — End-to-end tests for the v3.12 lineage→staleness wiring (Theme C).
- `test_lint_autofix.py` (~3142 tok, huge) — Tests for the lint -> repair-proposal path (mind_mem.lint / lint_autofix).
- `test_lint_wiring.py` (~4237 tok, huge) — ``lint`` is actually reachable — from ``mm lint`` and from the MCP surface.
- `test_llama_cpp_provider_contract.py` (~1294 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_llm_extractor_gate.py` (~2248 tok, huge) — Backend wiring — :func:`mind_mem.llm_extractor._gate_check_local`.
- `test_llm_extractor.py` (~1842 tok, huge) — Tests for the optional LLM entity/fact extractor module."""
- `test_llm_noise_profile.py` (~2359 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_llm_noise_profile_wiring.py` (~7741 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_locomo_suite.py` (~2743 tok, huge) — Tests for the LoCoMo recall harness (self-asserting adapters).
- `test_longmemeval_full_run.py` (~1696 tok, huge) — The full-run driver must score the same pool the canonical harness does.
- `test_maintenance_migrate.py` (~741 tok, large) — v3.2.0 §2.2 — tests for maintenance/ subdivision migration."""
- `test_maintenance_migrate_wiring.py` (~3993 tok, huge) — ``maintenance_migrate`` is actually reachable — from apply and from ``mm``.
- `test_maintenance_scripts_ship.py` (~1434 tok, large) — Every maintenance script listed must actually reach an install.
- `test_make_typecheck_gate.py` (~1661 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mcp_agent_inject.py` (~5423 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mcp_anchor_tools.py` (~2280 tok, huge) — The external-anchor tools wired onto the audit family in 5.0.0.
- `test_mcp_arch_mind_tools.py` (~2320 tok, huge) — Tests for the arch-mind MCP tool wrapper.
- `test_mcp_audit_verify_chain.py` (~2980 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mcp_compiled_truth_add_evidence.py` (~3677 tok, huge) — The one compiled-truth tool that WRITES, and the only one nothing tested.
- `test_mcp_compiled_truth_contradictions.py` (~4713 tok, huge) — ``compiled_truth_contradictions`` — the detective half of the truth pages.
- `test_mcp_compiled_truth_load.py` (~4697 tok, huge) — The compiled-truth read tool -- registered since v3.2.0, never tested.
- `test_mcp_db_error_backstop.py` (~1717 tok, huge) — The MCP tool decorator must not let a backend DB error crash the server.
- `test_mcp_graph_hitl.py` (~1868 tok, huge) — MCP-surface tests for the HITL typed-edge flow + provenance (v4.4.0 Finding 1)
- `test_mcp_http_gate_matches_enforcement.py` (~2152 tok, huge) — The MCP HTTP startup gate must agree with what actually enforces auth.
- `test_mcp_integration.py` (~5530 tok, huge) — MCP transport and auth integration tests (#474).
- `test_mcp_list_cores.py` (~3407 tok, huge) — ``list_cores`` — the read side of the ``.mmcore`` lifecycle, previously unpinned.
- `test_mcp_list_evidence.py` (~4569 tok, huge) — ``list_evidence`` — the audit family's only *reader* of the evidence chain.
- `test_mcp_ontology_load.py` (~4897 tok, huge) — ``ontology_load`` -- the MCP door onto the OWL-lite schema layer.
- `test_mcp_ontology_validate.py` (~4754 tok, huge) — ``ontology_validate`` — the MCP tool, as distinct from the ontology library.
- `test_mcp_pipeline.py` (~1495 tok, large) — Tests for the v3.9.0 pipeline-hash MCP tools."""
- `test_mcp_quality_preview_matches_enforcement.py` (~1206 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_mcp_server.py` (~5336 tok, huge) — Tests for mcp_server.py — tests the MCP server resources and tool logic.
- `test_mcp_thread_leak.py` (~3511 tok, huge) — Regression tests for the MCP-server thread leak (2026-07-04).
- `test_mcp_tools_model.py` (~2909 tok, huge) — Tests for ``mind_mem.mcp.tools.model`` — MCP wrappers for audit / sign / verify."""
- `test_mcp_tools.py` (~277 tok, medium) — Tests for MCP server tool definitions."""
- `test_mcp_tool_surface_v3_2.py` (~2006 tok, huge) — v3.2.0 — consolidated MCP public dispatcher tests."""
- `test_mcp_traverse_graph.py` (~5735 tok, huge) — ``traverse_graph`` — the causal-graph tool that nothing tested.
- `test_mcp_unload_core.py` (~3229 tok, huge) — ``unload_core`` — the MCP tool that had no test anywhere.
- `test_mcp_v140.py` (~6007 tok, huge) — Tests for MCP v1.4.0 features — issues #29, #31, #35, #36.
- `test_mcp_walkthrough_persona.py` (~1953 tok, huge) — Tests for the v3.9.0 MCP walkthrough + persona wrapper tools."""
- `test_medlow_batch12_regressions.py` (~3396 tok, huge) — Regression tests for the batch-12 medium/low audit findings.
- `test_memory_ab_analysis.py` (~2565 tok, huge) — Reps reduction, context poisoning, and disclosure — checked on known inputs.
- `test_memory_ab_harness.py` (~6569 tok, huge) — The A/B harness's own guarantees, tested rather than asserted in prose.
- `test_memory_ab_placebo.py` (~1051 tok, large) — The placebo arm must be a fair match, or it is a second memory arm.
- `test_memory_ab_report.py` (~3423 tok, huge) — Pooling stratum artifacts into one delta -- and refusing to pool dishonestly.
- `test_memory_index.py` (~3017 tok, huge) — Tests for the auto-generated hierarchical index (Group C).
- `test_memory_ops_backend_parity.py` (~2267 tok, huge) — ``delete_memory_item`` and ``memory_health`` on a non-Markdown backend.
- `test_memory_ops_postgres_backend.py` (~3255 tok, huge) — Backend-aware memory_ops tools — Postgres parity (audit bug 5).
- `test_memory_practical_e2e.py` (~2401 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_memory_tiers.py` (~3479 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_merkle_tree.py` (~3837 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mic_map_accel.py` (~1848 tok, huge) — Regression tests for the optional Cython accelerator at
- `test_mic_map_adversarial.py` (~3397 tok, huge) — Adversarial corpus for ``mind_mem.mic_map`` parsers.
- `test_mic_map_bench.py` (~4575 tok, huge) — pytest-benchmark suite for ``mind_mem.mic_map``.
- `test_mic_map_cli.py` (~1659 tok, huge) — Integration tests for the ``mm mic`` CLI subcommand.
- `test_mic_map_fuzz.py` (~2471 tok, huge) — Property-based fuzz tests for ``mind_mem.mic_map``.
- `test_mic_map_mcp.py` (~1812 tok, huge) — Integration tests for the MIC/MAP MCP tools (``mic_convert_tool``,
- `test_mic_map.py` (~3230 tok, huge) — Tests for ``mind_mem.mic_map`` — STARGA mic@2 / mic-b serialization.
- `test_mic_map_stream.py` (~2770 tok, huge) — Streaming-parser tests for ``mind_mem.mic_map.parse_micb_stream``.
- `test_mind_ffi.py` (~1632 tok, huge) — Tests for MIND FFI module."""
- `test_mind_kernels_v3_3.py` (~998 tok, large) — Kernel-loading tests for v3.3.0 features.
- `test_mind_kernels_wiring.py` (~5671 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mindllm_backend.py` (~1324 tok, large) — Regression tests for the MindLLM backend (roadmap v4.0.15).
- `test_mm_cli_audit_pinned_tilde.py` (~1077 tok, large) — ``mm audit-pinned`` must resolve a ``~`` config path before deriving the
- `test_mm_cli_bind_arms_the_gate.py` (~4579 tok, huge) — ``mm bind`` — the missing command that arms GovernanceGate step 1.
- `test_mm_cli_chain_recover.py` (~2197 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mm_cli_config_set.py` (~3637 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mm_cli_debug.py` (~3352 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_mm_cli_keyfile_perms.py` (~1473 tok, large) — ``mm sign-model --generate-key`` and the permissions it claims.
- `test_mm_doctor_postgres_hint.py` (~1128 tok, large) — Regression test: mm doctor must emit a clear hint when backend=postgres
- `test_model_audit.py` (~3557 tok, huge) — Tests for ``mind_mem.model_audit`` — checkpoint static-security audit.
- `test_model_audit_unreadable_files.py` (~1411 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_model_gate.py` (~3410 tok, huge) — Tests for ``mind_mem.model_gate`` — load-gate registry."""
- `test_model_provenance.py` (~2391 tok, huge) — Tests for ``mind_mem.model_provenance`` — base_model allowlist."""
- `test_model_signing.py` (~2301 tok, huge) — Tests for ``mind_mem.model_signing`` — Ed25519 manifest signing."""
- `test_mrs_wiring.py` (~6750 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_multi_file_recall.py` (~329 tok, medium) — Tests for recall across multiple files."""
- `test_multi_modal_wiring.py` (~8022 tok, huge) — ``multi_modal`` is actually reachable — and the door it opens quarantines.
- `test_namespace_relevance_floors.py` (~1645 tok, huge) — M3 — a relevance floor is a PER-NAMESPACE property, measured not chosen.
- `test_namespace_retrieval_reachability.py` (~1260 tok, large) — M2 — namespace retrieval reachability, asserted EMPIRICALLY.
- `test_namespaces.py` (~5844 tok, huge) — Tests for namespaces.py — zero external deps (stdlib unittest)."""
- `test_network_audit_headers.py` (~5723 tok, huge) — Audit headers propagate end-to-end — roadmap v4.0.0 Group D (RM-2291).
- `test_network_cert_pinning.py` (~5212 tok, huge) — Certificate pinning is opt-in, and when it is on it refuses — RM-2290 / RM-2382.
- `test_network_tls_floor.py` (~3100 tok, huge) — The TLS 1.3 floor holds by construction — roadmap v4.0.0 Group D (RM-2290).
- `test_niah.py` (~5014 tok, huge) — Needle In A Haystack (NIAH) benchmark for mind-mem recall.
- `test_no_silent_success_paths.py` (~3801 tok, huge) — Regressions for paths that used to report success while doing the wrong thing.
- `test_no_vacuous_skips.py` (~7101 tok, huge) — A skipped test reads as a pass — so the skip surface itself needs a gate.
- `test_novel_term_gate.py` (~2202 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_observability.py` (~791 tok, large) — Tests for observability.py — structured logging and metrics."""
- `test_observation_axis.py` (~3330 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_observation_compress.py` (~2754 tok, huge) — Tests for observation_compress module.
- `test_oidc_admin_enforcement.py` (~1830 tok, huge) — v3.2.1 — OIDC JWTs must pass through ``_require_admin`` checks.
- `test_oidc_auth.py` (~3323 tok, huge) — Tests for OIDCProvider / OIDCConfig in src/mind_mem/api/auth.py."""
- `test_oidc_key_discovery_and_audience.py` (~2358 tok, huge) — Regression tests: where the signing keys come from, and who a token is for.
- `test_oidc_pyjwt_security.py` (~2178 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_okf_bundle_round_trip_fidelity.py` (~1605 tok, huge) — An OKF bundle must survive its own writer, and a dropped concept must be loud.
- `test_okf_export.py` (~3745 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_ollama_host_resolver.py` (~2781 tok, huge) — Tests for the shared ollama base-URL resolver (v4.3.1).
- `test_one_corpus_definition.py` (~9425 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_online_trainer_wiring.py` (~8382 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_ontology_predicate_constraints.py` (~2315 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_ontology.py` (~2306 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_outcome_attribution_bounds.py` (~3311 tok, huge) — Abuse bounds for outcome attribution — one reporter, one vote.
- `test_outcome_attribution.py` (~4120 tok, huge) — Regression gate for outcome attribution — did the memory actually help?
- `test_paired_scorecard_gate.py` (~6587 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_payload_admission.py` (~5425 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_personas.py` (~1336 tok, large) — Tests for the v3.9 persona-aware recall projection."""
- `test_pg_block_store_ping.py` (~738 tok, large) — Tests for ``PostgresBlockStore.ping()`` — active backend health probe.
- `test_pg_pool_autocommit_isolation.py` (~3285 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_pg_restore_file_path.py` (~1187 tok, large) — restore() must preserve each block's file_path (routing metadata).
- `test_pipeline_hash.py` (~3381 tok, huge) — Tests for v3.9 hash-of-code pipeline invalidation."""
- `test_postgres_active_admission.py` (~6670 tok, huge) — R2-06 — on Postgres, ``blocks.active`` must mean what admission means.
- `test_postgres_block_store.py` (~8589 tok, huge) — v3.2.0 §1.4 PR-5 — PostgresBlockStore integration tests.
- `test_postgres_partial_import.py` (~992 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_postgres_pool_shutdown.py` (~1303 tok, large) — Regression coverage for process-wide Postgres pool shutdown."""
- `test_postgres_replica_routing.py` (~2241 tok, huge) — v3.2.0 — tests for read-replica routing in ReplicatedPostgresBlockStore."""
- `test_prefetch_context.py` (~1496 tok, large) — Tests for prefetch_context() in recall.py."""
- `test_prefix_cache.py` (~3140 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_protection.py` (~2068 tok, huge) — Tests for mind_mem.protection (v3.3.0+)."""
- `test_q1616_preimage.py` (~1496 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_quality_gate.py` (~1971 tok, huge) — Tests for the v3.11.0 deterministic block quality gate.
- `test_quality_gate_recent_window.py` (~2480 tok, huge) — quality_gate rule 6 (``near_duplicate``) must actually execute in the product.
- `test_quality_gate_skipped_rules.py` (~794 tok, large) — A rule that did not run must not look like a rule that passed.
- `test_quality_gate_strict_mode.py` (~2937 tok, huge) — Tests for v3.12.0 Theme B: quality-gate config plumbing + propose_update wiring.
- `test_quarantine_redteam.py` (~7739 tok, huge) — Red-team proof for the claim the whole product rests on.
- `test_query_decomposition.py` (~1604 tok, huge) — Tests for multi-hop query decomposition (#6)."""
- `test_query_expansion_auto_enable.py` (~1270 tok, large) — v3.3.0 Tier 2 #4 — query expansion auto-enables on ambiguous queries.
- `test_query_expansion_multi_provider.py` (~1237 tok, large) — Tests for multi-provider LLM query expansion (OpenAI-compatible endpoints)."""
- `test_query_expansion.py` (~3809 tok, huge) — Tests for query_expansion.py -- multi-query expansion for improved recall."""
- `test_query_planner.py` (~1348 tok, large) — v3.3.0 Tier 1 #1 — query decomposition for multi-hop questions.
- `test_query_term_stemming.py` (~1052 tok, large) — The query and the index must agree on what a word stems to.
- `test_read_surface_admission.py` (~6315 tok, huge) — Every read surface, swept with a three-status canary.
- `test_read_surface_classification.py` (~8832 tok, huge) — The registry-wide read-surface classification — the committed table.
- `test_read_surface_paths.py` (~3776 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_read_surface_resources.py` (~7355 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_recall_admissibility.py` (~8772 tok, huge) — Acceptance gate for recall admissibility — the servability allow-list.
- `test_recall_as_of.py` (~1297 tok, large) — Tests for the ``recall(..., as_of=)`` time-travel plumb-through (roadmap Group B).
- `test_recall_attestation_anchor.py` (~4081 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_recall_attestation_completeness.py` (~3691 tok, huge) — An attestation must distinguish two runs that served different answers.
- `test_recall_attestation.py` (~5631 tok, huge) — Tests for the per-run recall attestation (recall_attestation.py).
- `test_recall_attestation_served_backend.py` (~3590 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_recall_attestation_unrequested_leg.py` (~789 tok, large) — An unrequested vector leg must not be attested as degraded.
- `test_recall_attestation_v2.py` (~8107 tok, huge) — Acceptance gate for the ``RECALL_ATTEST_v2`` preimage.
- `test_recall_cache_chain_head.py` (~2442 tok, huge) — The recall cache belongs to a corpus state, not to a clock.
- `test_recall_cache.py` (~1916 tok, huge) — Tests for v3.2.0 distributed recall cache (LRU + Redis)."""
- `test_recall_cli_workspace.py` (~2346 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_recall_clock_guard.py` (~5537 tok, huge) — Every leg of the recall scoring path, proven to read no clock.
- `test_recall_concurrent.py` (~344 tok, medium) — Tests for concurrent recall queries."""
- `test_recall_context_field.py` (~263 tok, medium) — Tests for context field in blocks."""
- `test_recall_cross_encoder.py` (~1565 tok, huge) — Tests for cross-encoder reranker integration in recall pipeline."""
- `test_recall_date_field.py` (~315 tok, medium) — Tests for date field in recall results."""
- `test_recall_detection.py` (~1523 tok, huge) — Tests for _recall_detection.py — query type classification and text extraction."""
- `test_recall_determinism.py` (~4983 tok, huge) — The determinism seam: recall is a pure function of (corpus, config, scoring_instant).
- `test_recall_edge_cases.py` (~570 tok, large) — Edge case tests for recall engine."""
- `test_recall_empty_query_types.py` (~322 tok, medium) — Tests for various empty/minimal query types."""
- `test_recall_empty_workspace.py` (~134 tok, small) — Tests for recall on empty workspaces."""
- `test_recall_expansion_no_overbroad_synonyms.py` (~811 tok, large) — Regression tests for over-broad synonym entries in _QUERY_EXPANSIONS.
- `test_recall_explain.py` (~4352 tok, huge) — Tests for the explain=True flag on recall and hybrid_search MCP tools.
- `test_recall_filter_pushdown.py` (~5733 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_recall_format_cache_isolation.py` (~1598 tok, huge) — ``format`` is not in the recall-cache key, so it must not be applied inside it.
- `test_recall_hot_path_5_0_2.py` (~4239 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_recall_intent_router.py` (~1212 tok, large) — Tests for IntentRouter integration in recall pipeline."""
- `test_recall_large_workspace.py` (~343 tok, medium) — Tests for recall with large workspaces."""
- `test_recall_limit.py` (~395 tok, medium) — Tests for recall limit parameter behavior."""
- `test_recall_metadata.py` (~1345 tok, large) — Tests for A-MEM block metadata integration in recall pipeline."""
- `test_recall_pgvector_engagement.py` (~2392 tok, huge) — Regression tests for the Postgres pgvector recall-engagement audit.
- `test_recall_post_filters.py` (~310 tok, medium) — recall() applies lifecycle/event_id/min_maturity on EVERY dispatch path.
- `test_recall_postgres_backend.py` (~2914 tok, huge) — Backend-aware recall dispatch — Postgres parity (audit bug 1).
- `test_recall_priority.py` (~414 tok, medium) — Tests for priority boost in recall."""
- `test_recall.py` (~3898 tok, huge) — Tests for recall.py — zero external deps (stdlib unittest)."""
- `test_recall_quality_locomo.py` (~2613 tok, huge) — LoCoMo recall-quality regression gate.
- `test_recall_query_id_join.py` (~6647 tok, huge) — RA.1's residual, closed: the recall envelope publishes the run identity.
- `test_recall_recursion_fix.py` (~2069 tok, huge) — Regression tests for the recall ↔ query_index mutual recursion bug.
- `test_recall_references.py` (~270 tok, medium) — Tests for reference-based recall."""
- `test_recall_rerank_depth_and_score_contract.py` (~12176 tok, huge) — One score contract (F1) + rerank depth (F2).
- `test_recall_reranking.py` (~2740 tok, huge) — Tests for _recall_reranking.py — deterministic reranker + LLM rerank."""
- `test_recall_scoring_dates.py` (~1424 tok, large) — A slash-dated corpus must not silently lose its temporal signal.
- `test_recall_scoring_order.py` (~646 tok, large) — Tests for recall result scoring order."""
- `test_recall_scoring_timezone_determinism.py` (~2523 tok, huge) — Recency scoring must not depend on the host's timezone.
- `test_recall_source_field.py` (~279 tok, medium) — Tests for source field in recall results."""
- `test_recall_speaker.py` (~263 tok, medium) — Tests for speaker-based recall."""
- `test_recall_status_boost.py` (~399 tok, medium) — Tests for status boost in recall."""
- `test_recall_sufficiency.py` (~1295 tok, large) — Regression gate for Group I item 2 recall-sufficiency score.
- `test_recall_supersedes.py` (~216 tok, medium) — Tests for supersedes field in recall."""
- `test_recall_tags.py` (~320 tok, medium) — Tests for tag-based recall."""
- `test_recall_temporal.py` (~2800 tok, huge) — Tests for _recall_temporal.py — time-aware hard filters for temporal queries."""
- `test_recall_time_bounded.py` (~1977 tok, huge) — Regression tests for time-bounded recall (roadmap v4.0.0 Group E).
- `test_recall_vector.py` (~5496 tok, huge) — Tests for recall_vector.py — VectorBackend semantic search."""
- `test_recall_wire_owner.py` (~3251 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_recall_workspace_zero_index_probe.py` (~917 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_recompaction_bench.py` (~7349 tok, huge) — Tests for bench/recompaction_bench.py — the recompaction scalar metric.
- `test_recompaction.py` (~1962 tok, huge) — Tests for recompaction.py — iterative re-compression to a fixed point.
- `test_release_alerts_gate.py` (~6554 tok, huge) — Tests for the code-scanning alerts release gate.
- `test_release_preflight_gates.py` (~5444 tok, huge) — Tests for the release-preflight gates.
- `test_replay_check.py` (~4122 tok, huge) — Replay — does the ledger corroborate what one recall attestation claims?
- `test_repo_task_generation.py` (~3390 tok, huge) — Tests for the real-repo A/B task generator.
- `test_repo_task_mining_exclusion_scope.py` (~1303 tok, large) — The shared-service pre-exclusion must scan everything the harness executes.
- `test_repo_task_repeats_floor.py` (~1012 tok, large) — ``repeats`` below two silently deleted the determinism guarantee.
- `test_repo_task_tar_extraction_guard.py` (~811 tok, large) — The tar-extraction guard in ``bench.repo_task_validation``.
- `test_repro_cross_drive_paths.py` (~612 tok, large) — A package written to another drive must not take down the run.
- `test_repro_disclosure_consistency.py` (~1312 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_repro_package.py` (~2585 tok, huge) — The verifier must be able to FAIL.
- `test_required_named_controls.py` (~2126 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_rerank_debug.py` (~342 tok, medium) — Tests for rerank debug mode."""
- `test_rerank_ensemble.py` (~1531 tok, huge) — v3.3.0 Tier 4 #9 — reranker ensemble via Borda count.
- `test_reranker_model_cache.py` (~1909 tok, huge) — Reranker weights load once per process, not once per query.
- `test_reranking.py` (~246 tok, medium) — Tests for reranking module."""
- `test_rest_admin_gate_api_key_only.py` (~2102 tok, huge) — The REST admin gate must fire in an API-key-only deployment.
- `test_rest_api_oidc.py` (~2737 tok, huge) — Tests for OIDC callback + admin API key endpoints (v3.2.0)."""
- `test_rest_api.py` (~4314 tok, huge) — Tests for the mind-mem REST API layer (v3.2.0).
- `test_rest_audit_headers.py` (~1517 tok, huge) — Regression tests for the audit-header middleware (roadmap v4.0.0 Group D).
- `test_rest_docs_and_bucket_hardening.py` (~1129 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_rest_hardening_batch.py` (~2443 tok, huge) — REST-layer defects found inside files that also carried a HIGH finding.
- `test_restore44_parked_modules.py` (~6241 tok, huge) — The parked modules from the 5.0.0 restore: still here, still whole, still parked.
- `test_restore_does_not_rewind.py` (~10594 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_restore_is_gated_at_the_seam.py` (~5004 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_restore_record_manifest_containment.py` (~2236 tok, huge) — A snapshot MANIFEST.json cannot make the restore record name foreign blocks.
- `test_retention_class.py` (~2883 tok, huge) — RA.4 — the retention class, and the two things it must refuse to be.
- `test_retrieval_diagnostics.py` (~2428 tok, huge) — Tests for retrieval diagnostics (#428), corpus isolation (#429), and intent instrumentation (#430)."""
- `test_retrieval_graph.py` (~2242 tok, huge) — Tests for retrieval_graph.py — retrieval logging, co-retrieval graph, hard negatives."""
- `test_retrieval_trace.py` (~978 tok, large) — Tests for v3.3.0 per-feature retrieval attribution."""
- `test_retrieval_trace_wiring.py` (~3142 tok, huge) — ``retrieval_trace`` wired into the live recall pipeline.
- `test_review_batch.py` (~2787 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_review_cli.py` (~3748 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_review_evidence.py` (~1361 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_review_hardening.py` (~4758 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_review_metrics.py` (~1233 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_review_no_autoapprove.py` (~1499 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_review_preview.py` (~1291 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_review_preview_routed_writes.py` (~1530 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_review_queue.py` (~2038 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_review_render.py` (~2348 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_rm3_expand.py` (~321 tok, medium) — Tests for RM3 query expansion."""
- `test_roadmap_hygiene.py` (~4798 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_roadmap_ticks_gate.py` (~2911 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_schema_version.py` (~2454 tok, huge) — Tests for schema_version.py — zero external deps (stdlib unittest)."""
- `test_scope_outcome_is_truthful.py` (~6877 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_scoring_ledger_boundary.py` (~1120 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_scoring.py` (~337 tok, medium) — Tests for BM25 scoring functions."""
- `test_sdk_js_packaging.py` (~2404 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_sdk_openapi_drift.py` (~3346 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_sdk_release_versioning.py` (~1752 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_sdk_route_conformance.py` (~1778 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_security_floors_pinned.py` (~1059 tok, large) — Every advisory-bearing dependency keeps its declared security floor.
- `test_security_scanning_alerts.py` (~1395 tok, large) — Regression tests for code-scanning alerts #189 and #192.
- `test_self_editing_old_content.py` (~980 tok, large) — ``propose_edit`` must snapshot the block's real current content.
- `test_self_update.py` (~2727 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_served_ledger_concurrency.py` (~6866 tok, huge) — The served ledger under a SECOND WRITER — the two shapes production has.
- `test_served_ledger.py` (~12023 tok, huge) — RA.1 — the served-set ledger: proof of what was served, joinable to outcome.
- `test_session_boost.py` (~1488 tok, large) — v3.3.0 Tier 2 #5 — session-boundary preservation via recall-side boost.
- `test_session_summarizer_door.py` (~3747 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_session_summarizer.py` (~3976 tok, huge) — Comprehensive tests for mind_mem/session_summarizer.py.
- `test_sharded_pg.py` (~4415 tok, huge) — v4.0 prep — sharded Postgres routing tests (mock underlying stores)."""
- `test_silent_failure_regressions.py` (~3438 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_skeptical_query.py` (~194 tok, small) — Tests for skeptical query detection."""
- `test_skill_opt_adapters.py` (~943 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_skill_opt_analyzer.py` (~3122 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_skill_opt_config_orchestrator_path.py` (~798 tok, large) — The orchestrator location must be settable from OUTSIDE the installed package.
- `test_skill_opt_fleet_bridge.py` (~2105 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_skill_opt_governed_submission.py` (~1450 tok, large) — ``skill_opt.validator.submit_to_governance`` stages a governed block (AUD-06).
- `test_skill_opt.py` (~3373 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_skill_opt_test_runner.py` (~1506 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_skill_opt_validation_quorum.py` (~569 tok, large) — The skill-mutation acceptance vote needs an actual electorate.
- `test_smart_chunker_code.py` (~1135 tok, large) — Tests for code-aware chunking in smart_chunker.py."""
- `test_smart_chunker.py` (~8973 tok, huge) — Tests for smart_chunker.py — semantic-boundary document chunking."""
- `test_smart_chunker_wiring.py` (~4909 tok, huge) — ``smart_chunker`` wired into the BM25 chunk-scoring boost.
- `test_snapshot_path_confinement.py` (~1455 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_snapshot_snap_id.py` (~5713 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_spec_binding.py` (~3156 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_speculative_prefetch.py` (~3071 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_sqlite_handle_close_tool_output_hnsw.py` (~2197 tok, huge) — Every SQLite connection ``tool_output/store.py`` opens must be CLOSED before
- `test_sqlite_index_backends.py` (~2781 tok, huge) — Backend-parity regression tests for ``sqlite_index`` (audit bugs 4, 9, 13, 14).
- `test_sqlite_index.py` (~5547 tok, huge) — Tests for sqlite_index.py — SQLite FTS5 index for mind-mem recall."""
- `test_stopwords.py` (~247 tok, medium) — Tests for stopword handling."""
- `test_storage_factory.py` (~2902 tok, huge) — Tests for mind_mem.storage.get_block_store factory (v3.2.0)."""
- `test_streaming_front_gate.py` (~5460 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_streaming.py` (~2023 tok, huge) — v3.3.0 — back-pressure-aware streaming ingest queue."""
- `test_supersede_pointer.py` (~1697 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_tags_field_shapes.py` (~3424 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_task_frames.py` (~7349 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_telemetry.py` (~2829 tok, huge) — Tests for src/mind_mem/telemetry.py.
- `test_temporal_decay_scoring.py` (~863 tok, large) — v3.3.0 Tier 1 #3 — half-life decay on block ``Created``/``Date`` field.
- `test_temporal.py` (~223 tok, medium) — Tests for temporal filtering module."""
- `test_tenant_audit.py` (~2833 tok, huge) — v4.0 prep — per-tenant audit chain façade."""
- `test_tenant_kms.py` (~1916 tok, huge) — v4.0 prep — per-tenant KMS envelope encryption."""
- `test_tenant_kms_wiring.py` (~3617 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_text_io_is_utf8.py` (~4121 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_tier_axis_collapse.py` (~1611 tok, huge) — RA.0 — one tier axis, and the other ladders deleted rather than abstracted.
- `test_tier_decay.py` (~924 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_tier_manager_releases_descriptors.py` (~754 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_tmpdir_containment.py` (~702 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_tokenization.py` (~436 tok, medium) — Tests for tokenization module."""
- `test_token_rotation.py` (~1766 tok, huge) — Regression tests for the token rotation primitive (roadmap v4.0.x).
- `test_tool_output_postgres_backend.py` (~1615 tok, huge) — ``ToolOutputStore`` on the Postgres backend the module advertises.
- `test_tool_output.py` (~2168 tok, huge) — Tests for mind_mem.tool_output — the context-offload store (§5).
- `test_tracking_online_trainer_wiring.py` (~1519 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_tracking_wiring.py` (~8018 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_train_mind_mem_4b.py` (~962 tok, large) — Smoke tests for benchmarks/train_mind_mem_4b.py.
- `test_trajectory.py` (~2396 tok, huge) — Tests for trajectory.py — trajectory memory block operations."""
- `test_trajectory_wiring.py` (~6650 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_transcript_capture.py` (~3235 tok, huge) — Tests for transcript_capture.py — zero external deps (stdlib unittest)."""
- `test_trust_scores.py` (~4581 tok, huge) — Standalone trust surface — determinism, zero-regression, poisoning.
- `test_trust_signals.py` (~1433 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_truth_score.py` (~1930 tok, huge) — v3.3.0 — probabilistic truth score.
- `test_typed_edges_group_h.py` (~2875 tok, huge) — Tests for Group H typed-edge additions: supports, derived_from, edge_aware_boost.
- `test_typed_edges.py` (~3442 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_unanchored_blocks.py` (~4814 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_uncertainty_propagation.py` (~2158 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_uncertainty_propagation_wiring.py` (~5002 tok, huge) — ``uncertainty_propagation`` wired into the two multi-hop walks that need it.
- `test_unicode_edge_cases.py` (~2155 tok, huge) — Tests for Unicode and edge case handling across mind-mem modules."""
- `test_upsert_slots.py` (~2442 tok, huge) — M4 — enum-keyed upsert slots: prevent contradiction STRUCTURALLY.
- `test_usage_meter.py` (~4753 tok, huge) — Tests for `mm usage` — local per-day model-call token counter (Group G).
- `test_usage_meter_wiring.py` (~4878 tok, huge) — Model-call token metering AT THE CALL SITES (Group G — `mm usage` wiring).
- `test_v28_completion.py` (~4803 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v320_gaps.py` (~3257 tok, huge) — v3.2.0 gap tests — regression and edge-case coverage for new modules.
- `test_v34_features.py` (~3184 tok, huge) — Tests for v3.4.0 retrieval features.
- `test_v4_block_kinds.py` (~4158 tok, huge) — Tests for the v4 block-kind taxonomy module."""
- `test_v4_block_kinds_wiring.py` (~3522 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v4_block_versioning.py` (~3017 tok, huge) — Tests for v4 block versioning + time-travel (Group B, ``v4.self_editing``).
- `test_v4_circuit_breaker.py` (~5198 tok, huge) — Tests for v4 circuit breaker (round 5 audit, Mistral + GLM 9.9→10)."""
- `test_v4_cognitive_kernel.py` (~2770 tok, huge) — Tests for the v4 Cognitive Mind Kernel registry + dispatcher."""
- `test_v4_concurrency.py` (~1149 tok, large) — v4 concurrency / fuzz tests.
- `test_v4_embedding_pipeline_sources.py` (~1208 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_v4_embedding_pipeline_wiring.py` (~2228 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v4_feature_flags_config_error.py` (~1506 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v4_federation_self_editing_close_connections.py` (~3967 tok, huge) — ``v4.federation`` and ``v4.self_editing`` must close what they open.
- `test_v4_federation_wire.py` (~2829 tok, huge) — Wire-transport tests for v4 federation.
- `test_v4_health_wiring.py` (~2471 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v4_hnsw_kind_index.py` (~2209 tok, huge) — Tests for the HNSW kind-filtered ANN index."""
- `test_v4_hnsw_kind_index_wiring.py` (~3436 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v4_kernels.py` (~4297 tok, huge) — Tests for the v4 kernel strategy implementations.
- `test_v4_kernels_wiring.py` (~3137 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v4_kind_conn_close.py` (~2154 tok, huge) — Regression: ``v4.block_kinds`` / ``v4.kind_summaries`` must CLOSE connections.
- `test_v4_kind_summaries_wiring.py` (~2133 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v4_logging_context_wiring.py` (~4154 tok, huge) — ``v4.logging_context`` is WIRED — 5.0.1 restoration slice.
- `test_v4_metadata_pq_conn_close.py` (~3168 tok, huge) — Regression: ``v4.block_metadata`` / ``v4.pq`` must CLOSE their connections.
- `test_v4_pq.py` (~3447 tok, huge) — Tests for v4 product-quantization (PQ) encoding."""
- `test_v4_pq_wiring.py` (~3102 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_v4_round2_extensions.py` (~2992 tok, huge) — Tests for round-2 audit extensions: federation, self_editing."""
- `test_v4_round3_extensions.py` (~1453 tok, large) — Tests for round-3 audit extensions: observability.
- `test_v4_round4_concurrency.py` (~7591 tok, huge) — Concurrency + adversarial-input tests for round-4 v4 modules.
- `test_v4_round4_extensions.py` (~4530 tok, huge) — Tests for round-4 audit extensions.
- `test_v4_surprise_retrieval.py` (~2282 tok, huge) — Tests for the v4 surprise-weighted retrieval scoring module."""
- `test_v4_vocabulary.py` (~4094 tok, huge) — Tests for v4 vocabulary-bound fields (Group E, ``v4.vocabulary``).
- `test_validate_py.py` (~3560 tok, huge) — Tests for validate_py.py — workspace integrity validator."""
- `test_validate_py_vacuous_crossrefs.py` (~806 tok, large) — ``_check_cross_refs`` must not report an integrity property it never tested.
- `test_validate_sh_deprecation.py` (~573 tok, large) — Pin the runtime deprecation warning on validate.sh.
- `test_validity_gate_contradiction_list_fields.py` (~1553 tok, huge) — Regression gate: c3 must debit blocks named in a *real* CONTRADICTIONS.md.
- `test_validity_gate_extension_composition.py` (~1858 tok, huge) — Regression gate for the INDEPENDENCE of the validity gate's two opt-in
- `test_validity_gate.py` (~1253 tok, large) — Regression gate for the Phase-2 recall validity gate (Stage 2.65).
- `test_validity_provenance_class.py` (~2746 tok, huge) — Regression gate for the validity gate's FIFTH component (provenance class).
- `test_vault_allowlist_separator.py` (~868 tok, large) — The vault allowlist separator, and why a Windows drive letter broke it.
- `test_vault_wikilinks.py` (~1876 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_vector_index_admission.py` (~1350 tok, large) — # Copyright 2026 STARGA, Inc.
- `test_vector_index_provider_chain.py` (~1990 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_vector_index_shape_contract.py` (~4682 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_vector_inertness.py` (~1607 tok, huge) — The vector-leg honesty gauge.
- `test_verify_cli.py` (~4647 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_vocabulary_wiring.py` (~2980 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_walkthrough.py` (~2441 tok, huge) — Tests for the v3.9 dependency-ordered walkthrough."""
- `test_watcher.py` (~2037 tok, huge) — Tests for watcher.py — file change detection for auto-reindex."""
- `test_wide_retrieval.py` (~346 tok, medium) — Tests for wide retrieval parameter."""
- `test_withheld_not_in_stats.py` (~8119 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_witness_pin_race.py` (~2540 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_witness_writer_receipt_truth.py` (~3052 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_workspace_contextvar.py` (~1047 tok, large) — v3.2.1 — regression test for per-request workspace ContextVar scoping.
- `test_workspace_init.py` (~502 tok, large) — Tests for workspace initialization."""
- `test_workspace_structure.py` (~550 tok, large) — Tests for workspace directory structure."""
- `test_world_staleness_config_fallbacks.py` (~1465 tok, large) — A silent fallback is a config the operator thinks is in force and is not.
- `test_world_staleness.py` (~5183 tok, huge) — # Copyright 2026 STARGA, Inc.
- `test_write_path_guarantee.py` (~1559 tok, huge) — # Copyright 2026 STARGA, Inc.
- `_tls_certs.py` (~1957 tok, huge) — Throwaway CA, leaf certificates and a recording TLS peer for the TLS tests.
- `_toml_compat.py` (~462 tok, medium) — # Copyright 2026 STARGA, Inc.
- `_write_path_scan.py` (~11624 tok, huge) — # Copyright 2026 STARGA, Inc.
### `train/`

- `audit_canonical_coverage.json` (~5300 tok, huge) — Keys: threshold, total_probes, total_weak, by_group, weak
- `audit_canonical_coverage.py` (~1479 tok, large) — Audit canonical-answer coverage on every eval probe.
- `audit_semantic_correctness.json` (~5 tok, tiny) — Keys: findings
- `audit_semantic_correctness.py` (~1971 tok, huge) — Cross-check every 'file X ships Y' claim in the corpus against src/.
- `backport_sweep.py` (~1722 tok, huge) — Backport v2.9.0 audit fixes to every prior v2.x release as .post1.
- `build_model_card.py` (~4824 tok, huge) — Generate the HuggingFace model-card README for mind-mem-4b.
- `CORPUS_HASH_v3.11.0` (~21 tok, tiny) — 02b3ba6a1433e25bdbefe3cebf992ca961734850d1e3550e9496905abbadb3b7  build_corpus.p
- `CORPUS_HASH_v3.12.0-fullft` (~21 tok, tiny) — 568d1559631a590e44eeec6716081b4534a40ab5f3047feb622cc225ead9ad01  build_corpus.p
- `eval_harness.py` (~8643 tok, huge) — Eval harness for mind-mem-4b.
- `eval_holdout.py` (~2182 tok, huge) — Held-out paraphrase eval set — runs AFTER training, BEFORE ship.
- `export_gguf.py` (~1274 tok, large) — Export the trained model to GGUF for Ollama / LM Studio / llama.cpp.
- `HF_MODEL_CARD_v4.md` (~3467 tok, huge) — mind-mem-4b v4.1.1
- `merge_and_eval_v4.1.0.py` (~1488 tok, large) — Post-Kaggle: pull LoRA adapter, merge with v4.0.0-base, eval 131 probes.
- `Modelfile.v3.9.0` (~389 tok, medium) — FROM /data/checkpoints/mm-workspace/train-output/mind-mem-4b-Q4_K_M.gguf
- `Modelfile.v4.0.0` (~576 tok, large) — FROM /data/checkpoints/mm-workspace/train-output/mind-mem-4b-Q4_K_M.gguf
- `Modelfile.v4.1.0` (~395 tok, medium) — FROM /data/checkpoints/mm-workspace/gguf-v4.1.0/mind-mem-4b-v4.1.0-Q4_K_M.gguf
- `Modelfile.v4.1.1` (~440 tok, medium) — FROM /data/checkpoints/mm-workspace/gguf-v4.1.1/mind-mem-4b-v4.1.1-Q4_K_M.gguf
- `post_train_chain.sh` (~632 tok, large) — Post-training chain: wait for deploy → verify scp + SHA256 + pod-destroy markers → run eval.
- `post_train_pipeline.sh` (~592 tok, large) — Post-training pipeline for mind-mem-4b v3.9.2 (augmented-corpus retrain).
- `qlora_local_3080.py` (~1182 tok, large) — Local QLoRA fallback on RTX 3080 (10GB VRAM).
- `README.md` (~675 tok, large) — mind-mem-4b training pipeline
- `resume_pod_train.sh` (~1138 tok, large) — Recovery: pod uz2uajluzskmm2 was preempted mid-run. Wake it up,
- `RETRAIN_v3.9.0.md` (~1405 tok, large) — mind-mem-4b — v3.9.0 retrain plan
- `runpod_deploy.py` (~5340 tok, huge) — End-to-end RunPod driver for full-FT on Qwen3.5-4B.
- `runpod_full_ft.py` (~2551 tok, huge) — Full fine-tune of Qwen3.5-4B on RunPod (A100/H100) for mind-mem-4b.
- `ship_gguf_ollama_v4.1.0.py` (~1754 tok, huge) — GGUF + Ollama shipper for mind-mem-4b v4.1.0.
- `ship_gguf_ollama_v4.1.1.py` (~1804 tok, huge) — GGUF + Ollama shipper for mind-mem-4b v4.1.1.
- `spend_guard.py` (~2040 tok, huge) — spend_guard — mechanical interlock on cloud spend.
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

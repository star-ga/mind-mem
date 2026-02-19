# Changelog

All notable changes to mind-mem are documented in this file.

## 1.0.2 (2026-02-18)

**Category distillation, prefetch context, MIND kernel integration, full pipeline wiring**

### Added
- `scripts/category_distiller.py`: Deterministic category detection from block tags/keywords, generates `categories/*.md` thematic summaries with block references and `_manifest.json`
- `prefetch_context()` in `scripts/recall.py`: Anticipatory pre-assembly of likely-needed blocks using intent routing + category summaries
- 2 new MCP tools: `category_summary` (topic-based category retrieval), `prefetch` (signal-based context pre-assembly) — 14→16 total
- 16 MIND kernel source files (`.mind`) with C99 FFI bridge: 7 compiled scoring kernels + 9 configuration kernels
- MIND kernel batch categorization: `category_affinity` + `category_assign` C kernels integrated into category distiller with pure Python fallback
- `is_protected()` module-level function in `mind_ffi.py` for FORTRESS protection detection
- `mind_kernel_protected` field in `index_stats` MCP tool response
- Configurable prompts section in `mind-mem.example.json` (`prompts` + `categories` config keys)
- MemU added to README comparison chart (Full Feature Matrix)
- `tests/test_category_distiller.py`: 13 tests for category detection, distillation, context retrieval
- `tests/test_prefetch_context.py`: 7 tests for signal-based prefetch
- 3 new C category kernels in `lib/kernels.c`: `category_affinity`, `query_category_relevance`, `category_assign`
- FFI wrappers for category kernels in `scripts/mind_ffi.py`
- A-MEM block metadata wired into recall pipeline: importance boost on scoring, access tracking + keyword evolution on results
- IntentRouter wired into recall pipeline: 9-type classification replaces `detect_query_type()`, with backward-compatible mapping and fallback
- Cross-encoder reranking wired into recall pipeline: config-gated neural reranking stage with graceful degradation
- `mind/intent.mind`: Intent router configuration kernel (routing thresholds, graph boost, per-intent weights)
- `mind/cross_encoder.mind`: Cross-encoder configuration kernel (model, blend weight, normalization)
- `docs/api-reference.md`: Complete reference for 16 MCP tools + 8 resources
- `docs/configuration.md`: Every `mind-mem.json` key documented with defaults and examples
- `docs/architecture.md`: 10-section architecture deep dive with ASCII diagrams
- `docs/migration.md`: mem-os → mind-mem migration guide

### Changed
- MCP tool count: 14 → 16
- MIND kernel count: 6 → 16 (14 + 2 new config kernels)
- `reindex` MCP tool now regenerates category summaries automatically

### Security
- Workspace containment check in `_read_file` (path traversal guard)
- Path traversal guard on `FilesTouched` in `create_snapshot`
- Prefer installation scripts over workspace copies in `check_preconditions`
- Renamed `X-MemOS-Token` → `X-MindMem-Token` in server and tests

---

## Pre-fork History (inherited from mem-os)

The entries below document changes made during development as mem-os, before the project was renamed to mind-mem.

### 1.1.3 (2026-02-17)

**Audit round 2: evidence_packer tests, security, functional fixes**

#### Added
- 41 new tests for `evidence_packer.py` (was zero coverage) — covers all public functions, routing, packing strategies, edge cases
- Test for `"never"` pattern triggering negation penalty

#### Security
- Simplified `_ADVERSARIAL_SIGNAL_RE` to eliminate ReDoS risk (removed bounded-repeat branch)
- Moved `_load_env()` from module import scope to `main()` — prevents global env mutation on import
- Added `max_retries >= 1` guard in `_llm_chat()`

#### Fixed
- `"never"` now triggers `has_ever_pattern` penalty (was only `"ever"`) — functional gap in adversarial detection
- Removed duplicate `wasn't` entry in `_DENIAL_RE`
- Removed redundant local `from recall import detect_query_type` (uses module-level global)

#### Changed
- Test count: 520 → 562

### 1.1.2 (2026-02-17)

**Code quality: remaining pre-existing audit findings**

#### Fixed
- JSONL file handle leaked on exception — bare `open()` replaced with context manager
- Duplicate `global detect_query_type` declaration removed (dead code)
- Orphaned unreachable comment after `format_context()` removed
- `_strip_semantic_prefix()` duplication — now delegates to canonical `evidence_packer.strip_semantic_prefix()`

### 1.1.1 (2026-02-17)

**Hardening: pre-existing audit findings in locomo_judge.py**

#### Security
- Restrict `.env` loader to allowlisted API key names only (6 known keys)
- Cap environment variable values at 512 characters
- Clamp judge scores to `[0, 100]` in both JSON parse and regex fallback paths

#### Fixed
- JSONL bare key access (`r["category"]`) → safe `.get()` with try/except
- Malformed JSONL lines now logged and skipped instead of crashing
- Score type validation before aggregation

### 1.1.0 (2026-02-17)

**Adversarial abstention classifier + auto-ingestion pipeline**

Major retrieval quality improvement targeting LoCoMo adversarial accuracy (30.7% → projected 50%+).

#### Added

##### Adversarial Abstention Classifier
- New `scripts/abstention_classifier.py`: deterministic pre-LLM confidence gate
- Computes confidence from 5 features: entity overlap, BM25 score, speaker coverage, evidence density, negation asymmetry
- Below threshold → forces abstention ("Not enough direct evidence") without calling the LLM
- Integrated into `locomo_judge.py` benchmark pipeline (between pack_evidence and answer_question)
- Exposed via `evidence_packer.check_abstention()` for production MCP path
- Conservative default threshold (0.20) — tunable per benchmark run
- 31 new tests covering unit features, integration, and edge cases

##### Auto-Ingestion Pipeline
- `session_summarizer.py`: automatic session summary generation
- `entity_ingest.py`: regex-based entity extraction (projects, tools, people)
- `cron_runner.py`: scheduled transcript scanning and entity ingestion
- `bootstrap_corpus.py`: one-time backfill from existing transcripts
- SHA256 content-hash deduplication in `capture.py`
- JSONL transcript capture in `session-end.sh`
- Per-feature toggles in `mind-mem.json` `auto_ingest` section

### 1.0.1 (2026-02-17)

**Full 10-conv LoCoMo validated: 67.3% Acc>=50 (+9.1pp over 1.0.0)**

Generational improvement in retrieval quality, moving from keyword search to a deterministic reasoning pipeline.

| Metric     | 1.0.0 | 1.0.1     | Delta   |
| ---------- | ----- | --------- | ------- |
| Acc>=50    | 58.2% | **67.3%** | +9.1pp  |
| Mean Score | 54.3  | **61.4**  | +7.1    |
| Acc>=75    | 36.5% | **48.8%** | +12.3pp |

### 1.0.0

Initial release.

- BM25F retrieval with Porter stemming
- Basic query expansion
- 58.2% Acc>=50 on full 10-conv LoCoMo (1986 questions)
- Governance engine: contradiction detection, drift analysis, proposal queue
- Multi-agent namespaces with ACL
- MCP server with token auth
- WAL + backup/restore
- 478 unit tests

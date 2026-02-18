# Changelog

All notable changes to Mind-Mem are documented in this file.

## 1.1.3 (2026-02-17)

**Audit round 2: evidence_packer tests, security, functional fixes**

### Added
- 41 new tests for `evidence_packer.py` (was zero coverage) — covers all public functions, routing, packing strategies, edge cases
- Test for `"never"` pattern triggering negation penalty

### Security
- Simplified `_ADVERSARIAL_SIGNAL_RE` to eliminate ReDoS risk (removed bounded-repeat branch)
- Moved `_load_env()` from module import scope to `main()` — prevents global env mutation on import
- Added `max_retries >= 1` guard in `_llm_chat()`

### Fixed
- `"never"` now triggers `has_ever_pattern` penalty (was only `"ever"`) — functional gap in adversarial detection
- Removed duplicate `wasn't` entry in `_DENIAL_RE`
- Removed redundant local `from recall import detect_query_type` (uses module-level global)

### Changed
- Test count: 520 → 562

## 1.1.2 (2026-02-17)

**Code quality: remaining pre-existing audit findings**

### Fixed
- JSONL file handle leaked on exception — bare `open()` replaced with context manager
- Duplicate `global detect_query_type` declaration removed (dead code)
- Orphaned unreachable comment after `format_context()` removed
- `_strip_semantic_prefix()` duplication — now delegates to canonical `evidence_packer.strip_semantic_prefix()`

## 1.1.1 (2026-02-17)

**Hardening: pre-existing audit findings in locomo_judge.py**

### Security
- Restrict `.env` loader to allowlisted API key names only (6 known keys)
- Cap environment variable values at 512 characters
- Clamp judge scores to `[0, 100]` in both JSON parse and regex fallback paths

### Fixed
- JSONL bare key access (`r["category"]`) → safe `.get()` with try/except
- Malformed JSONL lines now logged and skipped instead of crashing
- Score type validation before aggregation

### Changed
- Test count: 509 → 520 (11 new tests from audit gap-filling)

## 1.1.0 (2026-02-17)

**Adversarial abstention classifier + auto-ingestion pipeline**

Major retrieval quality improvement targeting LoCoMo adversarial accuracy (30.7% → projected 50%+).

### Added

#### Adversarial Abstention Classifier
- New `scripts/abstention_classifier.py`: deterministic pre-LLM confidence gate
- Computes confidence from 5 features: entity overlap, BM25 score, speaker coverage, evidence density, negation asymmetry
- Below threshold → forces abstention ("Not enough direct evidence") without calling the LLM
- Integrated into `locomo_judge.py` benchmark pipeline (between pack_evidence and answer_question)
- Exposed via `evidence_packer.check_abstention()` for production MCP path
- Conservative default threshold (0.20) — tunable per benchmark run
- 31 new tests covering unit features, integration, and edge cases

#### Auto-Ingestion Pipeline
- `session_summarizer.py`: automatic session summary generation
- `entity_ingest.py`: regex-based entity extraction (projects, tools, people)
- `cron_runner.py`: scheduled transcript scanning and entity ingestion
- `bootstrap_corpus.py`: one-time backfill from existing transcripts
- SHA256 content-hash deduplication in `capture.py`
- JSONL transcript capture in `session-end.sh`
- Per-feature toggles in `mind-mem.json` `auto_ingest` section

### Changed
- `evidence_packer.py`: added `check_abstention()` wrapper function
- `locomo_judge.py`: abstention gate inserted in adversarial evaluation pipeline; fixed bare f-string lint warning
- Test count: 478 → 509

## 1.0.2 (2026-02-17)

**Cross-platform CI hardening + professional polish**

Fixes all CI failures across Ubuntu, macOS, and Windows. Every platform × Python version combination now passes.

### Fixes

#### macOS
- Fix `/var` → `/private/var` symlink path traversal in apply engine and WAL — `os.path.realpath()` now resolves workspace paths at all entry points

#### Windows
- Skip bash-dependent tests (`validate.sh`) on Windows
- Fix `FileLock` repr test to use `tempfile.gettempdir()` instead of hardcoded `/tmp`
- Skip `test_lock_file_contains_pid` (PermissionError on open fd)
- Fix path separator in namespace corpus path assertions
- Add monotonic counter to WAL entry IDs (prevents Windows timestamp collision)

#### Thread Safety
- Rewrite `FileLock` with two-layer locking: `threading.Lock` for intra-process + `O_CREAT|O_EXCL` + `flock`/`msvcrt` for cross-process contention
- Stale lock detection with PID-based cleanup

#### CI
- Install `fastmcp>=2.0` in CI (fixes 30+ ModuleNotFoundError failures)
- Drop Python 3.8/3.9 from matrix (FastMCP requires 3.10+)
- Add `fail-fast: false` to prevent single-platform cancellation cascade
- Fix 2 unused variable lint errors in recall.py

#### Docs
- Update Python requirement to 3.10+ in README badge, requirements, and troubleshooting
- Add Claude Code MCP configuration section to README
- Update security section: `os.path.abspath()` → `os.path.realpath()`

---

## 1.0.1 (2026-02-17)

**Full 10-conv LoCoMo validated: 67.3% Acc>=50 (+9.1pp over 1.0.0)**

This release represents a generational improvement in Mind-Mem retrieval quality, moving from keyword search to a deterministic reasoning pipeline.

### Benchmark Results

| Metric     | 1.0.0 | 1.0.1     | Delta   |
|------------|-------|-----------|---------|
| Acc>=50    | 58.2% | **67.3%** | +9.1pp  |
| Mean Score | 54.3  | **61.4**  | +7.1    |
| Acc>=75    | 36.5% | **48.8%** | +12.3pp |

Per-category Acc>=50: Open-domain +10.8pp, Single-hop +12.8pp, Temporal +7.3pp, Multi-hop +6.9pp, Adversarial +5.6pp.

### Changes

#### Retrieval Pipeline
- Wide retrieval: increased candidate pool to top-200 before rerank
- Deterministic rerank with speaker-match, time-proximity, entity-overlap, bigram-coherence, and recency-decay signals
- Speaker-aware extraction and boosting

#### Recall Hardening
- Month name normalization (January->1, etc.)
- Irregular verb lemmatization (went->go, said->say, etc.)
- Controlled synonym expansion with domain-aware terms
- Context packing (append-only post-retrieval):
  - Rule 1: Dialog adjacency (question-answer pair recovery)
  - Rule 2: Multi-entity diversity enforcement
  - Rule 3: Pronoun rescue (antecedent recovery)

#### Adversarial Gating
- Verification-intent regex for broader adversarial detection
- `morph_only` expansion mode for adversarial queries (lemma + months, no semantic synonyms)
- Gated synonym expansion based on query type classification

#### Infrastructure
- SQLite FTS5 backend with scan fallback
- Safe tar restore with path traversal protection
- Enforced MCP token authentication for HTTP transport
- Minimal snapshot apply (O(touched), copy2)
- CI: install pytest in workflow, fix 92 ruff lint warnings

---

## 1.0.0

Initial release.

- BM25F retrieval with Porter stemming
- Basic query expansion
- 58.2% Acc>=50 on full 10-conv LoCoMo (1986 questions)
- Governance engine: contradiction detection, drift analysis, proposal queue
- Multi-agent namespaces with ACL
- MCP server with token auth
- WAL + backup/restore
- 478 unit tests

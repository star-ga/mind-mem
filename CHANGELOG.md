# Changelog

All notable changes to mind-mem are documented in this file.

## 1.9.1 (2026-03-06)

**Stability patch: proposal apply, rollback safety, install bootstrap, and request-scoped MCP auth**

### Fixed
- `check_preconditions()` now runs the integrity scan via `python -m mind_mem.intel_scan` with an explicit package bootstrap path, so apply prechecks work from source checkouts and clean environments
- Minimal snapshot rollback now preserves unrelated pre-existing files by recording cleanup inventory before restore
- Rollback now marks applied proposals as `rolled_back` to keep proposal state aligned with restored workspace contents
- Source checkout entrypoints (`mcp_server.py`, `mind_mem.mcp_entry`, `install.sh`) now bootstrap `src/` correctly, fixing clean-install and script execution failures
- HTTP MCP auth now derives admin access from request token scopes instead of process-wide `MIND_MEM_SCOPE`, while preserving env-based fallback for local stdio usage

### Added
- Regression tests for clean install bootstrap, fresh-workspace prechecks, minimal snapshot orphan cleanup, rollback proposal status sync, and request-scoped MCP authorization

### Changed
- Version: 1.9.0 → 1.9.1

## 1.9.0 (2026-03-05)

**Governance deep stack: 8 new modules for audit, drift, causality, coding schemas, auto-resolution, benchmarks, and encryption**

### Added
- **Hash-chain mutation log** (`audit_chain.py`): SHA-256 chained append-only JSONL ledger with genesis block, tamper detection, chain verification, query/export APIs
- **Per-field mutation audit** (`field_audit.py`): SQLite-backed field-level change tracking with before/after diffs, agent attribution, chain integration
- **Semantic belief drift detection** (`drift_detector.py`): Character trigram Jaccard similarity (zero external deps), modality conflict detection, belief snapshots and timeline tracking
- **Temporal causal dependency graph** (`causal_graph.py`): Directed edges with cycle detection (BFS), staleness propagation, causal chain traversal (DFS)
- **Coding-native memory schemas** (`coding_schemas.py`): 5 block types (ADR, CODE, PERF, ALGO, BUG) with regex auto-classification, template generation, metadata extraction
- **Auto contradiction resolution** (`auto_resolver.py`): Extends conflict_resolver with preference learning, side-effect analysis via causal graph, confidence scoring
- **Governance benchmark suite** (`governance_bench.py`): Contradiction detection rate, audit completeness, drift detection performance, scalability metrics harness
- **Encryption at rest** (`encryption.py`): HMAC-SHA256 keystream (CTR-like), PBKDF2 key derivation (600k iterations), encrypt-then-MAC, file encryption, key rotation

### Testing
- 145 new tests across 8 test files (audit_chain: 28, field_audit: 12, drift_detector: 17, causal_graph: 22, coding_schemas: 23, auto_resolver: 11, governance_bench: 9, encryption: 23)

### Changed
- Version: 1.8.2 → 1.9.0

## 1.8.2 (2026-03-04)

**Cleanup: import hygiene, cross-encoder batching, integration tests**

### Fixed
- Removed dead `sys.path.insert` and `scripts/` references from 7 benchmark/test files
- Fixed bare imports in `locomo_judge.py` to use `mind_mem.*` prefix (6 modules)

### Added
- `batch_size` parameter on `CrossEncoderReranker.rerank()` (default: 32) — prevents OOM on large candidate sets
- `DeprecationWarning` on `hybrid_search` MCP tool (use `recall(backend="hybrid")` instead)
- 8 integration tests covering full pipeline: init → index → recall → propose
- CI now runs unit and integration tests as separate steps

### Changed
- CI dependency bumps: pytest <10.0, pytest-cov <8.0, pytest-benchmark <6.0, actions/setup-python 6.2.0, actions/upload-artifact 7.0.0

## 1.8.1 (2026-02-27)

**Polish: cross-platform fixes, docs alignment, project metadata**

### Fixed
- Windows CI: snapshot manifests now use POSIX separators for cross-platform portability
- Windows CI: `restore_snapshot()` and `_cleanup_orphans_from_manifest()` normalize paths correctly
- Windows CI: `snapshot_diff()` returns POSIX paths on all platforms
- macOS CI: thread-local connection test uses barrier to prevent `id()` collision from GC
- Mypy: suppressed false positive on nested dict indexed assignment

### Changed
- All `scripts/` references updated to `src/mind_mem/` or `python3 -m mind_mem.X` across docs, Makefile, hooks, install.sh, CODEOWNERS, source docstrings, and MCP error messages
- PyPI badge auto-fetches latest version (removed hardcoded `v=` parameter)
- Test count badge updated to 2027
- Added PEP 561 `py.typed` marker for type checker support
- Added classifiers: OS Independent, Python 3.10-3.14, Typing::Typed
- Added `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `*.log` to .gitignore
- SECURITY.md: bumped supported versions to 1.8.x, added ConnectionManager note
- Roadmap updated to v1.8.0 current / v1.9.0 planned

## 1.8.0 (2026-02-27)

**Architecture overhaul: 5 structural improvements, standard package layout, 74 new tests**

### Architecture
- **Package layout** (#467): Moved `scripts/` → `src/mind_mem/` — standard Python `src` layout. All imports, CI, docs, and templates updated. Package name and API unchanged.
- **SQLite connection manager** (#466): New `ConnectionManager` class with thread-local read connections (WAL concurrent readers) and a single serialized write connection. Integrated into `block_metadata.py` and `sqlite_index.py`. 19 new tests.
- **BlockStore abstraction** (#468): `BlockStore` protocol class and `MarkdownBlockStore` implementation decoupling block access from storage format. Enables future backend swap (SQLite, API). 16 new tests.
- **Adaptive intent router** (#470): Query performance feedback loop with persistent stats. Intents auto-adjust confidence weights based on result quality over time. Minimum 5 samples before adaptation. 31 new tests.
- **Delta-based snapshot rollback** (#471): Replaced `shutil.copytree` with file-level manifest-based snapshots. `MANIFEST.json` tracks snapshotted files for O(manifest) restore instead of O(workspace). Backward-compatible with legacy snapshots. 8 new tests.

### Changed
- CI workflows, CONTRIBUTING.md, docs, and README updated for `src/mind_mem/` layout
- `build_index()` now uses chunked commits (per-file instead of whole-rebuild lock)
- `query_index()` reuses connections via `ConnectionManager`
- Intent router persists adaptation weights to `memory/intent_router_stats.json`
- Recall pipeline records intent feedback after each query

## 1.7.3 (2026-02-27)

**Comprehensive security hardening and production reliability**

### Fixed
- **6 CRITICAL**: chunk_block off-by-one (#435), FTS5 wildcard injection (#436), CLI token exposure (#437), WAL post-check recovery (#438), DDL dimension validation (#439), intel state race condition (#440)
- **11 HIGH**: read-only pragma crash (#442), connection leaks in block_metadata (#444) and build_index (#446), PRF O(N*M) performance (#448), non-atomic proposal status write (#449), missing DB indexes (#450), block_metadata missing pragmas (#451), ACL startup warning (#441), plaintext API key removal (#443), export_memory caps (#447)
- **9 MEDIUM**: SSRF localhost validation (#452), block-header injection (#453), mid-block truncation (#454), sys.path restoration (#455), bare exception handlers (#456), index_status crash on fresh workspace (#457), intel_scan TOCTOU race (#458), vec_meta.json atomic write (#459), block_id validation (#460)
- **4 LOW**: kernel field weight passthrough (#461), delete audit log (#462), workspace permissions (#463), CI SHA pinning (#464)

### Security
- All CI actions pinned to immutable commit SHAs (supply chain hardening)
- Pinecone API key now requires env var only (removed config fallback)
- Workspace directories created with restrictive 0o700 permissions
- export_memory moved to ADMIN_TOOLS with 10k block cap
- HTTP transport warns when admin token is not set
- Deleted blocks now logged to deleted_blocks.jsonl for audit trail

## 1.7.2 (2026-02-27)

**Baseline snapshot, contradiction detection, and full type safety**

### Added
- **Baseline Snapshot**: `baseline_snapshot` MCP tool — freeze intent distribution baselines, detect drift via chi-squared test, compare baselines over time. CLI entry point: `mind-mem-baseline`.
- **Contradiction Detection**: `contradiction_detector` module — TF-IDF cosine similarity + Jaccard fallback, negation pattern detection, status reversal classification. Integrated into `approve_apply` governance gate.
- Terminal demo GIF (VHS recording) in README

### Fixed
- All 168 mypy typecheck errors across 32 files (zero errors, full strict pass)
- Windows CI encoding failure (em-dash in test fixture)
- Ruff format/lint issues in contributed code
- CodeQL high-severity findings in test files (URL sanitization, file permissions)
- Two stale "16 tools" references in README (now 19)

### Changed
- Copyright updated to "STARGA Inc and contributors"
- Benchmark example uses `mistral-large-latest` to match published results

## 1.7.1 (2026-02-25)

**Quality, testing, documentation, and developer tooling improvements**

### Added
- Structured `ErrorCode` enum with 29 codes across 8 categories for consistent error handling
- Test suites for `entity_ingest`, `session_summarizer`, `cron_runner`, `observation_compress`, `bootstrap_corpus`
- Unicode/i18n and stress tests for improved internationalization coverage
- MCP tool examples and API reference documentation
- Troubleshooting FAQ and performance tuning guide
- Pre-commit hooks, CODEOWNERS, issue/PR templates
- Dependabot configuration, `.editorconfig`, `.gitattributes`
- Benchmark CI workflow for automated performance regression tracking
- Shared pytest fixtures (`conftest.py`) for workspace setup
- `.python-version` file for pyenv/asdf compatibility

### Changed
- Replaced broad exception handlers with specific exception types across codebase
- Added structured logging to 4 additional modules
- Added return type hints to core modules

## 1.7.0 (2026-02-23)

**Recall quality — retrieval graph, fact indexing, knee cutoff, augmented embeddings, hard negatives**

### Added
- **Retrieval Logger + Co-retrieval Graph**: `scripts/retrieval_graph.py` — logs every `recall()` invocation to SQLite (`retrieval_log` table), builds co-retrieval edges between co-returned blocks (`co_retrieval` table), and propagates scores across the graph via damped PageRank-like iteration.
- **Fact Card Indexing (Small-to-Big Retrieval)**: `sqlite_index.py` — extracts atomic fact cards from Statement text at FTS5 index time, indexes as sub-blocks with `parent_id` linkage. Aggregation at query time folds fact scores into parent blocks.
- **Knee Score Cutoff**: `_recall_core.py` — adaptive top-K truncation at steepest score drop instead of fixed limit. Config: `recall.knee_cutoff` (default: on), `recall.min_score`.
- **Metadata-Augmented Embeddings**: `recall_vector.py` — prepends `[Category] [Speaker] [Date] [Tags]` to text before embedding for better vector disambiguation.
- **Abstention Hard Negative Mining**: `retrieval_graph.py` + `evidence_packer.py` — records misleading blocks when abstention fires (high BM25, low cross-encoder), demotes flagged blocks by 30% in future queries.
- New config keys: `knee_cutoff`, `min_score`.
- Schema: `parent_id` column on `blocks` table, `retrieval_log`, `co_retrieval`, `hard_negatives` tables in `recall.db`.

### Testing
- 37 new tests across 2 files:
  - `test_retrieval_graph.py`: 22 tests (logging, co-retrieval edges, propagation, hard negatives, knee cutoff)
  - `test_fact_indexing.py`: 15 tests (fact sub-blocks, aggregation, metadata augmentation, deletion cascade)
- Total: **1352 tests passing** (up from 1315)

### Changed
- `_recall_core.py`: integrated stages 2.6 (hard negative penalty), 2.8 (co-retrieval propagation), 2.9 (knee cutoff), retrieval logging
- `sqlite_index.py`: `_insert_block()` extracts + indexes fact sub-blocks, `_delete_blocks()` cascades to children, `query_index()` aggregates facts to parents
- `evidence_packer.py`: `check_abstention()` records hard negatives when abstention fires
- `_recall_constants.py`: added `knee_cutoff`, `min_score` to valid recall config keys
- Zero new dependencies — all features use Python stdlib + SQLite only

## 1.6.0 (2026-02-22)

**File watcher, LLM reranking, overlapping chunks — completes IMPL-PLAN Phase 1-3**

### Added
- **File Watcher Mode (1.5)**: `scripts/watcher.py` — `FileWatcher` class with mtime polling on background daemon thread. Detects new, modified, deleted `.md` files. Integrated into `mcp_server.py` with `--watch` and `--watch-interval` flags. Zero external deps (stdlib `threading`, `time`, `os`).
- **LLM Reranking Stage (1.6)**: Optional LLM-based reranking via local Ollama in `_recall_reranking.py`. Config-gated (`recall.llm_rerank: true`), uses `urllib.request` (stdlib). Sends query + candidates to LLM for relevance scoring, blends with existing scores. Silent fallback on failure.
- **Overlapping Chunks (1.7)**: `chunk_block()` in `block_parser.py` splits long blocks (>400 words) into overlapping windows for better recall at boundaries. `deduplicate_chunks()` merges chunk results by base block ID. Config-gated (`recall.chunk_overlap: 50`).
- New config keys: `llm_rerank`, `llm_rerank_url`, `llm_rerank_model`, `llm_rerank_weight`, `chunk_overlap`, `max_chunk_tokens`.

### Testing
- 32 new tests across 3 files:
  - `test_watcher.py`: 9 tests (new file detection, modification, deletion, ignore non-.md, ignore hidden dirs, stop, no-callback-on-unchanged, subdirectory, double-start)
  - `test_recall_reranking.py`: 11 tests (deterministic reranker + LLM rerank with HTTP mocks)
  - `test_block_parser_chunks.py`: 12 tests (chunking logic + dedup)
- Total: **1315 tests passing** (up from 1283)

### Changed
- `_recall_core.py`: integrated LLM rerank (Stage 2.7) + chunk expansion + chunk dedup
- `_recall_constants.py`: added 6 new valid recall config keys
- Zero new dependencies — all features use Python stdlib only

## 1.5.1 (2026-02-22)

**Block-level incremental FTS indexing — fixes #17 HIGH**

### Added
- Block-level incremental FTS indexing in `sqlite_index.py`: tracks per-block content hashes in `index_meta` table. On reindex, only NEW/MODIFIED/DELETED blocks are touched — unchanged blocks are skipped entirely. Turns O(blocks_per_file) into O(changed_blocks).
- `_compute_block_hash()`: SHA-256 hash of block content (excludes `_line` to avoid false positives when blocks shift).
- `_insert_block()` / `_delete_blocks()`: extracted helpers for clean block-level CRUD.
- Build summary now reports `blocks_new`, `blocks_modified`, `blocks_deleted`, `blocks_unchanged`.

### Testing
- 13 new tests: `TestBlockLevelIncremental` (10), `TestComputeBlockHash` (4)
- Total: **1274 tests passing** (up from 1261)

### Changed
- `sqlite_index.py`: `_index_file()` refactored from file-level to block-level incremental
- Zero new dependencies — uses stdlib `hashlib`, `json`, `sqlite3`

## 1.5.0 (2026-02-22)

**Embedding cache, incremental indexing, dimension safety, and provider fallback chain — closes #38, #39, #40, #41**

### Added
- **#38** — Embedding cache: SHA-256 content-hash cache in sqlite3 (`embedding_cache` table) avoids re-embedding unchanged blocks during reindex. Turns O(N) full reindex into O(changed).
- **#39** — Incremental vector indexing: only embeds cache-miss blocks during `reindex`. Cache hits are loaded directly from sqlite3, skipping the embedding provider entirely.
- **#40** — Dimension mismatch detection: `vec_meta_info` table tracks model name, embedding dimension, and build timestamp. Warns on search if query model differs from indexed model. Auto-invalidates cache on model change.
- **#41** — Embedding provider fallback chain with circuit breaker: cascades through llama_cpp → fastembed → sentence-transformers on failure. Circuit breaker (3 failures → 60s cooldown → auto-reset) prevents repeated calls to failed providers.

### Testing
- 20 new tests: `TestEmbeddingCache` (12), `TestDimensionMismatch` (7), `TestCircuitBreaker` (1)
- Total: **1261 tests passing** (up from 1241)

### Changed
- Version: 1.4.1 → 1.5.0
- `recall_vector.py`: 700 → 1352 lines (embedding cache, dimension tracking, fallback chain)
- `test_recall_vector.py`: 313 → 543 lines (20 new tests)
- Zero new dependencies — all features use Python stdlib only (hashlib, struct, time, sqlite3)

## 1.4.1 (2026-02-22)

**Build pipeline hardening — linker version script, source leak elimination**

### Fixed
- MIND build: added linker version script (exports.map) to control .dynsym — only 21 intended symbols exported
- MIND build: removed redundant mindc rebuild in Stage 3 that was undoing Stage 2's source stripping
- MIND build: .comment section now shows MIND toolchain attribution instead of GCC
- MIND build: runtime .so .comment cleaned during deploy

### Changed
- Version: 1.4.0 → 1.4.1
- Binary exports locked to 21 (15 scoring + 6 protection/auth) via version script
- All MIND internals (get_source, get_ir, protection functions) hidden via `local: *` in exports.map

## 1.4.0 (2026-02-22)

**Deep audit fixes + MCP completeness — closes #28, #29, #30, #31, #32, #33, #34, #35, #36, #37**

### Added
- **#35** — New MCP tools: `delete_memory_item` (admin-scope, removes block by ID) and `export_memory` (user-scope, exports workspace as JSONL)
- **#36** — `_schema_version` field in all MCP JSON responses for forward compatibility
- **#31** — Query-level observability: structured logging with tool_name, duration_ms, success/failure for every MCP tool call
- **#37** — Configurable limits via `mind-mem.json` `limits` section: max_recall_results, max_similar_results, max_prefetch_results, max_category_results, query_timeout_seconds, rate_limit_calls_per_minute

### Fixed
- **#29** — SQLite "database is locked" now returns structured `database_busy` error with `retry_after_seconds` instead of crashing
- **#30** — Corrupted blocks now log block line number and skip with warning (new `BlockCorruptedError` exception class)
- **#32** — `BlockMetadataManager` shared state protected with `threading.RLock` for concurrent access
- **#34** — FTS5 index now persists across queries; staleness check avoids redundant rebuilds
- **#28** — Hybrid fallback chain validates config schema (bm25_weight, vector_weight, rrf_k) before initializing HybridBackend

### Testing
- **#33** — Concurrency stress tests: 20-thread parallel recall with deadlock detection and explicit `join(timeout=10)`
- Total: **1241 tests passing** (up from 1157)
- New test files: `test_mcp_v140.py`, `test_core_v140.py`, `test_concurrency_stress.py`

### Changed
- Version: 1.3.0 → 1.4.0
- MCP tools: 16 → 18 (added delete_memory_item, export_memory)
- Documentation: `docs/configuration.md` updated with limits section

## 1.3.0 (2026-02-22)

**Security hardening + audit fixes — closes #20, #21, #22, #23, #24, #25, #26, #27**

### Security
- **#20** — MCP per-tool ACL: admin/user scope separation. Write tools (apply_proposal, write_memory, etc.) restricted to admin token. Read tools (recall, search, list) available to user scope.
- **#21** — Rate limiting (120 calls/min sliding window) and per-query timeouts (30s) added to MCP server
- **#25** — Optional dependencies pinned with exact versions (fastmcp==2.14.5, onnxruntime==1.24.1, tokenizers==0.22.2, sentence-transformers==5.2.3). Hash-verified install via requirements-optional.txt

### Fixed
- **#22** — Replaced 11 broad `except Exception` handlers with specific exceptions (OSError, ValueError, KeyError, ImportError). Added stack trace logging to previously silent error handlers.
- **#23** — Config numeric range validation: BM25 k1/b, rrf_k, limits, weights all validated and clamped on load
- **#24** — FFI .so version check on startup: compares library version against Python __version__, warns on mismatch
- **#26** — Malformed config (JSONDecodeError) caught at startup with line/column display, falls back to defaults

### Testing
- **#27** — 102 new error/edge case tests: DB lock, bad config, corrupted blocks, missing .so, invalid IDs, empty workspace, large limits
- Total: **1157 tests passing** (up from 1055)

### Changed
- Version: 1.2.0 → 1.3.0
- Optional deps now pinned to exact versions in pyproject.toml

## 1.2.0 (2026-02-22)

**Five enhancement features — closes #10, #11, #12, #13, #14**

### Added
- **#10 — BM25F weight grid search** (`benchmarks/grid_search.py`): One-at-a-time (11 combos) and full cartesian (243 combos) grid search over field weights. Evaluates against LoCoMo with MRR/R@k metrics. 14 new tests.
- **#11 — Fact key expansion** (`scripts/block_parser.py`): Enriches each block with `_entities`, `_dates`, `_has_negation` via `_enrich_fact_keys()`. Entity overlap boosts recall up to 1.45x; adversarial negation boost 1.2x. 14 new tests.
- **#12 — Chain-of-Note evidence packing** (`scripts/evidence_packer.py`): Structured `[Note N]` format with source, key facts, and relevance per block. Config toggle `evidence_packing: "chain_of_note"` (default) or `"raw"`. 38 new tests.
- **#13 — Temporal hard filters** (`scripts/_recall_temporal.py`): Resolves relative time references ("last week", "yesterday") to date ranges and hard-filters blocks. Integrated into recall pipeline. 36 new tests.
- **#14 — Cross-encoder A/B test** (`benchmarks/crossencoder_ab.py`): BM25 vs BM25+CE comparison. Result: +0.097 MRR (+24% relative), 58 questions improved, 17 regressed. Report section added.
- `temporal_hard_filter` key added to `_VALID_RECALL_KEYS`

### Changed
- Version: 1.1.2 → 1.2.0
- Total: **1055 tests passing** (up from 964)

## 1.1.2 (2026-02-22)

**Post-release audit hardening**

### Fixed
- **#15** — Stale `filelock.py` reference in `init_workspace.py` `MAINTENANCE_SCRIPTS` (renamed to `mind_filelock.py` in v1.0.6, never updated)
- **#16** — Coverage metric in `detect_drift()` did not subtract `dead_skipped_enforced` from denominator, deflating reported coverage
- **#17** — Unguarded `int()` on `priority` field in `generate_proposals()` crashes on non-numeric values
- **#18** — `apply_engine._get_mode()` returned `"unknown"` on failure, bypassing the `detect_only` mode gate. Now defaults to `"detect_only"` (safe default)
- **#19** — `intel_scan.py` appended `"Z"` to `datetime.now()` (local time), producing incorrect ISO 8601 timestamps. Now uses `datetime.now(timezone.utc).isoformat()`

### Changed
- Version: 1.1.1 → 1.1.2

## 1.1.1 (2026-02-22)

**Test coverage push + Full 10-conv benchmark**

### Test Coverage
- `tests/test_recall_vector.py` — 36 new tests covering vector backend initialization, embedding generation, similarity search, ONNX fallback, Pinecone/Qdrant providers, and error paths
- `tests/test_validate_py.py` — 30 new tests covering structural validation, schema checks, cross-reference integrity, and intelligence file validation
- Total: **964 tests passing** (up from 898)

### Benchmark
- Full 10-conversation LoCoMo LLM-as-Judge benchmark completed (Mistral Large answerer + judge, BM25-only, top_k=18)
- 1986 questions: **73.8% Acc≥50**, mean=70.5 (up from 67.3% / 61.4 on v1.0.0 baseline)
- Adversarial accuracy: **92.4%** (up from 36.3% on v1.0.0 baseline)
- Conv-0 detailed: mean=77.9, adversarial=82.3, temporal=88.5

### CI
- All 9 matrix jobs green (Ubuntu/macOS/Windows × Python 3.10/3.12/3.13)
- Fixed test isolation issues in vector backend tests

## 1.1.0 (2026-02-22)

**Multi-hop query decomposition + Recency decay**

### Multi-hop Query Decomposition (issue #6)
- Deterministic query splitting on conjunctions, wh-word boundaries, and question marks
- Context preservation: shared entities from first clause carried into sub-queries
- Recursion-safe: sub-queries do not re-decompose (prevents infinite recursion)
- Capped at 4 sub-queries, minimum 3 tokens per sub-query

### Recency Decay for Trajectory Similarity (issue #9)
- Exponential half-life decay on trajectory age (default 30 days)
- Configurable via `recency_halflife` in `trajectory.mind`
- Missing/unparseable dates receive no penalty (decay = 1.0)
- Zero halflife guard prevents division by zero

## 1.0.7 (2026-02-21)

**Retrieval quality push + Trajectory Memory foundation**

### Retrieval Improvements
- **top_k 10 → 18** — 80% more context blocks from RRF fusion pool (A/B tested: +3.0 mean, +6.3 acc@75 on conv-0)
- **Temporal extra_limit_factor 1.5 → 2.0** — Wider candidate retrieval for date-bearing queries
- **Temporal-multi-hop cross-boost** — "When did X do Y?" gets multi-hop signal boost in detection
- **Evidence-grounded answerer prompt** — Replaced hallucination-encouraging rules with evidence-citing instructions
- **Calibrated judge rubric** — 4-tier scoring replaces "core facts = 70+" anchor that inflated scores

### Trajectory Memory (v1.2.0 foundation)
- `mind/trajectory.mind` — Config kernel with schema, capture, recall, and consolidation settings
- `scripts/trajectory.py` — Block parser, validator, ID generator, Markdown formatter, similarity computation
- `tests/test_trajectory.py` — 19 tests covering ID generation, validation, parsing, roundtrip, similarity

### Testing
- `tests/test_recall_detection.py` — 32 tests for query type classification module
- `benchmarks/compare_runs.py` — A/B benchmark comparison utility
- Total: 873 tests passing (up from 822)

## 1.0.6 (2026-02-21)

**Hybrid retrieval pipeline + critical retrieval fixes**

### Fixed
- **Date field passthrough** — All 3 retrieval paths (BM25, FTS5, vector) now surface the Date field to the evidence packer. Previously blocks stored dates but never passed them through, causing 73% of multi-hop failures on LoCoMo (answerers couldn't resolve relative time like "yesterday" to absolute dates)
- **Module shadowing bug** — Renamed `filelock.py` to `mind_filelock.py` to stop shadowing the pip-installed `filelock` package. This silently broke `sentence_transformers.CrossEncoder` import in all benchmark contexts
- **Vector result enrichment** — `recall_vector.py` now passes speaker, DiaID, and Date in result dicts (previously showed `[SPEAKER=UNKNOWN]` for vector-only hits)

### Added
- **Cross-encoder reranking in hybrid path** — `ms-marco-MiniLM-L-6-v2` now runs post-RRF-fusion in `hybrid_recall.py` (previously only wired in the BM25-only path which hybrid mode bypasses). Config: `cross_encoder.enabled=true, blend_weight=0.6`
- **llama.cpp embedding provider** — `recall_vector.py` supports Qwen3-Embedding-8B via llama.cpp server for 4096-dimensional embeddings
- **sqlite-vec backend** — Local vector search via `sqlite-vec` extension (ONNX embeddings stored in recall.db)
- **Pinecone integrated inference** — Server-side embedding generation via Pinecone's model-on-index API
- **fastembed ONNX support** — Zero-torch embedding generation via fastembed

### Benchmark Impact
- Multi-hop accuracy: 55.5% → 74.4% (Date field fix)
- Adversarial accuracy: 36.3% → 86.6% (hybrid retrieval + strict judge)
- Overall: 61.4 → 62.3 mean score (3-conv partial, full run in progress)

## 1.0.5 (2026-02-19)

**Full security + code quality audit hardening**

### Security
- Removed workspace paths from all MCP server responses (health, scan, index_stats, reindex)
- Replaced raw `str(e)` exception leaks with generic error messages in MCP tools
- Fixed `startswith()` path traversal prefix collision in `mind_ffi.py` (added `os.sep` check)
- Removed absolute kernel paths from `list_mind_kernels` and `get_mind_kernel` responses
- Sanitized `_check_workspace` to not leak full paths in error messages

### Performance
- Fixed O(N²) RM3 re-scoring in `recall.py` with O(1) `result_by_id` dict lookup
- Fixed O(N) set rebuild in chain-of-retrieval with pre-built `existing_ids` set
- Hoisted `datetime` imports out of hot-path functions (`date_score`, `_extract_dates`)
- Added `threading.Lock` for thread-safe metrics in `observability.py`

### Fixed
- Split `except (ImportError, Exception):` into separate handlers in MCP server
- Made compaction source file writes atomic (write-to-tmp + `os.replace()`)
- Removed no-op `word = word` branches in recall.py (changed to `pass`)
- Removed dead comments and unused code paths
- Fixed f-string without placeholder in `intel_scan.py`

### Improved
- Extracted `_load_extra_categories()` helper to deduplicate CategoryDistiller config loading
- Migrated Pinecone from v2 to v3 API in `recall_vector.py`
- Added `PINECONE_API_KEY` environment variable support for vector search
- Updated `_VALID_KEYS` to include Pinecone/Qdrant configuration keys

### Changed
- Version: 1.0.4 → 1.0.5

### Post-release audit fixes (de1e747)
- Pre-computed IDF per query token before document loop (eliminates redundant `math.log` calls)
- Added `_id` tie-breaker to all 5 result sort calls for deterministic ordering across platforms
- Added `_VALID_RECALL_KEYS` whitelist for recall config section with unknown-key warnings
- Replaced silent `except: pass` with debug/warning logging in corpus parsing, RM3 config, vector backend
- Optimized `index_stats` MCP tool to use FTS index count (O(1)) instead of re-parsing all files (O(N))
- Cleaned up dead silent catch in `sqlite_index.py` xref scan

### FORTRESS binary hardening (ce33649)
- Expanded binary patching keyword lists from ~25 to 130+ patterns (8 leak categories)
- Added `patchelf --set-rpath '$ORIGIN'` to remove hardcoded build paths from ELF RPATH
- Build now fails if any of 8 leak categories (MIND source, attributes, TOML configs, hex auth key, RPATH, VM IR, protection internals, TOML comments) still have patterns in final binary
- String count: 412 → 186 (all remaining are exported symbols + system calls)

### Second audit pass — full fix
- **#5 Hidden coupling**: Added `_log.info` for missing optional subsystems (block_metadata, intent_router, llm_extractor) at import time
- **#9 Reranker latency**: Capped deterministic reranker and cross-encoder candidates at `MAX_RERANK_CANDIDATES` (200) in both BM25 and FTS paths
- **#11 FTS fallback silent**: MCP `recall` tool now returns envelope `{"_schema_version": "1.0", "backend": ..., "results": [...], "warnings": [...]}` with fallback warnings
- **#13 README accuracy**: Changed "Zero Dependencies" badge to "Zero Core Dependencies", added `ollama` to optional deps table, clarified optional deps in trust signals
- **#15 Config caps**: Added `MAX_BLOCKS_PER_QUERY` (50,000) cap with warning log on huge workspaces
- **Graph cap**: Capped graph neighbor expansion to `MAX_GRAPH_NEIGHBORS_PER_HOP` (50) per hop in both BM25 and FTS paths to prevent blowup on dense graphs
- **Path validation**: Extracted reusable `_validate_path()` helper in `mcp_server.py` for consistent workspace containment checks
- **Schema versioning**: MCP recall output now includes `_schema_version: "1.0"` for client compatibility detection
- **No-results feedback**: Empty recall results include a `"message"` hint instead of bare empty array

### In-depth audit — error transparency, test coverage, perf hardening
- **Silent ImportError logging**: Added `_log.debug` to 3 silent `except ImportError: pass` blocks (mind_ffi, namespaces, category_distiller)
- **HTTP auth warning**: MCP server logs warning when started on HTTP transport without token auth
- **Block size cap**: Added `MAX_PARSE_SIZE` (100KB) in `block_parser.py` — files over 100KB are truncated
- **Vector fallback escalation**: Upgraded vector backend unavailable message from `debug` to `warning`
- **Test coverage**: Added 60 new tests (821 total, up from 761):
  - `test_graph_boost.py` (29 tests): graph boost cross-refs, context packing, dialog adjacency, config validation, block cap
  - `test_fts_fallback.py` (23 tests): FTS fallback, envelope structure, block size cap, config key validation
  - `test_concurrency_stress.py` (8 tests): thread-safe parallel recall, 1000-2000 block stress tests, graph boost contention
- **Exception handler split**: Separated `(ImportError, Exception)` into distinct handlers in recall.py and hybrid_recall.py

---

## 1.0.4 (2026-02-19)

**MCP bug fixes + audit findings (security, error handling, DX)**

### Security
- Fixed path traversal guard in `create_snapshot` (`FilesTouched` containment)
- Prefer installation scripts over workspace copies in `check_preconditions`

### Fixed
- BRIEFINGS.md crash on missing briefings section
- `load_intel_state` crash on corrupt JSON
- MCP `reindex` error message leak on failure

### Improved
- MCP error messages include workspace validation hints
- Test count: 736 → 761

---

## 1.0.3 (2026-02-19)

**Documentation, CI/CD, MCP integration tests, LLM extraction prototype**

### Added
- Mermaid architecture diagrams (recall pipeline, governance flow, multi-agent)
- `docs/quickstart.md`: step-by-step tutorial for first-time setup
- Python MCP client examples in `docs/api-reference.md`
- 8 MCP integration tests (`tests/test_mcp_integration.py`)
- GitHub Actions CI workflow (3 OS x 3 Python version matrix)
- GitHub Actions release workflow for automated publishing
- LLM extraction prototype (optional, config-gated via `llm_extraction` key)

### Improved
- MCP error messages now include workspace validation and actionable hints
- README: added dated benchmark comparison table with infrastructure column
- README: added troubleshooting FAQ section

### Fixed
- Skipped test now uses proper `@pytest.mark.skipif` instead of bare `skip()`
- All E501 lint errors resolved (120-char line limit compliance)

### Changed
- `pyproject.toml`: added pytest/ruff configuration and `[test]` extras
- Test count: 736 -> 761

---

## 1.0.2 (2026-02-18)

**Category distillation, prefetch context, MIND kernel integration, full pipeline wiring**

### Added
- `scripts/category_distiller.py`: Deterministic category detection from block tags/keywords, generates `categories/*.md` thematic summaries with block references and `_manifest.json`
- `prefetch_context()` in `scripts/recall.py`: Anticipatory pre-assembly of likely-needed blocks using intent routing + category summaries
- 2 new MCP tools: `category_summary` (topic-based category retrieval), `prefetch` (signal-based context pre-assembly) — 14→16 total
- 16 MIND kernel source files (`.mind`) with C99 FFI bridge: 15 compiled scoring kernels + configuration parameters
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

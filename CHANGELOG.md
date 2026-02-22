# Changelog

All notable changes to mind-mem are documented in this file.

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

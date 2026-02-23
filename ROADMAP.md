# mind-mem Roadmap

## v1.0.6 — Hybrid Retrieval Pipeline ✅ Released

- [x] Date field passthrough in all retrieval paths
- [x] Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) in hybrid path
- [x] Module shadowing fix (filelock.py rename)
- [x] llama.cpp embedding provider (Qwen3-Embedding-8B, 4096d)
- [x] sqlite-vec local vector backend
- [x] Pinecone integrated inference
- [x] fastembed ONNX support

## v1.0.7 — Stability & Audit ✅ Released

- [x] Full 5-agent audit (security, code quality, performance, tests, docs)
- [x] FTS5 injection fixed, MD5→SHA256, limit capped, atomic writes
- [x] Dead code bugs fixed (extra_limit_factor, dead set comprehension, schema_version)
- [x] 873 tests passing, CI green on all platforms

## v1.1.0 — Adversarial Abstention + Auto-Ingestion + Multi-Hop ✅ Released (2026-02-17)

- [x] **Abstention classifier** — deterministic pre-LLM confidence gate (5 features, threshold 0.20)
- [x] **Answerer prompt tuning** — evidence-grounded instructions replacing hallucination-forcing rules
- [x] **Judge prompt calibration** — removed "core facts = 70+" anchor that inflated scores
- [x] **Multi-hop query decomposition** — decompose complex queries into sub-queries with parallel execution
- [x] **Recency decay** — time-weighted scoring for temporal relevance
- [x] **Trajectory memory** — `[TRAJECTORY]` block type for task execution traces
- [x] **Auto-ingestion pipeline** — session_summarizer, entity_ingest, cron_runner, bootstrap_corpus
- [x] **Content-hash dedup** — SHA256 on normalized text, 16-char hex prefix
- [x] **Entity extraction** — regex-based projects, tools, people extraction with alias dedup
- [x] **Detection test suite** — 32 tests for _recall_detection.py
- [x] **Benchmark comparison tool** — compare_runs.py for side-by-side A/B analysis
- [x] 898 tests passing, CI green on all platforms

## v1.1.1 — Test Coverage + Benchmark ✅ Released (2026-02-22)

- [x] **recall_vector.py test suite** — 36 tests covering VectorBackend init, cosine similarity, local index I/O, search_batch, provider routing
- [x] **validate_py.py test suite** — 30 tests covering Validator, file structure, decisions, tasks, entities, provenance, cross-refs, intelligence
- [x] **LoCoMo benchmark with Mistral Large** — full 10-conversation LLM-as-judge evaluation (1986 questions, 134 min)
  - Overall: mean=70.5, acc≥50=73.8%, acc≥75=65.6%
  - Adversarial: mean=87.2, acc≥50=92.4% (+43pp over v1.0.5 baseline)
  - BM25-only recall (v1.0.5 baseline used hybrid BM25+vector)
- [x] 964 tests passing, CI green on all platforms

## v1.2.0 — Retrieval Quality Push ✅ Released (2026-02-22)

- [x] **BM25F weight grid search** (`benchmarks/grid_search.py`) — one-at-a-time (11) + full cartesian (243) combo search
- [x] **Fact key expansion** — `_entities`, `_dates`, `_has_negation` per block; entity overlap boost up to 1.45x
- [x] **Chain-of-Note evidence packing** — structured `[Note N]` format with config toggle
- [x] **Temporal hard filters** (`scripts/_recall_temporal.py`) — relative time → date range → block filter
- [x] **Cross-encoder A/B test** — +0.097 MRR (+24% relative) with ms-marco-MiniLM-L-6-v2
- [x] 1055 tests passing, CI green on all platforms

## v1.3.0 — Security Hardening + Audit Fixes ✅ Released (2026-02-22)

- [x] **MCP per-tool ACL** — admin/user scope separation for all 16 MCP tools
- [x] **Rate limiting** — 120 calls/min sliding window + 30s per-query timeout
- [x] **Exception handling** — 11 broad `except Exception` replaced with specific exceptions
- [x] **Config validation** — numeric range clamping for BM25 k1/b, rrf_k, limits, weights
- [x] **FFI version check** — .so version validated against Python __version__ on startup
- [x] **Dependency pinning** — exact versions + hash-verified install path
- [x] **Malformed config handling** — JSONDecodeError caught with line/column display
- [x] **Error/edge case tests** — 102 new tests for failure modes
- [x] 1157 tests passing, CI green on all platforms

## v1.4.0 — Reflective Consolidation

- [ ] Sleep-time memory consolidation (periodic background pass)
- [ ] Pattern extraction from trajectory clusters
- [ ] Automatic contradiction detection across trajectories
- [ ] Memory importance scoring with decay

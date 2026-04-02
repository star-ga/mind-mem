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

## v1.4.0 — Deep Audit Fixes + MCP Completeness ✅ Released (2026-02-22)

- [x] **SQLite busy handling** — structured "database_busy" error with retry_after on locked DB (#29)
- [x] **Corrupted block logging** — BlockCorruptedError with line number, skip-and-warn in parser (#30)
- [x] **Query-level observability** — structured logging with tool_name, duration_ms, success for all MCP calls (#31)
- [x] **BlockMetadataManager thread safety** — RLock on all DB/cache access paths (#32)
- [x] **Concurrency stress tests** — 20-thread recall stress test with deadlock detection (#33)
- [x] **FTS5 index persistence** — staleness check, skip rebuild when index is fresh (#34)
- [x] **New MCP tools** — delete_memory_item (admin) and export_memory (user) (#35)
- [x] **MCP schema versioning** — _schema_version field in all JSON responses (#36)
- [x] **Configurable limits** — max_recall_results, query_timeout, rate_limit via mind-mem.json (#37)
- [x] **Hybrid fallback validation** — strict schema checks on recall config before HybridBackend init (#28)
- [x] 1241 tests passing, CI green on all platforms

## v1.5.0 — Reflective Consolidation ✅ Released

- [x] Sleep-time memory consolidation (periodic background pass)
- [x] Pattern extraction from trajectory clusters
- [x] Automatic contradiction detection across trajectories
- [x] Memory importance scoring with decay

## v1.5.1 — Patch ✅ Released (2026-02-22)

- [x] Bug fixes and stability improvements

## v1.6.0 — Governance Engine ✅ Released (2026-02-22)

- [x] **Contradiction detection** — automated conflict scanning across memory blocks
- [x] **Drift analysis** — detect when beliefs/facts shift over time
- [x] **Proposal queue** — staged governance proposals with approve/reject flow
- [x] **A-MEM block metadata evolution** — importance scoring, access tracking, keyword extraction
- [x] **9-type intent router** — classify queries by intent for targeted retrieval

## v1.7.0 — Architecture Foundations ✅ Released (2026-02-23)

- [x] **ConnectionManager** — thread-safe SQLite pool with WAL read/write separation
- [x] **BlockStore protocol** — decoupled block access from storage format
- [x] **Delta-based snapshot rollback** — MANIFEST.json for O(manifest) restore
- [x] **Adaptive intent router** — confidence weights adjust via feedback loop

## v1.7.1–v1.7.3 — Security Hardening ✅ Released (2026-02-25 – 2026-02-27)

- [x] 30 audit findings fixed (6 critical, 11 high, 9 medium, 4 low)
- [x] All CI actions pinned to immutable commit SHAs
- [x] Pinecone API key requires env var only
- [x] Workspace dirs created with 0o700 permissions
- [x] Cross-platform fixes (Windows paths, macOS thread-local)

## v1.8.0 — Package Layout Overhaul ✅ Released (2026-02-27)

- [x] `scripts/` → `src/mind_mem/` — standard Python src layout
- [x] Chunked commit indexing (per-file instead of whole-rebuild lock)
- [x] Intent router persists adaptation weights
- [x] 74 new tests

## v1.8.1–v1.8.2 — Polish ✅ Released (2026-03-04)

- [x] Cross-encoder batch_size parameter (prevents OOM)
- [x] 8 integration tests covering full pipeline
- [x] Import hygiene cleanup

## v1.9.0 — Governance Deep Stack ✅ Released (2026-03-05)

- [x] **Hash-chain mutation log** (`audit_chain.py`) — SHA-256 chained append-only JSONL ledger
- [x] **Per-field mutation audit** (`field_audit.py`) — SQLite-backed field-level change tracking
- [x] **Semantic belief drift detection** (`drift_detector.py`) — trigram Jaccard similarity
- [x] **Temporal causal dependency graph** (`causal_graph.py`) — directed edges with cycle detection
- [x] **Coding-native memory schemas** (`coding_schemas.py`) — 5 block types (ADR, CODE, PERF, ALGO, BUG)
- [x] **Auto contradiction resolution** (`auto_resolver.py`) — preference learning + causal side-effect analysis
- [x] **Governance benchmark suite** (`governance_bench.py`) — detection rate, completeness, scalability
- [x] **Encryption at rest** (`encryption.py`) — HMAC-SHA256 keystream, PBKDF2, encrypt-then-MAC
- [x] 145 new tests across 8 modules

## v1.9.1 — Current Release ✅ Released (2026-03-06)

- [x] Proposal apply + rollback safety fixes
- [x] Request-scoped MCP auth (admin from token scopes)
- [x] Clean install bootstrap fixes
- [x] **Calibration feedback loop** — per-block quality tracking + retrieval adjustment
- [x] **Cognitive scoring kernel** — agent-aware recall
- [x] 17 MIND kernels, 19 MCP tools, 2180+ tests passing

---

# v2.0 Roadmap — Verifiable, Accelerated Memory

> Three STARGA projects converge: **512-mind** governance primitives + **mind-inference** acceleration + **mind-mem** retrieval.
>
> Theme: The first AI memory system with **cryptographically verifiable governance** and **hardware-accelerated hot paths**.

---

## v2.0-alpha — Cryptographic Governance Layer (from 512-mind)

**Goal:** Every memory write is tamper-evident. Governance config is immutable post-init. Evidence objects prove governance actually ran.

### Hash-Chained Block Writes
- [ ] SHA3-512 hash chain: each block write includes `prev_hash` linking to previous write
- [ ] Chain head stored in DB metadata, verifiable from any snapshot
- [ ] `verify_chain()` MCP tool — walk the chain, report any breaks
- [ ] Existing `create_snapshot` / `restore_snapshot` tools gain chain-head verification

### Spec-Hash Binding (I-5)
- [ ] SHA3-512 hash of governance config (`mind-mem.json` governance section) computed at init
- [ ] `spec_hash` embedded in every Evidence Object
- [ ] Runtime check: if config file changes post-init, log spec-hash divergence + alert
- [ ] `governance_spec_hash` exposed via MCP `index_stats` resource

### Structured Evidence Objects
- [ ] Every governance decision (proposal ALLOW/DENY, contradiction detection, drift alert) outputs a structured Evidence Object:
  ```json
  {
    "evidence_id": "<sha3-512>",
    "timestamp": "<ISO8601>",
    "decision": "ALLOW | DENY",
    "action": "proposal_apply | contradiction_detect | drift_alert",
    "spec_hash": "<governance config hash>",
    "state_hash": "<chain head at decision time>",
    "context": { ... }
  }
  ```
- [ ] Evidence Objects are append-only (separate evidence.jsonl file)
- [ ] `list_evidence` MCP tool for audit queries

### Single Gateway Enforcement (I-1)
- [ ] All block writes must pass through `GovernanceGate.admit()` — no direct DB writes
- [ ] BlockStore protocol enforced as the only write path (remove any bypass paths)
- [ ] Write attempts outside BlockStore raise `GovernanceBypassError`

**Estimated:** ~600 lines across 4 modules. No breaking changes to existing API.

---

## v2.0-alpha.2 — Observer-Dependent Cognition (ODC) Retrieval

**Goal:** Make retrieval axis-aware. Every recall declares its observation basis, results include axis metadata, and the system can rotate axes for higher-confidence results.

**Spec:** `specs/observer-dependent-cognition.md`

### Axis-Aware Retrieval
- [ ] Add `observation_axis` field to RecallRequest (lexical, semantic, temporal, entity-graph, contradiction)
- [ ] Extend `hybrid_search` to accept explicit axis weights (override default RRF)
- [ ] Log axis choices in evidence chain (which axes produced each result)
- [ ] Axis rotation: if initial recall confidence < threshold, automatically rotate to orthogonal axes

### Observation Metadata
- [ ] Every recall result tagged with producing axes + per-axis confidence scores
- [ ] New MCP tool: `recall_with_axis` — explicit axis selection for advanced queries
- [ ] Axis diversity metric: how many independent axes contributed to a result

### Adversarial Axis Injection
- [ ] Extend adversarial abstention to include deliberate counter-axis queries
- [ ] Surface contradictions by measuring from opposing observation bases

---

## v2.0-beta — Inference Acceleration (from mind-inference)

**Goal:** Sub-millisecond hot paths. Predictive prefetch. KV cache for LLM-backed operations.

### KV Cache for LLM Operations
- [ ] Prefix caching for cross-encoder reranking (shared candidate context)
- [ ] Prefix caching for intent router (system prompt + governance context = cached)
- [ ] Multi-hop sub-queries share parent query prefix (90%+ overlap)
- [ ] Cache hit rate metric exposed via `index_stats`
- [ ] **TurboQuant-compressed prefix cache** — apply 3-bit vector quantization
  (arXiv:2504.19874) to cached KV embeddings for ~6x memory reduction. Enables
  caching far more prefix contexts in limited RAM/VRAM. PolarQuant rotation +
  Lloyd-Max codebook + QJL residual correction — quality-neutral at 3.5 bits/channel.
  Uses mind-inference's TurboQuant implementation when available (Phase 2), falls
  back to pure Python codebook lookup otherwise.

### Speculative Prefetch
- [ ] Predict next-needed blocks based on query pattern + access history
- [ ] Automatic prefetch during multi-hop decomposition (warm blocks before sub-query executes)
- [ ] Existing `prefetch` MCP tool becomes automatic (opt-in via config)
- [ ] Prefetch hit rate tracked in calibration feedback loop

### MIND-Compiled Hot Paths
- [ ] BM25F scoring kernel → `.mind` → native ELF via `mindc`
  - Porter stemming + term frequency + field weights in single compiled pass
  - Target: 1K blocks scored in <0.5ms (vs ~15ms Python)
- [ ] SHA3-512 hash chain verification → `.mind` → GPU kernel
  - Target: 81ns/hash (verified in mind-runtime benchmarks)
- [ ] Vector similarity (cosine/dot) → `.mind` → GPU kernel
  - Target: 1K vectors in <0.1ms (vs ~8ms Python)
- [ ] RRF fusion → `.mind` → native
  - Target: <0.01ms for 1K candidates
- [ ] FFI bridge: Python calls compiled `.mind` kernels via existing FFI path
- [ ] Automatic fallback to Python if compiled kernels unavailable

**Estimated:** ~1200 lines (MIND kernels) + ~400 lines (Python FFI bridge). Performance gains are opt-in — pure Python path remains default.

---

## v2.0-rc — External Verification (from 512-mind)

**Goal:** Third parties can verify memory integrity without full DB access.

### Merkle Tree over Block Store
- [ ] Merkle tree built over all blocks (leaf = block content hash)
- [ ] Merkle root anchored in snapshot metadata
- [ ] `verify_merkle` MCP tool — verify any single block's inclusion via proof
- [ ] Snapshot export includes Merkle root + proof paths

### Verification Without Operator Cooperation (I-4)
- [ ] Standalone `mind-mem-verify` CLI tool (reads snapshot + evidence.jsonl only)
- [ ] Verifies: hash chain integrity, spec-hash consistency, Merkle inclusion, evidence completeness
- [ ] Exit code 0 = verified, non-zero = specific failure code
- [ ] No database access required — works from snapshot alone

### Optional Ledger Anchoring
- [ ] Merkle root periodically anchored to external ledger (Ethereum L2 or similar)
- [ ] Anchoring is opt-in, not required for local verification
- [ ] `anchor_history` MCP tool shows all published roots + block heights

**Estimated:** ~800 lines. Fully backward-compatible — verification is additive.

---

## v2.0 Release Criteria

- [ ] All v2.0-alpha, beta, rc features complete
- [ ] Hash chain + spec-hash + evidence objects passing in CI
- [ ] MIND-compiled hot paths benchmarked (published in docs/benchmarks.md)
- [ ] `mind-mem-verify` CLI tool works on v1.x snapshots (backward compat)
- [ ] 2500+ tests passing
- [ ] LoCoMo benchmark re-run with acceleration (compare latency vs v1.9.x)
- [ ] Security audit of governance gate + hash chain implementation
- [ ] Migration guide from v1.9.x → v2.0 (no breaking changes expected)

---

## v2.1 — Self-Improving Retrieval via OpenClaw-RL

> **Paper:** "Train Any Agent Simply by Talking" (arXiv:2603.10165)
>
> Theme: mind-mem learns from every user interaction — corrections, re-queries, and rephrased searches become training signals that improve retrieval quality over time.

### Next-State Signal Recovery for Retrieval
- [ ] **Evaluative signal capture** — detect user re-queries (same intent, different phrasing) as negative feedback on previous recall
- [ ] **Directive signal extraction** — when user rephrases a query, extract the delta as a correction hint (OPD-style)
- [ ] **Signal taxonomy**: re-query = "result was wrong", refinement = "result was incomplete", explicit feedback = "that's not what I meant"
- [ ] **Signal store** — append-only JSONL log of (query, result, next_state, signal_type, timestamp)

### Local Fine-Tunable Retrieval Model
- [ ] **Local embedding model** — Qwen3-Embedding fine-tunable via LoRA on user interaction signals
- [ ] **Local reranker** — ms-marco-MiniLM fine-tunable on (query, passage, user_feedback) triples
- [ ] **Online training loop** — async: retrieval serves live, trainer updates model weights in background
- [ ] **Graceful weight swap** — new weights loaded without interrupting active recalls (SGLang-style)
- [ ] **Fallback** — if fine-tuned model degrades, auto-revert to base weights (governance-gated)

### Calibration Feedback Loop v2 (upgrade existing)
- [ ] **Per-block quality scores** feed into RL reward signal (existing infra → training signal)
- [ ] **Intent router adaptation** gains token-level OPD supervision (not just confidence weight adjustment)
- [ ] **A/B eval** — fine-tuned vs base model on held-out queries, auto-promote if MRR improves

### Metrics
- [ ] Recall MRR improvement over time (tracked per week)
- [ ] Signal capture rate (% of interactions that produce a training signal)
- [ ] Model revert rate (governance safety metric)

---

## v2.2 — Knowledge Graph Layer

> Theme: Relationships between facts are as retrievable as facts themselves.
> Ref: TrustGraph Context Core architecture, André Lindenberg "Memento Nightmare" analysis (2026-03-28)

### Entity-Relationship Graph Store
- [ ] **Graph backend** — pluggable: SQLite-based adjacency table (default), Neo4j, FalkorDB (optional)
- [ ] **Triple store** — (subject, predicate, object) with typed predicates: `AUTHORED_BY`, `DEPENDS_ON`, `CONTRADICTS`, `SUPERSEDES`, `PART_OF`, `MENTIONED_IN`
- [ ] **Entity registry** — canonical entity resolution: aliases, coreference, merge/split
- [ ] **Auto-extraction during ingestion** — entity pairs + relationships extracted per block (upgrade existing `entity_ingest`)
- [ ] **Graph-aware retrieval** — query hits block via BM25/vector → expand to N-hop neighbors via graph traversal → pack related entities into context
- [ ] **Multi-hop graph traversal** — "What are all projects that depend on tools authored by person X?" in <10ms for 100K nodes
- [ ] **Causal chain queries** — existing `causal_graph.py` promoted from governance-only to general retrieval
- [ ] **`graph_query` MCP tool** — Cypher-like query interface for direct graph access
- [ ] **`graph_stats` MCP resource** — node count, edge count, connected components, orphan detection

### Graph Reification (Statements About Statements)
- [ ] **Relationship-level provenance** — each edge carries: extraction_model, extraction_timestamp, source_block_id, confidence (0.0–1.0), temperature
- [ ] **Queryable provenance** — "Which model extracted the relationship between X and Y? At what confidence?"
- [ ] **Provenance-weighted retrieval** — edges from high-confidence sources ranked higher in graph expansion
- [ ] **Temporal validity windows** — edges can have `valid_from` / `valid_until` timestamps; expired edges excluded from retrieval by default

**Estimated:** ~2000 lines (graph store + extraction) + ~600 lines (reification + provenance). New dependency: none for SQLite backend.

---

## v2.3 — Context Cores: Portable Memory Bundles

> Theme: Docker for agent knowledge. Build once with a powerful model, deploy anywhere.
> Ref: TrustGraph Context Core concept

### Context Core Format
- [ ] **Bundle spec** — single archive (.mmcore) containing: blocks, graph edges, vector index, retrieval policies, ontology schema, metadata manifest
- [ ] **Versioned artifacts** — each core has semver + content hash; cores are immutable once published
- [ ] **Retrieval policies embedded** — BM25 weights, cross-encoder config, intent router weights, graph traversal depth — all travel with the bundle
- [ ] **Namespace isolation** — multiple cores loaded simultaneously with namespace prefixes; no cross-contamination in multi-tenant deployments
- [ ] **`build_core` MCP tool** — snapshot current memory (or filtered subset) into a .mmcore bundle
- [ ] **`load_core` / `unload_core` MCP tools** — hot-load/unload at runtime; no restart required
- [ ] **`list_cores` MCP resource** — active cores with stats (block count, graph size, load time)

### Edge Deployment
- [ ] **Lightweight runtime** — core loads in <2s on 1B-param model environments (no LLM needed for retrieval, only for answering)
- [ ] **Core diffing** — generate delta between core versions; deploy incremental updates instead of full bundle
- [ ] **Core rollback** — revert to previous core version when new knowledge proves flawed
- [ ] **Export to static formats** — .mmcore → JSON-LD, RDF/Turtle, or plain Markdown for interop

**Estimated:** ~1500 lines (bundle format + build/load) + ~400 lines (edge runtime). New file format, backward-compatible (cores are additive).

---

## v2.4 — Cognitive Memory Management

> Theme: Active forgetting, token-aware packing, and multi-modal memory.
>
> **Cross-ref:** Naestro's `consolidator.mind` (2026-03-31) implements the idle-time consolidation
> cycle for belief graphs (merge similar, resolve contradictions, promote repeated observations,
> decay stale). `write_discipline.mind` enforces the write-then-index invariant so failed writes
> never pollute the retrieval index. Both modules integrate with mind-mem via FFI. The v2.4
> features below formalize what those modules already enforce at the cognitive daemon level into
> mind-mem's own API surface.

### Active Cognitive Forgetting
- [ ] **Sleep consolidation cycle** — periodic background pass: mark → merge → archive → forget
  - **Mark**: blocks below importance threshold + no access in N days flagged for review
  - **Merge**: semantically similar blocks compressed into single summary block (provenance preserved)
  - **Archive**: merged blocks moved to cold storage (still queryable, not in hot index)
  - **Forget**: archived blocks past TTL permanently removed (governance-gated, requires explicit opt-in)
- [ ] **Compression ratio metric** — track block count reduction per consolidation cycle
- [ ] **Forgetting governance** — every forget decision produces an Evidence Object; reversible within 30-day grace period
- [ ] **Memory pressure alerts** — when block count exceeds configurable threshold, trigger consolidation cycle
- [ ] **`consolidate` MCP tool** — manual trigger with dry-run mode

### Token Budget Management
- [ ] **Context window awareness** — recall accepts `max_tokens` parameter; packer allocates budget across: system prompt, graph context, retrieved blocks, conversation history
- [ ] **Adaptive packing strategy** — given token budget:
  1. Reserve 15% for graph context (entity relationships)
  2. Reserve 10% for provenance metadata
  3. Pack remaining with blocks by relevance score, truncating lowest-scored
- [ ] **Packing quality metric** — % of packed tokens that user actually references in response (tracked via calibration loop)
- [ ] **Model-aware budgets** — auto-detect context window from model name (128K, 200K, 1M) and set defaults
- [ ] **`recall` gains `max_tokens` param** — backward-compatible, defaults to unlimited (current behavior)

### Multi-Modal Memory
- [ ] **Image block type** — `[IMAGE]` blocks store: description, embedding (CLIP/SigLIP), source path, dimensions, thumbnail hash
- [ ] **Audio block type** — `[AUDIO]` blocks store: transcript, embedding, duration, speaker labels, source path
- [ ] **Cross-modal retrieval** — text query retrieves relevant images/audio; image query retrieves relevant text blocks
- [ ] **Auto-extraction** — images/audio ingested via pipeline: transcribe/describe → embed → store with text + modal embedding
- [ ] **Modal-aware packing** — token budget accounts for image tokens (vision models) vs text-only models

**Estimated:** ~1800 lines (forgetting + packing) + ~1200 lines (multi-modal). No breaking changes.

---

## v2.5 — Ontology & Streaming

> Theme: Schema-enforced knowledge and real-time memory.

### Ontology / Schema Typing
- [ ] **OWL-lite schema support** — define entity types with required/optional properties
  - Example: `PERSON` must have `role`; `PROJECT` must have `status`, `repo`
- [ ] **Schema validation on write** — blocks referencing typed entities validated against ontology at ingestion
- [ ] **Schema evolution** — versioned ontologies; old blocks validated against schema version at write time
- [ ] **Domain ontology library** — pre-built schemas for: software engineering, legal, medical, financial
- [ ] **`ontology_load` / `ontology_validate` MCP tools**
- [ ] **Schema-guided retrieval** — "find all PERSONs with role=engineer" uses schema-aware index, not text search

### Streaming Ingestion
- [ ] **Event-driven write path** — new blocks written via async event queue (not synchronous DB write)
- [ ] **Write-ahead log** — blocks committed to WAL first, indexed asynchronously; queryable within <50ms of write
- [ ] **Webhook ingestion endpoint** — HTTP POST → block creation (for external event sources)
- [ ] **Change stream** — subscribers notified on new block/edge creation (for downstream consumers: dashboards, agents)
- [ ] **Backpressure** — configurable queue depth; shed load gracefully under burst writes
- [ ] **`stream_status` MCP resource** — queue depth, write latency, consumer lag

**Estimated:** ~1000 lines (ontology) + ~800 lines (streaming). Optional dependencies: none for core (aiohttp for webhook endpoint).

---

## Post-v2.5 — Future Directions

- [ ] **Agent-to-agent trust protocol** — agents verify each other's memory integrity via Merkle proofs before sharing context
- [ ] **Distributed memory mesh** — multiple mind-mem instances with hash-chain synchronization
- [ ] **Real-time governance dashboard** — web UI showing evidence stream, chain health, spec-hash status
- [ ] **512 Kernel full integration** — mind-mem as a governed resource within 512-mind production deployments
- [ ] **Hardware-specific compilation** — `mindc` targets for ARM (Apple Silicon), CUDA, ROCm
- [ ] **Multi-user retrieval adaptation** — per-user fine-tuning in multi-tenant deployments, isolated signal streams
- [ ] **Federated memory** — privacy-preserving retrieval across organizational boundaries (differential privacy + secure aggregation)
- [ ] **Continuous benchmark regression** — every PR runs LoCoMo subset + latency benchmarks; auto-reject if MRR drops or p99 increases >10%

---

## AGI Integration

This repo is part of the STARGA AGI stack. See `naestro-bot/specs/AGI-ROADMAP.md` for:
- Gap 1: CAUSAL block type for world model storage
- Gap 2: SKILL blocks for learned strategy tracking
- Gap 5: Cross-domain retrieval for transfer learning
- Gap 6: VISUAL blocks for grounded perception memory

# MIND-Mem Roadmap

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

> Three STARGA projects converge: **512-mind** governance primitives + **mind-inference** acceleration + **MIND-Mem** retrieval.
>
> Theme: The first AI memory system with **cryptographically verifiable governance** and **hardware-accelerated hot paths**.
>
> Versions follow PEP 440 (what PyPI actually accepts). The alpha → beta →
> rc → final progression maps to the milestone labels Cryptographic
> Governance → ODC Retrieval → Inference Acceleration → External
> Verification → v2.0 Final.

---

## v2.0.0a2 — Cryptographic Governance Layer (from 512-mind) ✅ Released as v2.0.0a2 (2026-04-13)

**Goal:** Every memory write is tamper-evident. Governance config is immutable post-init. Evidence objects prove governance actually ran.

### Hash-Chained Block Writes
- [x] SHA3-512 hash chain: each block write includes `prev_hash` linking to previous write
- [x] Chain head stored in DB metadata, verifiable from any snapshot
- [x] `verify_chain()` MCP tool — walk the chain, report any breaks
- [x] Existing `create_snapshot` / `restore_snapshot` tools gain chain-head verification

### Spec-Hash Binding (I-5)
- [x] SHA3-512 hash of governance config (`mind-mem.json` governance section) computed at init
- [x] `spec_hash` embedded in every Evidence Object
- [x] Runtime check: if config file changes post-init, log spec-hash divergence + alert
- [x] `governance_spec_hash` exposed via MCP `index_stats` resource

### Structured Evidence Objects
- [x] Every governance decision (proposal ALLOW/DENY, contradiction detection, drift alert) outputs a structured Evidence Object:
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
- [x] Evidence Objects are append-only (separate evidence.jsonl file)
- [x] `list_evidence` MCP tool for audit queries

### Single Gateway Enforcement (I-1)
- [x] All block writes must pass through `GovernanceGate.admit()` — no direct DB writes
- [x] BlockStore protocol enforced as the only write path (remove any bypass paths)
- [x] Write attempts outside BlockStore raise `GovernanceBypassError`

**Estimated:** ~600 lines across 4 modules. No breaking changes to existing API.

---

## v2.0.0a3 — Observer-Dependent Cognition (ODC) Retrieval ✅ Released as v2.0.0a3 (2026-04-13)

**Goal:** Make retrieval axis-aware. Every recall declares its observation basis, results include axis metadata, and the system can rotate axes for higher-confidence results.

**Spec:** `specs/observer-dependent-cognition.md`

### Axis-Aware Retrieval
- [x] `ObservationAxis` enum (lexical, semantic, temporal, entity_graph, contradiction, adversarial) + `AxisWeights` vector
- [x] `recall_with_axis` orchestrator dispatches per-axis passes with explicit weights, fused via weighted RRF
- [x] Axis choices recorded per-result in the `observation` metadata (foundation for evidence-chain integration in v2.0.0rc1)
- [x] Axis rotation: `should_rotate` fires when top-confidence < `DEFAULT_ROTATION_THRESHOLD (0.35)`, `rotate_axes` picks orthogonals

### Observation Metadata
- [x] Every recall result tagged with producing axes + per-axis confidence scores + rank
- [x] New MCP tool `recall_with_axis` with user-scope ACL, hardened arg parsing (length + count bounds, limit cap)
- [x] Axis diversity metric (`axis_diversity(results)`) returns count of distinct axes that contributed

### Adversarial Axis Injection
- [x] `adversarial=True` runs each active axis's adversarial pair (LEXICAL/SEMANTIC/TEMPORAL/ENTITY_GRAPH → CONTRADICTION; CONTRADICTION → ADVERSARIAL)
- [x] ADVERSARIAL axis wraps the query as `NOT "..."` (FTS5-safe phrase form) to surface dissent from the opposing basis

---

## v2.0.0b1 — Inference Acceleration (from mind-inference) — Python subset ✅ Released 2026-04-13 — all boxes checked in v2.8.0

**Goal:** Sub-millisecond hot paths. Predictive prefetch. KV cache for LLM-backed operations.

### KV Cache for LLM Operations
- [x] Prefix caching for cross-encoder reranking (shared candidate context)
- [x] Prefix caching for intent router (system prompt + governance context = cached)
- [x] Multi-hop sub-queries share parent query prefix (90%+ overlap)
- [x] Cache hit rate metric exposed via `index_stats`
- [x] **TurboQuant-compressed prefix cache** — apply 3-bit vector quantization
  (arXiv:2504.19874) to cached KV embeddings for ~6x memory reduction. Enables
  caching far more prefix contexts in limited RAM/VRAM. PolarQuant rotation +
  Lloyd-Max codebook + QJL residual correction — quality-neutral at 3.5 bits/channel.
  Uses mind-inference's TurboQuant implementation when available (Phase 2), falls
  back to pure Python codebook lookup otherwise.

### Speculative Prefetch
- [x] Predict next-needed blocks based on query pattern + access history
- [x] Automatic prefetch during multi-hop decomposition (warm blocks before sub-query executes)
- [x] Existing `prefetch` MCP tool becomes automatic (opt-in via config)
- [x] Prefetch hit rate tracked in calibration feedback loop

### MIND-Compiled Hot Paths
- [x] BM25F scoring kernel → `.mind` → native ELF via `mindc`
  - Porter stemming + term frequency + field weights in single compiled pass
  - Target: 1K blocks scored in <0.5ms (vs ~15ms Python)
- [x] SHA3-512 hash chain verification → `.mind` → GPU kernel
  - Target: 81ns/hash (verified in mind-runtime benchmarks)
- [x] Vector similarity (cosine/dot) → `.mind` → GPU kernel
  - Target: 1K vectors in <0.1ms (vs ~8ms Python)
- [x] RRF fusion → `.mind` → native
  - Target: <0.01ms for 1K candidates
- [x] FFI bridge: Python calls compiled `.mind` kernels via existing FFI path
- [x] Automatic fallback to Python if compiled kernels unavailable

**Estimated:** ~1200 lines (MIND kernels) + ~400 lines (Python FFI bridge). Performance gains are opt-in — pure Python path remains default.

---

## v2.0.0rc1 — External Verification (from 512-mind) ✅ Released 2026-04-13 — all boxes checked in v2.8.0

**Goal:** Third parties can verify memory integrity without full DB access.

### Merkle Tree over Block Store
- [x] Merkle tree built over all blocks (leaf = block content hash)
- [x] Merkle root anchored in snapshot metadata
- [x] `verify_merkle` MCP tool — verify any single block's inclusion via proof
- [x] Snapshot export includes Merkle root + proof paths

### Verification Without Operator Cooperation (I-4)
- [x] Standalone `mind-mem-verify` CLI tool (reads snapshot + evidence.jsonl only)
- [x] Verifies: hash chain integrity, spec-hash consistency, Merkle inclusion, evidence completeness
- [x] Exit code 0 = verified, non-zero = specific failure code
- [x] No database access required — works from snapshot alone

### Optional Ledger Anchoring
- [x] Merkle root periodically anchored to external ledger (Ethereum L2 or similar)
- [x] Anchoring is opt-in, not required for local verification
- [x] `anchor_history` MCP tool shows all published roots + block heights

**Estimated:** ~800 lines. Fully backward-compatible — verification is additive.

---

## v2.0.0 ✅ Released 2026-04-13 — stable promotion of the a2/a3/b1/rc1 train

Release criteria:
- [x] All v2.0.0a*, v2.0.0b*, v2.0.0rc* features complete
- [x] Hash chain + spec-hash + evidence objects passing (3197 tests green)
- [x] MIND-compiled hot paths benchmarked (published in docs/benchmarks.md)
- [x] `mind-mem-verify` CLI tool works on v1.x snapshots (backward compat)
- [x] 2500+ tests passing
- [x] LoCoMo benchmark re-run with acceleration (compare latency vs v1.9.x)
- [x] Security audit of governance gate + hash chain implementation
- [x] Migration guide from v1.9.x → v2.0.0 (no breaking changes — just `pip install --upgrade mind-mem`)

---

## v2.1.0 — Self-Improving Retrieval via OpenClaw-RL ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> **Paper:** "Train Any Agent Simply by Talking" (arXiv:2603.10165)
>
> Theme: MIND-Mem learns from every user interaction — corrections, re-queries, and rephrased searches become training signals that improve retrieval quality over time.

### Next-State Signal Recovery for Retrieval
- [x] **Evaluative signal capture** — detect user re-queries (same intent, different phrasing) as negative feedback on previous recall
- [x] **Directive signal extraction** — when user rephrases a query, extract the delta as a correction hint (OPD-style)
- [x] **Signal taxonomy**: re-query = "result was wrong", refinement = "result was incomplete", explicit feedback = "that's not what I meant"
- [x] **Signal store** — append-only JSONL log of (query, result, next_state, signal_type, timestamp)

### Local Fine-Tunable Retrieval Model
- [x] **Local embedding model** — Qwen3-Embedding fine-tunable via LoRA on user interaction signals
- [x] **Local reranker** — ms-marco-MiniLM fine-tunable on (query, passage, user_feedback) triples
- [x] **Online training loop** — async: retrieval serves live, trainer updates model weights in background
- [x] **Graceful weight swap** — new weights loaded without interrupting active recalls (SGLang-style)
- [x] **Fallback** — if fine-tuned model degrades, auto-revert to base weights (governance-gated)

### Calibration Feedback Loop v2 (upgrade existing)
- [x] **Per-block quality scores** feed into RL reward signal (existing infra → training signal)
- [x] **Intent router adaptation** gains token-level OPD supervision (not just confidence weight adjustment)
- [x] **A/B eval** — fine-tuned vs base model on held-out queries, auto-promote if MRR improves

### Metrics
- [x] Recall MRR improvement over time (tracked per week)
- [x] Signal capture rate (% of interactions that produce a training signal)
- [x] Model revert rate (governance safety metric)

---

## v2.2.0 — Knowledge Graph Layer ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> Theme: Relationships between facts are as retrievable as facts themselves.
> Ref: TrustGraph Context Core architecture, André Lindenberg "Memento Nightmare" analysis (2026-03-28)

### Entity-Relationship Graph Store
- [x] **Graph backend** — pluggable: SQLite-based adjacency table (default), Neo4j, FalkorDB (optional)
- [x] **Triple store** — (subject, predicate, object) with typed predicates: `AUTHORED_BY`, `DEPENDS_ON`, `CONTRADICTS`, `SUPERSEDES`, `PART_OF`, `MENTIONED_IN`
- [x] **Entity registry** — canonical entity resolution: aliases, coreference, merge/split
- [x] **Auto-extraction during ingestion** — entity pairs + relationships extracted per block (upgrade existing `entity_ingest`)
- [x] **Graph-aware retrieval** — query hits block via BM25/vector → expand to N-hop neighbors via graph traversal → pack related entities into context
- [x] **Multi-hop graph traversal** — "What are all projects that depend on tools authored by person X?" in <10ms for 100K nodes
- [x] **Causal chain queries** — existing `causal_graph.py` promoted from governance-only to general retrieval
- [x] **`graph_query` MCP tool** — Cypher-like query interface for direct graph access
- [x] **`graph_stats` MCP resource** — node count, edge count, connected components, orphan detection

### Graph Reification (Statements About Statements)
- [x] **Relationship-level provenance** — each edge carries: extraction_model, extraction_timestamp, source_block_id, confidence (0.0–1.0), temperature
- [x] **Queryable provenance** — "Which model extracted the relationship between X and Y? At what confidence?"
- [x] **Provenance-weighted retrieval** — edges from high-confidence sources ranked higher in graph expansion
- [x] **Temporal validity windows** — edges can have `valid_from` / `valid_until` timestamps; expired edges excluded from retrieval by default

**Estimated:** ~2000 lines (graph store + extraction) + ~600 lines (reification + provenance). New dependency: none for SQLite backend.

---

## v2.3.0 — Context Cores: Portable Memory Bundles ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> Theme: Docker for agent knowledge. Build once with a powerful model, deploy anywhere.
> Ref: TrustGraph Context Core concept

### Context Core Format
- [x] **Bundle spec** — single archive (.mmcore) containing: blocks, graph edges, vector index, retrieval policies, ontology schema, metadata manifest
- [x] **Versioned artifacts** — each core has semver + content hash; cores are immutable once published
- [x] **Retrieval policies embedded** — BM25 weights, cross-encoder config, intent router weights, graph traversal depth — all travel with the bundle
- [x] **Namespace isolation** — multiple cores loaded simultaneously with namespace prefixes; no cross-contamination in multi-tenant deployments
- [x] **`build_core` MCP tool** — snapshot current memory (or filtered subset) into a .mmcore bundle
- [x] **`load_core` / `unload_core` MCP tools** — hot-load/unload at runtime; no restart required
- [x] **`list_cores` MCP resource** — active cores with stats (block count, graph size, load time)

### Edge Deployment
- [x] **Lightweight runtime** — core loads in <2s on 1B-param model environments (no LLM needed for retrieval, only for answering)
- [x] **Core diffing** — generate delta between core versions; deploy incremental updates instead of full bundle
- [x] **Core rollback** — revert to previous core version when new knowledge proves flawed
- [x] **Export to static formats** — .mmcore → JSON-LD, RDF/Turtle, or plain Markdown for interop

**Estimated:** ~1500 lines (bundle format + build/load) + ~400 lines (edge runtime). New file format, backward-compatible (cores are additive).

---

## v2.4.0 — Cognitive Memory Management ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> Theme: Active forgetting, token-aware packing, and multi-modal memory.
>
> **Cross-ref:** an internal `consolidator.mind` module (2026-03-31) implements the idle-time consolidation
> cycle for belief graphs (merge similar, resolve contradictions, promote repeated observations,
> decay stale). `write_discipline.mind` enforces the write-then-index invariant so failed writes
> never pollute the retrieval index. Both modules integrate with MIND-Mem via FFI. The v2.4.0
> features below formalize what those modules already enforce at the cognitive daemon level into
> MIND-Mem's own API surface.

### Active Cognitive Forgetting
- [x] **Sleep consolidation cycle** — periodic background pass: mark → merge → archive → forget
  - **Mark**: blocks below importance threshold + no access in N days flagged for review
  - **Merge**: semantically similar blocks compressed into single summary block (provenance preserved)
  - **Archive**: merged blocks moved to cold storage (still queryable, not in hot index)
  - **Forget**: archived blocks past TTL permanently removed (governance-gated, requires explicit opt-in)
- [x] **Compression ratio metric** — track block count reduction per consolidation cycle
- [x] **Forgetting governance** — every forget decision produces an Evidence Object; reversible within 30-day grace period
- [x] **Memory pressure alerts** — when block count exceeds configurable threshold, trigger consolidation cycle
- [x] **`consolidate` MCP tool** — manual trigger with dry-run mode

### Token Budget Management
- [x] **Context window awareness** — recall accepts `max_tokens` parameter; packer allocates budget across: system prompt, graph context, retrieved blocks, conversation history
- [x] **Adaptive packing strategy** — given token budget:
  1. Reserve 15% for graph context (entity relationships)
  2. Reserve 10% for provenance metadata
  3. Pack remaining with blocks by relevance score, truncating lowest-scored
- [x] **Packing quality metric** — % of packed tokens that user actually references in response (tracked via calibration loop)
- [x] **Model-aware budgets** — auto-detect context window from model name (128K, 200K, 1M) and set defaults
- [x] **`recall` gains `max_tokens` param** — backward-compatible, defaults to unlimited (current behavior)

### Multi-Modal Memory
- [x] **Image block type** — `[IMAGE]` blocks store: description, embedding (CLIP/SigLIP), source path, dimensions, thumbnail hash
- [x] **Audio block type** — `[AUDIO]` blocks store: transcript, embedding, duration, speaker labels, source path
- [x] **Cross-modal retrieval** — text query retrieves relevant images/audio; image query retrieves relevant text blocks
- [x] **Auto-extraction** — images/audio ingested via pipeline: transcribe/describe → embed → store with text + modal embedding
- [x] **Modal-aware packing** — token budget accounts for image tokens (vision models) vs text-only models

**Estimated:** ~1800 lines (forgetting + packing) + ~1200 lines (multi-modal). No breaking changes.

---

## v2.5.0 — Ontology & Streaming ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> Theme: Schema-enforced knowledge and real-time memory.

### Ontology / Schema Typing
- [x] **OWL-lite schema support** — define entity types with required/optional properties
  - Example: `PERSON` must have `role`; `PROJECT` must have `status`, `repo`
- [x] **Schema validation on write** — blocks referencing typed entities validated against ontology at ingestion
- [x] **Schema evolution** — versioned ontologies; old blocks validated against schema version at write time
- [x] **Domain ontology library** — pre-built schemas for: software engineering, legal, medical, financial
- [x] **`ontology_load` / `ontology_validate` MCP tools**
- [x] **Schema-guided retrieval** — "find all PERSONs with role=engineer" uses schema-aware index, not text search

### Streaming Ingestion
- [x] **Event-driven write path** — new blocks written via async event queue (not synchronous DB write)
- [x] **Write-ahead log** — blocks committed to WAL first, indexed asynchronously; queryable within <50ms of write
- [x] **Webhook ingestion endpoint** — HTTP POST → block creation (for external event sources)
- [x] **Change stream** — subscribers notified on new block/edge creation (for downstream consumers: dashboards, agents)
- [x] **Backpressure** — configurable queue depth; shed load gracefully under burst writes
- [x] **`stream_status` MCP resource** — queue depth, write latency, consumer lag

**Estimated:** ~1000 lines (ontology) + ~800 lines (streaming). Optional dependencies: none for core (aiohttp for webhook endpoint).

---

## v2.6.0 — Memory Surface Expansion ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> Theme: STARGA-native expansion of the recall surface — staleness propagation across
> the dependency graph, working/episodic/semantic/procedural tiering, project intelligence
> profiles, P2P mesh, and a Model Reliability Score framework. Designed first-principles
> from the requirement that memory must be auditable, decay-aware, and shareable across
> agents without leaking authority.

### Cascading Staleness Propagation
_Rationale: when a block is invalidated, every block that depends on it inherits doubt — staleness must propagate transitively along the graph._

- [x] **Staleness propagation engine** — when a block is superseded or contradicted, automatically flag related blocks (edges, siblings, dependents) as potentially stale
- [x] **Staleness confidence decay** — propagation weakens with graph distance: direct relations get `stale=0.9`, 2-hop get `stale=0.5`, 3-hop get `stale=0.2`
- [x] **Staleness in retrieval scoring** — stale-flagged blocks penalized in BM25F scoring (configurable weight, default 0.3x)
- [x] **`propagate_staleness` MCP tool** — manual trigger with dry-run mode showing affected blocks
- [x] **Staleness audit log** — every propagation event recorded with source block, affected blocks, reason

### 4-Tier Memory Consolidation
_Rationale: not every memory deserves equal recall priority — working/episodic/semantic/procedural tiers with biologically-motivated decay match how agents actually use memory across a session._

MIND-Mem currently has append-only logs → manual promotion to MEMORY.md. This formalizes the pipeline:

- [x] **Tier 0 (Working)** — raw daily log entries (`memory/YYYY-MM-DD.md`), TTL 30 days before decay review
- [x] **Tier 1 (Episodic)** — compressed session summaries (`summaries/weekly/`), auto-generated from Tier 0
- [x] **Tier 2 (Semantic)** — verified facts and entity knowledge (`entities/`, `MEMORY.md`), promoted from Tier 1 after N repetitions or explicit confirmation
- [x] **Tier 3 (Procedural)** — learned patterns and strategies (`decisions/`), highest durability, governance-gated
- [x] **Ebbinghaus strength decay** — each block has `strength` field (0.0–1.0), decays exponentially with configurable half-life (default 30 days), reset on access
- [x] **Auto-promotion triggers** — block repeated 3+ times across sessions → auto-promote to next tier (governance proposal if Tier 2→3)
- [x] **Tier-aware retrieval** — higher tiers get retrieval priority boost (Tier 3: 2.0x, Tier 2: 1.5x, Tier 1: 1.0x, Tier 0: 0.7x)
- [x] **`consolidate` MCP tool** — trigger consolidation cycle with `--dry-run` and `--tier` filters

### Agent Hook Auto-Capture
_Rationale: a memory system that requires explicit calls to capture is never used — silent observation through CLI hooks is the only path to comprehensive coverage._

- [x] **Hook event schema** — standardized event format: `{type, timestamp, tool, input_hash, output_summary, project, session_id}`
- [x] **SessionStart hook** — inject recent context from MIND-Mem at conversation start (token-budgeted)
- [x] **PostToolUse hook** — capture tool name + output summary, SHA-256 dedup (5-min window)
- [x] **PreCompact hook** — re-inject critical memory context before context compaction
- [x] **SessionEnd hook** — trigger end-of-session summary compression
- [x] **Privacy filter** — strip API keys, secrets, `<private>` tagged content before storage
- [x] **Hook installer** — `mind-mem hooks install` CLI command, writes to `~/.claude/settings.json`
- [x] **Observation → block pipeline** — raw hook events compressed into structured blocks via LLM (Zod-validated, quality scored 0-100)

### Token Budget Context Injection
_Rationale: every recall result spends caller context — a configurable token budget with smart packing makes the cost explicit and bounded._

- [x] **`recall` gains `max_tokens` parameter** — backward-compatible, defaults to unlimited (current behavior)
- [x] **Adaptive packing strategy** — given token budget:
  1. Reserve 15% for graph context (entity relationships)
  2. Reserve 10% for provenance metadata (source citations)
  3. Pack remaining with blocks by relevance score, truncating lowest-scored
- [x] **Model-aware defaults** — auto-detect context window from model name and set sensible defaults
- [x] **Packing quality metric** — track % of packed tokens actually referenced in response (calibration loop)

### Project Intelligence Profiles
_Rationale: per-project profiles let an agent skip the ten-second "what is this codebase" warmup that bleeds tokens at every session start._

- [x] **Auto-generated project profiles** — aggregate from entity files + observations: top concepts, most-touched files, coding conventions, common errors, session count
- [x] **Profile as MCP resource** — `mindmem://project/{name}/profile` exposes structured project intelligence
- [x] **Profile injection at session start** — when project context detected, inject profile into system prompt
- [x] **Convention extraction** — LLM-powered extraction of implicit conventions from code observations (naming patterns, test patterns, error handling style)

### P2P Memory Mesh
_Rationale: when multiple agents work on the same project, isolated memories diverge — a P2P mesh with scope-typed sync keeps them coherent without forcing centralisation._

- [x] **Mesh protocol** — MIND-Mem instances discover peers via mDNS or explicit peer list
- [x] **7 sync scopes** — memories, actions, semantic, procedural, relations, graph, governance (each independently toggleable)
- [x] **Conflict resolution** — last-write-wins for Tier 0-1, governance-gated merge for Tier 2-3
- [x] **Namespace isolation** — shared vs private memory with per-scope access control
- [x] **Sync audit log** — every sync event recorded with peer ID, scope, blocks transferred, conflicts resolved
- [x] **`mesh_status` MCP resource** — connected peers, sync lag, scope health

### Model Reliability Score (MRS) Framework
_Rationale: model endpoints + retrieval backends are infrastructure — they need SLO-style reliability scoring (latency/quality/drift), not anecdotal "feels slow" judgement._

- [x] **MRS SLI definitions** — latency percentiles (p50/p95/p99), output quality drift, token throughput, error rate, cost per query
- [x] **Composite MRS (0-100)** — weighted aggregation of SLIs into single reliability score
- [x] **YAML SLO schema** — define per-model SLO thresholds, weights, alert conditions
- [x] **Memory retrieval MRS extension** — relevance decay rate, contradiction density, staleness ratio as retrieval-specific SLIs
- [x] **MRS dashboard** — real-time MRS per model endpoint + per retrieval backend
- [x] **Alert on MRS degradation** — configurable thresholds trigger warnings before quality impacts users

**Estimated:** ~3500 lines total. No breaking changes. All features are additive and config-gated.

---

## v2.7.0 — Universal Agent Bridge + Vault Sync ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> Theme: MIND-Mem becomes the shared memory layer for **every** coding agent — not just MCP-capable ones.
> Any CLI agent (Claude Code, codex, gemini, Cursor, Windsurf, Aider) reads and writes
> to the same memory through a unified interface. Plus bidirectional vault sync for Obsidian/file-based
> knowledge management.
>
> Rationale: the cost of a memory system is dominated by the agents that *can't* use it — universal CLI bridge eliminates that gap. Vault sync acknowledges that human knowledge bases (Obsidian-format markdown vaults) and agent memory should be one substrate, not two.

### Component 1: Universal Agent Bridge (`mm` CLI)

**Problem:** MCP-capable agents (Claude Code, other MCP-native runtimes) already have MIND-Mem access. Non-MCP agents (codex, gemini CLI, Cursor, Windsurf, Aider) have zero memory — every session starts blank. The `mm` CLI bridges this gap.

- [x] **`mm` unified CLI** — single binary (`~/.local/bin/mm`) wrapping all MIND-Mem operations:
  ```
  mm recall "query"                    # search memory (BM25F+vector hybrid)
  mm capture "text" --type decision    # store new block
  mm context "topic"                   # generate token-budgeted context blob for prompt injection
  mm scan                              # reindex workspace
  mm status                            # index stats, last scan, health
  mm inject --agent codex              # output context formatted for specific agent's system prompt
  mm hook install --agent <name>       # install agent-specific hooks/config
  ```
- [x] **Agent-specific formatters** — `mm inject` outputs context in the format each agent expects:
  - Claude Code: `CLAUDE.md` snippet injection
  - codex: `AGENTS.md` / `codex.md` injection
  - gemini: `GEMINI.md` / system instruction injection
  - Cursor: `.cursorrules` injection
  - Windsurf: `.windsurfrules` injection
  - Aider: `.aider.conf.yml` repo-map injection
  - Generic: stdout (pipe into any prompt)
- [x] **Pre-session context injection** — `mm context` generates a token-budgeted memory blob:
  1. Recall recent decisions (highest priority)
  2. Recall relevant entity context (by project detection)
  3. Recall open tasks
  4. Pack within configurable token budget (default 2000 tokens)
  5. Output as structured markdown ready for system prompt
- [x] **Post-session capture** — `mm capture --stdin` reads session transcript from stdin, extracts:
  - New decisions, corrections, preferences
  - Entity mentions (projects, people, tools)
  - Task state changes
  - Runs entity extraction + dedup before storage
- [x] **Shell integration** — optional shell hooks for automatic context injection:
  ```bash
  # .bashrc / .zshrc
  export MIND_MEM_WORKSPACE=/home/n/.openclaw/workspace
  alias codex='mm inject --agent codex --quiet && codex'
  alias gemini='mm inject --agent gemini --quiet && gemini'
  ```
- [x] **Agent config installer** — `mm hook install --agent claude-code` writes:
  - Claude Code: `~/.claude/settings.json` hooks (SessionStart + PostToolUse + Stop)
  - codex: `AGENTS.md` with memory recall instructions
  - gemini: `.gemini/settings.json` system instruction with recall context
  - Cursor: `.cursorrules` with memory-aware preamble
- [x] **Shared workspace env var** — `MIND_MEM_WORKSPACE` (default: `~/.openclaw/workspace`) ensures all agents write to the same index
- [x] **Conflict-free concurrent access** — WAL mode SQLite (already implemented) + advisory file locking for multi-agent concurrent reads/writes

### Component 2: Vault Bidirectional Sync

**Problem:** Obsidian (and similar PKM tools) provide visual graph navigation, backlinks, and manual curation that MIND-Mem doesn't. MIND-Mem provides hybrid retrieval, governance, and agent-accessible MCP that Obsidian doesn't. Users shouldn't have to choose.

- [x] **Vault scanner** — `mm vault sync /path/to/obsidian/vault`:
  - Reads all `.md` files in vault
  - Detects block types from frontmatter/headers (decisions, entities, tasks, notes)
  - Indexes into MIND-Mem with `source: vault` provenance tag
  - Respects `.obsidian/` and `.trash/` exclusions
  - Incremental: only re-indexes files modified since last sync (mtime-based)
- [x] **Reverse sync** — MIND-Mem → vault:
  - New decisions/entities created via `mm capture` or MCP get written back to vault as `.md` files
  - Maintains Obsidian-compatible frontmatter (tags, aliases, created, modified)
  - Creates `[[wikilinks]]` for entity cross-references
  - Respects vault folder structure (configurable mapping: decisions/ → vault/decisions/, etc.)
- [x] **Conflict resolution** — when both sides modify the same block:
  - Vault wins for manual edits (human curation > agent writes)
  - MIND-Mem wins for governance decisions (contradictions, drift alerts)
  - Conflicts logged with both versions preserved
- [x] **Vault config** — in `mind-mem.json`:
  ```json
  {
    "vault": {
      "path": "/path/to/obsidian/vault",
      "sync_dirs": ["decisions", "entities", "projects", "daily"],
      "exclude": [".obsidian", ".trash", "templates"],
      "reverse_sync": true,
      "conflict_policy": "vault_wins",
      "sync_interval_minutes": 5
    }
  }
  ```
- [x] **`mm vault status`** — last sync time, files indexed, pending reverse writes, conflicts
- [x] **`mm vault watch`** — filesystem watcher (inotify/fsevents) for real-time sync
- [x] **`vault_sync` MCP tool** — trigger sync from any MCP-connected agent
- [x] **Obsidian plugin (future)** — native Obsidian plugin that calls `mm` directly for in-editor recall

**Estimated:** ~1200 lines (mm CLI + formatters) + ~800 lines (vault sync). New dependency: `watchdog` (optional, for `vault watch`). No breaking changes.

---

## v3.0.0 — Architectural Release ✅ Released 2026-04-13

- [x] **Alerting layer** — `AlertRouter` + pluggable sinks (`LogSink`, `WebhookSink`, `SlackSink`, `NullSink`); intel-scan fires alerts on contradiction and drift spikes; config in `mind-mem.json` `alerts` section (GH #503, 13 tests)
- [x] **Transparent encryption at rest** — `EncryptedBlockStore` wrapper + `encrypt_workspace(ws)` one-shot migration; `get_block_store(ws)` factory dispatches on `MIND_MEM_ENCRYPTION_PASSPHRASE` env var (GH #504, 8 tests)
- [x] **Tier TTL/LRU decay** — `TierManager.run_decay_cycle()` demotes idle blocks and evicts never-accessed WORKING-tier blocks; wired into compaction alongside promotion; `max_idle_hours` + `ttl_hours` on `TierPolicy` (GH #502, 10 tests)
- [x] **Adversarial corpus harness** — 16 tests covering NUL injection, NaN smuggling, forged v1 hashes, SQL-flavour queries, oversized metadata (GH #507)
- [x] **Governance concurrency stress harness** — new `pytest -m stress` marker; 5 tests exercising N concurrent writers on audit_chain, hash_chain_v2, memory_tiers, evidence_objects (GH #506)
- [x] **16-client AI hook installer** — registry-driven `hook_installer.py`; new agent registrations: `openclaw`, `nanoclaw`, `nemoclaw`, `continue`, `cline`, `roo`, `zed`, `copilot`, `cody`, `qodo` in addition to existing `claude-code`, `codex`, `gemini`, `cursor`, `windsurf`, `aider` (28 tests)
- [x] **`mm detect` / `mm install` / `mm install-all`** CLI commands with auto-detection
- [x] **End-to-end memory test suite** — 9 tests: seeded corpus recall, contradiction lifecycle, audit chain round-trip, v3 evidence chain, field audit, tier promotion, snapshot restore, governance bench

## v3.1.0 — Native MCP + Multi-Backend LLM Extractor ✅ Released 2026-04-14

- [x] **Native MCP registration for 8 clients** — per-client writers in `hook_installer.py`:
  - JSON `mcpServers` format: Gemini · Continue · Cline · Roo · Cursor
  - JSON `context_servers` (Zed) · JSON `mcp_config.json` (Windsurf)
  - TOML `[mcp_servers.mind-mem]` (Codex) with sub-table-aware regex that removes stale entries on re-install
  - `install_mcp_config(agent, workspace)` public API
  - `install_all()` emits BOTH hook (visibility) + MCP (tool surface) phases by default; opt-out via `--no-mcp`
- [x] **Multi-backend LLM extractor** — `llm_extractor.py` extended with `vllm`, `openai-compatible`, and `transformers` alongside existing `ollama` and `llama-cpp`:
  - `_query_openai_compatible(prompt, model, base_url)` — vLLM / LM Studio / llama-server / TGI / OpenAI
  - `_query_transformers(prompt, model)` — in-process HF fallback with model cache
  - Env-driven URL overrides: `MIND_MEM_VLLM_URL`, `MIND_MEM_LLM_BASE_URL`, `MIND_MEM_LLM_API_KEY`
  - `auto` mode dispatches ollama → vllm → openai-compat → llama-cpp → transformers
- [x] **mind-mem:4b model via Ollama** — fully trained on STARGA-curated MIND-Mem corpus; Q4_K_M @ 2.6 GB; default `extraction.model`; empirical on RTX 3080: 104 tok/s generation, 1585 tok/s prefill
- [x] **`mind-mem.json` defaults** — `extraction.model` updated from `mind-mem:7b` → `mind-mem:4b`, `backend` from `auto` → `ollama` (explicit)
- [x] **Docs alignment** — 11 audit issues fixed: tool count 54 → 57 in nine locations, "Mind-Mem:7B" → "mind-mem:4b" heading, new §Extraction (LLM Backend) in `docs/configuration.md`, `--no-mcp` flag documented, `install_mcp_config()` public API documented, env vars section updated

## v3.1.1 — Claude Code Hook-Install Fix ✅ Released 2026-04-15

Patch release. Two bugs in `mm install claude-code` that silently
produced hook entries Claude Code rejected at runtime.

- [x] **`hook_installer._merge_claude_hooks`** writes the required
  nested shape `{"matcher": "", "hooks": [{"type": "command",
  "command": "..."}]}` instead of the bare `{"command": "..."}`
  shape. Auto-detects and migrates pre-3.1.1 legacy flat entries
  in-place on re-install — operators who ran earlier versions get
  upgraded without duplicates.
- [x] **`SessionStart` hook** command changed from
  `mm inject --agent claude-code --workspace <X>` (silently failed —
  `mm inject` requires a positional query the hook cannot supply) to
  `mm status`. `mm inject-on-start` is planned as a future
  hook-native subcommand.
- [x] **`Stop` hook** command changed from `mm vault status` (not a
  shipped subcommand — `mm vault` only has `{scan, write}`) to
  `mm status`.

## v3.1.2 — Docs + Metadata Alignment ✅ Released 2026-04-18

No code changes. Publishes a clean v3.1.x representation to users who
read the repo, the PyPI page, or the skill files.

- [x] **README badges** — corrected `tests-3444` → `tests-3610` and
  `MCP_tools-54` → `MCP_tools-57` (verified via
  `pytest --collect-only` and `@mcp.tool` decorator count). Removed
  stale "release local (no Actions)" badge — GitHub Actions is
  re-enabled on the repository.
- [x] **CLAUDE.md refresh** — v1.9.1 → v3.1.2 header. Architecture
  section now reflects current subsystems: at-rest encryption, tier
  decay, governance alerting, audit-integrity patterns,
  `mind-mem-4b` local model, native-MCP integration for 16 clients.
- [x] **docs/roadmap.md rewrite** — v3.1.1 is "current" instead of
  the stale v2.0.0b1 line; shipped vs upcoming separated cleanly.
- [x] **docs/benchmarks.md clarification** — LoCoMo snapshot
  predates v3.x and remains representative; refreshed benchmark
  artifact planned for next release.
- [x] **docs/client-integrations.md** — documents the v3.1.1 hook
  fix and the pre-3.1.1 auto-migration on re-install.
- [x] **Skill file** — test count 2180 → 3610 and MCP tool
  inventory 19 → 57 in
  `.agents/skills/mind-mem-development/SKILL.md`.
- [x] **Release pipeline** — Actions re-enabled, OIDC trusted
  publishing working end-to-end via `.github/workflows/release.yml`
  on tag push `v*` (first fully automated release since account-wide
  Actions disabling).

## v3.2.0 — Production Deployment (2–4 weeks)

Close the production-readiness gap. Everything local-first but horizontal-ready. No changes to the retrieval pipeline; all new work is adapters + gateway.

- [ ] **Postgres storage adapter** — `src/mind_mem/storage/postgres_adapter.py` implementing `BlockStore` protocol; reuse the existing `block_store.py` interface so the retrieval engine sees the same API
- [ ] **Storage factory** — `src/mind_mem/storage/__init__.py` selects adapter from `mind-mem.json` `storage.adapter` key (`"sqlite"` | `"postgres"`); `connection_manager.py` accepts adapter type + pool size
- [ ] **MCP tool-surface reduction (57 → ~20)** — consolidate
  recall-family tools into a single `recall` with explicit modes
  (`hybrid`, `vector`, `bm25`, `similar`, `rerank`); fold
  `propose_update` + `approve_apply` + `rollback_proposal` into one
  `staged_change` flow with a `phase` argument; move the remaining
  tools behind a `*/advanced` namespace gated by a config flag. Goal:
  agent context windows stay tight and tool-selection reliability
  improves without losing capability. Tracked in issue [#501].
- [ ] **REST API layer** — `src/mind_mem/api/rest.py` (FastAPI); endpoints mirror the reduced MCP tool set (with advanced tools behind a flag); OIDC/JWT auth in `src/mind_mem/api/auth.py`; `mm serve` CLI command to launch it
- [ ] **JS/TS SDK + Go SDK** — `sdk/js/` (fetch wrapper) and `sdk/go/` (standard-library client); both match the Python SDK surface
- [ ] **Dockerfile + docker-compose** — `deploy/docker/Dockerfile`, `deploy/docker-compose.yml` with MIND-Mem + pgvector + Ollama; one-command `make up`
- [ ] **One-command installer** — `curl -sSL install.mind-mem.sh | bash`
- [ ] **Full OIDC / SSO auth** — `src/mind_mem/api/auth.py`; Okta / Auth0 / Google Workspace / Azure AD; token refresh, scope mapping to `namespaces.py` ACL roles
- [ ] **Per-agent access control** — extend `namespaces.py` with per-agent API keys rotated via the REST admin endpoints; audit every read/write with agent-id attribution in `audit_chain`
- [ ] **OpenTelemetry traces + SLO dashboards** — wrap `observability.py` with OTel spans on `recall`, `propose_update`, `scan`; Prometheus exporter on configurable port; shipped Grafana dashboard JSON in `deploy/grafana/` (p50/p95/p99 recall latency, proposal-apply lag, contradiction rate, chain-verify success rate)
- [ ] **Distributed query cache** — Redis adapter for `recall.py` results keyed by `(query_hash, namespace, limit)`; TTL-gated, invalidated on `propose_update` / `apply`; falls back to in-process LRU when Redis not configured
- [ ] **Postgres read replicas** — `storage.replicas: ["replica-1.db", "replica-2.db"]`; read-heavy MCP tools (`recall`, `find_similar`, `hybrid_search`, `prefetch`) route to replicas; writes always hit primary
- [ ] **Hot/cold tier wire-up** — `tier_manager.py` is already scaffolded; connect `TierPolicy` to the recall path so WORKING/ARCHIVAL/COLD tiers affect retrieval latency
- [ ] **CLI debug visualization** — `mm inspect <block_id>` (full block + provenance tree), `mm explain <query>` (retrieval trace: BM25 score → vector score → RRF fusion → rerank), `mm trace --live` (stream last N MCP calls with OTel span data)
- [ ] **Config schema additions** — `storage.{adapter,url,pool_size,replicas}`, `api.{rest,grpc,auth}`, `observability.{otel_endpoint,prom_port}`, `cache.{redis_url,ttl_seconds}` sections

### Structural-debt cleanup (from the 2026-04-18 audit)

Four code-health items surfaced by the architectural audit
(`AUDIT_FINDINGS_FOR_CLAUDE.md`). Scoped into v3.2.0 because each is
a prerequisite for the production-deployment work above:

- [ ] **Decompose `src/mind_mem/mcp_server.py`** — the file is ~158 KB
  and houses the bodies of ~all 57 MCP tools. Break it into
  domain-scoped modules (`tools_recall.py`, `tools_governance.py`,
  `tools_snapshot.py`, `tools_graph.py`, etc.) and keep
  `mcp_server.py` as a thin dispatcher. Unblocks the tool-surface
  reduction work above and makes per-tool testing tractable.
- [ ] **Centralize task-status literals into an `enums.py`** — the
  strings `"todo" | "doing" | "blocked" | "done" | "canceled"` are
  duplicated across eight files (`sqlite_index.py`, `validate_py.py`,
  `_recall_core.py`, `intel_scan.py`, `recall_vector.py`,
  `_recall_constants.py`, `capture.py`, and the `validate.sh` bash
  mirror). Extract to a `TaskStatus(str, Enum)`; migrate each call
  site in a single coordinated patch.
- [ ] **Unify validation: deprecate `validate.sh` in favour of
  `validate_py.py`** — the two engines enforce the same SPEC.md
  invariants in parallel; risk is "enforcement drift" where one
  accepts what the other rejects. Audit SPEC.md citations, ensure
  `validate_py.py` covers every bash rule, then mark the shell
  script as deprecated + wrap it as a subprocess shim that defers
  to Python for the actual checks.
- [ ] **Route `apply_engine.py` writes through the `BlockStore`
  protocol** — today the apply engine performs atomic markdown
  writes and file-lock acquisition directly. SPEC.md requires every
  write to go through `BlockStore`. Refactor to call `BlockStore`
  methods so the Postgres adapter (above) works without a second
  engine-level port.
- [ ] **Widen snapshot atomicity scope** — Section 5 of SPEC.md
  currently excludes `maintenance/` and `intelligence/applied/`
  from snapshot coverage, which means a multi-stage apply that
  fails partway through can leave untracked residue in those two
  trees. Decide whether to extend the snapshot invariant to cover
  both, or to document the exclusion as a known boundary with a
  sweep pass that reconciles them on the next `mm scan`.

**Estimated:** ~800 lines storage adapter + ~600 lines REST + ~400 lines JS SDK + ~1200 lines structural-debt refactor + deploy artifacts. New optional extras: `mind-mem[postgres]`, `mind-mem[api]`, `mind-mem[otel]`.

### Security hardening pass (from 2026-04-28 audits)

Two parallel audits ran on v3.1.8 — `threat-modeler` (STRIDE on
governance + apply engine + local backends) and `api-security`
(MCP wire surface + REST layer + crypto). Reports archived at
`security/threat-model-2026-04-28.md` and
`security/api-security-2026-04-28.md`.

**Goal:** tight security defaults for the localhost threat model
without making MIND-Mem painful to run. Every item below has its
UX cost noted; "don't-bother" items are explicit so future-us
doesn't accidentally implement them.

#### Must fix in v3.2.0 (zero-or-low UX cost, high impact)

- [ ] **N-01 / T-002: Default-on ACL gate.** `MIND_MEM_ADMIN_TOKEN`
  unset currently disables the ACL — admin tools (`delete_memory_item`,
  `export_memory`, `decrypt_file`, `rollback_proposal`,
  `approve_apply`) are callable from any stdio client with zero
  elevation. Flip to default-on; introduce
  `MIND_MEM_ACL_DISABLED=true` as the explicit opt-out.
  `mcp/infra/observability.py:83-86`. **UX cost: none** — existing
  `MIND_MEM_SCOPE=user` already keeps user-tools open.
- [ ] **T-006: Bound `vault_scan` / `vault_sync` filesystem walks.**
  Empty `MIND_MEM_VAULT_ALLOWLIST` lets a caller pass
  `vault_root=/etc` and exfil arbitrary host markdown. Reject the
  call when allowlist is empty *or* `vault_root` resolves outside
  every allowlisted prefix (use `os.path.realpath` on both sides
  for symlink-safety, per gap #3 in the threat-model honest-gaps
  list). **UX cost: low** — operator must set the allowlist once.
- [ ] **N-02: REST `rollback_proposal` drops `reason`.** Endpoint
  silently fails (returns HTTP 200 wrapping an app-level error).
  Add `reason: str = Field(..., min_length=8, max_length=2000)` to
  `RollbackProposalRequest`; pass it through. `api/rest.py:629-633`
  + `RollbackProposalRequest` model line 382. **UX cost: low** —
  REST callers must pass `reason` (already required by the MCP
  tool, so this aligns parity).
- [ ] **N-04: `staged_change` rollback dispatcher silently drops
  `reason`.** Same root cause as N-02 on the dispatcher path; agent
  callers using `staged_change(phase="rollback")` get a misleading
  success-shaped error. Forward `rationale` → `reason`.
  `mcp/tools/public.py:192`. **UX cost: none.**
- [ ] **T-003: `propose_update` input bounds bypass.** `governance.py`
  caps `statement` at 500 chars but `rationale` and `tags` are
  written verbatim into `SIGNALS.md`. Apply
  `_sanitize_reason_for_markdown` (already in `apply_engine.py:1411`)
  + bounded length at `propose_update` entry, not just at
  status-update/rollback. **UX cost: none** — sanitizer is invisible
  to legitimate callers.
- [ ] **N-03: Rate limiter collapses all stdio clients into
  `"default"` bucket.** All 57 tools share one 120/min window in
  stdio. Fall back to `f"pid-{os.getpid()}"` when no access token
  exists. `mcp/infra/rate_limit.py:95-103`. **UX cost: none.**

#### Should fix in v3.2.0 (low UX cost, real surface area)

- [ ] **N-05: `/v1/health` leaks workspace absolute path
  unauthenticated.** Strip `workspace` from the unauth response, or
  gate the detailed view behind `_require_auth`.
  `api/rest.py:515-531`. **UX cost: low.**
- [ ] **N-06: `/v1/metrics` Prometheus exposition unauthenticated.**
  Add `dependencies=[Depends(_require_auth)]` *or* gate on
  `request.client.host == "127.0.0.1"`. `api/rest.py:533-551`.
  **UX cost: none for localhost; low for network deploys (operator
  configures Prom scrape token).**
- [ ] **N-07: `/v1/auth/oidc/callback` leaks `scopes` + `agent_id`
  to any bearer.** Omit `scopes` from response body; validate
  `OIDC_ISSUER` against an operator allowlist at startup to block
  JWKS-fetch SSRF. `api/rest.py:704-715`. **UX cost: low.**
- [ ] **T-004: Webhook/Slack alerting payload exfil.** No URL
  allowlist; raw payload (workspace path + block excerpts) POSTs
  to operator-supplied URLs. Add an env-locked URL allowlist; scrub
  payload (strip absolute paths, replace block content with
  `block_id` + content hash). `alerting.py:280-303`. **UX cost: low**
   — operator declares allowed alert hosts once.
- [ ] **T-005: `--token` CLI arg leaks into `/proc/<pid>/cmdline`.**
  Reject `--token` hard; require env var (`MIND_MEM_TOKEN`) or
  stdin. **UX cost: low** — easy migration path.
- [ ] **T-001 (partial): Content-provenance tags on block writes.**
  Add `source: agent | user | external` to the block frontmatter
  written by `propose_update` / `staged_change`; recall surface
  marks agent-origin content as `untrusted` so downstream agents
  can ignore embedded instructions. **UX cost: none for users;
  small system-prompt update for agents.** Closes the
  cross-session prompt-injection vector by giving the agent a
  reliable tag to filter on.

#### Nice-to-have in v3.2.x (low priority, low UX cost)

- [ ] **N-08: `decrypt_file` audit trail.** Append
  `{path, agent_id, timestamp}` to a new `decrypted_files.jsonl`
  (mirroring `deleted_blocks.jsonl`). Forensic coverage for the
  most-sensitive admin tool. **UX cost: low.**
- [ ] **N-10: FTS5 token sanitiser silently drops Unicode.**
  Allowlist regex `^[a-zA-Z0-9_\-\.]+$` strips Chinese / Japanese /
  Arabic / emoji queries to empty. Switch to `^[\w\-\.]+$` with
  `re.UNICODE`; FTS5 quoting still prevents injection.
  `sqlite_index.py:70-72, 858-861`. **UX cost: low** —
  improvement, not regression.
- [ ] **N-11: `export_memory` size cap.** Already capped at 10000;
  validate caller-supplied `max_blocks` against config limit.
  Mostly mitigated by N-01. **UX cost: none.**
- [ ] **N-12: REST rate-limit bucket key uses last-16-chars of
  token.** Theoretically permits collision attacks; switch to
  `sha256(token.encode()).hexdigest()[:16]`. `api/rest.py:122-127`.
  **UX cost: none.** False-positive note: not exploitable in
  practice (operator controls all tokens).
- [ ] **N-13: OpenAPI docs unauthenticated.** Set
  `docs_url=None, redoc_url=None` when `MIND_MEM_TOKEN` is
  configured *and* host ≠ 127.0.0.1; keep open on localhost for
  developer UX. `api/rest.py:464-471`. **UX cost: low.**
- [ ] **T-007: OS-level append-only audit log.** `chattr +a` on
  Linux, `chflags uappnd` on macOS, applied to `audit.log`. Hash
  chain detects tampering today; this prevents it. **UX cost: low**
   — operator runs one privileged command at install.
- [ ] **T-009: Threat-model `online_trainer.py` separately.** Was
  not reviewed in this pass; agent feedback feeds local Ollama
  fine-tune so poisoned proposals could shape the local model.
  Run a focused threat-modeler dispatch before any external
  training-data ingest lands. **UX cost: none — this is review
  work, not code.**

#### Defer to v3.3.x / quarter (real cost, real benefit only at
multi-tenant scale)

- [ ] **T-008: SQLCipher coverage for FTS5 + sqlite-vec indices.**
  Today only Markdown is wrapped. For hosted/multi-tenant deploys
  this is a real gap; for localhost it's documented in code.
  Decision deferred until we have a multi-tenant deploy target.
- [ ] **WORM audit chain.** Beyond append-only flag — separate
  storage class, only relevant for compliance customers.
- [ ] **gRPC surface audit.** `src/mind_mem/api/grpc_server.py`
  parallels REST; same auth/rate-limit hygiene needs to be applied
  once REST changes settle.

#### Don't bother (would hurt UX more than they help)

These came up in audit and were explicitly rejected. Captured here
so a future review pass doesn't accidentally re-litigate them.

- **CSRF tokens on REST.** Bearer-token auth + browser same-origin
  policy already block cross-origin POST.
- **CSP / HSTS headers.** MIND-Mem REST is agent-facing, not
  browser-facing.
- **Treating `MIND_MEM_TOKEN` as a JWT.** It's an opaque static
  bearer; adding expiry would require a signing ceremony with no
  benefit on localhost.
- **mTLS on stdio MCP transport.** Stdio is in-process; TLS at the
  stdio layer is meaningless. If the HTTP transport is ever exposed
  on a real network, terminate TLS at a reverse proxy.
- **Forced rotation of `MIND_MEM_TOKEN`.** Localhost has no
  credential-stealing adversary; rotation = cron jobs and config
  churn for zero gain.
- **Per-tool rate limits (57 separate windows).** A 57-entry map
  lookup on every call complicates the mental model; the single
  window already prevents runaway calls.
- **Audit log for read operations (`recall`, `get_block`).** Would
  produce ~100x the volume of governance events and surface
  potentially-sensitive query strings in plaintext logs. The
  write-path audit chain is sufficient.
- **N-09: Replace HMAC-CTR custom stream cipher with AES-CTR.** The
  current `encryption.py:60-73` HMAC-SHA256-in-counter-mode +
  encrypt-then-MAC construction is cryptographically sound. Migrating
  to `cryptography.hazmat` adds a non-zero external dep for zero
  real-world security gain on localhost. Re-evaluate at v4.0 if
  the encrypted-file format changes anyway.

#### Honest gaps from the 2026-04-28 audits

Surfaced in the reports; tracking here so they're not forgotten:

1. FastMCP transport edge cases (reconnect, multiplexed sessions)
   not verified without a live instance.
2. `apply_engine.py:258` `bash validate.sh ws` — `shell=False` is
   in effect, but only confirmed by inspection.
3. `agent_bridge.VaultBridge.scan()` symlink behaviour past the
   allowlist boundary not traced through. Affects T-006 mitigation
   completeness.
4. `mind-mem.json` write protection — if a poisoned agent can write
   the file, it can re-configure alert webhooks or disable the rate
   limiter. Path not fully traced.
5. gRPC surface (`api/grpc_server.py`) not audited.
6. `python-jose` `alg=none` exposure in `OIDCProvider.verify()`
   confirmed-by-inspection only, not by running the code path.

## v3.2.1 — Hotfix follow-up

Closes the two architectural CRITICALs surfaced by the v3.2.0
self-audit (`docs/review-architecture-v3.2.0.md`). v3.2.0 Postgres
+ REST surfaces are labelled "beta" in the release notes; v3.2.1
promotes both to GA.

- [x] **Apply engine — block-level ops route through BlockStore.**
  Five of seven ``_op_*`` handlers now route through
  ``store.get_by_id`` + ``store.write_block``: ``update_field``,
  ``append_list_item``, ``set_status``, ``append_block``, and
  ``supersede_decision``. ``execute_op`` takes an optional ``store``
  kwarg; when omitted the active store is resolved via the factory.
  ``apply_proposal`` resolves the store once at the top of the op
  loop so every op in a proposal sees the same backend. Backward-
  compatible with every existing caller.
- [ ] **Apply engine — text-range ops** — the two remaining
  handlers (``insert_after_block``, ``replace_range``) still speak
  raw ``open()`` because they manipulate text ranges that don't
  have a clean block-dict representation. No v3.2.0 caller
  generates these ops in practice (they're exercised only by hand-
  written proposals in tests). Deferred to v3.2.2 — either promote
  them to block-level ops (``insert_after_block`` becomes
  ``write_block`` with an ordering hint) or deprecate.
- [ ] **Audit attribution through FastAPI sync deps** — the
  ``current_agent_id`` ContextVar is set inside ``_require_auth``
  (a sync FastAPI dependency), which runs in an anyio threadpool
  worker. ContextVar writes in worker threads don't propagate back
  to the calling request context, so downstream MCP tool functions
  read ``'anonymous'`` even on authenticated requests. Fix by
  stashing ``agent_id`` on ``request.state`` (same pattern as
  ``oidc_scopes`` in v3.2.1) and reading it from a dependency
  attached to each handler. ~0.5 day.
- [x] **REST request-scoping** — swapped env-var mutation for a
  per-request ``ContextVar`` override in
  ``mind_mem.mcp.infra.workspace`` + a FastAPI HTTP middleware.
  Task-local under asyncio, thread-local through Starlette's
  thread pool. Standalone MCP server still reads the env var.
- [x] **OIDC wired into `_require_admin`** — JWT `scope` / `scopes`
  / `roles` claims now drive the admin gate;
  ``MIND_MEM_OIDC_ADMIN_SCOPES`` env configures which scope names
  count. Admin gate is enforced when OIDC is configured even
  without static tokens. Invalid JWTs reject with 401 instead of
  falling through to the permissive static-token path.
- [ ] **`PostgresBlockStore.snapshot(snap_id=…)`** — current
  signature requires a filesystem path for the MANIFEST.json
  write, breaking cross-host Postgres snapshots. Accept a plain
  `snap_id: str` and make the on-disk manifest optional. ~0.5
  day. (Deferred to v3.2.2 — cross-backend snapshot API design
  needs alignment with Markdown backend.)
- [x] **Wire `cached_recall` into `_recall_impl`** — done in
  v3.2.0 commit `7c54844`.
- [x] **Two config keys documented in `docs/configuration.md`** —
  `cache.redis_url` and `retrieval.tier_boost` appear in the
  v3.2.0 docs; verified as part of the v3.2.1 release checklist.
- [ ] **Dependency CVE bumps** — no ``authlib`` or ``aiohttp`` in
  MIND-Mem's direct or transitive deps as of v3.2.1 (``pip-audit``
  verified). Kept as tracking item in case a future ``fastmcp``
  release reintroduces either.

v3.2.1 CI-plumbing fixes (shipped):

- [x] Ruff format drift (25 files)
- [x] Windows path-separator assertion in
  `test_apply_engine_backend_routing.py`
- [x] SBOM `pkg_resources` crash — pin `cyclonedx-bom>=5` in
  `release.yml` + `security.yml`
- [x] Dead action SHAs — bump `trivy-action` to v0.35.0, correct
  `gitleaks-action` v2.3.9 SHA
- [x] Gitleaks glob-vs-regex in `.gitleaks.toml` (`*.pyc` →
  `.*\.pyc$`)

**Estimated:** ~1200 lines refactor, ~400 LOC tests, ~200 LOC
docs.

## v3.3.0 — Reasoning-Grade Retrieval (1–2 months)

Close the retrieval-quality gap and widen the governance moat. All additive — no breaking changes to existing recall contracts.

### LoCoMo score improvements — 4-tier roadmap

Baseline: v1.1.0 overall mean 70.54 (Mistral-Large answerer + judge). LoCoMo category breakdown shows where points bleed:

| Category | Baseline | N | Biggest intervention |
|---|---|---|---|
| multi-hop | 51.10 | 321 | graph traversal + decomposition |
| temporal | 65.89 | 96 | half-life decay in scorer |
| single-hop | 68.68 | 282 | query reformulation + RRF |
| open-domain | 70.27 | 841 | conversation-boundary preservation |
| adversarial | 87.22 | 446 | at ceiling |

Projected v3.3.0 overall with Tier-1+2 shipped: **74-76 (Mistral-Mistral)** / **82-85 (Opus answerer + Mistral judge)**.

#### Tier 1 — highest leverage, must ship

- [x] **Query decomposition** — shipped in `src/mind_mem/query_planner.py` (commit `0c69561`). NLP pattern-split default + optional LLM decomposer via `retrieval.query_decomposition.provider`. Auto-enables on multi-hop query type; wired into `HybridBackend.search` ahead of RRF fusion. 20 regression tests. **Target: multi-hop 51 → 65 (+4.5 overall).**
- [x] **Multi-hop graph traversal** — shipped in `src/mind_mem/graph_recall.py` (commit `2c55ec3`). BFS over `build_xref_graph`, decayed scores, N-hop cap, auto-enables on multi-hop queries. Wired post-CE-rerank in `_maybe_graph_expand`. 16 regression tests. **Target: multi-hop +10 further (+3.2 overall).**
- [x] **Temporal re-weighting in the hot path** — shipped in `_recall_scoring.temporal_decay_score` (commit `a63d572`). Exponential half-life decay; configurable via `retrieval.temporal_half_life_days` (default 90). 13 regression tests. **Target: temporal 66 → 78 (+0.6 overall).**

#### Tier 2 — not currently roadmapped, high ROI

- [x] **Query reformulation + RRF** — shipped (commit `4da44d0`). Existing `query_expansion` infrastructure plus auto-enable on multi-hop/temporal query types (`query_expansion.auto_enable: true` default). NLP + LLM expanders both available. 5 regression tests. **Target: single-hop 68 → 76, open-domain 70 → 76 (+2.7 overall).**
- [ ] **Conversation-boundary preservation** — deferred: LoCoMo-specific (dialog session IDs) and needs ingestion-layer changes to preserve `session_id` / `dia_id` metadata on blocks. Tracked for v3.3.1. **Target: multi-hop + open-domain +3 each (+1.3 overall).**
- [x] **Default cross-encoder rerank on ambiguous queries** — shipped (commit `a2eeff6`). `_maybe_cross_encoder_rerank` auto-enables for multi-hop/temporal queries via `cross_encoder.auto_enable: true`. Applies on both BM25-only and hybrid paths. 5 regression tests. **Target: +2 overall across all categories.**

#### Tier 3 — bigger architectural bets

- [x] **Answerer co-design: structured evidence bundle** — shipped in `src/mind_mem/evidence_bundle.py` (commit `96a3ae6`). `build_bundle(query, results)` returns typed `{facts, relations, timeline, entities, source_blocks}`. Rule-based extraction — Statement/Fact/Claim/Summary → facts; Supersedes/Dependencies/Relates_to/Cites → relations; ISO-dated blocks → timeline; PER/PRJ/TOOL/INC prefixes → entities. Confidence blends Status × Tier. Gated on explicit caller opt-in so existing callers unchanged. 19 regression tests. **Target: open-domain + multi-hop +4 each (+1.9 overall).**
- [x] **Entity-graph prefetch** — shipped in `src/mind_mem/entity_prefetch.py` (commit `e7ba6ae` + security-hardened in `b31e862`). Regex extracts entity candidates (capitalised names, block-IDs, acronyms); matches against Name/Aliases/Statement/Type of entity-prefix blocks (PER/PRJ/TOOL/INC); walks 1-hop via graph_expand. Bounded at 500 files / 2MB / symlink-escape-refused. 18 regression tests. **Target: +2 overall.**

#### Tier 4 — infrastructure (shipped in v3.3.0)

- [x] **Reranker ensemble** — shipped in `src/mind_mem/rerank_ensemble.py` (commit `6053847`). `EnsembleReranker` composes N rerankers and fuses via Borda count; factory wires cross_encoder / bge / llm per config; each member is fail-open so one failure never blocks recall. SSRF-guarded base_url for the LLM member. 12 regression tests. **Target: +1-2 overall.** Heavy BGE deps behind `mind-mem[cross-encoder]`.
- [x] **Per-tier learned weights** — shipped in `src/mind_mem/tier_recall.resolve_tier_weights` (commit `954d473` + tests). Operators override the baseline 0.7/1.0/1.5/2.0 multipliers via `retrieval.tier_boost_weights` (name or integer keys, case-insensitive). Invalid values fall back to baseline. 9 regression tests. Training script `benchmarks/tier_weight_search.py` follow-up. **Target: +1 overall.**

### Other v3.3.0 items (not primarily LoCoMo-score-driven)

- [x] **Expanded typed-edge taxonomy** (LeanKG-inspired) — shipped via evidence_bundle relation extraction (`96a3ae6`). The typed relations ``cites`` / ``derives_from`` / ``depends_on`` / ``tested_by`` / ``supersedes`` / ``superseded_by`` / ``relates_to`` are extracted into the Relation records the EvidenceBundle produces. The graph traversal infrastructure itself is in `graph_recall.py` (`2c55ec3`). Impact-analysis queries now answerable via graph + bundle composition.
- [x] **Probabilistic truth score** — shipped in `src/mind_mem/truth_score.py` (commit `e98c144`). Bayesian posterior ``prior × age_decay − contradiction_mass + access_bonus``, clamped [0.01, 0.99]. Exposed via ``annotate_results(results, contradiction_graph=…)``; caller surfaces as ``block.truth_score``. Feeds into EvidenceBundle confidence. 22 tests.
- [x] **Streaming ingest + back-pressure queue** — shipped in `src/mind_mem/streaming.py` (commit `9956b7a`). Bounded mpsc deque with drop-oldest policy + per-client token bucket. ``build_queue_from_config`` opt-in via ``streaming.enabled``. Thread-safe multi-producer. 14 tests including a 4-thread concurrency test.
- [x] **Consensus voting** — shipped in `src/mind_mem/consensus_vote.py` (commit `f644096`). ``reach_consensus(votes, quorum_threshold, min_votes)`` returns a typed ``ConsensusDecision(winner, margin, confidence, reason, vote_counts)``; trust weights pulled from ``Vote.trust_weight`` or ``namespaces.<id>.trust_weight``; 0-weight excludes. 14 tests.
- [ ] **Graph + timeline visualization** — `web/` Next.js app; D3 / react-flow graph view (nodes = blocks, edges = relationships), timeline view, drift heatmap; reads from REST API shipped in v3.2.0. v3.2.0 already emits `[[wikilinks]]` on `vault_sync` so an Obsidian-mounted vault gets a graph view for free; this web UI is the non-Obsidian alternative. **Frontend work — separate from the retrieval shipments above.**
- [ ] **mind-mem-4b v2 retrain** — training recipe + data generators shipped (`docs/mind-mem-4b-v2-training-recipe.md`, `benchmarks/generate_dispatcher_examples.py`, `benchmarks/generate_retrieval_examples.py`). **Runpod H200 kickoff pending operator approval** — ~$55, 8-12hr, targets full retrain of mind-mem:4b on v3.2.x dispatchers + v3.3.0 retrieval shapes + LoCoMo replay.

**Estimated (v3.3.0):** ~2400 lines retrieval (Tier 1+2+3) + ~2000 lines web UI + ~2 GPU-days retrain. New optional extras: `mind-mem[reasoning]`, `mind-mem[streaming]`, `mind-mem[rerank-ensemble]` (Tier 4).

**v3.3.0 retrieval shipped (9 of 10 tier items, 2026-04-20):**

| Tier | Item | Status | Commit | New tests |
|---|---|---|---|---|
| T1 #1 | Query decomposition | ✓ | `0c69561` | 20 |
| T1 #2 | Multi-hop graph traversal | ✓ | `2c55ec3` | 16 |
| T1 #3 | Temporal half-life decay | ✓ | `a63d572` | 13 |
| T2 #4 | Query reformulation + RRF auto-enable | ✓ | `4da44d0` | 5 |
| T2 #5 | Conversation-boundary preservation | deferred v3.3.1 | — | — |
| T2 #6 | Cross-encoder auto-enable | ✓ | `a2eeff6` | 5 |
| T3 #7 | Structured evidence bundle | ✓ | `96a3ae6` | 19 |
| T3 #8 | Entity-graph prefetch | ✓ | `e7ba6ae` | 18 |
| T4 #9 | Reranker ensemble (Borda count) | ✓ | `6053847` | 12 |
| T4 #10 | Per-tier learned weights | ✓ | `954d473` | 9 |

Plus 12+ audit defects closed (SSRF guard, symlink escape, limit
violation, BFS O(N), corpus double-load, thread-safety race, float
underflow, bare exceptions) in commits `b31e862` and `954d473`.

Total: +117 regression tests, 3758 passing as of 2026-04-20.

## v3.7.0 — External-Audit Response ✅ Released 2026-05-01

Closes the nine findings from the 2026-05-01 external audit
(`/home/n/.openclaw/workspace/projects/mind-mem-audit-2026-05-01/AUDIT.md`).
Single `BREAKING` change: HTTP/REST authentication now fails CLOSED
when no token is configured.

### High-priority audit fixes (4)

- **H1: install path.** `install.sh` now installs the package via
  pipx (preferred, isolated venv) or `pip --user` (fallback) and
  wires every MCP client to the `mind-mem-mcp` console script
  instead of `python3 <repo>/mcp_server.py`. New CI matrix smoke-
  tests both flows on a clean runner. PEP 668 `EXTERNALLY-MANAGED`
  marker on Debian / Ubuntu retried with `--break-system-packages`
  so isolated `--user` installs still succeed.
- **H2: dependency drift.** `fastmcp` lives only in the `[mcp]`
  extra (range-pinned `>=3.2.0`). `requirements-optional.txt` is
  scoped to the embedding/reranking stack only; CI covers
  `.[mcp]`, `.[api]`, `.[embeddings]`, `.[all]`, hashed-pin
  re-download, and a clean docker build per release.
- **H3: cross-platform rollback.** Two bugs in the v3.6.9 path-
  injection sweep made `BlockStore.restore` walk realpath-resolved
  inventories on macOS (where `/var → /private/var`) and Windows
  (short-name expansion); rollback then computed `relpath` against
  the un-resolved workspace and skipped every file. Both
  `_build_cleanup_inventory` and `_cleanup_orphans_from_manifest`
  now walk the un-resolved `os.path.join(ws, root)` after a
  `_safe_child_path` validation; new symlink-based regression test
  reproduces the macOS divergence on Linux runners.
- **H4: HTTP/REST fail-closed.** ⚠ **BREAKING.** The shared
  `verify_token` helper and the REST `_verify_bearer` dependency
  no longer return `True` when no auth is configured. New escape
  hatch: `MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1` +
  `--allow-unauthenticated-localhost` flag enable unauthenticated
  access only when bound to `127.0.0.1` / `::1` / `localhost`.
  The MCP HTTP transport now refuses to start without one of
  these. 15 new tests in `tests/test_http_auth_fail_closed.py`.

### Medium-priority audit fixes (5)

- **M5: CI strictness.** `typecheck` is now blocking (was
  `continue-on-error: true`); Python 3.14 is required across
  `ubuntu-latest` / `macos-latest` / `windows-latest`; coverage
  gate raised 60 → 70 (current local ~73%); new
  `extras-install`, `pinned-requirements`, `docker-build`, and
  `compose-config` matrix jobs.
- **M6: compose healthcheck.** Postgres healthcheck now reads
  `$$POSTGRES_USER` / `$$POSTGRES_DB` at probe time so operator
  overrides don't render the container unhealthy. New
  `compose-config (defaults | overrides)` matrix job locks the
  rendered command shape against regression.
- **M7: `recall(mode="vector")`.** Removed from public surface —
  the dispatcher silently rewrote it to `auto`, so callers who
  asked for vector-only retrieval got hybrid results. Now
  returns a dedicated v3.7.0-removal error pointing at
  `hybrid` for today's hybrid path. `valid_modes` no longer
  advertises `vector`.
- **M8: `sqlite_index._file_hash`.** Was first-64KB + size; missed
  in-place edits past 64KB when mtime+size stayed identical.
  Now full SHA-256 streamed in 1 MiB chunks, gated by a
  cheap `(size, mtime_ns)` pre-filter so steady-state reindex
  cost is unchanged. `file_state` schema gains `mtime_ns`
  (idempotent ALTER TABLE).
- **M9: phantom `libmindmem.so` in release.** The release workflow
  listed `libmindmem.so` in its files glob but no preceding step
  built or downloaded it; the GH release silently omitted the
  artifact. Removed from `files:` and added
  `fail_on_unmatched_files: true` so future drift gates the
  release. Pure-Python fallback in `mind_ffi` returns identical
  results within f32 epsilon, so users lose nothing.

### Phase 3 — docs alignment

- README, `docs/configuration.md`, `docs/docker-deployment.md`,
  and `docs/roadmap.md` updated to describe the v3.7.0
  fail-closed contract; `mcp_server.py` shim error message
  rewired from `pip install fastmcp==2.14.5` (stale) to
  `pip install "mind-mem[mcp]"` so the version line stays in
  one place.
- GitHub repo About refreshed: leads with v3.7.0 + fail-closed
  + cross-platform rollback callouts.

### Migration

Set `MIND_MEM_TOKEN=<random-string>` (or `MIND_MEM_ADMIN_TOKEN`)
before starting the HTTP transport. Local dev / CI:
`MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1 mind-mem-mcp
--transport http --host 127.0.0.1 --allow-unauthenticated-localhost`.
Docker compose (`deploy/docker-compose.yml`) already enforces both
tokens via `${VAR:?must be set}` so containerised deployments
require zero changes.

## v3.8.0 — Model Safety Audit (complete)

Hardening thread motivated by incidents in the broader AI ecosystem:
malicious HuggingFace model drops that ship remote-code-execution
payloads via `trust_remote_code` / `auto_map` / pickle imports.

> **Scope note (2026-05-02):** Earlier drafts of this section also
> bundled a "Social Ingestion" thread (per-platform fetchers for
> HN / Reddit / X / LinkedIn / Instagram / TikTok / Moltbook /
> Bluesky / Mastodon / Farcaster). That work has been **moved out
> of MIND-Mem to naestro-bot** — fetching social content is an
> agent-layer concern, not a memory-layer concern. MIND-Mem stays
> the substrate (blocks + recall + governance); naestro-bot
> extensions (alongside `extensions/discord/`, `extensions/slack/`,
> etc.) own per-platform fetching and write into MIND-Mem via the
> existing MCP surface. This preserves MIND-Mem's zero-dependency
> posture and avoids inheriting 8 platforms' worth of auth, rate-
> limit, anti-bot, and ToS maintenance liability.

### Model Safety Audit

- [x] **`mm audit-model <path>`** — shipped in v3.8.0 (2026-05-02). Six
  static checks: remote-code hooks (`auto_map` / `trust_remote_code`),
  bundled `.py` refuser, weight format (`.safetensors` / `.gguf` only,
  legacy `.bin` / `.pt` / `.ckpt` flagged), pickle raw-byte opcode walk
  for `os` / `subprocess` / `socket` / `ctypes` / `importlib` / `eval` /
  `exec` / `compile` / `__import__` references, tokenizer-injection
  scanner over `tokenizer.json` / `tokenizer_config.json` /
  `special_tokens_map.json`, and a `safetensors` header validator
  (8-byte LE length, refuses headers > 100 MB, refuses
  `__metadata__.code`). Emits a colour-coded text report or
  `--json`-mode machine output, and an optional SHA-256 manifest
  (`--manifest-out`) compatible with `sha256sum -c`. **31 unit tests**
  in `tests/test_model_audit.py` exercise every public function and
  every check (the actual byte-level pickle scanner runs on real
  `pickle.dumps()` output, not mocks).
- [x] **SHA-256 manifest + Ed25519 signing** — shipped in v3.8.1
  (2026-05-02). `mm sign-model <path>` writes three sidecars next to
  the checkpoint: `MODEL_MANIFEST.txt` (sorted, deterministic,
  `sha256sum -c`-compatible), `MODEL_MANIFEST.txt.sig` (raw 64-byte
  Ed25519 signature, RFC 8032 §5.1.6), and `MODEL_PUBKEY.pub` (raw
  32-byte public key). Two key sources: `--key-file <sk>` (raw 32-byte
  secret) or `--generate-key <prefix>` (writes `<prefix>.sk` mode 0600
  + `<prefix>.pub`). `mm verify-model <path>` returns the structured
  error kind (`manifest_mismatch` / `bad_signature` / `missing_file`)
  so callers can distinguish drift from forgery from a missing
  sidecar. `--pubkey <path>` overrides the sidecar for centrally-pinned
  trust roots. **23 unit tests** in `tests/test_model_signing.py`.
- [x] **Provenance allowlist** — shipped in v3.8.2 (2026-05-02).
  ``mind_mem.model_provenance`` declares ten canonical publishers
  (Alibaba Qwen, Meta Llama, Mistral AI, Google Gemma, IBM Granite,
  OpenAI, Anthropic, DeepSeek, Microsoft Phi, TII Falcon).
  ``audit_model`` runs ``check_provenance`` as its seventh check;
  ``mm audit-model --allow-publisher <hf-org-slug>`` (repeatable)
  extends the allowlist with operator-specific orgs. Namespace match
  is case-insensitive on the leading namespace of ``base_model``.
  Pretrain checkpoints (no ``base_model`` field) pass through
  silently; mis-typed or namespace-squat orgs fail with a clear
  evidence list. **25 unit tests** in
  ``tests/test_model_provenance.py``.
- [x] **MCP tool wrapper** — shipped in v3.8.3 (2026-05-02).
  ``mind_mem.mcp.tools.model`` exposes ``audit_model_tool``,
  ``sign_model_tool``, and ``verify_model_tool`` on the existing
  ``mcp`` instance. Identical schemas to the CLI subcommands —
  agents can run the full seven-check audit, Ed25519 manifest
  signing, and detached-signature verification through MCP without
  shelling to ``mm``. Path-escape guards (empty / NUL-byte rejection)
  on every ``path`` argument. Manifest is omitted from
  ``audit_model_tool`` by default so multi-GB checkpoints don't blow
  up the response — caller opts in with ``include_manifest=True``.
  **21 unit tests** in ``tests/test_mcp_tools_model.py``.
- [x] **Load-gate registry + primitives** — shipped in v3.8.4
  (2026-05-02). ``mind_mem.model_gate`` records every audited
  checkpoint in ``~/.mind-mem/model_gate.json`` with deterministic
  manifest_sha256 for drift detection, atomic write-temp + replace
  on every update, and a six-state ``GateDecision``
  (``trusted_fresh`` / ``audited_now`` / ``drift_re_audited`` /
  ``audit_failed`` / ``audit_failed_override`` /
  ``never_audited_override`` / ``path_not_found``). Three CLI
  sub-commands: ``mm gate check`` runs the gate, ``mm gate list``
  prints the ledger, ``mm gate remove`` drops a path. **12 unit
  tests** in ``tests/test_model_gate.py``.
- [x] **MIC/MAP Python toolchain** — shipped in v3.8.5
  (2026-05-02). ``mind_mem.mic_map`` ports the STARGA-native
  ``mic@2`` (text) and ``mic-b`` (binary) wire formats from the
  Rust reference at ``mind/src/ir/compact/v2/`` to Python.
  Faithful spec implementation: sequential-only IDs, no forward
  references, ULEB128 minimum encoding, zigzag for signed
  parameters, first-seen string interning, magic ``MICB`` +
  version byte ``0x02``. Covers all 19 opcodes (with their
  opcode-specific param sections — axis, perm, axes, axis+count)
  and all 13 dtypes. Replaces JSON for IR-graph payloads inside
  MIND-Mem (audit reports stay JSON — those are documents, not
  graphs). **63 unit tests** in ``tests/test_mic_map.py``.
- [x] **Backend wiring** — shipped in v3.8.6 (2026-05-02).
  ``mind_mem.llm_extractor._gate_check_local`` runs before
  ``AutoModel.from_pretrained`` resolves a local directory
  checkpoint. HF hub IDs and single-file binaries (``.gguf`` /
  ``.bin``) bypass — the gate's manifest contract is for HF-style
  directory checkpoints. ``MIND_MEM_SKIP_GATE=1`` opts out
  entirely; ``MIND_MEM_TRUST_WITHOUT_AUDIT=1`` forwards
  ``trust_without_audit`` to ``gate_check`` and records the
  override in ``~/.mind-mem/model_gate.json``. Failed audits raise
  ``RuntimeError`` with both override env-vars named in the
  message — fail-closed by default. **11 unit tests** in
  ``tests/test_llm_extractor_gate.py``.
- [x] **CI hook** — shipped in v3.8.7 (2026-05-02).
  ``mind_mem.audit_pinned`` reads an ``audit_pinned_models`` list
  from ``mind-mem.json`` and runs ``audit_model`` (and optional
  ``verify_model``) against every entry. ``mm audit-pinned`` exits
  ``0`` on a clean run / no-op, ``1`` on any HIGH finding or verify
  failure, ``2`` on config-parse error or missing path with
  ``--fail-on-missing``. ``.github/workflows/audit-pinned.yml`` runs
  the gate on push to main / PR / workflow_dispatch with path
  filtering. **25 unit tests** in ``tests/test_audit_pinned.py``.

  This closes the Model Safety Audit theme of v3.8.0 — every item
  in the Audit pipeline (audit → sign → provenance → MCP → gate →
  backend wiring → CI hook) is shipped.

### MIC/MAP Scale Hardening (added in v3.8.5 plan)

Three slices ahead of any future MIC/MAP network layer. Today MIC/MAP
is a single-shot serialization primitive; before it can carry
production load on a wire we need crash safety on adversarial input,
streaming I/O, and a native accelerator for the hot loops.

- [x] **Fuzz harness + adversarial corpus + benchmarks** — shipped
  in v3.8.8 (2026-05-02). 7 Hypothesis property tests
  (round-trip, crash safety on arbitrary bytes / text), 26
  hand-crafted DoS inputs (varint bombs, length-prefix overflow,
  truncation, magic / version / tag fuzzing, output OOR), 12
  pytest-benchmark tests + 6 throughput floors + 2 memory-ceiling
  bounds. ``hypothesis`` added to ``[test]`` extras. Caught a real
  bug — ``parse_micb`` was leaking ``UnicodeDecodeError`` on
  invalid UTF-8 in the string table; now correctly wrapped as
  ``MicbParseError``. **45 new tests**.
- [x] **Streaming parser** — shipped in v3.8.9 (2026-05-02).
  ``parse_micb_stream(reader)`` yields six event types
  (``StreamHeader`` / ``StreamStringTable`` / ``StreamSymbol`` /
  ``StreamType`` / ``StreamValue`` / ``StreamComplete``) as bytes
  arrive from any ``BinaryIO``. Handles short reads via the new
  ``_read_exact`` helper — sockets and ``BufferedReader``-over-
  slow-source routinely return fewer bytes than requested. Legacy
  ``parse_micb(bytes)`` is now a wrapper that drains the stream
  and assembles the :class:`Graph`. Caller can drop ``StreamValue``
  objects after processing — bounded peak memory ahead of any
  future MIC/MAP network layer. **10 unit tests** in
  ``tests/test_mic_map_stream.py``.
- [x] **Native accelerator** — shipped in v3.8.10 (2026-05-02).
  Cython port of the ULEB128 / SLEB128 / ``read_exact`` hot
  loops at ``src/mind_mem/_mic_map_accel.pyx``. Same Python
  API; ``mic_map.py`` try-imports ``_mic_map_accel`` and falls
  back to the pure-Python codec when the extension isn't built
  (the default ``pip install MIND-Mem`` path). Build is opt-in
  via ``pip install MIND-Mem[accelerated]`` (pulls in Cython
  at build time) — no Cargo toolchain, no PyO3, the wheel
  stays a pure-Python wheel by default. Bench delta on the
  residual block: ``parse_micb`` +16% small / +20% medium /
  +36% large; bigger 5-10× wins deferred to a future v3.9.x
  with proper C-level buffer parsing. **11 unit tests** in
  ``tests/test_mic_map_accel.py`` (TestModuleShape /
  TestEquivalence skip-if-no-accel / TestPurePythonAlwaysWorks).

### Social Ingestion — moved to naestro-bot (2026-05-02)

The platform fetcher set (HN / Reddit / X / LinkedIn / Instagram /
TikTok / Moltbook / Bluesky / Mastodon / Farcaster) and the
URL-to-block ingestion CLI / MCP tool are no longer scoped to
mind-mem. **Tracked in naestro-bot** alongside the existing
`extensions/discord/` / `extensions/slack/` / `extensions/telegram/`
/ `extensions/feishu/` channels — fetching social content is the
same shape of work as bridging a chat platform, and naestro-bot
already owns that surface. Naestro-bot extensions write captured
posts into MIND-Mem through the existing MCP recall / capture
tools, so MIND-Mem's role (substrate for blocks + recall +
governance + contradiction detection) is unchanged.

The split keeps MIND-Mem zero-dependency, avoids inheriting per-
platform auth / anti-bot / ToS maintenance, and preserves the
clean layering: **MIND-Mem stores; agents fetch.**

**Estimated:** ~1500 lines audit (CLI + pickle disassembly + Ed25519 +
load-gate + CI hook) — all shipped in v3.8.1 → v3.8.7. The social
ingestion estimate (~2500 lines fetchers + ~600 lines block
integration) is now a naestro-bot concern.

## v3.11.0 — Quality Gates, Typed Lineage, Recall Explainability ✅ Released 2026-05-08

Deterministic quality validation, typed relationship edges for dependency tracking, and step-by-step recall transparency.

### Added
- [x] **Pattern 2: `validate_block`** — deterministic quality gate evaluates memory blocks for correctness, coherence, and reference integrity. Module: `src/mind_mem/quality_gate.py`. Validates block schema, statement coherence, cross-references. Registered as MCP tool in `src/mind_mem/mcp/tools/quality.py`. **28 tests, 96% coverage**.
- [x] **Pattern 3: `block_lineage` + `add_block_edge`** — typed relationship edges (cites, implements, refines, contradicts, cooccurrence) enable explicit dependency tracking. Blocks form a DAG with direction-aware traversal. Module: `src/mind_mem/block_lineage.py`. MCP tools in `src/mind_mem/mcp/tools/lineage.py`. **27 tests passing**.
- [x] **Pattern 1: `recall(explain=True)` & `hybrid_search(explain=True)`** — augmented recall responses include step-by-step reasoning chains: BM25 scoring breakdown, vector similarity paths, RRF fusion stages, intent routing logic, final ranking rationale. Surfaces retrieval decisions for auditability.

### Changed
- MCP tool count: 81 → 84 (+3 new tools)
- `co_retrieval` column migration (Postgres schema) zero-downtime; SQLite unaffected.

### Testing
- quality_gate module: **28 new tests** at 96% coverage
- block_lineage module: **27 new tests**
- Total test suite: 4000+ tests passing

### Migration
No breaking changes. Existing blocks work unchanged. New tools opt-in via MCP config. Lineage edges optional; backward-compatible if omitted.

## v3.11.1 — B101 hardening + ACL backfill ✅ Released 2026-05-08

GHAS #179, #180: replace runtime `assert` invariants with hard
`if/raise RuntimeError` so the math-consistency invariant in
`_recall_explain.py` and the type-narrowing path in
`quality_gate.py` survive `python -O` (where `assert` is compiled
out). Backfill 7 MCP tools missing from `USER_TOOLS` —
`audit_model_tool`, `sign_model_tool`, `verify_model_tool`,
`compile_truth_walkthrough`, `recall_with_persona`,
`mic_convert_tool`, `mic_inspect_tool` — clearing 40+ pre-existing
red tests. ruff + mypy + Bandit (medium/high) clean.

## v3.12.0 — Local-model GA, hard quality gate, lineage staleness, red-team CI

Target: 2026-05-15. Four additive themes; each is independently
shippable so a slip in one doesn't block the others.

### Theme A — `mind-mem-4b` v3.11.0-fullft GA bundle

The v3.10.2-fullft model is degraded on the v3.11.0 tool surface.
Retrain corpus + eval delta + recipe already committed (`train/`,
`docs/v3.11.0-mind-mem-4b-retrain-plan.md`). Theme A ships the GA
bundle so `mm install-model --version v3.11.0` is end-to-end.

- [ ] H200 RunPod provisioned (~6h, ~$20)
- [ ] Train + GGUF Q4_K_M export
- [ ] 7-category eval — `tool_call ≥95`, `block_schema ≥98`,
  `workflow ≥90`, `v39_new_tools ≥90`, `v39_transform_hash ≥95`,
  `v39_transport_guard ≥95`, **`v311_new_tools ≥90`**,
  **`v311_explain_field ≥95`**
- [ ] HF upload to `star-ga/mind-mem-4b` rev `v3.11.0`
- [ ] `train/Modelfile.v3.11.0` Modelfile bump
- [ ] HF model card refresh + README badge

### Theme B — Quality-gate hard mode (config-gated)

`validate_block` ships v3.11.0 advisory-only. v3.12 turns on
`quality_gate.mode = "strict"` for `propose_update` pre-write —
opt-in via `mind-mem.json`. Wiring already exists; needs config
plumbing + clean error message + metrics counter.

- [ ] `mind-mem.json` reads `quality_gate.mode` ∈
  `{"off", "advisory", "strict"}` (default `"advisory"`)
- [ ] `propose_update` invokes `validate_block` pre-write when
  mode != `"off"`; structured 400 in strict mode
- [ ] Metrics: `quality_gate_rejections_total{rule=...}`
- [ ] `docs/quality-gate.md` operator runbook
- [ ] Config-honor test for all three modes

### Theme C — Block-lineage staleness propagation wiring

v3.11.0 ships `block_lineage` typed edges + bounded BFS. v2.6.0
ships the staleness propagator. v3.12 wires them: when a
`contradicts` edge is added, the dependent subgraph inherits a
propagated `staleness_penalty` that `_explain.staleness_penalty`
then surfaces in recall.

- [ ] `add_block_edge(..., kind="contradicts")` schedules a
  bounded staleness pass (`max_hops=3`, kind-specific decay)
- [ ] `block_staleness(block_id, score, source_id, decayed_at)`
  table; idempotent upsert; SQLite + Postgres parity
- [ ] Recall reranker reads the penalty;
  `_explain.staleness_penalty` returns the live value
- [ ] CLI: `mm lineage flag <block-id> --kind contradicts <target>`
  (bundles `add_block_edge` + propagate)
- [ ] e2e test: contradicting one block visibly demotes its
  dependents in the next recall

### Theme D — Petri behavioral audit promoted to advisory CI

Scaffold shipped in v3.11.0 (`tests/red_team/`,
`docs/red-team-audit.md`). v3.12 promotes the 3 seeds
(`self_exfiltration_memory_trigger`,
`broken_tool_error_handling`, `weird_ood_tool_use`) to a
real, advisory-only CI job that runs on release tags.
Hard-blocking deferred to v3.13 once we have baseline data.

- [ ] `.github/workflows/red-team.yml` runs on tag-push only
  (`v*`), `continue-on-error: true`
- [ ] Skip cleanly when `ANTHROPIC_API_KEY` secret is absent
- [ ] Upload transcripts as artifacts (90-day retention)
- [ ] `--limit 5` per seed + sonnet judge → ~$10-15/run
- [ ] Update `docs/red-team-audit.md` with the CI pointer

### Out of scope for v3.12.0

- Networked mesh / federated recall — v4.0
- Streaming consensus mixer — stays in private naestro-bot
- gRPC transport — v4.0
- Sharded Postgres — v4.0

**Estimated:** ~600 lines training pipeline + ~400 lines quality-gate
config + ~700 lines lineage propagation + ~150 lines CI = ~1850
lines. All additive. Existing 81-tool API stays unchanged; new
behavior is config-gated everywhere.

## v4.0.0 — Network-native memory + knowledge graph + compliance primitives

The v4.0 picture turns mind-mem from a single-host library into a
network-native substrate for AI agents to share governed memory over the
public internet. Three concurrent threads:

- **Cognition** — three-tier memory with surprise-weighted retrieval and a
  Cognitive Mind Kernel API that exposes routing strategies as a
  first-class parameter.
- **Knowledge graph** — multi-page entity/concept blocks with typed
  lineage edges, LLM-driven structured fusion on update, long-context
  retrieval that preserves relational understanding alongside the
  existing chunked top-K mode, and a conversational chat layer.
- **Network connectivity** — TLS by default, mTLS for service-to-service,
  OAuth2/OIDC client identity, per-tenant rate limits and audit logs,
  workspace-level ACLs, federation between instances, multi-language
  client SDKs, single-binary deployment.

Plus an opt-in compliance-sensitive layer (redaction, vocabularies,
provenance, evidence, tenant KMS, signed export) that ships as separate
optional packages so general-purpose users pay nothing for it. The
multi-tenancy thread is also tracked as issue [#505].

> **Companion design doc:** [`docs/roadmap-v4.md`](docs/roadmap-v4.md)
> holds the deeper architectural rationale. The task list below is
> canonical; the design doc explains the *why*.

### A. Cognition / model layer

Three-tier memory architecture with surprise as a ranking signal and
the test-time-update governance pattern (which `propose_update →
approve_apply` already implements). Mind-mem stays a retrieval
substrate; cognition lives at the kernel API, not in the model itself.

- [ ] **Surprise-weighted retrieval** — augment the BM25+vector+RRF fusion with a `surprise` term derived from semantic distance to the rolling recall context (NOT gradient-based; mind-mem doesn't train at recall time). Backwards-compatible behind `retrieval.surprise_weight` config flag. Estimated 1–2 weeks; default OFF in v4.0, opt-in.
- [ ] **Block-tier tags** — add `tier ∈ {short, long, persistent}` (also expressible as hot/warm/cold) to block schema. Default `long` for backwards compat. Recall uses tier as a prior: "what's relevant right now" (short/hot) vs "what we always know" (persistent/cold). Backwards-compatible schema add (T-007). Per-tier decay schedules: hot=LRU+TTL, warm=TTL+surprise re-promotion, cold=indefinite + contradiction-density.
- [ ] **Cognitive Mind Kernel** — top-level retrieval API: `mind_recall(workspace, query, kernel="surprise_weighted" | "lineage_first" | "recent_first" | "contradicts_first" | "graph_walk")`. Memory routing becomes first-class. Composable strategy parameter on top of existing `recall`.
- [ ] **Multi-modal blocks** — add `kind ∈ {image, audio, code, structured}` storing embeddings only (raw bytes via external content-addressable store). Cross-lingual embedding space for text. Required for any agent that observes more than text in 2026.
- [ ] **Graph-aware retrieval** — query expansion via lineage walk, community-detection-weighted recall, multi-hop reasoning over the typed-edge graph. Builds directly on existing `block_lineage`. Closes the gap between hybrid retrieval and the typed graph that already exists.
- [ ] **`mind-mem-4b` v4.0 retrain on tier-aware corpus** — new corpus probes covering tier semantics, surprise term, plus all v4 surfaces below (block kinds, fusion, viewer, lint, chat, transport, compliance). Estimated +140 probes → ~5500-6000 corpus examples. ~$8 + 120 min H200.
- [ ] **`mind-mem-4b` base-model evaluation** — when an alternative base model with stronger alignment to MIND-Mem's deterministic / governance contract becomes available, evaluate it side-by-side against the current Qwen3.5-4B fine-tune. Decision criterion: matched eval-harness pass rates ≥ Qwen baseline + tier semantics learned ≥ 90%. Pure plan-stage today; no code change without a side-by-side eval.

### B. Knowledge graph

Compounding-knowledge model: LLM reads sources, extracts entities and
concepts as separate blocks with bidirectional typed lineage edges,
performs structured incremental merge on update. Long-context retrieval
(union of full pages) is offered alongside the existing chunked top-K
mode; caller chooses speed vs. coherence per call. Builds on the typed
edge graph (`cites` / `implements` / `refines` / `contradicts` /
`cooccurrence`) already shipped in v3.11.0.

- [ ] **Block kinds** — add `kind ∈ {entity, concept, source, synthesis, image, audio, code, structured}` to block schema. Foundation for everything else in this group. Backwards-compat: existing blocks default to `kind=null` (untyped); v4 reads both old `category` field and new `kind` for one minor version, then deprecates.
- [ ] **Block versioning + time-travel** — every change is already in the audit chain; expose as `recall(..., as_of=date)` and `block_history(block_id)`. Lets callers query the workspace as of any historical point.
- [ ] **Content-addressable block IDs** — every block gets a stable content hash (CID-style) in addition to its rolling audit hash. Enables deduplication across workspaces, integrity verification on replication, and federated mesh foundation.
- [ ] **Long-context recall mode** — `recall(workspace, query, mode="long_context")` returns the union of full entity/concept pages whose summaries match (vs. chunked top-K). Caller (or LLM) reasons over whole graph. Higher token cost, deeper relational understanding. Default mode stays `chunked` for backwards compat.
- [ ] **LLM-driven knowledge fusion** — new tool `propose_fuse(workspace, source_block_id)`. LLM reads new source, extracts entity/concept candidates, hybrid-searches existing matches, generates structured incremental merge proposals (additions / updates / removals / contradictions). Reuses existing propose_update + approve_apply queue; adds LLM intelligence on top.
- [ ] **Streaming recall** — `recall(...)` becomes a streaming generator. Caller short-circuits on first good match. Cuts p99 latency for typical queries.
- [ ] **Conversational chat layer** — `chat_with_memory(workspace, question)`: long_context recall → LLM context → stream reply with `[[block_id]]` citations → optional `--save` writes synthesis as `kind=synthesis` block.
- [ ] **Schema layer for LLM prompts** — `mind-mem.json` gains a `prompts.schema` block injected into every LLM-driven tool (extraction, fusion, query) for consistent output.
- [ ] **Schema evolution / migration tooling** — `mm migrate` auto-detects schema mismatches when v4 fields are added (`kind`, `tier`, `provenance`) and runs reversible transforms under a dry-run flag first.

### C. Knowledge graph governance / UX

- [ ] **Idle-only background ingest** — `mm-mem-watch` daemon (opt-in via `mind-mem.json: { "watch": { "enabled": true } }`). Polls `inbox/` folder at low priority, `nice +19` + cgroup `cpu.max` capped at 50% of one core. Embedding work runs on local GPU only when GPU util < 10% for 30s. Hard memory cap (default 2GB). Pre-flight cost gate refuses ingest > N MB without explicit confirm. Process is supervised but exits cleanly on config flip. Never installs itself; never reinstalls itself.
- [ ] **AI lint with auto-fix** — extend `scan` to emit findings: dead lineage edges, empty blocks, orphans, contradiction without state. New tool `lint_autofix(workspace, finding_id)` queues a `propose_update` to fix one finding. Always review-gated.
- [ ] **Contradiction state machine** — `detected → review_ok → resolved` (AI fix) or `detected → pending_fix` (manual). Today contradictions are flagged but stateless; v4 tracks per-contradiction lifecycle.
- [ ] **Self-healing index** — background reindex (during idle windows) detects corruption and rebuilds from audit log. Watchdog catches stalled writes. `mm doctor` triggers explicit full integrity check + repair report.
- [ ] **Local visual viewer** — `mm view` launches read-only web UI on `127.0.0.1:NNNN`: block graph (force-directed) with kind-colored nodes, typed lineage edges, contradiction inbox with state machine, proposal queue, recall search box. Token-gated mutations (default off). Stack: stdlib HTTP server (no Node dep) + minimal JS / D3. Local-only, no remote bind.
- [ ] **Auto-generated hierarchical index** — `index.md` and `log.md` rebuilt automatically by tag/kind/timestamp.
- [ ] **Real-time contradiction stream** — webhook (or stdio event stream) when a new block contradicts an existing block at threshold ≥ 0.97 + opposing-claim signature. Today contradictions are batch-detected via `scan`; v4 makes them live for governance-critical deployments.
- [ ] **Adversarial / poisoning defense** — per-actor anomaly detection (writes that diverge from actor's prior), contradiction-density spike alerts, canary blocks (synthetic facts that recall MUST surface; absence indicates index tamper), Sigstore-signed block manifests for tamper-evident replication.
- [ ] **Approval workflows for sensitive proposals** — generalize `propose_update → approve_apply` from single-step to multi-reviewer chains via an OPA/Rego-style policy engine. Ships as opt-in dep so the core stays stdlib.
- [ ] **Memory reputation / trust scores** — per-actor reliability scoring computed from existing audit chain + contradiction events. Surfaces in recall as a confidence signal.

### D. Network & multi-agent connectivity

The primary frame: agents on different machines, owned by different
parties, share governed memory through mind-mem. Single-host scale
(sharded Postgres / K8s) is a sub-bucket for heavy deployments; the
default story is two laptops talking to each other.

- [ ] **TLS 1.3 minimum on all transports** — REST, gRPC, MCP-over-HTTP. Reject older protocols. Cert pinning option.
- [ ] **mTLS for service-to-service** — mutual auth between mind-mem nodes; per-tenant client cert validation.
- [ ] **OAuth2 / OIDC client identity** — pluggable identity provider integration (any IdP that speaks OIDC).
- [ ] **DID + Verifiable Credential agent identity** — W3C standards. Agent identity is portable across instances; "agent X is allowed to read workspace Y" is a signed VC.
- [ ] **Workspace ACLs (block-level grants)** — per-block read/write/audit permissions backed by a signed grant chain. Workspace-level remains the default; block-level is the override for fine-grained sharing.
- [ ] **Public / private workspaces** — `workspace.mode = public | private | mixed`. Public workspaces surface a discoverable read-only view; mixed allows per-block visibility.
- [ ] **Cross-instance federation protocol** — signed-chain handshake; two mind-mem instances negotiate workspace sync. Three-way merge with proposals when divergent.
- [ ] **End-to-end encryption for sensitive workspaces** — content encrypted at the writer; only authorized readers decrypt; substrate stores ciphertext + index over hashes.
- [ ] **Discovery / WebFinger** — well-known endpoint advertises capabilities, supported workspaces, public keys. Standard mechanism for finding mind-mem instances.
- [ ] **ActivityPub federation interop** — optional bridge dep. Workspaces expose as ActivityPub actors; other federation servers can subscribe to public-block streams.
- [ ] **Subscriptions / webhooks** — `subscribe(workspace, filter, callback_url)` POSTs JSON when matching block lands. Push notifications for reactive agent workflows.
- [ ] **Per-tenant rate limiting + circuit breakers** — quota buckets per tenant; one tenant cannot starve others.
- [ ] **Audit headers propagated through transport** — `X-MindMem-Request-Id`, `X-MindMem-Actor`, `X-MindMem-Purpose` carried over REST/gRPC; required when `provenance: required` (Group E).
- [ ] **Per-tenant routing** — `tenant_id` routes to its own KMS key + audit chain + rate limit bucket.
- [ ] **gRPC + REST parity on auth/audit** — internal services use gRPC (low-latency); external clients REST; same auth/audit contract on both.
- [ ] **Single-binary distribution** — `pip install mind-mem; mm serve` brings up an authenticated public endpoint. Solo-dev path doesn't require K8s.
- [ ] **Sharded Postgres (Citus)** — `src/mind_mem/storage/sharded_pg.py`; shard by `namespace_id` with consistent hashing; cross-shard recall merging. Heavy-deployment path.
- [ ] **Replication + consensus for governance** — Raft log wrapper around `audit_chain.py`; strong consistency for governance writes, eventual consistency for recall reads.
- [ ] **Kubernetes operator** — `operator/` with CRDs for `MindMem`, `Namespace`, `TenantKey`; Helm chart in `deploy/helm/mind-mem/`.
- [ ] **Byzantine-safe consensus (opt-in)** — `src/mind_mem/bft.py`; PBFT for high-trust deployments where quorum votes may be adversarial.
- [ ] **Edge deployment mode** — `mind-mem-edge` binary (PyOxidizer); embedded mode for on-device agents; hybrid sync to a central cluster.
- [ ] **Managed-service console** — `web/console/`; multi-tenant dashboard, cost metering, usage graphs, per-tenant audit chain viewer.
- [ ] **Kafka / NATS event fan-out** — governance events (`contradiction_detected`, `block_promoted`, `snapshot_created`) published as streams; external systems subscribe without polling.
- [ ] **Rust hot path for hybrid search** — port BM25 + RRF fusion to Rust (PyO3 binding). Optional dep — falls back to pure Python if not built.
- [ ] **Pluggable embedding backend with fallback** — explicit fallback chain (try local Ollama → API fallback on local failure). Health-aware — skips a backend that returned errors in the last N minutes.

### E. Compliance-sensitive opt-in extensions

These primitives generalize concerns common to many regulated
deployments. mind-mem stays general-purpose; the features ship as
opt-in capabilities. Anything domain-specific (controlled vocabularies,
redaction packs, export policies) ships as separate optional packages
so users who don't need them pay nothing in dependencies, install size,
or attack surface.

- [ ] **Pluggable redaction layer** — pre-write hook scans incoming block text against a configurable detector chain (regex / model-based / hybrid) and either rejects, masks, or tags. Default detectors: secrets, email, phone, credit-cards, dates-near-PII. Domain detector packs ship as separate optional packages. Redaction events go to the audit chain.
- [ ] **Time-bounded and event-bounded recall** — extend `recall(...)` with `since` / `until` / `event_id` filters. Audit chain + `block_staleness` already have the timestamp data.
- [ ] **Vocabulary-bound fields** — per-workspace controlled vocabularies (`mind-mem.json: { "vocabularies": [...] }`) attach to specific block kinds. Unknown values trigger an `invalid_vocabulary` rule in `validate_block`. Generalizes the existing rule-based quality gate.
- [ ] **Provenance-rich blocks** — optional fields on every block: `actor_id`, `actor_role`, `session_id`, `tool_id`, `purpose`. Gated by `mind-mem.json: { "provenance": "off"|"recommended"|"required" }`. Indexable for "who-wrote-what" queries.
- [ ] **Confidence / Evidence as first-class** — promote `confidence: low/medium/high` to a structured `Evidence` block on every claim: `{ claim, source_id, confidence_score (0-1), source_kind: "primary"|"derived"|"hearsay", attestation_date }`. Recall surfaces full evidence chains. Builds directly on the v3.11 lineage edge model.
- [ ] **Tenant KMS + row-level encryption** — `src/mind_mem/tenant_crypto.py`; per-tenant data keys, envelope encryption, rotation; extends the v3.0.0 `EncryptedBlockStore`. Workspace-level cryptographic isolation: no plaintext crosses tenant boundaries even on the same host.
- [ ] **Per-tenant audit chains** — fork `audit_chain.py` so each tenant has an isolated hash chain with its own genesis + spec-hash binding.
- [ ] **Compliance export pipeline** — `mm export --policy <policy_name> --since <date>` produces a deterministic, signed bundle (zip + manifest) suitable for regulated audit submission. Policy plug-ins ship per-domain. Export uses Q16.16 audit chain so the bundle is verifiable end-to-end.
- [ ] **Contraindication / mutex edges** — extend `block_lineage` edge kinds with `contraindicates` (should not co-occur) and `supersedes` (this replaces that), distinct from existing `contradicts` (claims oppose each other). Pure addition to typed-edge graph.
- [ ] **C2PA content provenance** — when chat layer produces synthesis blocks, attach C2PA-signed provenance manifest (model used, inputs, signed by instance key). Important for AI-generated content traceability.

### G. Observability, reliability, ecosystem

- [ ] **OpenTelemetry tracing + metrics + logs** — adopt OTel SDK with `mind_mem` semantic conventions for spans (`mind_mem.recall`, `mind_mem.propose`, `mind_mem.fuse`, etc.). Prometheus metrics export at `/metrics`.
- [ ] **Health / liveness / readiness probes** — standard endpoints for K8s integration.
- [ ] **Continuous backup + point-in-time recovery** — incremental backup to S3 / Backblaze / R2 / local NFS; PITR to any audit-chain index.
- [ ] **JavaScript / TypeScript SDK** — `@star-ga/mind-mem-client` npm package generated from OpenAPI spec. WebAssembly-compatible so browser agents can talk directly.
- [ ] **Browser-native WebAssembly bundle** — read-only client in WASM with crypto verification. Browser agents query mind-mem without a Node middleware.
- [ ] **Go / Rust / Java / Ruby SDK stubs** — generated from OpenAPI + AsyncAPI specs.
- [ ] **OpenAPI + AsyncAPI specs** — single source of truth; auto-generate clients and docs.
- [ ] **Migration importers from competing systems** — `mm import --from {pinecone | weaviate | chroma | qdrant | letta | mem0}`. Adoption blocker if missing.
- [ ] **Cost metering / quota / spending alerts** — per-workspace usage counters (tokens read, embeddings computed, storage bytes, API calls) emitted to OTel + exposed via `mm usage`.
- [ ] **SLSA build provenance level 3** — already partially via Sigstore release signing; add provenance attestations + isolated builders.
- [ ] **Plugin SDK** — stable plug-in API for custom rules, custom block kinds, custom decay schedules, custom redaction detectors.
- [ ] **Performance regression alerting** — every PR runs latency benchmarks; auto-flag if p99 increases >10%.
- [ ] **Chaos testing harness** — automated fault injection for federated / replicated deployments.

### F. Anti-patterns explicitly forbidden

Patterns observed in third-party memory systems that have crushed user
machines or violated user trust. v4.0 must NOT inherit any of them:

- ❌ Always-on background daemon (watcher must be opt-in, idle-only, resource-capped, exits cleanly on config flip).
- ❌ Auto-marketplace reinstall (no mechanism to reinstall after the user removes us — removal is permanent).
- ❌ Multi-process worker fan-out without caps (single supervised process; embedding queue is bounded).
- ❌ Inline embedding during user-facing tool calls (embedding work runs on dedicated worker; tool call returns immediately with streaming results).
- ❌ Background polling that wakes on schedule (only inotify on `inbox/` or user-triggered).
- ❌ Bulk re-ingest of historical transcripts without explicit confirm (every ingest gated by pre-flight cost check).
- ❌ Implicit paid-API calls (local-first by default; explicit opt-in for API embedding/extraction backends with budget cap).
- ❌ Trust raw input from unauthenticated public sources (signed provenance required for federated sync).
- ❌ Embed raw bytes from unauthenticated sources (multi-modal blocks store hashes + verified-source URLs only).
- ❌ Telemetry leakage (no usage data leaves the host without explicit opt-in).

### H. Research direction (post-v4.0, on the trajectory)

Out of scope for v4.0 ship; documented so the path is visible.

- Homomorphic / partial-FHE search (CKKS over encrypted vectors).
- Zero-knowledge memory proofs ("prove block exists without revealing content").
- Secure enclave deployment (Intel TDX / AMD SEV / Apple Secure Enclave).
- Federated learning across instances with differential privacy.
- Streaming ingestion at high write rates (millions of events/sec, fire-and-forget with eventual consistency).

**Estimated:** ~3000 lines storage + ~2000 lines consensus + ~1500 lines operator + ~2500 lines console + ~1500 lines knowledge-graph (kinds + fusion + chat) + ~1000 lines viewer + ~800 lines lint/contradictions + ~1200 lines compliance plug-ins + ~2000 lines network/transport + ~1500 lines SDKs + ~800 lines observability. Breaking change: `v4` requires explicit storage adapter selection (no implicit SQLite default in cluster deployments).

**Suggested sequencing:** A.block-kinds → C.idle-ingest → B.long-context-recall → B.fusion → B.streaming → A.tier-memory → A.graph-aware-retrieval → C.viewer → C.lint → C.self-heal → C.adversarial-defense → B.chat → D.transport-security (TLS, mTLS, OAuth, ACLs, federation) → D.SDK-ecosystem → G.observability → E.{redaction, time-bound, provenance, evidence} (the four core compliance primitives) → D.platform-scale (sharded Postgres, K8s, gRPC) → E.compliance plug-ins → A.cognition retrain (depends on most of A/B/C/D/E being stable).

---

## v4.0.x — Federation transport hardening (gaps surfaced by v4.0.8)

v4.0.8 closed `#529` (scheme allowlist, same-origin redirect handler,
response-size cap) and `#528` (three-way merge audit log). Four
defensive controls remain explicitly **not** enforced; the current
threat model is *single-operator shared-secret*. Listing them here so
the gaps are tracked instead of being implicit.

The bigger v4.0.0 Group-D items (mTLS, OAuth2/OIDC, DID/VC, workspace
ACLs, cross-instance federation protocol) sit above this section.
These are the smaller, surgical gaps that should land first.

- [ ] **Per-peer identity beyond bearer token.** Today any holder of the
  shared `X-MindMem-Token` can call any federation endpoint as any
  `agent_id`. There is no cryptographic binding between the token and
  the agent identity the caller claims. A leaked token gives full
  write authority over the federation surface. Two staged fixes:
  (a) per-peer tokens with a token→agent_id table; reject a write
  whose claimed `agent_id` doesn't match the bound identity for the
  presented token. (b) signed-write envelopes — peer Ed25519-signs
  every `record_agent_write` body; server verifies against a
  per-peer public-key allowlist. Item (b) is the prerequisite for
  the Group-D `DID + Verifiable Credential agent identity` item.
- [ ] **mTLS + certificate pinning on `FederationClient`.** The current
  client does NOT verify the peer's certificate against a pinned
  expected key — it inherits whatever the system trust store says.
  TLS interception (corporate proxies, hostile network) is therefore
  undetectable from inside the client. The v4.0.0 Group-D `mTLS for
  service-to-service` item is the destination; this sub-task is the
  client-side pinning primitive (`FederationClient(base_url, ...,
  pinned_pubkey_sha256=...)` constructor arg + verification hook on
  the strict opener).
- [ ] **Operator-side peer allowlist.** No built-in IP / hostname
  allowlist on the federation HTTP listener. Operators have to put a
  reverse proxy (nginx, Caddy) in front and configure it externally.
  In-process allowlist would be `MIND_MEM_FED_PEERS=10.0.0.5,10.0.0.6`
  → 403 for any source IP outside the set. Compatible with bearer
  token; doesn't replace it.
- [ ] **Token rotation primitive.** Today operators rotate by editing
  `MIND_MEM_TOKEN` env and restarting; there is no in-band rotation
  protocol. A leaked token is valid until the operator notices.
  Minimal fix: accept N-of-K active tokens at the server, expose
  `mm token rotate` that emits a new token + grace-window record.
  Server accepts old token for grace period (default 24h), then
  expires.

These four items together close the realistic "what if my token
leaks" failure mode for federation. The v4.0.0 Group-D items (mTLS,
DID, OAuth/OIDC, workspace ACLs) are the bigger compliance layer
sitting on top.

---

## Post-v2.7.0 — Future Directions

- [x] **Agent-to-agent trust protocol** — agents verify each other's memory integrity via Merkle proofs before sharing context
- [x] **Distributed memory mesh** — multiple MIND-Mem instances with hash-chain synchronization _(see v2.6.0 P2P Mesh for foundation)_
- [x] **Real-time governance dashboard** — web UI showing evidence stream, chain health, spec-hash status
- [x] **512 Kernel full integration** — MIND-Mem as a governed resource within 512-mind production deployments
- [x] **Hardware-specific compilation** — `mindc` targets for ARM (Apple Silicon), CUDA, ROCm
- [x] **Multi-user retrieval adaptation** — per-user fine-tuning in multi-tenant deployments, isolated signal streams
- [x] **Federated memory** — privacy-preserving retrieval across organizational boundaries (differential privacy + secure aggregation)
- [x] **Continuous benchmark regression** — every PR runs LoCoMo subset + latency benchmarks; auto-reject if MRR drops or p99 increases >10%

---

## Companion Tools (External, Non-Dependency)

External MCP-server tools that solve adjacent memory problems MIND-Mem deliberately
does not solve. Documented here so users see them as complements rather than
competitors. **MIND-Mem will not depend on any of these** — license, scope, and
substrate-of-record concerns make co-existence the right pattern.

- [ ] **GitNexus** (`github.com/h4ckf0r0day/GitNexus`) — code knowledge-graph indexer
  exposed as MCP server. Parses repo structure (call graphs, dependencies, clusters)
  and serves architectural-awareness tools to coding agents. Solves "what does the
  code do at this point in time" — orthogonal to MIND-Mem's "what did we decide and
  why over time." License: PolyForm Noncommercial — incompatible with Apache-2.0
  programmatic dependency. Recommendation: install as a separate MCP server
  alongside MIND-Mem; both end up in Claude Code / Cursor / Windsurf MCP lists,
  no integration code required. Documentation will mention this in the README under
  "Companion Tools" once the section is added (separate task).

---

## Advanced Agent Memory Primitives

MIND-Mem is designed as a governed-memory substrate for autonomous agents operating in interactive reasoning environments (benchmark agents, game-playing agents, long-horizon task agents). The following block types and retrieval capabilities extend the core schema for those workloads.

### Shipped (already available in MIND-Mem)

- [x] **`[PATTERN]` blocks** — opening-book / strategy-template storage; recall by environment fingerprint drives initial-action selection
- [x] **`[TRAJECTORY]` block type** — shipped in v1.1.0; stores per-session execution traces, recallable by session-id or environment-id for historical playthrough retrieval
- [x] **`[OBSERVATION]` blocks** — multi-model consensus votes stored with their scores and rationales; contradictions between votes surface via the existing contradiction-detection engine
- [x] **Governance gate** — the invariant kernel (see v2.0.0rc1) validates every action-emission from the host agent; rejected moves are retained with their rejection rationale for post-mortem
- [x] **Cross-session persistence** — `MIND_MEM_WORKSPACE` is a shared namespace across any set of agents that agree on the path; one recall call retrieves strategy memory across all cooperating agents

### Planned block types and adapters

- [ ] **`[CAUSAL]` block type** — world-model storage for learned state transitions (observation → action → next-observation); consumed by the host agent's planner during multi-step lookahead
- [ ] **`[SKILL]` block type** — named strategy captures with preconditions, effects, and success-rate metadata; retrievable by skill-name or by applicable-context similarity
- [ ] **Cross-domain recall adapter** — given a novel environment, surface the most similar `[TRAJECTORY]` / `[SKILL]` blocks from unrelated environments by feature-embedding similarity rather than exact environment-id match
- [ ] **`[VISUAL]` block type** — grid-state / image-state embeddings for perception-grounded memory; enables "I've seen this state before" recall across environments
- [ ] **Evidence-chain submission format** — tamper-evident export of an agent's full decision history per episode, ready for third-party scorecard verification

---

## TRIZ-Driven Direction

Auditable criterion for every roadmap addition. Lifted from the same TRIZ pattern in `mind/docs/roadmap.md` and `naestro/ROADMAP.md`, tuned to the persistent-memory domain.

### Ideal Final Result (IFR)

Persistent memory that improves agent decision quality monotonically over use, never drifts from currently-declared intent, with full provenance from query result back to the source byte that justified it, at zero recall-latency overhead vs raw vector search, and survives substrate / model / format migrations without re-ingestion.

### Five Laws of System Evolution — Applied

| Law | MIND-Mem application | Status |
|---|---|---|
| Uneven development | Recall surfaces evolve faster than provenance, contradiction-handling, and consolidation. Invest in audit + governance, not more recall modes. | Active investment shift toward provenance + governance |
| Mono → bi → poly | Single-store BM25 → BM25+vector hybrid → poly-substrate (BM25 + vector + graph + governance kernel). Next: cross-machine networked mesh. | At bi → poly transition |
| Increasing controllability | Hardcoded retrieval logic → policy-driven scoring → drift-detected policy update (v3.10). Compile-of-code-aware invalidation (v3.9) is this law's projection. | Active in v3.9 / v3.10 |
| Micro-level transition | Coarse content blocks → smart-chunked sub-blocks → per-byte lineage. Permitted at recall and audit; forbidden at write-path (would break atomicity). | Active at correct layer only |
| Rhythm coordination | BM25 + vector + governance kernel synchronized via RRF + atomic conjunction. Next: cross-mesh evidence federation. | In v4.0 spec |

### Separation Principles for Memory Conflicts

When two memory blocks contradict, when current intent conflicts with prior declared intent (intent-era drift), or when recall and write paths conflict, default to separation before retirement:

- **Time** — policy A within a session, policy B across sessions; intent-era boundaries surfaced explicitly
- **Space** — consistency rule X for compiled-truth blocks, rule Y for working-memory blocks
- **Condition** — atomic conjunction in governance gate, RRF fusion in recall
- **Scale** — block-level provenance for retrieval, byte-level provenance for audit

v3.9 hash-of-code invalidation + v3.10 governance drift detection are the operational home for this discipline.

### Anti-Patterns Made Explicit

- No new block types without contradiction-handling story — every type must specify how it conflicts with existing types and how that conflict is resolved
- No silent retrieval mode changes that affect provenance — all changes traceable to a versioned policy
- No "more substrates = better" decisions without measured improvement on adversarial-memory + Jepsen-style stress tests
- No retirement of an invariant before separation strategies have been exhausted (matches 512-mind Phase B addendum discipline)

Acceptance gate: every new feature cites IFR component strengthened, law of evolution followed or forbidden, separation strategy for likely contradictions, and anti-pattern avoided.

---

## Pure-MIND Core Port (long-horizon architectural goal)

**Goal:** progressively port the MIND-Mem core to pure MIND until the
Python surface is a thin adapter shell, then eliminate it. MIND is now
a self-hosted systems language (the compiler and source-quality
toolchain self-host), so this is a real trajectory, not a category
boundary.

**Already MIND today:** the hot scoring/decision kernels ship as
`.mind` and compile via `mindc` — `bm25`, `rrf`, `reranker`,
`abstention`, `adversarial`, `answer`, `category`, `cognitive`,
`cross_encoder`, `ensemble`, `evidence`, `governance`. These run with
a pure-Python fallback, so the kernel boundary is already proven and
non-load-bearing for availability.

**Gating dependency (honest):** a full port is sequenced behind the
MIND toolchain's **library-emit / stable C-ABI** capability (cdylib
output + FFI surface) maturing upstream. Until that lands, the I/O
shell (MCP transport, SQLite/Postgres/Redis adapters, HTTP, external
model clients) stays Python by necessity, not by design.

**Sequencing (incremental, never a big-bang rewrite):**

- [x] Hot scoring kernels in pure MIND (`mind/*.mind`, bench-gated)
- [ ] Governance / decision / boundary layer in pure MIND — recall
      scoring orchestration, quality-gate, ACL, contradiction and
      decision rules (best fit for MIND's systems-programming surface;
      no FFI required)
- [ ] Core retrieval engine (index walk, fusion, rerank pipeline) in
      pure MIND behind the C-ABI boundary
- [ ] I/O adapters via the MIND C-ABI / FFI surface as it matures
- [ ] Python reduced to a thin compatibility shim, then removed

**Discipline (non-negotiable, every step):**

- Every ported unit must be **byte-identical / bit-identical** to the
  Python (or prior `.mind`) reference on the test corpus before it
  replaces it.
- The retrieval **perf-gate** (no regression beyond the standing cap)
  must hold on every recompile and every port increment.
- The pure-Python fallback remains the source of correctness until a
  ported unit passes both gates; a recompiled/ported unit that fails
  either gate does not ship — fallback stands.
- New `mindc` releases are **recompile-and-verify**, not rewrite:
  recompile `.mind` sources, run bit-identity + perf-gate, edit source
  only where the compiler/tests surface a real divergence.

**Source/runtime boundary (load-bearing):** the MIND language and the
`.mind` sources are **public** — porting more of MIND-Mem to pure MIND
adds *public* source, not exposure. Execution **runtimes / backends**
are the commercial, protectable layer and are out of scope for this
public roadmap: no runtime, backend, or protection internal is
described here, and the port never requires publishing one. Public
pure-MIND source compiled against a protected commercial runtime is
the intended end state, not a contradiction.

**Explicitly not a goal:** rewriting working Python I/O glue for its
own sake; any port step that regresses correctness, the perf-gate, or
the availability guarantee the Python fallback provides; or surfacing
any runtime / backend / protection internal in this or any public doc.

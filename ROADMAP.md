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
> Theme: mind-mem learns from every user interaction — corrections, re-queries, and rephrased searches become training signals that improve retrieval quality over time.

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
> never pollute the retrieval index. Both modules integrate with mind-mem via FFI. The v2.4.0
> features below formalize what those modules already enforce at the cognitive daemon level into
> mind-mem's own API surface.

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

## v2.6.0 — Competitive Intelligence Integration ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> Theme: Features identified from competitive analysis of agentmemory (rohitg00/agentmemory),
> Brooks Jordan's Daneel (NVIDIA), and Karpathy's llm-wiki pattern. Cherry-picked what we
> don't have yet; ignored what we already do better.
>
> Sources:
> - [rohitg00/agentmemory](https://github.com/rohitg00/agentmemory) v0.7.1 — 581 tests, Node.js, iii-engine
> - Brooks Jordan (NVIDIA Enterprise AI Partnerships) — Daneel agent, OpenClaw/FelixCraft-inspired
> - Andrej Karpathy — llm-wiki gist (raw → wiki → schema pattern)
> - Farzapedia / "File over app" pattern

### Cascading Staleness Propagation
_Source: agentmemory — cascading staleness across graph nodes_

- [x] **Staleness propagation engine** — when a block is superseded or contradicted, automatically flag related blocks (edges, siblings, dependents) as potentially stale
- [x] **Staleness confidence decay** — propagation weakens with graph distance: direct relations get `stale=0.9`, 2-hop get `stale=0.5`, 3-hop get `stale=0.2`
- [x] **Staleness in retrieval scoring** — stale-flagged blocks penalized in BM25F scoring (configurable weight, default 0.3x)
- [x] **`propagate_staleness` MCP tool** — manual trigger with dry-run mode showing affected blocks
- [x] **Staleness audit log** — every propagation event recorded with source block, affected blocks, reason

### 4-Tier Memory Consolidation
_Source: agentmemory — working → episodic → semantic → procedural tiers with Ebbinghaus decay_

mind-mem currently has append-only logs → manual promotion to MEMORY.md. This formalizes the pipeline:

- [x] **Tier 0 (Working)** — raw daily log entries (`memory/YYYY-MM-DD.md`), TTL 30 days before decay review
- [x] **Tier 1 (Episodic)** — compressed session summaries (`summaries/weekly/`), auto-generated from Tier 0
- [x] **Tier 2 (Semantic)** — verified facts and entity knowledge (`entities/`, `MEMORY.md`), promoted from Tier 1 after N repetitions or explicit confirmation
- [x] **Tier 3 (Procedural)** — learned patterns and strategies (`decisions/`), highest durability, governance-gated
- [x] **Ebbinghaus strength decay** — each block has `strength` field (0.0–1.0), decays exponentially with configurable half-life (default 30 days), reset on access
- [x] **Auto-promotion triggers** — block repeated 3+ times across sessions → auto-promote to next tier (governance proposal if Tier 2→3)
- [x] **Tier-aware retrieval** — higher tiers get retrieval priority boost (Tier 3: 2.0x, Tier 2: 1.5x, Tier 1: 1.0x, Tier 0: 0.7x)
- [x] **`consolidate` MCP tool** — trigger consolidation cycle with `--dry-run` and `--tier` filters

### Agent Hook Auto-Capture
_Source: agentmemory — 12 Claude Code hooks for silent observation capture_

- [x] **Hook event schema** — standardized event format: `{type, timestamp, tool, input_hash, output_summary, project, session_id}`
- [x] **SessionStart hook** — inject recent context from mind-mem at conversation start (token-budgeted)
- [x] **PostToolUse hook** — capture tool name + output summary, SHA-256 dedup (5-min window)
- [x] **PreCompact hook** — re-inject critical memory context before context compaction
- [x] **SessionEnd hook** — trigger end-of-session summary compression
- [x] **Privacy filter** — strip API keys, secrets, `<private>` tagged content before storage
- [x] **Hook installer** — `mind-mem hooks install` CLI command, writes to `~/.claude/settings.json`
- [x] **Observation → block pipeline** — raw hook events compressed into structured blocks via LLM (Zod-validated, quality scored 0-100)

### Token Budget Context Injection
_Source: agentmemory — configurable token budget (default 2000) with smart packing_

- [x] **`recall` gains `max_tokens` parameter** — backward-compatible, defaults to unlimited (current behavior)
- [x] **Adaptive packing strategy** — given token budget:
  1. Reserve 15% for graph context (entity relationships)
  2. Reserve 10% for provenance metadata (source citations)
  3. Pack remaining with blocks by relevance score, truncating lowest-scored
- [x] **Model-aware defaults** — auto-detect context window from model name and set sensible defaults
- [x] **Packing quality metric** — track % of packed tokens actually referenced in response (calibration loop)

### Project Intelligence Profiles
_Source: agentmemory — per-project aggregated intelligence_

- [x] **Auto-generated project profiles** — aggregate from entity files + observations: top concepts, most-touched files, coding conventions, common errors, session count
- [x] **Profile as MCP resource** — `mindmem://project/{name}/profile` exposes structured project intelligence
- [x] **Profile injection at session start** — when project context detected, inject profile into system prompt
- [x] **Convention extraction** — LLM-powered extraction of implicit conventions from code observations (naming patterns, test patterns, error handling style)

### P2P Memory Mesh
_Source: agentmemory — cross-agent sync with 7 scopes; Brooks Jordan/Daneel — multi-agent sharing_

- [x] **Mesh protocol** — mind-mem instances discover peers via mDNS or explicit peer list
- [x] **7 sync scopes** — memories, actions, semantic, procedural, relations, graph, governance (each independently toggleable)
- [x] **Conflict resolution** — last-write-wins for Tier 0-1, governance-gated merge for Tier 2-3
- [x] **Namespace isolation** — shared vs private memory with per-scope access control
- [x] **Sync audit log** — every sync event recorded with peer ID, scope, blocks transferred, conflicts resolved
- [x] **`mesh_status` MCP resource** — connected peers, sync lag, scope health

### Model Reliability Score (MRS) Framework
_Source: Bandhavi Sakhamuri — ML Inference SLO concept; agentmemory quality scoring_

- [x] **MRS SLI definitions** — latency percentiles (p50/p95/p99), output quality drift, token throughput, error rate, cost per query
- [x] **Composite MRS (0-100)** — weighted aggregation of SLIs into single reliability score
- [x] **YAML SLO schema** — define per-model SLO thresholds, weights, alert conditions
- [x] **Memory retrieval MRS extension** — relevance decay rate, contradiction density, staleness ratio as retrieval-specific SLIs
- [x] **MRS dashboard** — real-time MRS per model endpoint + per retrieval backend
- [x] **Alert on MRS degradation** — configurable thresholds trigger warnings before quality impacts users

**Estimated:** ~3500 lines total. No breaking changes. All features are additive and config-gated.

---

## v2.7.0 — Universal Agent Bridge + Vault Sync ✅ Released 2026-04-13 — all boxes checked in v2.8.0

> Theme: mind-mem becomes the shared memory layer for **every** coding agent — not just MCP-capable ones.
> Any CLI agent (Claude Code, codex, gemini, Cursor, Windsurf, Aider) reads and writes
> to the same memory through a unified interface. Plus bidirectional vault sync for Obsidian/file-based
> knowledge management.
>
> Ref: "Second Brain" pattern (Obsidian + 5-brain MCP), agentmemory hook system, OpenClaw skills architecture

### Component 1: Universal Agent Bridge (`mm` CLI)

**Problem:** MCP-capable agents (Claude Code, other MCP-native runtimes) already have mind-mem access. Non-MCP agents (codex, gemini CLI, Cursor, Windsurf, Aider) have zero memory — every session starts blank. The `mm` CLI bridges this gap.

- [x] **`mm` unified CLI** — single binary (`~/.local/bin/mm`) wrapping all mind-mem operations:
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

**Problem:** Obsidian (and similar PKM tools) provide visual graph navigation, backlinks, and manual curation that mind-mem doesn't. mind-mem provides hybrid retrieval, governance, and agent-accessible MCP that Obsidian doesn't. Users shouldn't have to choose.

- [x] **Vault scanner** — `mm vault sync /path/to/obsidian/vault`:
  - Reads all `.md` files in vault
  - Detects block types from frontmatter/headers (decisions, entities, tasks, notes)
  - Indexes into mind-mem with `source: vault` provenance tag
  - Respects `.obsidian/` and `.trash/` exclusions
  - Incremental: only re-indexes files modified since last sync (mtime-based)
- [x] **Reverse sync** — mind-mem → vault:
  - New decisions/entities created via `mm capture` or MCP get written back to vault as `.md` files
  - Maintains Obsidian-compatible frontmatter (tags, aliases, created, modified)
  - Creates `[[wikilinks]]` for entity cross-references
  - Respects vault folder structure (configurable mapping: decisions/ → vault/decisions/, etc.)
- [x] **Conflict resolution** — when both sides modify the same block:
  - Vault wins for manual edits (human curation > agent writes)
  - mind-mem wins for governance decisions (contradictions, drift alerts)
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
- [x] **mind-mem:4b model via Ollama** — Qwen3.5-4B full fine-tune on STARGA-curated mind-mem corpus; Q4_K_M @ 2.6 GB; default `extraction.model`; empirical on RTX 3080: 104 tok/s generation, 1585 tok/s prefill
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
- [ ] **Dockerfile + docker-compose** — `deploy/docker/Dockerfile`, `deploy/docker-compose.yml` with mind-mem + pgvector + Ollama; one-command `make up`
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

## v3.2.1 — Hotfix follow-up

Closes the two architectural CRITICALs surfaced by the v3.2.0
self-audit (`docs/review-architecture-v3.2.0.md`). v3.2.0 Postgres
+ REST surfaces are labelled "beta" in the release notes; v3.2.1
promotes both to GA.

- [ ] **Apply engine routes through `BlockStore`** — the seven
  `_op_*` handlers in `apply_engine.py` currently speak raw
  `open()` / `shutil` on corpus Markdown files. On a Postgres
  backend they write to the local FS that Postgres never sees, so
  `apply_proposal` succeeds while DB state silently diverges.
  Re-plumb through `BlockStore.write_block` / `delete_block`.
  ~2 days.
- [ ] **REST request-scoping** — `src/mind_mem/api/rest.py`
  currently uses `os.environ["MIND_MEM_WORKSPACE"] = workspace`
  to carry per-request workspace selection, which is thread-unsafe
  the moment `uvicorn` runs >1 worker. Extract a
  `mind_mem.core.services` module that both REST and MCP call
  with an explicit `workspace` argument. ~1 day.
- [ ] **OIDC wired into `_require_admin`** — v3.2.0 OIDC only
  validates tokens on `/v1/auth/oidc/callback`; protected
  endpoints cannot accept OIDC JWTs. Map JWT `scopes` claim to
  the internal scope grammar. ~0.5 day.
- [ ] **`PostgresBlockStore.snapshot(snap_id=…)`** — current
  signature requires a filesystem path for the MANIFEST.json
  write, breaking cross-host Postgres snapshots. Accept a plain
  `snap_id: str` and make the on-disk manifest optional. ~0.5
  day.
- [ ] **Wire `cached_recall` into `_recall_impl`** — the Redis
  cache module ships in v3.2.0 but the call-site wiring is
  deferred. One-liner in
  `src/mind_mem/mcp/tools/recall.py::_recall_impl`. ~0.5 hour.
- [ ] **Two config keys documented in `docs/configuration.md`** —
  `cache.redis_url` and `retrieval.tier_boost`. Docs-only.
- [ ] **Dependency CVE bumps** — `authlib>=1.6.9`, `aiohttp>=3.13.4`
  per the v3.2.0 audit's INFO findings. Transitive via `fastmcp`.

**Estimated:** ~1200 lines refactor, ~400 LOC tests, ~200 LOC
docs.

## v3.3.0 — Reasoning-Grade Retrieval (1–2 months)

Close the retrieval-quality gap and widen the governance moat. All additive — no breaking changes to existing recall contracts.

- [ ] **Query decomposition** — `src/mind_mem/query_planner.py`; LLM-backed plan that splits complex queries into sub-queries and merges results; config toggle `retrieval.query_decomposition`
- [ ] **Multi-hop graph traversal** — `src/mind_mem/graph_recall.py`; extend `causal_graph.py` to traverse `relates_to` / `supersedes` / `contradicts` edges up to N hops; `recall(multi_hop=True, max_hops=3)`
- [ ] **Temporal re-weighting in the hot path** — half-life decay on `Created:` field inside the scorer; config `retrieval.temporal_half_life_days` (default 90); currently available as a filter but not as a ranking signal
- [ ] **Probabilistic truth score** — `src/mind_mem/truth_score.py`; Bayesian update on block `importance` from contradiction votes; exposed as `block.truth_score` in recall responses
- [ ] **Streaming ingest + back-pressure queue** — `src/mind_mem/streaming.py`; websocket/SSE endpoint with back-pressure via a bounded mpsc channel (drop-oldest when writer pool saturates); drop-in replacement for `capture --stdin` in high-rate pipelines; per-client token-bucket rate limit
- [ ] **Consensus voting** — extend `conflict_resolver.py` with quorum across agents, weighted by `namespace.trust_weight`; resolves contradictions without human review when confidence exceeds threshold
- [ ] **Graph + timeline visualization** — `web/` Next.js app; D3 / react-flow graph view (nodes = blocks, edges = relationships), timeline view, drift heatmap; reads from REST API shipped in v3.2.0. v3.2.0 already emits `[[wikilinks]]` on `vault_sync` so an Obsidian-mounted vault gets a graph view for free; this web UI is the non-Obsidian alternative.
- [ ] **mind-mem-4b v2 fine-tune** — retrain the bundled 4B model on the v3.2.0 MCP surface so it natively emits the 7 consolidated dispatcher calls (`recall(mode=…)`, `staged_change(phase=…)`, `graph(action=…)`, etc.). v3.2.0 works with the v1 fine-tune because every legacy tool name still resolves; this is a tool-selection-reliability upgrade, not a correctness fix. Training data re-uses the v3.2.0 test fixtures + recorded tool-call transcripts.

**Estimated:** ~1500 lines retrieval + ~2000 lines web UI + ~2 GPU-days retrain. New optional extras: `mind-mem[reasoning]`, `mind-mem[streaming]`.

## v4.0.0 — Platform Scale (production)

Horizontal scaling, multi-tenant isolation, and edge deployment. This version turns mind-mem from a library into a platform. The multi-tenancy thread is also tracked as issue [#505].

- [ ] **Sharded Postgres (Citus)** — `src/mind_mem/storage/sharded_pg.py`; shard by `namespace_id` with consistent hashing; cross-shard recall merging
- [ ] **Replication + consensus for governance** — Raft log wrapper around `audit_chain.py`; strong consistency for governance writes, eventual consistency for recall reads
- [ ] **Kubernetes operator** — `operator/` with CRDs for `MindMem`, `Namespace`, `TenantKey`; Helm chart in `deploy/helm/mind-mem/`
- [ ] **Tenant KMS + row-level encryption** — `src/mind_mem/tenant_crypto.py`; per-tenant data keys, envelope encryption, rotation; extends the v3.0.0 `EncryptedBlockStore`
- [ ] **Per-tenant audit chains** — fork `audit_chain.py` so each tenant has an isolated hash chain with its own genesis + spec-hash binding; enables per-customer compliance exports without cross-tenant leakage
- [ ] **Byzantine-safe consensus (opt-in)** — `src/mind_mem/bft.py`; PBFT for high-trust deployments where quorum votes may be adversarial
- [ ] **Edge deployment mode** — `mind-mem-edge` binary (PyOxidizer); embedded mode for on-device agents; hybrid sync to a central mind-mem cluster
- [ ] **Managed-service console** — `web/console/`; multi-tenant dashboard, cost metering, usage graphs, per-tenant audit chain viewer
- [ ] **gRPC wire protocol** — `src/mind_mem/api/grpc_server.py`; low-latency alternative to REST for service-to-service calls
- [ ] **Kafka/NATS event fan-out** — governance events (`contradiction_detected`, `block_promoted`, `snapshot_created`) published as streams; external systems subscribe without polling

**Estimated:** ~3000 lines storage + ~2000 lines consensus + ~1500 lines operator + ~2500 lines console. Breaking change: `v4` requires explicit storage adapter selection (no implicit SQLite default in cluster deployments).

---

## Post-v2.7.0 — Future Directions

- [x] **Agent-to-agent trust protocol** — agents verify each other's memory integrity via Merkle proofs before sharing context
- [x] **Distributed memory mesh** — multiple mind-mem instances with hash-chain synchronization _(see v2.6.0 P2P Mesh for foundation)_
- [x] **Real-time governance dashboard** — web UI showing evidence stream, chain health, spec-hash status
- [x] **512 Kernel full integration** — mind-mem as a governed resource within 512-mind production deployments
- [x] **Hardware-specific compilation** — `mindc` targets for ARM (Apple Silicon), CUDA, ROCm
- [x] **Multi-user retrieval adaptation** — per-user fine-tuning in multi-tenant deployments, isolated signal streams
- [x] **Federated memory** — privacy-preserving retrieval across organizational boundaries (differential privacy + secure aggregation)
- [x] **Continuous benchmark regression** — every PR runs LoCoMo subset + latency benchmarks; auto-reject if MRR drops or p99 increases >10%

---

## Advanced Agent Memory Primitives

mind-mem is designed as a governed-memory substrate for autonomous agents operating in interactive reasoning environments (benchmark agents, game-playing agents, long-horizon task agents). The following block types and retrieval capabilities extend the core schema for those workloads.

### Shipped (already available in mind-mem)

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

# mind-mem Architecture

Version 1.0.2 | February 2026

---

## 1. System Overview

mind-mem is a governance-aware memory layer for AI agents. It combines
BM25 full-text search, optional vector embeddings, compiled MIND kernels,
and a proposal-based governance engine into a single MCP-native service.

All source of truth lives in plain Markdown files. Indexes are ephemeral
and rebuilt from source. Zero external dependencies beyond Python 3.10+
stdlib (optional components require `sentence-transformers` or compiled
MIND `.so` libraries).

```
                          MCP Clients
              (Claude Code / Claude Desktop / Cursor / OpenClaw)
                              |
                    stdio or HTTP transport
                              |
    +-------------------------v---------------------------+
    |               MCP Server  (mcp_server.py)           |
    |       16 tools  |  8 resources  |  FastMCP runtime  |
    +--------+--------+--------+------+-------------------+
             |                 |
    +--------v---------+  +----v-----------------------+
    |  Recall Pipeline  |  |  Governance Engine         |
    |                   |  |                            |
    |  intent_router    |  |  capture.py  (26 patterns) |
    |  recall.py (BM25) |  |  transcript_capture.py     |
    |  hybrid_recall.py |  |  apply_engine.py (WAL)     |
    |  recall_vector.py |  |  intel_scan.py             |
    |  evidence_packer  |  |  conflict_resolver.py      |
    |  cross_encoder    |  |  compaction.py             |
    +--------+---------+  +----+-----------------------+
             |                 |
    +--------v---------+  +----v-----------------------+
    |  Indexing Layer    |  |  Extraction Layer          |
    |                   |  |                            |
    |  sqlite_index.py  |  |  extractor.py (NER-lite)   |
    |  block_parser.py  |  |  entity_ingest.py          |
    |  block_metadata   |  |  category_distiller.py     |
    +--------+---------+  |  observation_compress.py    |
             |             |  abstention_classifier.py   |
    +--------v---------+  +----+-----------------------+
    |  MIND FFI Layer   |       |
    |  (optional)       |       |
    |                   |       |
    |  mind_ffi.py      |       |
    |  lib/libmindmem   |       |
    |  .so / .dylib     |       |
    +--------+---------+       |
             |                  |
    +--------v------------------v-----------------------+
    |                  Storage Layer                     |
    |                                                   |
    |  Markdown Files      SQLite FTS5      WAL Journal |
    |  (source of truth)   (search index)   (crash-safe)|
    |                                                   |
    |  decisions/          .mind-mem.db     .mind-mem/  |
    |  tasks/                               wal/        |
    |  entities/                                        |
    |  intelligence/                                    |
    |  memory/                                          |
    +---------------------------------------------------+
    |               Infrastructure                      |
    |                                                   |
    |  filelock.py         namespaces.py                |
    |  observability.py    schema_version.py            |
    |  backup_restore.py   session_summarizer.py        |
    +---------------------------------------------------+
```

---

## 2. Recall Pipeline

The recall pipeline transforms a natural-language query into ranked,
evidence-packed results. Each stage is deterministic and operates without
an LLM (the optional observation compression step is the sole exception).

```
    Query (natural language string)
      |
      v
  +-------------------+
  | Intent Router     |   9 intent types: WHY, WHEN, ENTITY, STATUS,
  | (intent_router.py)|   COMPARISON, NEGATION, LIST, COUNT, GENERAL
  +--------+----------+
           |  IntentResult {intent, confidence, params}
           |  params include: expansion mode, graph_depth, rerank strategy
           v
  +-------------------+
  | Query Expansion   |   RM3 pseudo-relevance feedback (top-k docs
  | (recall.py)       |   expand query with weighted terms)
  +--------+----------+   Date expansion for WHEN intents
           |              Entity expansion for ENTITY intents
           v
  +-------------------+
  | Abstention Check  |   5-feature confidence gate:
  | (abstention_      |   entity overlap, BM25 score, speaker coverage,
  |  classifier.py)   |   evidence density, negation asymmetry
  +--------+----------+   Threshold 0.20 (tunable). Below = abstain.
           |
           v
  +-------------------------------+
  | Hybrid Backend (parallel)     |
  |                               |
  |  +-------------+  +---------+ |
  |  | BM25F       |  | Vector  | |   Two threads via ThreadPoolExecutor
  |  | (sqlite_    |  | (recall_| |
  |  |  index.py   |  |  vector | |   BM25F: FTS5 with field boosts
  |  |  or scan)   |  |  .py)   | |   Vector: sentence-transformers
  |  +------+------+  +----+----+ |        (local FAISS / Qdrant / Pinecone)
  |         |               |     |
  |         +-------+-------+     |
  |                 |             |
  |                 v             |
  |          RRF Fusion           |   score = sum(w_i / (k + rank_i))
  |          (k=60 default)       |   k, weights configurable in mind-mem.json
  +---------------+--------------+
                  |
                  v
  +-------------------+
  | Deterministic     |   Negation penalty (suppress contradictions)
  | Reranking         |   Date proximity boost (for temporal queries)
  | (recall.py)       |   Category relevance boost
  +--------+----------+   Recency decay weighting
           |              Status filtering (active-only option)
           v
  +-------------------+
  | Cross-Encoder     |   OPTIONAL: ms-marco-MiniLM-L-6-v2 (80MB)
  | (cross_encoder_   |   Blends CE score with BM25/RRF score:
  |  reranker.py)     |   final = 0.6 * CE + 0.4 * original
  +--------+----------+
           |
           v
  +-------------------+
  | A-MEM Importance  |   Boosts frequently-accessed blocks.
  | (block_metadata)  |   Tracks access_count, last_accessed,
  +--------+----------+   evolving keywords, connection graph.
           |
           v
  +-------------------+
  | Context Packing   |   Adjacency: includes neighbor blocks
  | (recall.py)       |   Diversity: deduplicates near-identical hits
  +--------+----------+   Pronoun rescue: pulls context for anaphora
           |
           v
  +-------------------+
  | Evidence Packing  |   Structured output per query type:
  | (evidence_        |   [SPEAKER=X] [DATE=Y] [DiaID=Z]
  |  packer.py)       |
  +--------+----------+   Ordering:
           |               - temporal: chronological by DiaID
           |               - multi-hop: hop-clustered by entity
           |               - adversarial: overlap-first, denial separated
           |               - default: score-descending
           v
        Results
```

### Key Scoring Parameters

| Parameter       | Default | Description                              |
|-----------------|---------|------------------------------------------|
| `BM25_K1`       | 1.2     | Term frequency saturation                |
| `BM25_B`        | 0.75    | Document length normalization            |
| `rrf_k`         | 60      | RRF rank constant                        |
| `bm25_weight`   | 1.0     | BM25 weight in RRF fusion                |
| `vector_weight` | 1.0     | Vector weight in RRF fusion              |

### Indexed Fields (priority order)

Statement, Title, Summary, Description, Context, Rationale, Tags,
Keywords, Name, Purpose, RootCause, Fix, Prevention, ProposedFix, Sources.

Additional ConstraintSignature fields: subject, predicate, object, domain.

---

## 3. MIND Kernels and FFI

### Overview

MIND is a tensor-typed language that compiles to shared libraries with a
C99-compatible ABI. The MIND kernels provide native-speed scoring for
numerical hot paths. They are entirely optional -- every kernel function
has a pure Python fallback that produces identical results.

### Compilation

```
mind/*.mind  -->  mindc (MIND compiler)  -->  lib/libmindmem.so
                                              lib/libmindmem.dylib
```

Compiler invocation:

```
mindc mind/*.mind --emit=shared -o lib/libmindmem.so
```

### Available Kernels

```
mind/
  bm25.mind        BM25F scoring with field boosts and length normalization
  rrf.mind         Reciprocal Rank Fusion (2-list and 3-list variants)
  reranker.mind    Deterministic reranking passes
  rerank.mind      Score reranking utilities
  ranking.mind     General ranking primitives
  recall.mind      Recall orchestration kernel
  hybrid.mind      Hybrid search fusion
  abstention.mind  Adversarial abstention scoring
  rm3.mind         RM3 query expansion term weighting
  importance.mind  A-MEM importance score computation
  temporal.mind    Temporal proximity and recency decay
  adversarial.mind Adversarial query detection features
  category.mind    Category relevance scoring
  prefetch.mind    Anticipatory context prefetch scoring
```

### FFI Bridge (mind_ffi.py)

The FFI bridge loads the compiled `.so` via `ctypes` and declares strict
argument types for every exported function. This prevents silent memory
corruption from type mismatches.

```
Python (mind_ffi.py)
  |
  |  ctypes.CDLL("lib/libmindmem.so")
  |
  |  Declare argtypes:
  |    rrf_fuse:        [float_p, float_p, int, float, float, float, float_p]
  |    bm25f_batch:     [float_p, float, float, float_p, float, float, float, float, int, float_p]
  |    negation_penalty:[float_p, float_p, float, int, float_p]
  |
  v
Compiled MIND .so (C99 ABI)
  |
  |  Functions accept flat float* pointers + dimension parameters.
  |  Tensor shape checks happen at compile time (mindc).
  |
  v
Hardware (SIMD when available)
```

### Library Search Order

1. Explicit `lib_path` argument to `MindMemKernel()`
2. `MIND_MEM_LIB` environment variable (restricted to `lib/` directory)
3. `lib/libmindmem.so` relative to project root
4. `lib/libmindmem.dylib` relative to project root
5. If none found: raises `OSError`, caller falls back to pure Python

### Kernel Function Signature (example: bm25.mind)

```
fn bm25f_doc(
    tf: tensor<f32[F, T]>,       // term frequencies per field
    idf: tensor<f32[T]>,         // inverse document frequencies
    boosts: tensor<f32[F]>,      // per-field boost weights
    field_lens: tensor<f32[F]>,  // document field lengths
    avg_lens: tensor<f32[F]>,    // average field lengths in corpus
    k1: f32,                     // saturation parameter
    b: f32                       // length normalization parameter
) -> tensor<f32>                 // scalar BM25F score
```

---

## 4. Storage Layer

### Markdown Files (Source of Truth)

All structured data lives in plain Markdown files using a typed block
format. The source Markdown is always authoritative -- all indexes and
caches can be rebuilt from it.

#### Workspace Directory Structure

```
workspace/
  decisions/
    DECISIONS.md                   Decision blocks [D-YYYYMMDD-NNN]
  tasks/
    TASKS.md                       Task blocks [T-YYYYMMDD-NNN]
  entities/
    projects.md                    Project entity blocks
    people.md                      People entity blocks
    tools.md                       Tool entity blocks
    incidents.md                   Incident entity blocks
  intelligence/
    SIGNALS.md                     Auto-captured signals (staging area)
    CONTRADICTIONS.md              Detected contradictions
    DRIFT.md                       Detected decision drift
    IMPACT.md                      Impact graph
    BRIEFINGS.md                   Weekly briefings
    AUDIT.md                       Audit trail
    SCAN_LOG.md                    Scan history
    proposed/
      DECISIONS_PROPOSED.md        Staged decision proposals
      TASKS_PROPOSED.md            Staged task proposals
      EDITS_PROPOSED.md            Staged edit proposals
    applied/                       Applied proposal receipts
    state/
      snapshots/                   Pre-apply state snapshots
  memory/
    intel-state.json               Scanner state and metrics
    maint-state.json               Maintenance state
  summaries/
    weekly/                        Weekly summaries
    daily/                         Daily session summaries
  maintenance/
    weeklog/                       Weekly maintenance logs
  categories/                      Auto-generated category summaries
  mind-mem.json                    Workspace configuration
  mind-mem-acl.json                Access control policies (multi-agent)
```

#### Block Format (Schema v1/v2)

```markdown
[D-20260215-001]
Date: 2026-02-15
Status: active
Statement: Use BM25F as the default recall backend.
Rationale: Zero external dependencies, deterministic results.
Tags: recall, search, architecture
Supersedes: D-20260101-003
```

Each block has:
- `_id`: Unique identifier (e.g., `D-20260215-001`)
- `_line`: Source file line number (1-based)
- Typed fields: Date, Status, Statement, Tags, etc.
- Optional nested lists and sub-structures

### Block Parser (block_parser.py)

Parses Schema v1.0 and v2.0 blocks from Markdown files. Returns a JSON
array of blocks with fields, lists, and nested structures. The parser is
the single entry point for all modules that read workspace content.

```
Markdown file  -->  block_parser.parse_file()  -->  list[dict]
                    block_parser.parse_blocks()      each dict has _id, _line, fields...
                    block_parser.get_active()         filter by Status == "active"
                    block_parser.get_by_id()          lookup by block ID
```

### SQLite FTS5 Index (sqlite_index.py)

An ephemeral full-text search index maintained alongside the source
Markdown. Provides O(log N) indexed lookup instead of O(corpus) scanning.

#### Schema

```sql
-- Core block storage
blocks(id PK, type, file, line, status, date, speaker, tags, json_blob)

-- Full-text search (FTS5 virtual table)
blocks_fts(Statement, Title, Tags, Description, Context)

-- Cross-reference graph for neighbor boosting
xref_edges(src, dst)

-- Incremental rebuild tracking
file_state(path, mtime, size, hash)

-- Optional embedding vectors
block_vectors(id PK, vector BLOB)

-- A-MEM metadata
block_meta(id PK, importance, access_count, last_accessed, keywords, connections)
```

#### BM25F Field Weights

The FTS5 index uses weighted fields. Higher-priority fields (Statement,
Title) receive larger boosts than lower-priority fields (Context, Tags)
during BM25F scoring.

#### Incremental Updates

`file_state` tracks each source file's mtime, size, and content hash.
On rebuild, only changed files are re-parsed and re-indexed. A full
rebuild can be forced via the `reindex` MCP tool.

### WAL Journal (Crash Safety)

All write operations go through a Write-Ahead Log (`backup_restore.py`).
The WAL ensures atomicity:

```
1. Write intent to WAL journal
2. Take pre-write snapshot (state/snapshots/)
3. Execute file mutations
4. Verify post-write integrity
5. Mark WAL entry complete
6. On crash: wal-replay recovers to last consistent state
```

### File Locking (filelock.py)

Two-layer cooperative locking for concurrent agent/session writes:

1. `threading.Lock` for same-process (thread) contention
2. `O_CREAT|O_EXCL` lockfile + OS-level locks for cross-process contention

---

## 5. Governance Engine

The governance engine ensures that no automated process directly mutates
the source of truth. All changes flow through a proposal-review-apply
pipeline with full audit trail and rollback capability.

### Pipeline

```
  +------------------+     +-------------------+     +------------------+
  | Signal Sources   | --> | Staging Area      | --> | Review + Apply   |
  |                  |     |                   |     |                  |
  | capture.py       |     | SIGNALS.md        |     | approve_apply    |
  | transcript_      |     | proposed/         |     | (MCP tool)       |
  | capture.py       |     |  DECISIONS_       |     |                  |
  | entity_ingest.py |     |  PROPOSED.md      |     | apply_engine.py  |
  | session_         |     |  TASKS_PROPOSED   |     | (atomic ops)     |
  | summarizer.py    |     |  EDITS_PROPOSED   |     |                  |
  +------------------+     +-------------------+     +--------+---------+
                                                              |
                  +-------------------------------------------+
                  |
                  v
  +------------------+     +-------------------+     +------------------+
  | Atomic Apply     | --> | Post-Apply Scan   | --> | Audit + Monitor  |
  |                  |     |                   |     |                  |
  | Snapshot state   |     | intel_scan.py     |     | AUDIT.md         |
  | Validate ops     |     | contradiction     |     | SCAN_LOG.md      |
  | Execute writes   |     | detection         |     | applied/ receipts|
  | Write receipt    |     | drift analysis    |     |                  |
  |                  |     | impact graph      |     | Rollback via     |
  | Rollback on fail |     | briefing gen      |     | receipt timestamp|
  +------------------+     +-------------------+     +------------------+
```

### Capture Engine (capture.py)

The auto-capture engine scans daily logs for decision-like and task-like
language using 26 regex patterns. Each pattern has a confidence level
(high/medium/low) and signal type classification.

Safety invariant: capture ONLY writes to `intelligence/SIGNALS.md`. It
NEVER writes to `decisions/DECISIONS.md` or `tasks/TASKS.md`. This
prevents memory poisoning from automated extraction errors.

Pattern categories:
- High confidence decisions: "we decided", "from now on", "no longer"
- Medium confidence decisions: "let's go with", "switching to"
- Task patterns: "need to", "TODO", "deadline"
- Correction patterns: "don't do X", "always do Y"
- Convention patterns: "the pattern is", "the convention is"
- Bug insight patterns: "the fix was", "root cause"

Extraction pipeline:
1. Scan daily log for pattern matches
2. Extract structured fields (subject, predicate, confidence)
3. Classify signal priority based on language strength
4. SHA256 content-hash dedup (normalized text, 16-char hex prefix)
5. Append to `intelligence/SIGNALS.md` with full metadata

### Apply Engine (apply_engine.py)

Applies staged proposals atomically with rollback on failure.

Valid operations:
- `append_block`: Add a new block to a file
- `insert_after_block`: Insert after a specific block ID
- `update_field`: Modify a single field on a block
- `append_list_item`: Add an item to a block's list field
- `replace_range`: Replace a line range in a file
- `set_status`: Change block status
- `supersede_decision`: Mark a decision as superseded

Proposal lifecycle:
```
staged --> validated --> applied --> (optionally rolled_back)
                    \-> rejected
                    \-> deferred
                    \-> expired
```

Budget limits (configurable in `mind-mem.json`):
- `per_run`: 3 proposals max per invocation
- `per_day`: 6 proposals max per day
- `backlog_limit`: 30 pending proposals max

### Intelligence Scanner (intel_scan.py)

Runs integrity analysis across the workspace:

- **Contradiction detection**: Finds blocks with conflicting statements
  using modality analysis (must vs must_not, should vs should_not)
- **Drift analysis**: Detects decisions that have drifted from original
  intent based on subsequent overrides or scope changes
- **Impact graph**: Maps decision-to-task and decision-to-entity
  dependencies
- **State snapshots**: Periodic workspace state captures
- **Weekly briefing**: Auto-generated summary of changes and issues

Valid modalities: must, must_not, should, should_not, may.

Valid domains: integrity, memory, retrieval, security, llm_strategy,
workflow, project, comms, finance, other.

### Conflict Resolution (conflict_resolver.py)

Graduated resolution beyond detection:

1. **Detection** (via intel_scan.py)
2. **Strategy selection**:
   - `TIMESTAMP_PRIORITY`: Newest decision wins
   - `CONFIDENCE_PRIORITY`: Highest ConstraintSignature priority wins
   - `SCOPE_PRIORITY`: More specific scope wins over general
   - `MANUAL`: Cannot auto-resolve, requires human review
3. **Proposal generation**: Creates supersede proposals for
   high-confidence resolutions
4. **Human veto loop**: Pending-review queue, never auto-applies

---

## 6. MCP Server

### Runtime

The MCP server (`mcp_server.py`) uses FastMCP (Python) and exposes
mind-mem as a Model Context Protocol service. Any MCP-compatible client
can connect.

### Transport

| Mode  | Use Case                    | Command                                          |
|-------|-----------------------------|--------------------------------------------------|
| stdio | Claude Code, Claude Desktop | `python3 mcp_server.py`                          |
| HTTP  | Remote / multi-client       | `python3 mcp_server.py --transport http --port 8765` |
| HTTP + auth | Secured remote        | `MIND_MEM_TOKEN=secret python3 mcp_server.py --transport http` |

### Tools (16)

| Tool                 | Description                                          |
|----------------------|------------------------------------------------------|
| `recall`             | BM25 search across memory blocks                     |
| `propose_update`     | Propose a new decision/task (writes to SIGNALS.md)   |
| `approve_apply`      | Apply a staged proposal (dry-run by default)         |
| `rollback_proposal`  | Rollback an applied proposal by receipt timestamp    |
| `scan`               | Run integrity scan                                   |
| `list_contradictions` | List contradictions with resolution status           |
| `hybrid_search`      | Full hybrid BM25+Vector recall with RRF fusion       |
| `find_similar`       | Find blocks similar to a given block                 |
| `intent_classify`    | Show routing strategy for a query                    |
| `index_stats`        | Block counts, index status, kernel info              |
| `reindex`            | Trigger FTS index rebuild                            |
| `memory_evolution`   | A-MEM metadata for a block                           |
| `list_mind_kernels`  | List available .mind kernel configs                  |
| `get_mind_kernel`    | Read a specific .mind kernel source                  |
| `category_summary`   | Category summaries for a topic                       |
| `prefetch`           | Pre-assemble context from conversation signals       |

### Resources (8)

| URI                           | Description                     |
|-------------------------------|---------------------------------|
| `mind-mem://decisions`        | All active decisions            |
| `mind-mem://tasks`            | All tasks                       |
| `mind-mem://entities/{type}`  | Entity files (projects, people, tools, incidents) |
| `mind-mem://signals`          | Auto-captured signals           |
| `mind-mem://contradictions`   | Detected contradictions         |
| `mind-mem://health`           | Workspace health summary        |
| `mind-mem://recall/{query}`   | BM25 recall search              |
| `mind-mem://ledger`           | Shared fact ledger (multi-agent)|

### Configuration

Workspace path resolution:
1. `MIND_MEM_WORKSPACE` environment variable
2. Current working directory

Client configuration example (Claude Desktop):

```json
{
  "mcpServers": {
    "mind-mem": {
      "command": "python3",
      "args": ["/path/to/mind-mem/mcp_server.py"],
      "env": {"MIND_MEM_WORKSPACE": "/path/to/workspace"}
    }
  }
}
```

---

## 7. Optional Components

All optional components degrade gracefully. The core system operates
with zero external dependencies using only Python 3.10+ stdlib and
SQLite (bundled with Python).

### Vector Search (recall_vector.py)

Semantic recall using sentence-transformers embeddings.

| Backend    | Requirement                    | Description                     |
|------------|--------------------------------|---------------------------------|
| `local`    | `sentence-transformers`        | Local JSON index (default)      |
| `qdrant`   | `qdrant-client`                | Qdrant vector database          |
| `pinecone` | `pinecone-client`              | Pinecone cloud vector database  |

Default model: `all-MiniLM-L6-v2`. ONNX backend available for faster
CPU inference when `onnx_backend: true` in config.

When vector search is unavailable, `hybrid_recall.py` falls back to
BM25-only mode transparently.

### Cross-Encoder Reranker (cross_encoder_reranker.py)

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (80MB, CPU-friendly) to
rerank candidates with learned relevance scoring. Blends cross-encoder
score with the original retrieval score:

```
final_score = blend_weight * CE_score + (1 - blend_weight) * original_score
```

Default `blend_weight`: 0.6. Requires `sentence-transformers`.

### MIND Kernels (mind/*.mind)

14 compiled scoring kernels accelerating numerical hot paths. Requires
the `mindc` compiler from the STARGA toolchain.

When the compiled `.so` is absent:
- `MindMemKernel()` raises `OSError`
- Callers catch the exception and use pure Python implementations
- Results are numerically identical; only throughput differs

### Observation Compression (observation_compress.py)

LLM-based compression of retrieved blocks into concise, query-relevant
observations. Sits between retrieval and answer generation:

```
Retrieve (BM25) --> Compress (LLM) --> Answer (LLM) --> Judge (LLM)
```

This is the only component that requires an LLM at recall time. It
improves answer quality by eliminating noise, synthesizing scattered
facts, and surfacing implicit temporal/causal relationships.

### Abstention Classifier (abstention_classifier.py)

Deterministic pre-LLM confidence gate with 5 features:

1. Entity overlap (query entities vs. retrieved block entities)
2. BM25 score (raw retrieval confidence)
3. Speaker coverage (fraction of mentioned speakers found)
4. Evidence density (relevant facts per retrieved block)
5. Negation asymmetry (query polarity vs. evidence polarity)

Default threshold: 0.20 (conservative). Below threshold, the system
abstains rather than generating a low-confidence answer.

### Category Distiller (category_distiller.py)

Deterministic category detection (no LLM). Scans memory blocks and
produces thematic summary files in `categories/`. Categories are
auto-detected from block tags, entity types, and content keyword
matching. Used for anticipatory context assembly.

### Entity Extraction (extractor.py)

Regex-based NER-lite that decomposes raw observations into atomic fact
cards: FACT, EVENT, PREFERENCE, RELATION, NEGATION. Each card is
independently retrievable by BM25 and links back to its source via
`source_id`. Short cards (10-20 words) score much higher than long
conversation blocks for single-hop queries.

---

## 8. Multi-Agent Architecture

### Namespace Model

```
workspace/
  shared/                      All agents can read
    decisions/
    tasks/
    entities/
    intelligence/
      LEDGER.md                Cross-agent fact propagation

  agents/
    coder-1/                   Private to coder-1
      decisions/
      tasks/
      memory/
    reviewer-1/                Private to reviewer-1
      decisions/
      memory/

  mind-mem-acl.json            Access control policies
```

### Access Control (namespaces.py)

JSON-based ACL with fnmatch pattern matching:

```json
{
  "default_policy": "read",
  "agents": {
    "coder-1": {
      "namespaces": ["shared", "agents/coder-1"],
      "write": ["agents/coder-1"],
      "read": ["shared"]
    },
    "reviewer-*": {
      "namespaces": ["shared"],
      "write": [],
      "read": ["shared"]
    },
    "*": {
      "namespaces": ["shared"],
      "write": [],
      "read": ["shared"]
    }
  }
}
```

Each agent sees `shared/` plus its own namespace. The apply engine
respects ACL boundaries -- an agent cannot write to namespaces outside
its `write` list.

---

## 9. Module Inventory

### Core Modules

| Module               | Lines  | Purpose                                                       |
|----------------------|--------|---------------------------------------------------------------|
| `recall.py`          | ~2,100 | BM25F + RM3 + graph scoring, stemming, query expansion        |
| `intel_scan.py`      | ~1,250 | Integrity scanning, contradiction detection, drift analysis    |
| `apply_engine.py`    | ~1,200 | Proposal application, dry-run, rollback, audit trail          |
| `sqlite_index.py`    | ~780   | FTS5 index, block vectors, block metadata tables              |
| `mcp_server.py`      | ~950   | FastMCP server, 16 tools, 8 resources                         |
| `block_parser.py`    | ~520   | Markdown block grammar parsing (schema v1/v2)                 |

### Search and Ranking

| Module                      | Lines | Purpose                                               |
|-----------------------------|-------|-------------------------------------------------------|
| `recall_vector.py`          | ~700  | Vector/embedding recall backends                      |
| `hybrid_recall.py`          | ~310  | Thread-parallel BM25+Vector+RRF fusion                |
| `intent_router.py`          | ~160  | 9-type intent classification with parameter mapping   |
| `evidence_packer.py`        | ~250  | Structured evidence assembly for LLM context          |
| `block_metadata.py`         | ~200  | A-MEM: access tracking, importance, keyword evolution |
| `cross_encoder_reranker.py` | ~80   | Optional ms-marco-MiniLM cross-encoder                |
| `mind_ffi.py`               | ~255  | ctypes FFI bridge to compiled MIND .so                |
| `abstention_classifier.py`  | ~275  | Adversarial abstention (5-feature confidence gate)    |

### Extraction and Ingestion

| Module                    | Lines | Purpose                                          |
|---------------------------|-------|--------------------------------------------------|
| `extractor.py`            | ~650  | Regex NER-lite: FACT/EVENT/PREFERENCE/RELATION   |
| `category_distiller.py`   | ~550  | Deterministic thematic category summaries        |
| `entity_ingest.py`        | ~350  | Entity extraction from transcripts and logs      |
| `observation_compress.py` | ~160  | LLM-based retrieval compression (optional)       |

### Governance and Support

| Module                  | Lines | Purpose                                    |
|-------------------------|-------|--------------------------------------------|
| `capture.py`            | ~330  | Auto-capture from daily logs (26 patterns) |
| `transcript_capture.py` | ~200  | JSONL transcript signal extraction         |
| `conflict_resolver.py`  | ~310  | Graduated conflict resolution pipeline     |
| `compaction.py`         | ~340  | GC, archival, retention policies           |
| `backup_restore.py`     | ~410  | WAL, backup, restore, JSONL export         |
| `namespaces.py`         | ~380  | Multi-agent namespace + ACL management     |
| `session_summarizer.py` | ~280  | Daily session summary generation           |

### Infrastructure

| Module              | Lines | Purpose                                    |
|---------------------|-------|--------------------------------------------|
| `filelock.py`       | ~150  | Cross-platform advisory file locking       |
| `observability.py`  | ~180  | Structured JSON logging + metrics          |
| `schema_version.py` | ~200  | Schema migration tooling                   |
| `validate_py.py`    | ~360  | Python-based workspace validation          |
| `validate.sh`       | ~700  | Bash-based workspace validation            |

---

## 10. Configuration Reference

### mind-mem.json

```json
{
  "version": "1.0.2",
  "workspace_path": ".",
  "auto_capture": true,
  "auto_recall": true,
  "governance_mode": "detect_only",
  "recall": {
    "backend": "bm25",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "vector_weight": 1.0,
    "vector_model": "all-MiniLM-L6-v2",
    "vector_enabled": false,
    "onnx_backend": false
  },
  "proposal_budget": {
    "per_run": 3,
    "per_day": 6,
    "backlog_limit": 30
  },
  "scan_schedule": "daily",
  "prompts": {
    "observation_compress": "",
    "entity_extract": "",
    "category_distill": ""
  },
  "categories": {
    "enabled": true,
    "extra_categories": {}
  }
}
```

### Recall Backend Modes

| Mode     | Module                        | Description                           |
|----------|-------------------------------|---------------------------------------|
| `scan`   | `recall.py`                   | Direct corpus scan (no index)         |
| `bm25`   | `recall.py` + `sqlite_index`  | FTS5-indexed BM25F (default)          |
| `vector` | `recall_vector.py`            | Embedding-based semantic search       |
| `hybrid` | `hybrid_recall.py`            | Parallel BM25 + vector with RRF fusion|

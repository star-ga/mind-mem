# Roadmap

> This is the short-form roadmap. The canonical, detailed version lives
> in [`../ROADMAP.md`](../ROADMAP.md) at the repo root and includes the
> full milestone breakdown.

## v3.8.2 (Current — released 2026-05-02)

Provenance allowlist for `mm audit-model`. Adds the seventh check —
`base_model` namespace must match an allowlisted upstream publisher.
Default list covers Alibaba Qwen, Meta Llama, Mistral AI, Google
Gemma, IBM Granite, OpenAI, Anthropic, DeepSeek, Microsoft Phi, and
TII Falcon. `mm audit-model --allow-publisher <slug>` (repeatable)
extends the allowlist for internal fine-tune orgs. 25 unit tests in
`tests/test_model_provenance.py`. No breaking changes.

## v3.8.1 (Released 2026-05-02)

Ed25519 manifest signing for `mm audit-model` checkpoints. Adds the
`mind_mem.model_signing` module plus `mm sign-model <path>` and
`mm verify-model <path>` CLI subcommands. Three sidecars are written
next to the checkpoint root: `MODEL_MANIFEST.txt` (sorted, deterministic
SHA-256 manifest, `sha256sum -c`-compatible), `MODEL_MANIFEST.txt.sig`
(raw 64-byte Ed25519 signature, RFC 8032 §5.1.6), and `MODEL_PUBKEY.pub`
(raw 32-byte public key). `verify_model` returns a structured
`error_kind` enum (`manifest_mismatch` / `bad_signature` /
`missing_file`) so callers can distinguish drift from forgery from a
missing sidecar. 23 unit tests in `tests/test_model_signing.py`.

## v3.8.0 (Released 2026-05-02)

Model Safety Audit — first slice. Static security scan of any local
model checkpoint via `mm audit-model <path>`: remote-code hooks
(`auto_map` / `trust_remote_code`), bundled `.py` refuser, weight
format guard (`.safetensors` / `.gguf` only), pickle raw-byte opcode
walk for dangerous-import references, tokenizer-injection scanner, and
a `safetensors` header validator. 31 unit tests in
`tests/test_model_audit.py`. Subsequent v3.8.x patches add Ed25519
signing (v3.8.1, shipped), provenance allowlist, MCP wrapper, and
load-gate integration into the extractor backends.

## v3.7.0 (Released 2026-05-01)

External-audit response (see CHANGELOG.md "3.7.0" for the full list).
**Breaking change:** HTTP/REST authentication now fails CLOSED — any
deployment that relied on the implicit "no token configured →
anonymous access" path must set `MIND_MEM_TOKEN` /
`MIND_MEM_ADMIN_TOKEN` (production) or
`MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1` (loopback dev).

Other audit fixes: cross-platform atomic snapshot rollback (Linux +
macOS + Windows now share one path, no realpath divergence); CI
matrix tightened (typecheck blocking, Python 3.14 required, coverage
gate raised to 70%, new extras-install / hashed-requirements / docker
/ compose-config jobs); recall `mode="vector"` removed from the public
surface (was silently rewriting to `auto`); `sqlite_index._file_hash`
now full-content SHA-256 with nanosecond mtime precision; install.sh
defaults to pipx with PEP 668 fallback to `pip --user`; release
workflow no longer pretends to ship a precompiled `libmindmem.so` it
doesn't build.

## v3.1.7 (Released 2026-04-18)

Static-typing cleanup. The 39 pre-existing mypy errors across 14
modules are resolved (explicit constructor casts, corrected
argument types on two call sites, and narrowed annotations where
`Any` was leaking into the signature). `typecheck` is now a real
gate in CI. GitHub repository **About** refreshed to match v3.1.x
reality (57 MCP tools, 17 native AI-client integrations, 4B local
model, zero core deps).

## v3.1.6 (Released 2026-04-18)

Range-pin `onnxruntime`, `tokenizers`, `sentence-transformers` in
the `[test]` extra (the prior `1.24.2` pin was not resolvable on
macOS wheels); `TemporaryDirectory(ignore_cleanup_errors=True)`
across every test to avoid Windows `PermissionError` on SQLite
handles still open at teardown.

## v3.1.5 (Released 2026-04-18)

Restored the optional-extras visibility for tests that import
`onnxruntime` / `tokenizers` / `sentence-transformers` directly.
Stale `demo.gif` removed from the README pending a v3.1.x refresh.

## v3.1.4 (Released 2026-04-18)

**Mistral Vibe CLI** added as a first-class client (`mm install-all`
now wires 17 clients). Fixes the Windows path-separator round-trip
in `agent_bridge.VaultBridge.scan`. Adds `sqlite-vec` to the
`[test]` extra so CI test matrices install it.

## v3.1.3 (Released 2026-04-18)

CI-layer patch release: ruff lint + format cleared across the repo,
`fastmcp` added to the `[test]` extra. No runtime change.

## v3.1.2 (Released 2026-04-18)

Docs + metadata alignment to v3.1.x. README badges corrected
(`tests-3610`, `MCP_tools-57`), stale "release local (no Actions)"
badge removed. `CLAUDE.md` and `docs/roadmap.md` refreshed.

## v3.1.1 (Released 2026-04-15)

Patch release. Claude Code hook-installer fix: `install claude-code`
now writes the required nested hook shape and migrates legacy flat
entries in-place. `mm inject` / `mm vault status` hooks replaced with
`mm status` where the old commands pointed at unshipped subcommands.

## v3.1.0 (Released 2026-04-15)

Native MCP integration for 8 additional AI clients plus a
multi-backend LLM extractor. `mm install-all` auto-wires 16 clients
total.

## v3.0.0 (Released 2026-04-14)

Governance alerting hooks (webhook / Slack), transparent at-rest
encryption (SQLCipher + BlockStore), TTL/LRU tier decay, and
full-fine-tune local model `star-ga/mind-mem-4b` (Qwen3.5-4B base).
Adversarial-memory corpus tests and Jepsen-style concurrency stress
tests merged. MCP tool surface now 57.

## v2.10.0 (Released 2026-04-14)

Audit-integrity patch series. TAG_v1 NUL-separated hash preimages for
collision resistance, Q16.16 fixed-point scores in audit hash
preimages.

## v2.9.0 and earlier

See `CHANGELOG.md` for the full history. Highlights across v2.x:
incremental reindexing, delta-snapshot block versioning with WAL,
sqlite-vec HNSW-compatible vector search, prefix cache + speculative
prefetch, workspace federation via namespaces, FTS5 index,
MIND-kernel plugin system for custom scoring, ODC axis-aware
retrieval, cryptographic governance layer, GBrain enrichment,
inference acceleration for the Python subset.

## v1.9.x (Earlier stable line, superseded)

Foundational retrieval and governance work:

- BM25F scoring with field weights
- Co-retrieval graph boost
- Fact card sub-block indexing
- Knee score cutoff, hard negative mining
- LoCoMo benchmark suite
- Cross-platform CI (Ubuntu / macOS / Windows)
- Baseline snapshot with chi-squared drift detection
- Contradiction detection at governance gate
- Hash-chain mutation audit log
- Per-field mutation tracking, semantic belief drift detection
- Temporal causal dependency graph
- Coding-native memory schemas (ADR / CODE / PERF / ALGO / BUG)
- Auto contradiction resolution with preference learning
- Governance benchmark suite, AES-256 encryption at rest
- LLM-free multi-query expansion with RRF fusion
- 4-layer search deduplication
- Semantic-aware smart chunking
- Compiled truth pages with evidence trails
- Dream cycle (autonomous memory enrichment / repair)
- Calibration feedback loop with Bayesian weight computation
- Graph traversal tool
- Block compaction and stale block detection

## Upcoming (v3.2 / v3.3 / v4.0)

Tracked in open GitHub issues and in the root `ROADMAP.md`.
Direction:

- MCP tool-surface reduction (57 → ~20 as a stable public surface, with
  the rest moving behind a `*/advanced` namespace)
- Multi-tenancy foundation (orgs / users / RBAC)
- Real-time workspace watching (inotify / FSEvents)
- Web UI for memory browsing
- REST API server mode (distinct from MCP)
- Distributed workspace sync (mDNS-discovered mesh)
- LoRA retrain loop wired into production pipeline

## v3.9.0 candidates — Inbox / Auto-Consolidate / Extended HTTP API

> **Note (2026-05-01):** these were originally drafted as v3.7.0
> candidates, but v3.7.0 shipped as the external-audit response
> (HTTP/REST fail-closed auth, cross-platform rollback fix, CI
> hardening — see CHANGELOG.md). v3.8.0 is now the Model Safety
> Audit + Social Ingestion minor (see `ROADMAP.md` root); the
> Always-On Memory Agent patterns below are deferred to v3.9.0
> candidates and live alongside the walkthrough/persona theme below.

Inspired by Google Cloud's Always-On Memory Agent reference architecture
(MIT, `GoogleCloudPlatform/generative-ai/gemini/agents/always-on-memory-agent`).
Their reference validates the category but lacks governance, hybrid search,
multi-LLM backends, and MCP compatibility — features mind-mem already has.
Three patterns from their design are worth adopting on top of mind-mem's
production-grade foundation:

### 1. Inbox folder ingestion

Drop any file into `./inbox/`, mind-mem detects, classifies by extension,
routes to the right ingestion path:

- text (`.txt`, `.md`, `.json`, `.csv`, `.log`, `.xml`, `.yaml`) →
  markdown block (existing path)
- image (`.png`, `.jpg`, `.gif`, `.webp`) → ImageBlock
  (existing `multi_modal.py` schema; embedding via optional CLIP / SigLIP)
- audio (`.mp3`, `.wav`, `.flac`, `.m4a`) → AudioBlock with transcript
  (existing `multi_modal.py` schema; transcription via optional Whisper)
- documents (`.pdf`) → text-extract → markdown block
  (via optional `pypdf` / `pdfplumber`)

CLI: `mm watch ./inbox/`. Extends existing `watcher.py` (currently `.md`-only)
with multi-format routing. Heavy dependencies (CLIP / Whisper / pdf-extract)
stay optional behind extras: `pip install mind-mem[multimodal]`.

Solves the "how do I get content into mind-mem" friction for non-technical
users. Files = universal interface, no API knowledge needed.

### 2. Auto-scheduled dream cycle

mind-mem already has `dream_cycle.py` (consolidation logic) and
`cron_runner.py` (scheduler). Missing: default-on automatic schedule.

Add:

- `mm config set dream_cycle.auto_interval_seconds 1800` (default off,
  enable per-deployment)
- Background daemon mode: `mm daemon` runs `cron_runner` internally on
  a thread, no external cron required
- Documented in README as the "set it and forget it" mode

"Drift prevention" is a core mind-mem promise — but only if dream cycle
actually runs. Most users never set up cron, so dream cycle never fires.
One config flag flips the default.

### 3. Extended HTTP API surface

mind-mem already has `/ingest` HTTP endpoint (`ingestion_pipeline.py`).
Missing endpoints:

- `GET /status` — health, memory count, last-scan timestamp,
  dream-cycle last-run timestamp
- `POST /query` — natural-language search (wraps `recall` / `hybrid_search`)
- `GET /memories` — list/browse with filtering (by category, age, axis)
- `POST /consolidate` — trigger dream cycle on demand
- `DELETE /memories/{id}` — remove specific memory
- `POST /clear` — wipe workspace (governance-protected, requires
  rationale per v3.6.x mandatory rationale binding)

MCP is great for AI agents; HTTP is required for non-MCP integrations
(Slack bots, dashboards, web apps, monitoring tools, Streamlit/Gradio
front-ends built by users). Most endpoints are 5-line wrappers around
existing functions.

### What we explicitly do NOT borrow

- Streamlit dashboard. Adding Streamlit as a core dependency violates the
  "zero core dependencies" badge that's part of mind-mem's positioning.
  If a dashboard is wanted, ship as a separate package
  (`mind-mem-dashboard`) or document how users build one with the new
  HTTP API.
- Gemini hardcoding. mind-mem stays embedding-model agnostic and
  multi-LLM compatible.
- SQLite-only backend. We keep markdown + Postgres + hybrid search.

### Estimated effort

1-2 weeks for one engineer. All three play together: a deployment using
all three becomes "drop a file in inbox → ingested → consolidated
automatically → queryable via HTTP" — the Google reference architecture
pitch with mind-mem's governance, hybrid search, and audit chain
underneath.

## v3.9.0 candidates — Dependency-ordered walkthrough / Persona-aware recall

Second v3.9.0 theme (the first is the Inbox / Auto-Consolidate / HTTP-API
trio above). Both themes are additive and could ship together as v3.9.0
or be split across v3.9.0 / v3.10.0.

Inspired by `Lum1104/Understand-Anything` (MIT, ~10k stars). Their distinctive
value is the *human browsing UX* — guided dependency-ordered walkthroughs and
persona-adaptive detail. mind-mem already has graph traversal, hybrid search,
and compiled-truth pages, so the dashboard / multi-agent extraction pipeline /
JSON-in-repo convention don't fit. Two ideas do.

### 1. Dependency-ordered walkthrough

Today `recall` and `hybrid_search` return blocks ranked by relevance. For
"explain the state of project X" queries, an agent gets a flat result list and
has to assemble its own learning order. Add:

- New MCP tool: `compile_truth_walkthrough(topic, depth=auto)` — returns blocks
  in dependency order (foundations → derived → current state) by walking the
  existing co-retrieval graph (`graph_traversal_tool`) and topo-sorting.
- Returns a sequence, not a set: `[{step: 1, block_id: ..., role: "foundation"},
  {step: 2, block_id: ..., role: "context"}, ...]`.
- Falls back to relevance order if the topic has no graph structure (single
  isolated block, no connections).

Distinct from existing `compile_truth` (which produces one consolidated page)
and existing `graph_traversal_tool` (which returns neighborhoods, not ordered
sequences). Builds on both.

### 2. Persona-aware detail level

Same recall, different summary granularity per caller:

- `recall(query, persona="brief")` — 1-line summaries per block, IDs only
- `recall(query, persona="detailed")` — current default, full block content
- `recall(query, persona="technical")` — full content + axis scores +
  governance state + provenance hash chain

Implemented as a post-recall projection layer over the existing block format,
not a separate index. Zero index cost. Useful when one agent (e.g. routing
layer) wants a 1-line answer and another (e.g. audit / governance check) wants
the full evidence trail.

### What we explicitly do NOT borrow

- Web dashboard. mind-mem stays headless; if a dashboard is wanted, ship as a
  separate package on top of the future Extended HTTP API (see Theme A above).
- Multi-agent code-extraction pipeline. mind-mem already has its own ingestion;
  we don't re-extract from source repos.
- `.understand-anything/` JSON-in-repo convention. mind-mem's per-workspace
  model is the right unit, not per-repo committed metadata.
- Persona-adaptive UI. UI is out of scope; persona affects API output shape
  only.

### Estimated effort

3-5 days for one engineer. Walkthrough is the heavier lift (topo-sort + cycle
handling on the co-retrieval graph). Persona is a 1-day projection wrapper.
Both are additive — no breaking change to existing `recall` / `hybrid_search`
contracts.

### Cross-MIND-ecosystem consumers

The walkthrough and persona primitives are designed for cross-repo consumption,
not just mind-mem-internal use. Once shipped, the following MIND-ecosystem
projects can adopt without parallel implementation:

- **Naestro** — primary consumer. Spec'd as `R5 - Naestro Lens` in
  `~/naestro/ROADMAP.md`. Surfaces walkthroughs through CLI (`naestro lens`),
  Cockpit panel, Telegram channel, and governance evidence chain. Gated on
  this v3.9.0 work.
- **512-mind** — module-level walkthroughs over the 38-module governance
  kernel (DOS / CVS / ICL / Five Anchors / payment_rail) for new contributors
  and external auditors. Optional, opt-in.
- **mind-inference** — pipeline walkthroughs over the LLM-inference DAG for
  operators tuning quantization, batching, or backend selection. Optional.
- **rfn-mind / MindLLM** — research-artifact walkthroughs once the Phase
  truthfulness work (audit findings F1-F14) is closed; not blocked on that
  work, but only useful once the codebase is buildable.
- **mind-fleet** — swarm-orchestration walkthroughs as governance evidence
  when human operators review autonomous fleet decisions.

mind-mem ships the *primitives* (MCP tools); each consumer ships its own
*surface* (CLI flag, panel, channel command). No per-consumer code in mind-mem.
Cross-repo adoption happens via the existing MCP integration list — no new
transport, no new auth, no new convention.

## v4.0 candidates — Networked Mesh + Parallel Pipelines

mind-mem today is single-process and single-host. Two adjacent surfaces have
been deferred long enough: cross-machine sync (already stubbed in
`memory_mesh.py` v2.6.0 as a transport-less core) and intra-process parallelism
(ingestion, recall fan-out, dream cycle all single-threaded today). v4.0 lights
up the transport layer behind `memory_mesh.py`, formalizes peer-aware
governance and audit-chain reconciliation, and parallelizes the three hottest
pipelines. Local-first, zero-core-deps, and the v3.x MCP/HTTP contracts all
stay unchanged — peering is opt-in. This is v4.0 not v3.x because it's the
first cross-process and cross-host surface in the codebase: the blast radius
of a mistake (audit-chain forks, governance race conditions, dream cycles
fighting each other) justifies a major bump.

### 1. Peer transport + federated workspaces (Phase 1)

`memory_mesh.py` already ships peer registry, scope tracking, and
conflict-resolution primitives — but no wire. v4.0 phase 1 wires it with
**HTTP/2 over mTLS**, extending the v3.7 HTTP server rather than introducing
gRPC. Rationale: gRPC pulls in protobuf + a code-gen step and breaks the
zero-core-deps story; HTTP/2 streams (via `httpx[http2]` as an optional extra)
cover request/response, server-sent events for change streams, and reuse the
v3.7 token auth and route conventions. WebSocket and QUIC are deferred —
HTTP/2 is sufficient up to ~hundreds of peers on a LAN.

**Data model: federated, not replicated.** Each peer owns its workspace; no
peer holds a full mirror of any other. A query against peer A may fan out to
peers B/C/D, results merged with the existing RRF fusion (`recall.py`).
Mutations are local-only by default; cross-peer mutations require an explicit
`mm replicate <block-id> --to <peer>` action that lands on the target peer as
a *governance proposal*, not a direct write. This keeps `governance.py`'s
proposal/apply contract intact across machines: the receiving peer's human
(or auto-resolver) decides whether to apply, and the audit-chain entry on peer
B lists peer A as the agent and includes the original rationale verbatim.

**Discovery: both.** mDNS for zero-config LAN discovery (the v3.x roadmap
entry on line 124), plus a static `peers.json` for explicit declaration in
CI/cluster setups where mDNS is blocked. Peers authenticate with **per-peer
Ed25519 keypairs** generated on `mm peer init`, exchanged on `mm peer add
<addr> --pin <fingerprint>` (TOFU model with explicit pinning, no CA
infrastructure). Tokens are derived from the keypair, so the v3.7 token-auth
code path still works on the wire — the keys just rotate it.

**Topology: hub-and-spoke for v4.0 phase 1, full-mesh deferred to v4.1.**
Recommended because (a) audit-chain reconciliation across N peers is O(N²)
edges with full-mesh and O(N) with a hub; (b) governance proposals are easier
to reason about with one canonical authority; (c) most real deployments (a
dev's laptops + a workstation, or a 3-node team setup) map naturally to one
hub. Full-mesh adds CRDT-grade conflict math we don't need to ship at v4.0.
The hub is just another peer with `role: hub` set — no special binary.

**Failure modes.** Network partition: each peer keeps writing locally, change
streams buffer, sync resumes on reconnect with vector-clock comparison from
`memory_mesh.py`. Audit chain: each peer's `audit_chain.py` ledger remains
independently valid (genesis hash + per-peer seq); cross-peer apply events are
*anchored* — the receiving peer records `prev_hash_remote` alongside its own
`prev_hash`, so `chain.verify()` validates locally and a new
`chain.verify_federation()` validates the join points. Governance conflicts
(peer A and peer B both propose contradictory updates to a replicated block)
surface through the existing `contradiction_detector.py` once they meet at
the hub. **Open question:** does v4.0 ship CRDT block bodies (so concurrent
edits auto-merge) or stick with last-writer-wins + contradiction surfacing?
Recommend the latter for v4.0; CRDT block bodies are a v4.2 stretch.

### 2. Parallel pipelines (ingestion, recall fan-out, extraction)

"Parallel pipelines" is three distinct wins. v4.0 picks the three with the
highest user-visible payoff and leaves the rest serial.

**Ingestion fan-out.** `watcher.py` today processes one event per loop
iteration; `ingestion_pipeline.py` has a bounded queue but a single drainer.
Move the drainer to a `concurrent.futures.ThreadPoolExecutor` with N workers
(default `min(8, os.cpu_count())`), keyed by block-id so two events touching
the same block stay ordered. Extraction (the heaviest step) is I/O-bound when
using a remote LLM and CPU-bound when using local Ollama — the threadpool
covers both. Measured win on a 500-file inbox dump: ~6× on `mind-mem.json`'s
`backend: ollama` config, ~10× on `backend: anthropic`. Audit chain stays
consistent because `mind_filelock.py` already serializes writes to the chain
JSONL.

**Recall fan-out — local.** `recall.py` runs BM25, then vector, then graph,
then RRF — sequentially. The three signals are independent. Run them on three
threads, await all, then RRF. Sub-100ms recall today drops to ~40-50ms on
warm cache; the win is bigger on cold cache where vector lookup dominates.
Zero API change — the `recall()` signature is identical, parallelism is
internal.

**Recall fan-out — federated.** Once peers exist, `recall(query,
scope="federation")` queries all reachable peers in parallel via the HTTP/2
transport, each peer runs its own (already-parallelized) local recall, results
stream back, and the calling peer RRF-merges across peers. Per-peer timeout
(default 500ms) plus partial-result returns: a slow peer doesn't block the
query, it just doesn't contribute. New MCP tool: `recall_federated(query,
peers=None, timeout_ms=500)`. Existing `recall` stays local-only by default —
opt-in is explicit.

**What stays serial.** Dream cycle (`dream_cycle.py`) — consolidation passes
have read-modify-write dependencies that aren't worth untangling at v4.0; a
single peer's dream cycle is already <30s on typical workspaces. Multi-LLM
extraction dispatch — the `extractor.py` backend selector picks one model per
block; running multiple models per block in parallel and ensembling is a
separate research project, not a v4.0 plumbing fix. **Open question:** should
the federated dream cycle run on the hub, on each peer independently, or
coordinated? Recommend each peer independently for v4.0 — federated
consolidation is a v4.1 design exercise.

### 3. MCP and CLI surface

Six new MCP tools, all under a `mm peer * / mm replicate * / mm shard *`
namespace so the public 20-tool surface (per v3.2 roadmap) stays clean —
federation tools live in `*/advanced`:

- `peer_list` — enumerate known peers with health/last-sync timestamps
- `peer_add(addr, fingerprint)` — register and key-pin a peer
- `peer_remove(id)` — drop a peer and revoke its key
- `replicate(block_id, peer_id, rationale)` — emit a cross-peer governance
  proposal
- `recall_federated(query, peers, timeout_ms)` — fan-out recall with RRF
  merge
- `chain_verify_federation()` — validate cross-peer audit-chain join points

CLI mirrors: `mm peer list/add/remove`, `mm replicate`, `mm recall
--federation`, `mm chain verify --federation`. New module `peer_sync.py` (new
module) holds the HTTP/2 client + key management; `memory_mesh.py` stays the
conflict-resolution core; `audit_chain.py` gets a `verify_federation()` method
but no breaking change to existing API. The v3.7 HTTP server gains `/peer/*`
routes mounted under the same auth.

**Open question:** does the existing `namespaces.py` agent-namespace ACL
extend to peers (peer-as-agent), or does peer-trust live in a separate ACL?
Recommend peer-as-agent for v4.0 — the ACL grammar already handles wildcards
and it keeps one mental model.

### What we explicitly do NOT borrow / What is out of scope at v4.0

- **Strong consistency / Raft / Paxos.** v4.0 is eventual consistency with
  vector-clock LWW + governance-gated merges for cold tiers (per
  `memory_mesh.py` v2.6.0 design). If users need linearizable writes they
  should run a single peer.
- **CRDT block bodies.** Concurrent edits to the same block surface as
  contradictions, not auto-merges. v4.2 candidate.
- **Multi-region / WAN.** LAN and small-cluster only. The HTTP/2 + mTLS
  design works over WAN but we don't tune timeouts, retries, or compression
  for high-latency links at v4.0.
- **OIDC / SSO / SAML.** Auth stays per-peer Ed25519 keypairs + tokens.
  Enterprise SSO is a v4.x extra package.
- **GPU coordination / model serving.** mind-mem isn't a model serving
  framework. Each peer runs its own `mind-mem-4b` (or remote LLM) — no
  cross-peer model sharing, no central inference coordinator. Parallel
  extraction means parallel local calls, not distributed inference.
- **Full-mesh topology.** Phase 1 is hub-and-spoke. Full-mesh requires CRDT
  or consensus protocols we're not shipping at v4.0.
- **Dream cycle coordination across peers.** Each peer dreams independently.
  Federated consolidation is v4.1.
- **Parallel dispatch within `extractor.py` (multi-LLM ensemble per block).**
  Out of scope; one model per block stays the contract.

### Estimated effort

**10-14 weeks for one strong engineer**, broken by phase:

- **Phase 1 (4-5 weeks):** HTTP/2 transport + mTLS + Ed25519 key management
  on top of `memory_mesh.py`; `peer_sync.py` new module; mDNS discovery via
  `zeroconf` extra; six new MCP tools; CLI surface.
- **Phase 2 (3-4 weeks):** federated recall with RRF merge + per-peer
  timeout/partial-result handling; cross-peer governance proposal flow;
  `audit_chain.verify_federation()`; full integration tests with 3-node
  hub-and-spoke fixture.
- **Phase 3 (3-5 weeks):** intra-process parallelism (ingestion threadpool,
  parallel BM25/vector/graph in `recall.py`); benchmarks proving the wins;
  documentation; the `mind-mem[mesh]` extra packaging.

Distributed systems testing dominates the back half — Jepsen-style partition
tests, audit-chain fuzzing across forks, and governance race conditions need
real coverage before v4.0 ships. Budget another 2-3 weeks of stabilization on
top of the 10-14 if shipping to production users on day one is the goal. The
intra-process parallelism (Phase 3) is independent of the network work and
could ship as v3.9 if v4.0 slips — recommend keeping them coupled so the
federated recall path benefits from local parallel recall on day one.

## Design Principles

1. **Zero dependencies** — No external services required for the core
2. **Auditable** — Every mutation goes through the proposal system
3. **Fast** — Sub-500ms recall on commodity hardware
4. **Portable** — Plain Markdown files, any filesystem
5. **Extensible** — MIND kernels for custom scoring

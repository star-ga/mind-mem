# Roadmap

> This is the short-form roadmap. The canonical, detailed version lives
> in [`../ROADMAP.md`](../ROADMAP.md) at the repo root and includes the
> full milestone breakdown.

## v3.1.7 (Current — released 2026-04-18)

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

## v3.7.0 candidates — Inbox / Auto-Consolidate / Extended HTTP API

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

## Design Principles

1. **Zero dependencies** — No external services required for the core
2. **Auditable** — Every mutation goes through the proposal system
3. **Fast** — Sub-500ms recall on commodity hardware
4. **Portable** — Plain Markdown files, any filesystem
5. **Extensible** — MIND kernels for custom scoring

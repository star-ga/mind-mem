# MIND-Mem — Persistent AI Memory System

## Overview
BM25F + vector hybrid search memory system for AI agents.
Published on PyPI: `pip install mind-mem`

**v4.0.17** (released 2026-06-05) — security-hardening companion to v4.0.16
(same audit). At-rest encryption relabeled honestly (256-bit HMAC-SHA256
keystream + encrypt-then-MAC, **not** AES-256/SQLCipher; recall index not
encrypted) + 256 MiB encrypt/decrypt DoS cap + `.mind-mem-keys` 0700 / salt
0600 / decrypt-audit-jsonl 0600. `verify_model` warns on in-tree-key
(self-describing) verification + manifest skips sidecars only at root (no
smuggling). MIC-b parser bounds `n_syms`/`n_types`. gRPC binds 127.0.0.1 by
default (was `[::]`). `tenant_kms` (real AESGCM) unchanged.

**v4.0.16** (released 2026-06-05) — correctness-hardening release from a
six-agent audit. Postgres backend: `delete_block` silent-rollback (missing
`deleted_blocks` table aborted the txn), `diff()` AND/OR precedence + `active`
metadata leak, `restore()` `file_path` wipe, new `ping()` health probe surfaced
in `mm doctor`. Replication: `ReplicatedPostgresBlockStore` now forwards
`embedding` and implements `hybrid_search`/`backfill_embedding`/`ping`. Recall:
FTS5 `bm25()` weight shift (`block_id` stole `statement`'s weight) + unified
`_apply_post_filters()` so sqlite/vector paths no longer drop
`lifecycle`/`event_id`/`min_maturity`. Governance/concurrency:
`federation.resolve_conflict` rowcount guard, `apply_proposal` re-validate
under lock (double-apply), `conflict_resolver` winner/loser hash mapping,
`ConnectionManager.close()` write-lock, tier-schema ALTER race. MCP: DB errors
become structured responses instead of crashing the stdio server. 5465 tests
(live PG). No `mind-mem-4b` retraining. Builds on **v4.0.15** (MindLLM backend +
token rotation + time-bounded recall + N-08/T-007).

**v4.0.12** (released 2026-05-19) — `build_index` perf fix (#530:
55.9 s → 0.19 s by bounding `extract_facts` scan text) + Windows CI
green (cross-platform `ConnectionManager` teardown in the recursion
regression test). Builds on **v4.0.11** (security code-scanning
alerts #181–#189: CRLF log-sanitization + audited nosec), **v4.0.10**
(recall↔query_index recursion fix, 100 KB block_parser truncation
fix, observability/logging crash-safety, telemetry kill-switch,
query-expansion synonym narrowing), and **v4.0.9** — Predicate.register()
runtime API + CI matrix fully green across 12 OS×Python-version rows.

* **Added**: `Predicate.register(name)` on `knowledge_graph` — returns
  a `_RuntimePredicate` sentinel (str subclass with `.name`/`.value`
  enum-shaped attrs) so downstream adapters can extend the predicate
  set without forking the closed enum.
* **Fixed**: live LLM HTTP calls in two test files
  (`test_cross_encoder_auto_enable`, `test_query_expansion_auto_enable`)
  that hung Windows CI; circuit-breaker timing flake on Windows
  (recovery 0.05s → 0.1s; sleep 0.06s → 0.15s — margin > Windows
  15.6ms tick); mypy `_RuntimePredicate` slot annotations.
* **CI infra**: `--cov` on ubuntu-3.12 only (other matrix rows skip
  instrumentation to fit GitHub's 7 GB budget); `pytest-timeout=120s
  --timeout-method=thread` surfaces hangs by name; OOM-prone
  concurrency files + slow `build_index` regression (refs #530)
  marked file-level `pytestmark = pytest.mark.stress`. CI run id +
  job count are recorded in CHANGELOG.md per release, not pinned here
  (they go stale on every flake — see `feedback_ci_green_claim_discipline`).

Builds on **v4.0.8** (released 2026-05-14) — closes 4 open issues. **#526** ACL
`_get_request_scope` fail-closed: introspection exceptions return
`"deny"` sentinel (was silent `None`→`"user"` downgrade); decorator
short-circuits to reject the call; `acl_introspection_failed` log +
`mcp_acl_introspection_failed_total` metric (Critical). **#527**
THREE_WAY_MERGE upserts winner_version into `block_tier_vclock` for
the synthetic merge agent AND both fork agents — resolved conflicts
no longer recur (Critical, functional). **#528** every three-way
merge emits a `three_way_merge_resolved` log with SHA-256 hashes of
left/right/merged payloads for operator audit (Critical, HTTP
transport). **#529** FederationClient hardened — scheme allowlist
(rejects `file://`/`ftp://`/etc.), same-origin redirect handler
(blocks SSRF pivot to cloud metadata), `MAX_RESP_BYTES` 1 MiB
response-size cap (High). 18 new regression tests in
`tests/test_issue_52[679]_*.py` + 75 existing pass. Plus 5 missing
file-level `pytestmark = pytest.mark.stress` markers added to OOM-
prone concurrency files (`test_concurrency_stress.py`,
`test_filelock_stress.py`, `test_v4_concurrency.py`,
`test_v4_round4_concurrency.py`, `test_concurrent_integration.py`).
Builds on **v4.0.7** (released 2026-05-14) — test-only fix: the post-#508 ACL
hardening (defence-in-depth) gates every decorator call against
`ADMIN_TOOLS ∪ USER_TOOLS` even with `MIND_MEM_ACL_DISABLED=true`,
so `tests/test_mcp_v140.py::TestObservabilityDecorator::test_failure_increments_failure_counter`'s
ad-hoc `failing_tool` was rejected before its body ran (the test
expected `ValueError` to propagate). Patched the test to register
`failing_tool` in `USER_TOOLS` for the test scope (both
`mind_mem.mcp.infra.acl.USER_TOOLS` and the import-time binding in
`mind_mem.mcp.infra.observability`). Decorator unchanged — the
defence-in-depth behaviour is correct.
Builds on **v4.0.6** (released 2026-05-14) — PyPI badge alignment + CI green.
README badge block was rendering uncentred on PyPI because lines
9–26 used 2/4-space leading indentation; PyPI's strict CommonMark
treats 4-space indented lines as a code block. Flushed left. Also:
applied `ruff format` to 10 drifted files (closes the `lint`
Format-check failure); added `-m "not stress"` to both CI pytest
steps (closes the OOM kills on ubuntu 3.12/3.14 from `test_niah`-
class stress tests); marked Python 3.14 matrix rows as
`continue-on-error` (still pre-release). No source/test changes.
Builds on **v4.0.5** (released 2026-05-14) — Docs/badges aligned + release
workflow idempotent. README badges + comparison table now match
ground truth (`tests-5155+`, `clients-18`, `audit-10-LLM`, `84` MCP
tools); CLAUDE.md drift cleared (`MCP Tools (81) → (84)`,
`16 → 15` clients). `.github/workflows/release.yml` `pypa/gh-action-pypi-publish`
step passes `skip-existing: true` so tag re-pushes and local-twine
races stop turning the Release badge red. Same wheel surface as
v4.0.3, no code/test changes.
Builds on **v4.0.4** (released 2026-05-14) — docs-only patch: PyPI README logo
now resolves (relative `assets/logo.png` rewritten to absolute GitHub
raw URL — PyPI does not resolve relative paths against the source
repo). Same wheel surface as v4.0.3. No code/test changes.
Builds on **v4.0.3** (released 2026-05-14) — Postgres-backed recall pipeline fix.
`recall()` in `_recall_core.py` now dispatches to the configured backend
at the library entry-point (previously only `python -m mind_mem.recall`
honored it, so `mm recall` against a PG workspace returned `[]`).
`_load_backend` now tolerates non-dict `recall` config (falls through
to BM25 scan instead of crashing). `mm doctor --rebuild-cache` creates
the SQLite `recall.db`, runs `_init_schema`, and populates the FTS5
`blocks_fts` virtual table on first run against a PG-backed workspace.
Closes #524 + #525. Targeted recall/doctor/rebuild/error_paths/
sqlite_index suite: 636 passed, 6 skipped, 0 failed. `mind-mem-4b`
weights unchanged — CLI/library fix only, zero retraining required.
Builds on **v4.0.2** (released 2026-05-13) — Security + correctness audit pass
over the v4.0.1 surface: 1 Critical / 12 High / 18 Medium / 12 Low / 3
Info findings closed. HMAC-equal token compare, Origin allowlist, OPTIONS
rejection, per-client sliding-window rate limit, symlink TOCTOU close,
arch-mind flag-injection guard, `--token` CLI removal, LWW wall-clock
semantic, BEGIN IMMEDIATE rowid pin, `±Inf`/`None` preimage rejection,
RRF date-freshness dedup, copy-on-write temporal decay, lru-cached
query-type detection. `mind-mem-4b` weights unchanged — zero probe-surface
overlap.
Builds on **v4.0.1** (2026-05-11) — federation wire transport
(`/federation/{vclock,write,resolve,conflicts}` over HTTP + stdlib
`FederationClient`).
Builds on **v4.0.0** (released 2026-05-10) — Cognitive kernel, knowledge
graph, resilience suite, observability layer. All surfaces flag-gated
under `v4.<flag>` in `mind-mem.json`. No breaking changes. `mind-mem-4b`
retrained at **109/109 = 100%** on the un-softened harness (14 new
`V4_SURFACES` probes; `qg.escape_hatch` and `lin.cites=0.8` gaps from
v3.12.1 confirmed fixed). 376 v4 unit + 38 concurrency + 22 held-out
paraphrase tests. Architecture audited at unanimous **10/10** across 4
LLMs (Grok 4.3, DeepSeek v4-pro, Mistral large, GLM-5).
Builds on **v3.12.1** (2026-05-10) — patched eval, model card honesty.
Builds on **v3.12.0** (2026-05-09) — strict quality gate, lineage→staleness.
Builds on **v3.11.0** (2026-05-08) — typed lineage edges, recall explainability.
Builds on **v3.9.0** — 4000+ tests, native MCP for 17 AI clients, **84 tools**,
Postgres+pgvector, full-fine-tune local model, at-rest encryption, tier
decay, governance alerting, MIC/MAP wire format, HTTP transport, daemon,
inbox ingestion, pipeline hash, persona projection, Kahn walkthrough.

## Architecture
```
src/mind_mem/           — Main package (src layout)
  core/                 — BlockStore, ConnectionManager, retrieval
  governance/           — Contradiction detection, drift, proposals,
                          audit chain, alerting hooks
  mcp_server.py         — MCP server (84 tools, 8 resources)
  ingestion/            — Auto-ingestion pipeline
  skill_opt/            — Skill optimization
  hook_installer/       — Client hook installation (v3.1.1+)
  http_transport.py     — Stdlib HTTP REST adapter (v3.9.0)
  daemon.py             — Background dream-cycle/intel-scan loop (v3.9.0)
  inbox.py              — File-drop folder ingestion (v3.9.0)
  pipeline_hash.py      — Hash-of-code invalidation (v3.9.0)
  personas.py           — brief/detailed/technical projection (v3.9.0)
  walkthrough.py        — Dependency-ordered learning sequence (v3.9.0)
  quality_gate.py       — Deterministic quality validation (v3.11.0)
  block_lineage.py      — Typed relationship edges (v3.11.0)
  lineage_staleness.py  — BFS staleness propagation (v3.12.0)
  — v4.0.0 surfaces (all flag-gated) —
  tier_memory.py        — Hot/warm/cold block tiers + CAS (v4.0.0)
  cognitive_kernel.py   — KernelKind enum + mind_recall (v4.0.0)
  surprise_retrieval.py — compute_surprise + FallbackPolicy (v4.0.0)
  block_kinds.py        — Multi-label block_kind_tags table (v4.0.0)
  block_metadata.py     — Tag + TTL + schema validators (v4.0.0)
  kind_summaries.py     — Per-kind global summaries (v4.0.0)
  embedding_pipeline.py — Pluggable embedder + 3-gram default (v4.0.0)
  consolidation_worker.py — plan_consolidation pure fn (v4.0.0)
  eviction.py           — LRU/LOW_SURPRISE/AGE/COMPOSITE (v4.0.0)
  federation.py         — VClock + conflict log + MergeStrategy (v4.0.0)
  self_editing.py       — block_edits + propose/approve/reject (v4.0.0)
  pq.py                 — Product Quantization M=32 K=256 (v4.0.0)
  hnsw_kind_index.py    — sqlite-vec HNSW + brute-force fallback (v4.0.0)
  circuit_breaker.py    — CircuitBreaker + @circuit_breaker (v4.0.0)
  backpressure.py       — BackpressureController + hysteresis (v4.0.0)
  health.py             — health_check + 7 probes + register (v4.0.0)
  observability.py      — counter/gauge/histogram + @timed (v4.0.0)
  logging_context.py    — contextvar stack + StructuredLogFilter (v4.0.0)
  feature_flags.py      — 35 flags + is_enabled/require_enabled (v4.0.0)
tests/                  — pytest suite (4000+ tests + 376 v4 unit
                          + 38 concurrency + 22 paraphrase probes)
mind/                   — MIND scoring kernels (.mind)
docs/                   — User + integration docs (35+ files)
```

### Key Components
- **BM25F retrieval** with Porter stemming + RM3 query expansion
- **Hybrid search**: BM25 + vector search with RRF fusion (sqlite-vec
  on the SQLite backend; pgvector + HNSW on the Postgres backend)
- **Backends**: SQLite (default, zero-deps) or Postgres (pgvector +
  HNSW + GIN, with replicated read/write routing in v3.9)
- **A-MEM blocks**: metadata evolution (importance, access, keywords)
- **9-type intent router** with adaptive confidence weights
- **Cross-encoder reranking** (opt-in, config-gated)
- **Governance engine**: contradiction detection, drift, proposal queue,
  alerting hooks (webhook / Slack)
- **ConnectionManager**: Thread-safe SQLite pool with WAL read/write
  separation
- **Delta-based snapshot rollback**: MANIFEST.json for O(manifest)
  restore
- **At-rest encryption** (v3.0.0+): SQLCipher + BlockStore ciphertext
- **Tier decay** (v3.0.0+): TTL/LRU aging for lower tiers
- **Audit-integrity patterns** (v2.10.0+): Q16.16 fixed-point scoring in
  audit hash preimages, TAG_v1 NUL-separated composition for collision
  resistance
- **Local model** (v4.0.0): `star-ga/mind-mem-4b` — full fine-tune of
  Qwen3.5-4B retrained for v4.0.0 (v4 weights revision). Knows all 84
  tools, v4 surfaces (cognitive kernel, block kinds, tier memory,
  self-editing), and the corrected `KIND_DECAY['cites']=0.8` value.
  Prior v3.12.0-fullft weights pinned at `v3.12.0` HF revision. Prior
  v3.0.0 QLoRA at `v3.0.0`. See `docs/mind-mem-4b-setup.md`.
- **Native MCP integration** (v3.1.0+): 15 AI clients auto-wired via
  `mm install-all` (Claude Code, Claude Desktop, Codex CLI, Gemini CLI,
  Cursor, Windsurf, Zed, OpenClaw, and 8 more). See
  `docs/client-integrations.md`.

### MCP Tools (84)
Grouped surfaces (full list in `docs/api-reference.md` and
`src/mind_mem/mcp_server.py`):
recall, hybrid_search, prefetch, propose_update, approve_apply,
rollback_proposal, scan, list_contradictions, reindex, index_stats,
create_snapshot, list_snapshots, restore_snapshot, briefing,
category_summary, cross_encoder_rerank, find_similar, memory_evolution,
delete_memory, export_memory, import_memory, intent_classify,
retrieval_diagnostics, get_mind_kernel, list_mind_kernels,
verify_chain, audit_replay, tier_decay_apply, encrypt_status,
alerts_subscribe, mic_convert_tool, mic_inspect_tool,
compile_truth_walkthrough, recall_with_persona, pipeline_status,
reindex_dirty, and more.

## Config
- Config file: `mind-mem.json` (NOT `mem-os.json` — renamed)
- Auth header: `X-MindMem-Token` (NOT `X-MemOS-Token` — renamed)
- `governance_mode` (NOT `self_correcting_mode` — renamed)
- LLM backend: default `ollama`; switch to `openai-compatible` to point
  at a vLLM/exllamav2 endpoint running `mind-mem-4b`.

## Testing
```bash
pytest                           # full suite (4000+ tests)
pytest tests/test_retrieval.py   # specific module
pytest -x --tb=short             # stop on first failure
pytest --collect-only -q | tail  # verify test count
```

## Conventions
- Python 3.10+ with type hints everywhere
- src layout: `src/mind_mem/`
- All public functions have docstrings
- No `print()` in library code — use `logging`
- SQLite WAL mode for concurrent reads

## Benchmarks (LoCoMo, Mistral Large)
Mean: 77.9, Adversarial: 82.3, Temporal: 88.5

## Git
- Remote: star-ga/mind-mem (public)
- Author: `STARGA Inc <noreply@star.ga>`
- CI: GitHub Actions (enabled); release workflow on tag push with
  trusted PyPI publishing via OIDC (`environment: pypi`)

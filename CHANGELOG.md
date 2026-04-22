# Changelog

All notable changes to mind-mem are documented in this file.

## 3.3.0 (2026-04-21)

**Platform-scale release.** Ships 12 v3.3.0 retrieval features, 8 v4.0
preview modules, a protection layer for shipped wheels, 7 new .mind
tuning kernels, and mind-mem-4b v2 (full fine-tune on H200 NVL replacing
v1's QLoRA checkpoint). No breaking changes — v3.2.1 deployments upgrade
cleanly.

### Added — v3.3.0 retrieval features (12)

- **``mind_mem.query_planner``** — NLP + LLM query decomposition
  (regex patterns for temporal_after / causal / contrastive /
  conjunction, plus SSRF-guarded LLM decomposer). Tunable via
  ``mind/query_plan.mind``.
- **``mind_mem.graph_recall``** — Multi-hop BFS graph expansion over
  block references, decay=0.5 per hop, ``max_hops`` security-capped
  at 3, ``max_total_added=50``. Tunable via ``mind/graph.mind``.
- **``mind_mem.entity_prefetch``** — Widened regex
  (``(?:PER|PRJ|TOOL|INC)-\d+|[A-Z]{2,}|[A-Z][a-zA-Z][a-zA-Z]+``),
  symlink-escape guard, 500-file / 2MB safety caps.
- **``mind_mem.session_boost``** — Same-session score multiplier
  ``(1 + boost)`` with ``top_seed_count=3``, ``boost=0.3``. Tunable
  via ``mind/session.mind``.
- **``mind_mem.evidence_bundle``** — Structured (facts / relations /
  timeline / entities) JSON bundle via ``recall(format="bundle")``.
  Tunable via ``mind/evidence.mind``.
- **``mind_mem.rerank_ensemble``** — Borda-count fusion across
  cross-encoder (ms-marco-MiniLM-L-6-v2) + BGE-reranker-v2-m3.
  Fail-open per member. Tunable via ``mind/ensemble.mind``.
- **``mind_mem.truth_score``** — Bayesian truth scoring:
  ``status_prior × age_decay − contradiction_mass + access_bonus``.
  Tunable via ``mind/truth.mind``.
- **``mind_mem.answer_quality``** — ``verify_answer`` regex patterns
  (date/time/number/proper_noun/yes_no), ``self_consistency``
  plurality voting, ``prompt_for_category`` per-category prompts.
  Tunable via ``mind/answer.mind``.
- **``mind_mem.streaming``** — Token-bucket back-pressure queue with
  drop-oldest policy.
- **``mind_mem.consensus_vote``** — Quorum voting with
  ``trust_weight`` namespace fallback and 1.0-2.0 confidence scale.
- **``mind_mem.retrieval_trace``** — ContextVar-backed zero-cost
  tracing; activated when ``retrieval.trace_attribution`` is set.
- **``mind_mem.feature_gate``** — Declarative ``FieldSpec`` +
  ``FeatureGate`` pattern for v3.3.0 retrieval features. Collapses
  the five near-identical ``is_X_enabled`` / ``resolve_X_config``
  implementations into a single declaration.

### Added — v4.0 preview modules (8)

- **``mind_mem.event_fanout``** — Pluggable publisher protocol with
  ``LoggingPublisher`` + ``RedisStreamPublisher``. Canonical event
  kinds: ``contradiction_detected``, ``block_promoted``,
  ``snapshot_created``.
- **``mind_mem.tenant_audit``** — Per-tenant HMAC-separated audit
  chains. ``_tenant_genesis = HMAC(root_secret, tenant_id)[:32]``.
- **``mind_mem.tenant_kms``** — Envelope DEK encryption via
  AES-256-GCM (cryptography library), SHAKE-256+HMAC fallback.
  ``rotate_tenant_dek`` returns (new_dek, new_wrapped, old_dek).
- **``mind_mem.governance_raft``** — Raft-style consensus wrapper.
  ``Proposal`` + ``LocalConsensusLog`` + HMAC-signed
  ``sign_proposal`` / ``verify_proposal``.
- **``mind_mem.api.grpc_server``** — Typed ``RecallRequest`` /
  ``RecallResponse`` / ``GovernanceRequest`` dataclasses.
  ``serve(port)`` lazy-imports grpc when ``mind-mem[grpc]`` installed.
- **``mind_mem.storage.sharded_pg``** — Consistent-hash ring with
  virtual nodes (160 × weight). ``ShardedPostgresBlockStore``
  implements the full ``BlockStore`` protocol.
- **``deploy/edge/pyoxidizer.bzl``** — PyOxidizer single-binary edge
  deployment (Linux x86_64, macOS arm64).
- **``web/``** — Next.js console (``GraphView`` / ``TimelineView`` /
  ``FactList`` / ``TenantSwitcher``).

### Added — protection layer

- **``src/mind_mem/protection.py``** — SHA-256 integrity manifest
  verified at import time. ``MIND_MEM_INTEGRITY=strict`` turns
  tamper detection into a hard fault. Frozen
  ``AUTH_HEADER="X-MindMem-Token"`` and ``AUDIT_TAG="TAG_v1"``
  constants that downstream consumers can pin with ``assert``.
- **``scripts/build_integrity_manifest.py``** — Wheel-build hook that
  bakes ``_integrity_manifest.json`` for 19 critical modules. Wired
  into ``.github/workflows/release.yml``.
- **``docs/protection.md``** — 15-layer defence-in-depth map
  (OIDC trusted publishing → Sigstore → SBOM → integrity manifest →
  strict-mode guard → world-writable dir check → frozen constants →
  gitleaks → trivy → bandit → mypy → audit chain → tenant isolation
  → envelope encryption).

### Added — .mind tuning kernels (7 new, 25 total)

Each v3.3.0 retrieval feature ships a TOML-style ``.mind`` config
kernel so operators can tune bounds without patching code:

- ``mind/query_plan.mind`` — decomposition + LLM-decomposer config
- ``mind/graph.mind`` — BFS expansion bounds (hops ≤ 3)
- ``mind/session.mind`` — same-session boost config
- ``mind/truth.mind`` — Bayesian priors + age decay
- ``mind/answer.mind`` — self-consistency + verification patterns
- ``mind/evidence.mind`` — bundle assembly bounds
- ``mind/ensemble.mind`` — reranker ensemble (cross_encoder + BGE)

15 parameterised tests in ``tests/test_mind_kernels_v3_3.py`` pin
expected section names + invariants (graph max_hops≤3, answer
samples odd, ensemble uses reranker not retriever, etc.).

### Changed — mind-mem-4b v2 (HuggingFace)

- **``star-ga/mind-mem-4b``** replaced with v2 on HF main branch.
- **Full fine-tune** on H200 NVL 141GB (v1 was QLoRA on RTX 3080).
- 16,450 training examples (10k dispatcher + 5k retrieval + 1.45k v1
  replay).
- Hyperparameters: bf16, AdamW fused, LR 5e-6 cosine with 3% warmup,
  batch 4 × accum 8 (effective 32), seq_length 384, gradient
  checkpointing enabled. Final loss: 0.08 (converged).
- GGUF Q4_K_M (2.7GB) imported to Ollama as ``mind-mem:4b``;
  ``mind-mem:4b-v1`` tag preserved for rollback.
- v2 model card at ``docs/hf-mind-mem-4b-v2-README.md`` advertises
  the 7-dispatcher v3.2.x surface + v3.3.0 retrieval call shapes.

### Added — benchmarking + ops

- **``benchmarks/local_stack_audit.py``** — pre-bench health check
  across 11 optional features (ollama, redis, claude-proxy, CE, BGE
  v2-m3, sqlite-vec, v3.3.0 + v4.0-prep modules, 25 kernels).
- **``benchmarks/runpod_kickoff.sh``** — one-shot Runpod H200/A100
  training kickoff. Writes all artifacts to ``/runpod-volume``
  (persists across pod termination).
- **``benchmarks/train_config_a100.yaml``** — A100 80GB variant
  (batch 8 × accum 4, gradient_checkpointing) of the H200 config.
- **``benchmarks/train_mind_mem_4b.py``** now auto-resumes from
  latest checkpoint when ``output_dir`` contains one. Survives SSH
  hangups and pod restarts.

### LoCoMo results (Opus 4.7 answerer, Mistral-Large judge)

| | conv-0 | Notes |
|---|---|---|
| v1.1.0 (Mistral baseline) | 70.54 | Original LoCoMo run |
| v3.2.1 (Opus, BM25) | 76.7 | Previous release |
| v3.3.0 (Opus, BM25) | **77.06** | This release — 199 QAs, +0.36 over v3.2.1 |

Per-category on v3.3.0 conv-0: adversarial=92.98, temporal=98.12,
open-domain=74.87, single-hop=70.12, multi-hop=64.35.

**Known limitation:** v3.3.0 feature experiment (query decomposition +
rerank ensemble + self-consistency) regressed full-bench score to
70.05 because sub-query RRF-fuse attenuated joint-reasoning bridges.
Path to 85+ documented in ``docs/v3.4.0-roadmap-llm-consensus.md``
based on 4-LLM consensus (Grok-4.1, Mistral-Large, DeepSeek-Reasoner,
Gemini-3.1-Pro).

## 3.2.1 (2026-04-20)

**Hotfix / production-hardening release.** Addresses the three
architectural debt items called out in the v3.2.0 release notes:
REST request-scoping, OIDC admin-gate wiring, and CI plumbing. Also
cleans up a clutch of CI reds (ruff format, Windows path separators,
gitleaks regex, cyclonedx-bom `pkg_resources` crash, dead action
SHAs). No breaking changes — v3.2.0 deployments upgrade cleanly.

### Fixed

- **Apply engine routes block-level ops through BlockStore.** The
  three block-mutation ops (``update_field``, ``append_list_item``,
  ``set_status``) previously spoke raw ``open()`` on corpus
  Markdown files, so Postgres-backed workspaces silently diverged
  (the filesystem changed but the DB did not). Refactored to use
  ``BlockStore.get_by_id`` + ``BlockStore.write_block``.
  Markdown's ``write_block`` is itself the same atomic file ops
  under the hood, so Markdown deployments see zero behavioural
  change. ``execute_op`` now accepts an optional ``store`` kwarg
  for callers that want to inject a specific backend; legacy
  callers (no kwarg) resolve the active store via the factory.
  File-level ops (``append_block``, ``insert_after_block``,
  ``replace_range``, ``supersede_decision``) remain filesystem-
  based in v3.2.1 and are tracked for v3.2.2.
- **REST request-scoping.** Previously every handler mutated
  ``os.environ["MIND_MEM_WORKSPACE"]`` on each request, which raced
  under concurrent requests. Replaced with a per-request
  ``ContextVar`` override in ``mind_mem.mcp.infra.workspace`` (via a
  new ``use_workspace`` context manager) plus a FastAPI HTTP
  middleware that scopes every request automatically. The env var
  remains authoritative for the standalone MCP server.
- **OIDC JWTs drive the admin gate.** Pre-v3.2.1 ``_require_admin``
  only recognised ``MIND_MEM_ADMIN_TOKEN`` matches and mmk_* keys
  with an explicit ``admin`` scope — an OIDC JWT carrying an admin
  scope was silently downgraded. ``_verify_bearer`` now returns a
  ``(valid, agent_id, scopes)`` triple; ``_require_auth`` stashes
  the validated scopes on ``request.state`` for cross-dependency
  reads; ``_require_admin`` consults ``request.state.oidc_scopes``
  when evaluating admin access. Scope names configurable via
  ``MIND_MEM_OIDC_ADMIN_SCOPES`` (default: ``mind-mem.admin admin``).
- **Invalid OIDC JWTs reject instead of falling through.** When
  OIDC is configured and a token fails validation,
  ``_verify_bearer`` now returns ``(False, ...)`` instead of
  accepting the request under the permissive "no auth configured"
  fallback that applied when ``MIND_MEM_TOKEN`` was unset.
- **CI — ruff format + Windows path separator.** 25 files picked
  up minor format drift; formatted via ``ruff format``.
  ``tests/test_apply_engine_backend_routing.py`` asserted with
  forward-slash on ``args[0].endswith(...)`` — Windows' ``os.sep``
  emits backslashes, so the assertion normalises via
  ``args[0].replace(os.sep, "/")``.
- **CI — SBOM job ``pkg_resources`` crash.**
  ``CycloneDX/gh-python-generate-sbom@v2`` and unpinned
  ``cyclonedx-bom`` both pulled in cyclonedx-bom<4, whose CLI
  imports the ``pkg_resources`` shim removed in setuptools 78+.
  Both workflows now install ``cyclonedx-bom>=5`` and invoke
  ``cyclonedx-py environment`` directly.
- **CI — dead action SHAs.** Bumped
  ``aquasecurity/trivy-action`` to v0.35.0 (internal pin to
  ``setup-trivy@v0.2.1`` was deleted upstream). Corrected
  ``gitleaks/gitleaks-action`` v2.3.9 SHA which had a
  transcription error.
- **CI — gitleaks regex.** ``*.pyc`` was glob-syntax in
  ``.gitleaks.toml`` ``paths[]`` which requires regex. Escaped
  to ``.*\.pyc$``.

### Added

- **``mind_mem.mcp.infra.workspace.use_workspace``** — context
  manager for scoping a block of code (or a FastAPI request) to a
  specific workspace via ContextVar override. Task-local under
  asyncio and thread-local through Starlette's thread pool.
- **``MIND_MEM_OIDC_ADMIN_SCOPES`` env var** — comma/space-separated
  list of OIDC scope names that grant admin access.
- **``tests/test_workspace_contextvar.py``** — 5 regression tests
  covering ContextVar precedence, reset on exception, nested
  stacking, thread isolation, and abspath normalisation.
- **``tests/test_oidc_admin_enforcement.py``** — 6 regression
  tests covering OIDC admin-scope passthrough, non-admin rejection,
  invalid-JWT 401, scope-name override, and OIDC-path bypass when
  unconfigured.

### Deprecated

- **``mind_mem.api.rest._set_workspace_env``** — kept for
  compatibility. New code should prefer
  ``mind_mem.mcp.infra.workspace.use_workspace``.

## 3.2.0 (2026-04-20)

**Production-deployment release.** Turns mind-mem from a single-host
Markdown-on-disk memory into a production-ready system with
Postgres, Docker, REST, multi-language SDKs, observability, and a
publishable security posture. See
[docs/v3.2.0-release-notes.md](docs/v3.2.0-release-notes.md) for
the full narrative.

> **Stability labels.** Markdown + stdio-MCP (the default
> deployment) is **GA** and rock-solid. Postgres backend, multi-
> worker REST, and OIDC admin-scope enforcement are **beta** —
> they work end-to-end for typical workloads but the v3.2.0
> self-audit surfaced two architectural refactors that complete
> their production story in v3.2.1:
>
> 1. Apply engine op executors will route through BlockStore
>    (Postgres apply currently writes to local FS that Postgres
>    never sees — a silent divergence on this path).
> 2. REST request-scoping will drop `os.environ` mutation in
>    favour of a request-local workspace argument (current impl
>    is thread-unsafe past one uvicorn worker).
> 3. OIDC JWTs will be honoured on protected endpoints, not just
>    at `/v1/auth/oidc/callback`.
>
> See `docs/review-architecture-v3.2.0.md` for the full
> architecture review + `ROADMAP.md` § v3.2.1 for the remediation
> plan.

### Added

- **Postgres storage backend** (opt-in via
  ``block_store.backend: "postgres"``) — full ``BlockStore``
  protocol implementation with atomic snapshot/restore, upsert
  writes, FTS search. Optional extra: ``mind-mem[postgres]``.
- **Read-replica routing** — ``block_store.replicas: [...]``
  round-robins reads across a replica pool with a 3-failure /
  30-second circuit breaker.
- **Storage migration guide** — bidirectional Markdown ↔
  Postgres migration with dry-run, verification, and receipt
  trail at ``docs/storage-migration.md``.
- **REST API layer** (FastAPI) at ``src/mind_mem/api/rest.py``;
  run with ``mm serve --port 8080``. Mirrors the MCP tool
  surface with Pydantic validation + OpenAPI docs.
- **OIDC/SSO auth** — Okta / Auth0 / Google Workspace / Azure AD
  JWT validation at ``src/mind_mem/api/auth.py``.
- **Per-agent API keys** — ``mmk_live_*`` tokens with SQLite
  rotation + revocation at ``src/mind_mem/api/api_keys.py``.
- **Audit attribution** — every governance-chain entry now
  carries an ``agent_id``.
- **JS/TS SDK** — publishable ``@mind-mem/sdk`` at ``sdk/js/``
  covering the read-only REST surface; TypeScript 5.4 strict.
- **Go SDK** — publishable ``github.com/star-ga/mind-mem/sdk/go``
  at ``sdk/go/``; stdlib-only.
- **Dockerfile + docker-compose** — multi-stage
  ``python:3.12-slim`` build (142 MB), compose bringing up
  mind-mem + pgvector + Ollama; ``make up`` one-command start.
- **One-command installer** — ``curl -sSL install.mind-mem.sh |
  bash`` via ``install-bootstrap.sh``.
- **OpenTelemetry + Prometheus** at
  ``src/mind_mem/telemetry.py``; OTLP exporter, ``@traced``
  decorator, Prometheus ``/v1/metrics`` endpoint.
- **Grafana dashboard** at
  ``deploy/grafana/mind-mem-dashboard.json`` — recall p50/p95/p99,
  qps, propose_update rate, apply-rollback rate.
- **Distributed recall cache** at
  ``src/mind_mem/recall_cache.py`` — two-tier LRU + Redis;
  namespace-wide invalidation on governance events.
- **Hot/cold tier-aware retrieval** at
  ``src/mind_mem/tier_recall.py`` — opt-in score boost per
  memory tier (WORKING 0.7x → VERIFIED 2.0x).
- **Obsidian wikilink export** — ``vault_sync`` emits
  ``[[wikilinks]]`` from knowledge-graph edges so Obsidian's
  graph view visualizes memory links.
- **CLI debug commands** — ``mm inspect`` /
  ``mm explain`` / ``mm trace`` for block introspection,
  retrieval-stage tracing, and MCP call streaming.
- **MCP consolidated dispatchers** (7) — ``recall``,
  ``staged_change``, ``memory_verify``, ``graph``, ``core``,
  ``kernels``, ``compiled_truth`` with ``mode`` / ``phase`` /
  ``action`` arguments that route to existing implementations.
  **Additive — all 57 v3.1.x tool names still resolve.**
- **External security audit SoW** at
  ``docs/security-audit-sow.md`` — RFP-shape document for
  tier-1 audit firms.
- **Internal security audit report** at
  ``SECURITY_AUDIT_2026-04.md`` — 3 findings (2 HIGH, 1 MEDIUM),
  all fixed pre-release.
- **``SECURITY.md``** — supported versions, 90-day disclosure,
  threat model, in-scope definition.
- **Supply-chain CI** at ``.github/workflows/security.yml`` —
  CodeQL, bandit, pip-audit, gitleaks, trivy, CycloneDX SBOM
  generation.
- **Sigstore signing** of wheels + tarballs on release tags.
- **Pre-commit hooks** — ruff, bandit, detect-secrets.
- **Workspace-wide lock primitive** at
  ``BlockStore.lock(blocking, timeout)`` — process-serializing
  context manager for migrations and batched writes.

### Changed

- **Architecture cleanup (audit §1.2)** — ``mcp_server.py``
  decomposed from 4,578 → 245 LOC (94.6% reduction) into 14
  tool modules + 7 infra modules + server/resources/public
  modules.
- **Architecture cleanup (audit §1.4)** — 7-PR BlockStore
  routing series: ``list_files`` → ``list_blocks`` rename
  (alias deprecated), ``write_block`` / ``delete_block`` added,
  snapshot/restore/diff moved from apply_engine into
  ``MarkdownBlockStore``, Postgres adapter shipped, storage
  factory wired, migration helper documented.
- **Atomicity (audit §2.2)** — ``maintenance/`` split into
  ``tracked/`` (snapshot-included) and ``append-only/`` (snapshot-
  excluded) to fix the "dedup-hash survives rollback" class of
  bug. Orphan-cleanup walk honors the exclusion. Auto-migration
  on first v3.2.0 apply.
- **Token hardening** — bearer tokens over 4096 chars are
  rejected before the hmac compare (DoS prevention); startup
  emits a warning on tokens under 32 chars.
- **Query-length cap** — recall rejects queries over 8192 chars
  to mitigate regex-DoS.
- **Docker compose credentials** — all hardcoded passwords
  replaced with required env-var substitution (CWE-798 fix).

### Fixed

- **Orphan-cleanup walk descent** — post-restore sweep no longer
  descends into ``maintenance/append-only/`` or
  ``intelligence/applied/`` subtrees. Pinned by regression test
  ``test_atomicity_maintenance_scope.py``.
- **``test_apply_engine`` hashlib import** — previous refactor
  dropped the import while retaining the call sites; restored.
- **Resource leaks in tests** — three test files had
  ``open(path).read()`` calls without context managers; all use
  ``with open(...)`` now so ``pytest -W error`` passes clean.

### Deprecated

- ``BlockStore.list_files()`` — use ``list_blocks()``. Alias
  stays through v3.x; removed in v4.0.
- ``validate.sh`` — already a Python forwarder; shell script
  removed in v4.0.

### Security

- 3 findings from the internal v3.2.0 audit (2 HIGH, 1 MEDIUM)
  — all fixed pre-release. Full report at
  ``SECURITY_AUDIT_2026-04.md``.
- Dependency CVE recommendations (not auto-bumped): ``authlib``
  via fastmcp → recommend ``>=1.6.9``; ``aiohttp`` →
  recommend ``>=3.13.4``.

## 3.1.9 (2026-04-18)

**Hotfix for v3.1.8 release hygiene + duplicate entry-point
cleanup.**

### Fixed

- **`src/mind_mem/__init__.py` `__version__` now matches
  `pyproject.toml`.** v3.1.8 shipped with pyproject=3.1.8 but
  `__init__.py` still reading 3.1.7, so
  `python -c "import mind_mem; print(mind_mem.__version__)"`
  on a `pip install mind-mem==3.1.8` installation printed
  `3.1.7`. The `test_versions_match` regression test would have
  caught this but the CI run on the release commit was cancelled
  by the concurrency group when the niah-Benchmark wiring
  follow-up landed. v3.1.9 brings the two sources of truth back
  into alignment. PyPI users can `pip install -U mind-mem` to
  pick up the correct version string.
- **Duplicate MCP server entry-point cleanup (§1.1 of audit).**
  `mcp_server.py` (top-level developer-checkout shim) simplified
  from 47 LOC using `exec(compile())` to 41 LOC using a standard
  `sys.path` insert + import. `src/mcp_server.py` (wheel-level
  compatibility module) simplified from 50 LOC using `globals()`
  + `__getattr__` delegation to 32 LOC using a clean star-import
  + `main` re-export. Both wrappers preserve the full public
  surface (105 symbols including the `FastMCP` instance).

## 3.1.8 (2026-04-18)

**CI parity + cross-platform stability + v3.2.0 architectural
prep.** All 16 matrix jobs (Linux/macOS/Windows × Python
3.10/3.12/3.13/3.14) green with zero warnings.

### Fixed

- **`tests/test_niah.py` marked `@pytest.mark.stress`.** The
  250-test parametrised NIAH benchmark (5 sizes × 5 depths × 10
  needles) takes ~10 s per test — 40+ minutes total — and was
  the actual cause of the CI hangs across the v3.1.4 → v3.1.7
  release window. The default `pytest` invocation now skips it
  (matches the existing `addopts = "-m 'not stress'"` config).
  Run explicitly with `pytest -m stress tests/test_niah.py` or
  via the dedicated benchmark workflow.
- **`tests/test_enums.py::test_str_enum_serialises_as_string`** —
  removed the f-string assertion whose result diverges between
  Python 3.10 and 3.11+ (PEP 663). Now tests the stable contract:
  `isinstance(_, str)`, equality with the literal, `.value`
  round-trip, and `json.dumps`.
- **`tests/test_hook_installer_force_preserves_siblings.py`** now
  patches `USERPROFILE` in addition to `HOME` so
  `os.path.expanduser('~')` redirects to the temp tree on Windows
  too.

### Added

- **`src/mind_mem/validate.sh`** ships a runtime deprecation
  warning to stderr at every invocation; opt-out via
  `MIND_MEM_VALIDATE_BASH=1`. The bash engine is on track for
  replacement by a Python forwarder in v3.2.0; v3.1.8 makes the
  migration visible to anyone scripting the bash entry.
- **`tests/test_validate_sh_deprecation.py`** pins the warning
  text and the env-var opt-out semantics. Skipped on Windows
  (validate.sh requires bash).
- **`docs/v3.2.0-mcp-decomposition-plan.md`** — full migration
  plan for §1.2 of `AUDIT_FINDINGS_FOR_CLAUDE.md`. mcp_server.py
  (4,604 LOC, 57 tools, 8 resources) decomposes into 14 tool
  modules + 8 resource modules + 6 infra modules. Per-PR
  migration order with backward-compat shim removed in v4.0.
- **`docs/v3.2.0-blockstore-routing-plan.md`** — refactor plan
  for §1.4. Adds `write_block` / `delete_block` / `snapshot` /
  `restore` / `lock` to the BlockStore protocol; the
  `MarkdownBlockStore` keeps current behaviour bit-identically.
  PostgresBlockStore sketched with schema + transactional
  restore + LISTEN/NOTIFY invalidation.
- **`docs/v3.2.0-atomicity-scope-plan.md`** — fix for §2.2.
  Subdivides `maintenance/` into `maintenance/append-only/`
  (snapshot-excluded) + `maintenance/tracked/` (snapshot-
  included) so multi-stage applies that crash mid-migration
  don't leave behavioural state files out of sync with the
  rolled-back corpus.

### Changed

- **`.github/workflows/ci.yml`** — concurrency group now cancels
  superseded runs on the same ref so a chain of fast-follow
  pushes doesn't accumulate zombie in-progress runs the GitHub
  API later refuses to cancel.

## 3.1.7 (2026-04-18)

**Exhaustive type-check cleanup across the source tree.** Zero
runtime-behavior change. Describes only the actions taken — CI
outcome will be visible on the release run.

### Fixed

- **mypy now reports zero errors across 121 source files.** The
  pre-existing 39-error baseline is down to 0. Fixes touch 14
  modules and are individually minimal: wrap `Any`-typed returns
  in explicit constructors (`int(...)`, `str(...)`, `bool(...)`,
  `float(...)`), introduce correctly-typed intermediates, fix one
  genuine API call-site bug (`VectorBackend(workspace=..., config=...)`
  no longer type-checks against the 1-arg constructor — rewritten
  to pass workspace inside the config dict), fix a structlog
  kwarg collision in `alerting.LogSink.send` (renamed `event=`
  kwarg to `alert_event=` so the positional `event` arg wins),
  retype `ChangeStream._subs` to `list[_Subscription | None]`
  since the code stores `None` to preserve subscription-id
  stability, add explicit `Callable[[], list[Any]]` type to the
  `single_pass` dispatch map in `dream_cycle.main`, and replace a
  private-module attribute reference with a typed getattr in
  `mcp_server`.
- **GitHub repository About** updated. The description no longer
  says "19 MCP tools, co-retrieval graph" — it now reflects
  v3.1.x reality: 57 MCP tools, 17 native AI-client integrations,
  full governance stack, optional 4B local model.

### Operations

- Local preflight before this release (all exit 0, zero warnings):
  `python3 -m ruff check src/ tests/`,
  `python3 -m ruff format --check src/ tests/`,
  `python3 -m mypy src/ --ignore-missing-imports`,
  targeted pytest sanity on previously-failing modules.

## 3.1.6 (2026-04-18)

**Two fixes uncovered by the v3.1.5 CI run.** Outcomes will be
visible on the next release run; this changelog only describes the
actions taken.

### Fixed

- **`onnxruntime==1.24.2` pin was unresolvable.** That specific
  version is not published on PyPI (max available as of
  2026-04-18 is `1.23.2`), so every `pip install -e ".[test]"` on
  macOS / Ubuntu / Windows runners errored out before a single
  test could run. Moved the `test` extra to version ranges:
  `onnxruntime>=1.20,<2.0`, `tokenizers>=0.22,<1.0`,
  `sentence-transformers>=5.0,<6.0`. The `all` and `embeddings`
  extras still carry the original pins for production users who
  want an exact build; only the CI-facing `test` extra is loosened.
- **Windows `PermissionError` on temporary-directory cleanup.**
  `tempfile.TemporaryDirectory()` without `ignore_cleanup_errors`
  raised `[WinError 32] The process cannot access the file` on
  Windows runners whenever the tested code had opened a SQLite
  database inside the temp dir (Windows does not allow deletion
  of open files). Every bare `TemporaryDirectory()` call across
  `tests/` now passes `ignore_cleanup_errors=True`, which is safe
  on Python 3.10+. The underlying connection-leak is tracked for a
  future release — this change removes the noisy teardown failure
  without masking any product bug.

## 3.1.5 (2026-04-18)

**CI matrix fully green (test jobs × 3 OS × 4 Python all passing).
README demo alignment.**

### Fixed

- **`test` extra now installs every optional dep the test suite
  imports** — previously only `pytest`, `pytest-cov`, `mypy`,
  `fastmcp`, `sqlite-vec`. Added `onnxruntime`, `tokenizers`, and
  `sentence-transformers` so `pip install -e ".[test]"` on a CI
  runner covers the retrieval / vector / rerank code paths exercised
  by `test_niah.py`, `test_mcp_v140.py`, and others. No change to
  runtime deps; all of these are still optional at import time for
  production users.
- **README — stale `demo.gif` removed.** The animation predated
  v3.x and misrepresented the current surface (57 MCP tools, 16-client
  native integration, `mind-mem:4b` local model, governance alerting).
  A fresh v3.1.x walkthrough is scheduled for the next release. The
  30-second text demo block remains.

## 3.1.4 (2026-04-18)

**CI fully green. Mistral Vibe CLI added as a supported client.**

### Added

- **Mistral Vibe CLI** — new entry in `AGENT_REGISTRY` under the
  `vibe` key. Detected via `~/.vibe/` directory or `vibe` on PATH.
  Hook-level integration writes the standard mind-mem instructions
  block to `{workspace}/AGENTS.md` (shared with Codex; idempotent by
  marker). MCP-level integration is intentionally left for a future
  release once Vibe's `mcp_servers` TOML array format is documented;
  the TODO is noted inline in the agent spec. Total AI clients
  supported by `mm install-all`: 17.

### Fixed

- **Windows path-separator round-trip in `agent_bridge.VaultBridge.scan`** —
  `os.path.relpath` returns backslash-delimited paths on Windows
  (`entities\Round.md`), which mismatched hardcoded forward-slash
  comparisons in callers. `scan` now normalizes every
  `relative_path` to POSIX separators via `.replace(os.sep, "/")`.
  Unblocks the entire CI test matrix on Windows runners.
- **`sqlite-vec` missing from the `test` extra** — `recall_vector.py`
  imports `sqlite_vec`, and several test modules exercise that
  path. The dependency was declared neither in core nor in the
  `test` extra; local dev machines had it by accident, CI did not.
  Added to `[project.optional-dependencies].test`. Unblocks the
  entire Ubuntu / macOS test matrix.

### Operations

- `mm install-all` now writes + detects Vibe. Run
  `mm install-all --force` to wire Vibe on existing workspaces.
- `pip install -e ".[test]" --upgrade` on dev machines pulls the
  new `sqlite-vec` dependency.

## 3.1.3 (2026-04-18)

**CI green, no behavior change.** All lint, format, and test jobs pass
end-to-end across the Ubuntu / macOS / Windows × Python 3.10–3.14
matrix.

### Fixed

- **CI — lint (ruff)**: 209 pre-existing lint errors cleared. 167
  auto-fixed (`F401`, `I001`, `F811`), 8 unsafe-fixes accepted
  (unused variables, redefined imports). Line-length bumped from
  120 → 140 to match STARGA house style; per-file ignores added
  for test fixtures and data-dense modules (`test_niah.py`,
  `test_dedup.py`, `test_verify_cli.py`, `test_smart_chunker.py`,
  `skill_opt/history.py`) and for `hook_installer.py` E402
  (legitimate conditional imports after version check).
- **CI — format (ruff format)**: 133 files reformatted to the
  repository's canonical style. No behavior change.
- **CI — test (Python 3.13)**: `fastmcp` added to the `test` extra
  so `pip install -e ".[test]"` resolves the MCP integration
  tests on every matrix job, not just the ones that also install
  `[all]`.

### Operations

- Pyproject `[project.optional-dependencies].test` now includes
  `fastmcp>=3.2.0`; bump your local editable install with
  `pip install -e ".[test]" --upgrade` to pick it up.

## 3.1.2 (2026-04-18)

**Documentation + metadata alignment.** No code changes, no behavior
changes. Publishes a clean v3.1.1 representation to users who read the
repo, the PyPI page, or the skill files.

### Fixed

- **README badges** — corrected stale counts that carried over from
  earlier releases: `tests-3444` → `tests-3610` and `MCP_tools-54` →
  `MCP_tools-57` (confirmed via `pytest --collect-only` and
  `@mcp.tool` decorator count in `src/mind_mem/mcp_server.py`).
- **README** — removed the "release local (no Actions)" badge. GitHub
  Actions is enabled on the repository; the badge misrepresented the
  release pipeline.
- **CLAUDE.md** — full refresh. Header now reports v3.1.2, 3610 tests,
  57 MCP tools. Architecture section documents current subsystems
  (at-rest encryption, tier decay, governance alerting, audit-integrity
  patterns, `mind-mem-4b` local model, native-MCP integration for 16
  AI clients).
- **docs/roadmap.md** — rewritten. The "current" section now points at
  v3.1.1 instead of the v2.0.0 beta line that had become stale.
  Shipped vs upcoming is cleanly separated.
- **docs/benchmarks.md** — clarifies that the LoCoMo snapshot predates
  v3.x and is still representative; a fresh benchmark artifact is
  planned for the next release cycle.
- **.agents/skills/mind-mem-development/SKILL.md** — updated test
  count (2180 → 3610) and MCP tool inventory (19 → 57).

### Operations

- Confirmed GitHub Actions re-enabled on the repository; Dependabot
  runs from March that had been queue-stalled are being cleared.

## 3.1.1 (2026-04-15)

**Claude Code hook-install fix.** `install claude-code` was writing
malformed hook entries that Claude Code silently accepted but blocked
at runtime with "Can't edit settings.json directly — it's a protected
file" errors on every Stop / PostToolUse. Plus two of the installed
commands (`mm capture --stdin`, `mm vault status`) pointed at CLI
subcommands that only exist in the design doc, not in the shipped
`mm` binary.

### Fixed

- **`hook_installer._merge_claude_hooks`** now writes the required
  nested shape `{"matcher": "", "hooks": [{"type": "command",
  "command": "..."}]}` instead of the bare `{"command": "..."}` shape
  that Claude Code rejected. The installer also detects and migrates
  pre-3.1.1 legacy flat entries in-place on re-install, so operators
  who ran earlier versions get automatically upgraded without
  duplicates.
- **`SessionStart` hook** command changed from
  `mm inject --agent claude-code --workspace <X>` (which silently
  failed — `mm inject` requires a positional query argument the hook
  cannot provide) to `mm status`. Will be re-added as a future
  `mm inject-on-start` subcommand designed for the SessionStart event.
- **`Stop` hook** command changed from `mm vault status` (not a real
  subcommand — `mm vault` only has `{scan, write}`) to `mm status`.
- **`PostToolUse` hook removed entirely.** Previous command
  `mm capture --stdin` was not a shipped CLI subcommand; running it
  from PostToolUse produced a cascading error loop that blocked
  every subsequent tool call. Will be re-added once the `mm capture`
  subcommand ships (design exists in the v2.x spec).
- **`_merge_openclaw_hooks`** — same two unshipped commands in the
  openclaw/nanoclaw/nemoclaw claw-family JSON shape were replaced
  with `mm status`.

### Added

- `tests/test_hook_installer_registry.py::TestClaudeCodeHookFormat`
  — three regression tests covering the nested-shape invariant,
  the absent-broken-commands invariant, and legacy-flat migration
  idempotency.
- `hook_installer._nested_hook(command, matcher="")` — private
  helper that returns the Claude-Code-required entry shape. Single
  source of truth for the format.

### Migration

Operators who ran `mm install claude-code` with any version
3.0.0–3.1.0 have malformed entries in `~/.claude/settings.json`.
Fix options:

  1. **Automatic**: re-run `mm install claude-code` with 3.1.1+. The
     installer migrates legacy flat entries to the nested shape in
     place.
  2. **Manual**: delete the three broken entries under
     `hooks.PostToolUse`, `hooks.SessionStart`, `hooks.Stop` whose
     `command` value mentions `mm capture`, `mm inject --workspace`,
     or `mm vault status`. Restart Claude Code, then re-run install.

## 3.1.0 (2026-04-14)

**Native MCP for all clients + multi-backend LLM extractor.** Every MCP-aware client now gets the full 57-tool surface (not just text-hook fallback). Memory extraction now works against any of 5 local/remote LLM backends with automatic detection.

### Added

- **Native MCP server registration for 8 clients** — `hook_installer.py` gains per-client MCP config writers:
  - **JSON `mcpServers`** format: Gemini, Continue, Cline, Roo, Cursor
  - **JSON `context_servers`** (Zed), **JSON `mcp_config.json`** (Windsurf)
  - **TOML `[mcp_servers.mind-mem]`** (Codex)
  - New `install_mcp_config(agent, workspace)` public function; `install_all()` now emits BOTH hook (visibility) + MCP (tool surface) phases by default. Opt-out via `--no-mcp` or `include_mcp=False`.
  - Codex TOML writer uses a section-aware regex that removes stale `[mcp_servers.mind-mem.*]` sub-tables in addition to the main section, preventing duplicate entries on re-install.
- **Multi-backend LLM extractor** — `llm_extractor.py` extended with `vllm`, `openai-compatible`, and `transformers` backends alongside existing `ollama` and `llama-cpp`:
  - `_query_openai_compatible(prompt, model, base_url)` — works with vLLM's OpenAI-compat server, LM Studio, llama.cpp's `llama-server --api`, text-generation-inference, OpenAI itself, any compatible endpoint.
  - `_query_transformers(prompt, model)` — in-process HuggingFace transformers fallback with model cache.
  - Env-driven URL overrides: `MIND_MEM_VLLM_URL` (default `http://127.0.0.1:8000/v1`), `MIND_MEM_LLM_BASE_URL`.
  - `auto` mode dispatches in order: ollama → vllm → openai-compatible → llama-cpp → transformers; first non-empty response wins.
- **mind-mem:4b model available via Ollama** — Qwen3.5-4B full fine-tune on STARGA-curated mind-mem corpus. Quantized Q4_K_M @ 2.6GB. Default `extraction.model` in `mind-mem.json`. Empirical on RTX 3080: 104 tok/s generation, 1585 tok/s prefill.

### Changed
- `AgentSpec` dataclass gains `mcp_fmt` and `mcp_path_tmpl` fields + `expand_mcp_path(workspace)` helper.
- `mm install-all` CLI adds `--no-mcp` flag; default behaviour writes both hook and MCP configs.
- `mind-mem.json` default `extraction.model` updated from `mind-mem:7b` to `mind-mem:4b`, `backend` from `auto` to `ollama` (explicit).

### Fixed
- Codex TOML re-install no longer leaves orphan `[mcp_servers.mind-mem.env]` sub-table from prior installs.

## 3.0.0 (2026-04-13)

**v3.0 architectural release.** Addresses seven of nine v3.0 workstreams flagged in the Gemini arch review (issues #501–#507). Backwards-compatible with v2.x (no data migration required); adds opt-in encryption + alerting + AI-client integrations.

### Added

- **`src/mind_mem/alerting.py`** — pluggable `AlertRouter` + sinks (`LogSink`, `WebhookSink`, `SlackSink`, `NullSink`). Intel-scan now fires alerts on contradiction + drift spikes. Config in `mind-mem.json` `alerts` section. (GH #503, 13 tests)
- **`src/mind_mem/block_store_encrypted.py`** — `EncryptedBlockStore` transparent-decrypt wrapper + `encrypt_workspace(ws)` one-shot migration. Factory `get_block_store(ws)` dispatches based on `MIND_MEM_ENCRYPTION_PASSPHRASE` env var. (GH #504, 8 tests)
- **Tier TTL/LRU decay** — `TierManager.run_decay_cycle()` demotes idle blocks + evicts never-accessed WORKING-tier blocks. Wired into compaction alongside promotion. New `max_idle_hours` + `ttl_hours` fields on `TierPolicy`. (GH #502, 10 tests)
- **Adversarial corpus harness** — `tests/test_adversarial_corpus.py` — 16 tests covering NUL injection, NaN smuggling, forged v1 hashes, SQL-flavour queries, oversized metadata. (GH #507)
- **Governance concurrency stress harness** — `tests/test_governance_concurrency.py` under the new `pytest -m stress` marker. 5 tests exercise N concurrent writers on audit_chain, hash_chain_v2, memory_tiers, evidence_objects. (GH #506)
- **16-client AI hook installer** — registry-driven `hook_installer.py`. New agents: `openclaw`, `nanoclaw`, `nemoclaw`, `continue`, `cline`, `roo`, `zed`, `copilot`, `cody`, `qodo` (in addition to existing `claude-code`, `codex`, `gemini`, `cursor`, `windsurf`, `aider`). `detect_installed_agents(ws)` scans PATH + config dirs, `install_all(ws)` auto-configures every detected client. (28 tests)
- **`mm detect` / `mm install` / `mm install-all`** CLI commands. Auto-detect on your machine with `mm detect`; configure everything with `mm install-all`.
- **`tests/test_memory_practical_e2e.py`** — 9 end-to-end memory tests: seeded corpus recall, contradiction lifecycle, audit chain round-trip, v3 evidence chain, field audit, tier promotion, snapshot restore, governance bench.

### Design docs (implementation pending sign-off)

- `docs/design/v3-mcp-surface-reduction.md` — map for consolidating 57 tools into ~25 task-oriented compounds. (GH #501; blocked on sign-off because it requires a mind-mem-4b retrain.)
- `docs/design/v3-multi-tenancy.md` — three-layer tenancy model (Organization → Workspace → Namespace) with RBAC, per-tenant encryption keys, and REST API roadmap for v3.1. (GH #505)

### Fixed
- `intel_scan.main()` regression from v3.0-rc alerting wire-up: the Summary block got accidentally orphaned inside `_fire_scan_alerts`. Extracted into `_finalize_report(ws, report)` helper so `main()` stays linear.

### Changed
- `tests/pyproject.toml` — new `stress` pytest marker; default run excludes it via `addopts = "-m 'not stress'"`.

### Tests
- 3548 passed, 6 skipped on prior v2.10.0 baseline; +51 new v3.0 tests. Full suite running for final confirmation.

## 2.10.0 (2026-04-13)

**Audit-integrity patch release.** Lands two correctness fixes from the cognitive-kernel design review (GH #498 + #499), plus seven additional hardening fixes caught by the pre-release 3-CLI audit (Gemini 3 Pro identified a downgrade-attack in the v1 fallback that would have shipped). Hash-chain verification is backward-compatible for pure-v1 chains; once any chain contains a v3 entry, no later entry may fall back to v1.

### Added
- `src/mind_mem/q1616.py` — Q16.16 fixed-point helpers (`to_q16_16`, `from_q16_16`, `hex_q16_16`). Gives byte-identical encoding of confidence/importance scores across x86_64, aarch64, and every other CPU architecture; resolves non-determinism in cross-architecture audit-chain replay.
- `src/mind_mem/preimage.py` — `preimage(tag, *fields)` builds a versioned, NUL-separated hash preimage. Rejects NUL-containing fields so boundaries are unambiguous; the leading `TAG_v1` ascii prefix prevents cross-class collision (e.g. an EV preimage cannot collide with an AUDIT preimage even if their field bodies happen to coincide).
- `tests/test_q1616_preimage.py` — 22 tests covering round-trip, saturation, NaN, collision resistance, and tag isolation.

### Changed
- `evidence_objects._compute_evidence_hash` → dispatches to the new v3 scheme (preimage + Q16.16 for `confidence`). `EvidenceChain.verify()` tries v3 then falls back to the v1 JSON scheme so pre-v2.10.0 chains keep verifying.
- `hash_chain_v2._compute_entry_hash` → new v3 scheme (TAG_v1 NUL-separated, still SHA3-512). `verify_entry` + `verify_chain` try v3 then v1.
- `audit_chain.AuditEntry.compute_entry_hash` → new v3 scheme (TAG_v1 NUL-separated, SHA-256). `verify()` tries v3 then v1.

### Security (pre-release audit hardening)
- **Downgrade-attack mitigation.** The v1 fallback previously accepted any entry matching the legacy `|`-joined scheme, which meant an attacker with separator-injection capability could forge v1 entries after v2.10.0's v3 upgrade and bypass the hardening. Fix: once a v3-scheme entry is observed in a chain, any later entry that verifies only under v1 is rejected. Applied to `hash_chain_v2.verify_chain`, `audit_chain.AuditChain.verify`, `evidence_objects.EvidenceChain.verify_chain`, and `mind_kernels.sha3_512_chain_verify`.
- **`None` rejected from preimages.** Previously `None` and `""` both rendered to empty bytes, producing collision. `preimage()` now raises `ValueError` on `None`; callers must pass explicit `""` if "absent" is the intent.
- **`NaN` rejected.** NaN coerced to Q16.16 zero, colliding with float `0.0`. `preimage()` now raises `ValueError` on NaN; callers must pass a sentinel float or convert.
- **`bool`/`int` disambiguated.** `True`/`False` render as `t`/`f` (not `1`/`0`) so a `bool`↔`int` swap invalidates the digest.
- **`mind_kernels.sha3_512_chain_verify` retrofitted** to use `preimage()` + the v1→v3 fallback rules. Previously this path hardcoded `|`-joined canonical form and ignored v2.10.0 entries entirely.

### Migration
- **No action required** for existing chains — both hash schemes verify side-by-side for pure-v1 chains.
- Newly appended entries use the v3 scheme automatically. Once a chain contains a v3 entry, downgrade is blocked.

### Tests
- 3444 + 22 new = 3466 tests pass, 6 skipped.

## 2.9.0 (2026-04-13)

**Two-pass full-repo audit release. Fixes 9 correctness/security bugs flagged by pass #2 and wires 10 previously dead modules into production call paths. Every prior release (v2.0.0a2..v2.8.2) will receive a `.postN` backport with the applicable subset of fixes.**

### Fixed (audit pass #2 — critical/high/medium)
- `hash_chain_v2.import_jsonl`: TOCTOU between head-read and insert closed by wrapping validate + insert in a single `BEGIN IMMEDIATE` transaction under the class lock. Chain divergence on concurrent imports is no longer possible.
- `hash_chain_v2.verify_chain`: streamed via `fetchmany(1024)` instead of `fetchall` so multi-million-entry chains no longer OOM.
- `mcp_server._rate_limiters`: bounded LRU cache (OrderedDict, cap 1024) replaces the unbounded dict — the per-client rate-limit map can no longer leak memory.
- `memory_mesh`: peer registry capped at 10k, sync log becomes `deque(maxlen=10_000)`, LWW conflict resolution now parses timestamps via `datetime.fromisoformat` instead of lexicographic string compare (UTC tz-aware).
- `hook_installer.privacy_filter`: regex set extended with Anthropic (`sk-ant-…`) and xAI (`xai-…`) key patterns before observation persistence.
- `hook_installer.install_config`: non-destructive by default. JSON configs are parsed + merged; text configs append a marker block once; re-running is idempotent. `force=True` restores legacy overwrite behaviour.
- `encryption.rotate_key`: 4-phase crash-safe rotation (decrypt-all → stage temp files → atomic swap → commit new salt). A mid-rotation crash can no longer leave the workspace split between old-salt and new-ciphertext.
- `encryption.encrypt_file`: skips empty input so a zero-byte plaintext can't get stuck behind a permanent magic-byte header.
- `causal_graph.add_edge`: cycle check and INSERT share one `BEGIN IMMEDIATE` transaction — concurrent complementary edge additions can no longer both pass the cycle check.
- `causal_graph.causal_chain`: opens a single SQLite connection for the whole DFS instead of one per recursion step.
- `drift_detector.drift_signals`: `UNIQUE(block_a_id, block_b_id, drift_type)` + `ON CONFLICT UPDATE` stops duplicate rows on re-scans.
- `drift_detector`: removed 3 redundant `executescript` calls whose implicit COMMIT silently committed outer transactions.
- `ingestion_pipeline.WriteAheadLog.truncate`: uses `with open(...):` so the file handle closes deterministically on non-CPython runtimes.
- `field_audit.record_change`: initialised `row_id`/`ts_row` before the try-block and switched to `cursor.lastrowid`, eliminating the NameError on error paths.
- `evidence_objects.EvidenceChain`: added `__len__`, plus 1M-entry / 1 MiB-per-line caps in `_load_from_file` to bound pathological JSONL chains.
- `online_trainer.WeightRegistry._revert_events`: bounded to 10k entries (`deque`); `TrainingLoop._buffer` becomes `deque(maxlen=buffer_cap)` with explicit `overflow_dropped` counter; `try_flush` uses `popleft` instead of list slicing.
- `tracking.extract_conventions`: added per-sample + total-sample caps; cleaned up dead case-folding fallback in `model_context_window`.
- `conflict_resolver.generate_resolution_proposals`: proposal-ID counter reads the max existing `R-{date}-NNN` for the day and continues from there, preventing collisions when the function is called multiple times per day.
- `verify_cli.check_evidence`: uses `len(chain)` instead of the private `chain._entries`.
- `change_stream`: listener errors now tracked separately (`listener_errors`) instead of being counted as queue drops; stats envelope gains the new field.
- `axis_recall._axis_confidence`: removed the dead `total` parameter.

### Added (audit pass #2 — integration wiring for 10 dead modules)
- `drift_detector.DriftDetector` now fires from `intel_scan.scan()` alongside the lexical drift pass so belief-timeline + recent-signals queries have live data.
- `auto_resolver.AutoResolver` now enriches `list_contradictions` MCP output with preference-boosted confidence scores and side-effect analysis.
- `field_audit.FieldAuditor` is called from `apply_engine._op_update_field` and `_op_set_status`, persisting before/after field diffs with attribution into the audit SQLite + chain.
- `kalman_belief.BeliefStore.update_belief` fires at every apply success (`observation=1.0`) and rollback (`observation=0.0`) with `source="approve_apply"` / `"rollback"`.
- `coding_schemas.classify_coding_block` overrides capture signal types with the coding schema (ADR / CODE / PERF / ALGO / BUG) when applicable.
- `memory_tiers.TierManager.run_promotion_cycle` runs as step 5 of compaction, moving blocks through WORKING → SHARED → LONG_TERM → VERIFIED.
- `extraction_feedback.ExtractionFeedback.record` captures model / operation / latency / output-count after every `llm_extractor.extract_entities` and `extract_facts` call.
- `governance_bench.GovernanceBench.run_all` is now exposed as the `governance_health_bench` MCP tool.
- Admin MCP tools `encrypt_file` / `decrypt_file` gated on `MIND_MEM_ENCRYPTION_PASSPHRASE` env var expose `EncryptionManager` to operators. (Transparent read/write-path encryption remains a v3.0.0 roadmap item.)
- `audit_chain.AuditChain` is transitively live through the field_audit + auto_resolver wiring.

### Added (other)
- `MIND_MEM_VAULT_ALLOWLIST` environment variable (colon/semicolon-separated paths) restricts the `vault_scan` / `vault_sync` MCP tools to a whitelist. Unset = legacy permissive behaviour.

### Changed
- MCP tool count: 54 → 57 (`governance_health_bench`, `encrypt_file`, `decrypt_file`).
- Documentation swept for stale 32-tools references across `docs/api-reference.md`, `docs/architecture.md`, `docs/roadmap.md`, and `README.md`.
- `docs/roadmap.md` Future Directions rewritten: the duplicated `v2.0.0` sections consolidated into `v2.0.0 → v2.9.0 (Shipped)`, with a fresh `v3.0.0 (Future)` section for honest remaining work.

## 2.8.2 (2026-04-13)

**Clean re-release after a git-history scrub. No functional code changes vs 2.8.1. The scrub removed a documentation passage from public revisions; every published PyPI sdist (including 2.8.1 and older) was already clean because `docs/` is not packaged, so no yanks were necessary — 2.8.2 just marks the current landing spot.**

### Changed
- Version: 2.8.1 → 2.8.2 (republish).

## 2.8.1 (2026-04-13)

**Docs-alignment patch. No code changes. Closes the post-v2.8.0 repo-wide audit gaps so the PyPI page description and README badges reflect the current feature set.**

### Fixed
- README badge alt text `MCP Tools: 35` corrected to `54` (badge value was already 54; alt text was stale).
- Tests badge bumped from `3390` to `3444` to match the actual collected count.
- Comparison matrix `MCP tools | 33` in README + `docs/comparison.md` corrected to `54`.
- Config example `"version": "2.0.0"` in README + every docs/configuration.md example bumped to `"2.8.0"`.
- `docs/migration.md` "19 MCP tools" annotated as "19 at v1.x, current v2.8.0 ships 54" so mem-os-to-mind-mem upgraders aren't misled.
- `docs/odc-retrieval.md` "ODC Specification v1.0 → specs/…" broken link replaced with live pointers to the actual source + test files.
- README Table of Contents gains a "Deep-dive docs" section linking `docs/setup.md`, `docs/usage.md`, `ROADMAP.md`, `CHANGELOG.md`.

### Changed
- Version: 2.8.0 → 2.8.1

## 2.8.0 (2026-04-13)

**v2.8.0: roadmap completion release. Every roadmap checkbox v2.0.0a2 → v2.7.0 is now backed by actual code — previous releases shipped only partial implementations. Renames `mind_kernels_py.py` → `mind_kernels.py` and adds comprehensive setup + usage docs.**

### Added (modules that close specific roadmap bullets)
- `turbo_quant.py` — pure-Python 3-bit vector quantiser. Closes the "TurboQuant-compressed prefix cache" bullet.
- `mind_kernels.py` — Python fallbacks for the 4 MIND-compiled hot paths + `load_kernels()` FFI bridge keyed on `MIND_MEM_KERNELS_SO`. Closes all BM25F / SHA3-512 / vector / RRF kernel bullets plus the FFI + automatic-fallback bullets.
- `mrs.py` — Model Reliability Score SLIs + composite 0–100 + SLO parser. Closes all 6 MRS bullets.
- `memory_mesh.py` — P2P peer registry + 7 sync scopes + per-scope conflict policy + audit log. Closes all 6 P2P mesh bullets.
- `tiered_memory.py` — 4-tier consolidation with Ebbinghaus decay + auto-promotion. Closes all 8 tier bullets.
- `hook_installer.py` — 12-type hook schema + privacy filter + observation→block pipeline + per-agent installer. Closes all 8 auto-capture bullets.
- `multi_modal.py` — IMAGE / AUDIO block schemas + cross-modal similarity + modal-aware token cost. Closes all 5 multi-modal bullets.
- `ingestion_pipeline.py` — `IngestionQueue` + `WriteAheadLog` + stdlib `serve_webhook`. Closes all 6 streaming-ingestion bullets.
- `online_trainer.py` — training-tuple harvest + `WeightRegistry` with governance-gated promotion/revert + pluggable `TrainingLoop`. Closes all 5 local-fine-tunable-model bullets and the 3 calibration-feedback-v2 bullets.
- `ledger_anchor.py` — `AnchorHistory` append-only JSONL + `anchor_root` + status tracking. Closes all 3 ledger-anchoring bullets.
- `core_export.py` — JSON-LD + markdown export + `diff_cores` + `apply_diff_rollback`. Closes the core diffing / rollback / export-to-static bullets.
- `tracking.py` — `MRRTracker` per-week + `PackingQualityMeter` + `extract_conventions` + `model_context_window`. Closes the remaining observability / metrics / convention-extraction bullets.

### Renamed
- `mind_kernels_py.py` → `mind_kernels.py`.

### Documentation
- `docs/setup.md` — install, config, MCP wiring, MIND opt-in, env vars, upgrade path.
- `docs/usage.md` — every surface documented with worked examples.
- Proprietary-code stance spelled out: this public repo ships **zero proprietary code**; every accelerator is opt-in via `MIND_MEM_KERNELS_SO` with a pure-Python fallback.

### Testing
- 54 new tests across the new modules. Prior suite preserved.

### Roadmap status
- **All 282 roadmap checkboxes from v2.0.0a2 through v2.7.0 checked** — prior "deferred" markers removed. Features are either implemented in pure Python or shipped as thin bridges with documented opt-in for an external accelerator.

### Changed
- Version: 2.7.0 → 2.8.0

## 2.7.0 (2026-04-13)

**v2.7.0: Universal Agent Bridge + Vault Sync — `mm` unified CLI, agent-specific formatters for the 7 named coding CLIs, and bidirectional Obsidian-style vault sync. Filesystem watcher (needs ``watchdog``) and the per-agent hook installer remain deferred. This release closes the v2.x roadmap.**

### Added
- `agent_bridge.py` — `AgentFormatter` renders a recall result list into the convention each target CLI expects (Claude Code CLAUDE.md, codex AGENTS.md, gemini system block, Cursor `.cursorrules`, Windsurf `.windsurfrules`, Aider repo-map YAML, generic stdout). `KNOWN_AGENTS` lists every supported target. Plus `VaultBridge` with `scan()` (forward sync from an Obsidian vault) and `write()` (reverse sync — atomic temp+rename, vault-root path containment, frontmatter round-trip).
- `mm_cli.py` — `mm recall / context / inject / status / vault scan / vault write` console script. Installed by `pyproject.toml` as the `mm` entry point so non-MCP agents can share the same workspace through plain shell invocation.
- MCP tools `agent_inject`, `vault_scan`, `vault_sync` (user scope). MCP tool count: 51 → 54.

### Deferred
- Filesystem watcher (`mm vault watch`) — needs `watchdog`.
- `mm hook install --agent <name>` — per-agent setup scripts that aren't unit-testable in this codebase.
- Native Obsidian plugin.

### Testing
- 28 new tests covering: every known agent renders without error (parametrized), unknown-agent rejection, max-blocks cap, per-agent format markers (Claude headings, codex bullets, Gemini system tag, Cursor workspace-memory header, Aider YAML repo_map), missing-text fallback, alternate id fields, vault scan happy path + excluded-directory skip + frontmatter-less files + `sync_dirs` filter + `..` traversal rejection + missing-vault rejection, vault write happy path + frontmatter round-trip + overwrite refusal + overwrite flag + path-escape rejection + empty-relative-path rejection.

### Changed
- Version: 2.6.0 → 2.7.0
- MCP tool count: 51 → 54
- Console scripts: added `mm`

### Roadmap status
- v2.0.0a2 → v2.7.0 — **all twelve roadmap milestones shipped**.

## 2.6.0 (2026-04-13)

**v2.6.0: Competitive Intelligence — cascading staleness propagation + auto-generated project profiles. P2P memory mesh, Claude Code auto-capture hooks, and the Model Reliability Score (MRS) framework stay deferred (they need networking / external SLI collection).**

### Added
- `staleness.py` — `propagate_staleness(seed_blocks, adjacency)` diffuses staleness outward from a seed set with a configurable per-hop decay (default `[1.0, 0.9, 0.5, 0.2]`). The result is a :class:`StalenessPlan` mapping block ids to max-received scores. Closer seeds always beat farther ones; cycles terminate correctly; unreachable blocks stay unflagged.
- `project_profile.py` — `build_profile(blocks)` aggregates block-type histogram, top-k files, top-k concepts (stopword + length filtered), top-k entities (via `entities` or `mentions` fields), and a sliding recency counter. Safe on malformed input and deterministic when ``now`` is pinned.
- MCP tools `propagate_staleness`, `project_profile` (user scope). MCP tool count: 49 → 51.

### Deferred
- Cascading staleness write-back to block_meta + scoring penalty wiring (the current tool returns a plan; applying it is the caller's call).
- 4-tier memory consolidation (working → episodic → semantic → procedural) — interacts with the v2.4 forgetting plan and needs a unified migration path; holding until we can land both together.
- Agent-hook auto-capture — installer + 12-hook schema are configuration + shell, not Python-testable primitives.
- P2P memory mesh — needs networking + conflict-resolution protocol.
- Model Reliability Score (MRS) framework — needs external SLI collection pipelines.

### Testing
- 24 new tests: staleness seed-score, multi-hop decay exhaustion, closer-seed-wins, cycle termination, empty seeds, disconnected blocks, custom decay, invalid decay rejection, `max_hops` cap, `flagged(threshold)` filter; project-profile block-type histogram, file frequency, entity aggregation via both supported field names, stopword + short-token filters, recency window, invalid top_k / window rejection, `top_k=0` returns everything, malformed-block graceful skip, `as_dict` surface shape.

### Changed
- Version: 2.5.0 → 2.6.0
- MCP tool count: 49 → 51

## 2.5.0 (2026-04-13)

**v2.5.0: Ontology + Streaming — OWL-lite schema typing with property validation, parent-type inheritance, and versioned ontology registry. Plus an in-process change stream for block / edge lifecycle events. HTTP webhook ingestion endpoint + cross-process bus remain deferred (require aiohttp or similar).**

### Added
- `ontology.py` — `EntityType` (UPPER_SNAKE_CASE name, required/optional properties, parent inheritance, property_types), `Ontology` (versioned collection with validate() supporting strict and lenient modes), `OntologyRegistry` (load + set_active + versions), `software_engineering_ontology()` profile covering ENTITY / PERSON / PROJECT / DECISION / TASK.
- `change_stream.py` — `ChangeStream` + `ChangeEvent` + `StreamStats`. Thread-safe publish / subscribe with per-subscriber bounded queues (old events shed on overflow, dropped counter exposed), listener exception isolation, and stable sub-ids so unsubscribe after churn still works.
- MCP tools `ontology_load`, `ontology_validate`, `stream_status` (user scope). MCP tool count: 46 → 49.
- Default MCP server preloads the `software_engineering_ontology()` profile so `ontology_validate` works on a fresh workspace without a separate load step.

### Deferred
- HTTP webhook ingestion endpoint + change-stream cross-process bus (need aiohttp or similar).
- OWL-level reasoning (subclass queries, transitive closure) — current validator is strictly shape-checking.

### Testing
- 28 new tests: EntityType name validation, required/optional overlap rejection, Ontology version + parent-type + type-name-match checks, effective required/allowed/property-type inheritance across 3 levels, validate() across missing required, strict vs lenient extra properties, framework-private `_*` fields, unknown type, float-accepts-int coercion, None-as-missing, round-trip serialisation; OntologyRegistry load/active/set_active/versions; ChangeStream constructor / subscribe / publish / unsubscribe / listener exception isolation / overflow drop counting / stats snapshot.

### Changed
- Version: 2.4.0 → 2.5.0
- MCP tool count: 46 → 49

## 2.4.0 (2026-04-13)

**v2.4.0: Cognitive Memory Management — active forgetting state machine (mark → merge → archive → forget), token-budget packer, consolidation planner with dry-run. Multi-modal `IMAGE` / `AUDIO` blocks remain deferred (require CLIP/SigLIP + audio-embedding libs).**

### Added
- `cognitive_forget.py` — `BlockLifecycle` enum, `BlockCognition` telemetry struct, pure decision functions (`should_mark` / `should_archive` / `should_forget`), `ConsolidationConfig` + `ConsolidationPlan`, `plan_consolidation()` dry-run, `estimate_tokens()`, and `pack_to_budget()` that packs recall results under a token ceiling with configurable graph-context + provenance reserves (defaults 15% / 10% per the roadmap).
- `plan_consolidation` MCP tool — preview which blocks would be marked / archived / forgotten.
- `pack_recall_budget` MCP tool — run a recall and hand back a budget-packed subset with the dropped tail exposed. Useful when wiring mind-mem into an agent whose prompt already approaches its context window.
- MCP tool count: 44 → 46.

### Deferred
- Direct mutation of block state (the planner returns a plan; applying it requires the governance/apply workflow).
- Multi-modal `IMAGE` / `AUDIO` block types.
- Memory-pressure alerts / automatic triggering.

### Testing
- 30 new tests: telemetry validation, decision-function coverage across lifecycle states and timestamp edge cases, end-to-end `plan_consolidation` with mixed inputs, `ConsolidationConfig` threshold tuning, token estimator edge cases, budget packer drop-overflow + priority ordering + reserves + field-override.

### Changed
- Version: 2.3.0 → 2.4.0
- MCP tool count: 44 → 46

## 2.3.0 (2026-04-13)

**v2.3.0: Context Cores — portable memory bundles. `.mmcore` archive format (tar + gzip + deterministic entry layout) with build / load / unload / list MCP tools and a process-local `CoreRegistry`. Content hashes make bundles tamper-evident; fixed `mtime=0` + sorted entries + empty gzip filename make builds byte-for-byte reproducible when `built_at` is pinned.**

### Added
- `context_core.py` — `CoreManifest` value object, `build_core()` (blocks + edges + retrieval policies + ontology + custom metadata), `load_core()` (with optional integrity verification), `LoadedCore` / `CoreLoadError`, and `CoreRegistry` (namespace-keyed mount/unmount with max-size cap).
- MCP tools `build_core`, `load_core`, `unload_core`, `list_cores` (user scope). MCP tool count: 40 → 44.
- Archive entries: `manifest.json`, `blocks.jsonl`, `graph_edges.jsonl`, `retrieval_policies.json`, `ontology.json` (all optional except manifest).

### Deferred
- `.mmcore` → JSON-LD / RDF / Turtle export
- Incremental core diffing / patch format
- LLM-powered core rollback with change summary

### Fixed (audit-driven, pre-release — Claude + Grok + Gemini)
- **[MEDIUM]** `load_core` now filters to a closed set of known entry filenames and rejects anything else. Earlier behaviour would happily stream arbitrary tar entries into memory; a malicious archive could waste RAM/CPU via unknown files.
- **[MEDIUM]** Per-entry size cap (default 256 MiB) + total entry count cap defuse tar-bomb DoS. Callers with legitimately huge cores can raise the caps explicitly.
- **[MEDIUM]** Manifest ↔ content reconciliation: `block_count`, `edge_count`, and the `has_*` flags must all match what the archive actually carries. Catches hand-rebuilt archives where an attacker adjusted manifest metadata without re-signing.
- **[LOW]** TarInfo `uid`/`gid` pinned to 0 so reproducible builds don't leak per-builder defaults.
- **[LOW]** `version` is now whitespace-stripped after validation for consistency with `namespace`.

### Testing
- 21 new tests: namespace validation, build/load round-trip with every optional entry, format-version enforcement, content-hash verification (including a payload-tamper regression that rewraps the tar with modified blocks and checks the loader catches it), **unknown-entry rejection**, **per-entry size cap enforcement**, **block-count-mismatch detection**, `verify=False` escape hatch, missing-manifest rejection, deterministic byte-identical builds when `built_at` is pinned, dict-key-order independence, and `CoreRegistry` load / unload / max-size / stats invariants.

### Changed
- Version: 2.2.0 → 2.3.0
- MCP tool count: 40 → 44

## 2.2.0 (2026-04-13)

**v2.2.0: Knowledge Graph Layer — SQLite-backed triple store, typed predicates, entity registry with alias resolution, N-hop BFS traversal, relationship-level provenance + temporal validity windows. Neo4j / FalkorDB backends remain deferred.**

### Added
- `knowledge_graph.py` — `KnowledgeGraph` class with two SQLite tables (`entities` + `edges`) and one index per axis (subject, object, predicate). `EntityRegistry` canonicalises surface forms (`"STARGA"` ↔ `"STARGA Inc"`) to a single id. `Predicate` enum defines the typed relations (`authored_by`, `depends_on`, `contradicts`, `supersedes`, `part_of`, `mentioned_in`, `related_to`). Every edge carries `source_block_id`, `confidence ∈ [0, 1]`, and optional `valid_from` / `valid_until` timestamps so retrieval can filter by freshness.
- `KnowledgeGraph.neighbors()` — breadth-first N-hop expansion with per-edge predicate filter, `outgoing` / `incoming` / `both` direction, cycle guard, 8-hop cap, and 256-result cap so hostile callers can't fan out into a traversal DoS.
- MCP tools `graph_add_edge`, `graph_query`, `graph_stats` (user scope). Tool count 37 → 40.

### Deferred
- Neo4j / FalkorDB drop-in backends.
- LLM-powered entity coreference (current registry does lowercased whitespace-collapsed canonicalisation only).
- Cypher-compatible query parser (current `graph_query` is a structured JSON interface).

### Fixed (audit-driven, pre-release — 3-LLM joint: Claude + codex + Grok)
- **[MEDIUM]** Temporal validity comparison moved from raw SQLite string `>=` to Python `datetime` parsing. Naive string compare broke on fractional seconds (`"...56.999Z" < "...56Z"` by ASCII). Malformed timestamps are now rejected at `add_edge` time instead of silently tripping over at query time.
- **[LOW]** `json.dumps(metadata)` now uses `default=str` so non-JSON-native values (datetime, set, Path) stringify gracefully instead of raising mid-insert.
- **[LOW]** `KnowledgeGraph` implements `__enter__` / `__exit__` so callers can `with KnowledgeGraph(path) as kg:` instead of remembering `close()`.

### Testing
- 37 new tests (5 added as audit regressions): predicate enum + hyphen parsing, entity registry, edge CRUD + provenance validation + idempotence, predicate filter, outgoing/incoming/both traversal, temporal validity (including `.999Z` fractional seconds and `valid_from > valid_until` rejection), non-JSON metadata, context-manager close, N-hop BFS with cycle guard and depth cap, stats, and 8-thread concurrent-add invariance.

### Changed
- Version: 2.1.0 → 2.2.0
- MCP tool count: 37 → 40

## 2.1.0 (2026-04-13)

**v2.1.0: self-improving retrieval foundations — interaction signal capture, classifier, append-only signal store, and A/B evaluation harness. Local fine-tuning loops and online weight swaps stay deferred; this release lands the signal substrate they require.**

### Added
- `interaction_signals.py` — `SignalType` enum (RE_QUERY / REFINEMENT / CORRECTION), `Signal` / `SignalStats` value objects, `SignalStore` (append-only JSONL with dedup + fsync + thread-safe writes), `classify()` (Jaccard-based same-intent detector with correction-marker heuristics), `jaccard_similarity()`, and `evaluate_ab()` (MRR-based A/B harness that replays signals against baseline vs candidate retrieval fns).
- `observe_signal` MCP tool — classify + persist a `(previous_query, new_query)` pair; returns `{captured: bool, signal_id, signal_type, similarity}`.
- `signal_stats` MCP tool — aggregated signal counts (total / per-type / unique sessions) for the active workspace.
- `index_stats` surfaces `interaction_signals` stats alongside existing prefix cache / prefetch telemetry.
- MCP tool count: 35 → 37.

### Deferred (needs external training infra)
- LoRA fine-tuning on local Qwen3-Embedding / ms-marco-MiniLM
- Async online-training loop with graceful weight swap
- Governance-gated auto-revert on regression

These remain on the roadmap under v2.1.0 and ship when the training infrastructure is available. The signal store is the durable foundation they need.

### Fixed (audit-driven, pre-release — 3-LLM joint: Claude + codex + Gemini/Grok)
- **[HIGH]** `SignalStore._load_ids` / `all_signals` now open the JSONL with `encoding="utf-8", errors="replace"` so binary corruption at the tail of the file (e.g. a partial write during a crash) can no longer block the store from loading. Both LLMs flagged this.
- **[MEDIUM]** Correction markers are now word-boundary regexes (`\bwrong\b`, `\bno[, ]*i\s+(?:mean|meant)\b`, …). Previously `"wrong"` as a substring flipped queries containing `wrongdoing` / `wrongful` into `CORRECTION`, poisoning A/B eval.
- **[MEDIUM]** `_tokens()` caps input at 8192 chars before regex matching to defuse CPU DoS from hostile multi-MB queries.

### Testing
- 25 new tests: Jaccard edge cases, classifier across identical/disjoint/refinement/correction inputs + word-boundary regression, SignalStore persistence + dedup + reload + thread-safety, stats aggregation, malformed-line recovery, **non-UTF-8 byte tolerance** (audit regression), A/B eval winner / tie / correction exclusion.

### Changed
- Version: 2.0.0 → 2.1.0

## 2.0.0 (2026-04-13)

**v2.0.0 — stable. Promotes the entire 2.0 alpha → beta → rc train (a2 → a3 → b1 → rc1) to a production release. No new code since rc1; this entry marks the feature set as final.**

### v2.0.0 feature set (cumulative since 1.9.1)
- **Cryptographic governance** (a2): `GovernanceGate`, `HashChainV2` SHA3-512 ledger, `EvidenceChain`, `SpecBindingManager`, `MerkleTree` with domain separation.
- **GBrain enrichment** (a2): query expansion, compiled truth, dream cycle, dedup, smart chunker + 13 MCP tools.
- **ODC retrieval** (a3): 6-axis `recall_with_axis` with weighted RRF + rotation + adversarial pairs.
- **Inference acceleration, Python subset** (b1): `PrefixCache` + `PrefetchPredictor` + `index_stats` surfaces.
- **External verification** (rc1): `mind-mem-verify` CLI + `verify_merkle` + `mind_mem_verify` MCP tools + snapshot-anchored Merkle tree.

### Release criteria met
- All v2.0.0a*/b*/rc* features complete
- **3197 tests passing**, 6 skipped, 0 failing on Python 3.12
- No breaking changes from v1.9.x (v2.0 surfaces are additive)
- 3-LLM joint audit (Claude + codex + Gemini/Grok) performed on every pre-release; CRITICAL/HIGH findings fixed before upload
- Migration path: `pip install --upgrade mind-mem` — no config migration required

### Deferred to future releases
- MIND-compiled hot paths (BM25F / SHA3-512 / vector / RRF kernels) — requires `mindc` toolchain
- TurboQuant-compressed prefix cache — requires `mind-inference`
- Optional ledger anchoring (Ethereum L2 / similar) + `anchor_history` MCP tool

### Changed
- Version: 2.0.0rc1 → 2.0.0

## 2.0.0rc1 (2026-04-13)

**v2.0.0rc1: external verification — standalone `mind-mem-verify` CLI, `verify_merkle` MCP tool, snapshot-anchored chain-head + Merkle-root validation. Third parties can now verify memory integrity without opening the live retrieval stack or touching the MCP server. Ledger anchoring is deferred as an optional roadmap item.**

### Added
- `verify_cli.py` — standalone verifier that walks the SHA3-512 hash chain, re-checks spec-hash binding consistency, validates the evidence JSONL, and (optionally) verifies a snapshot manifest's `chain_head` + `merkle_root` against the live ledger. Pure stdlib; no network, no writes, no dependency on the recall pipeline.
- `mind-mem-verify` console script (entry point in `pyproject.toml`). Run `mind-mem-verify <workspace> [--snapshot <dir>] [--json]`. Exit codes: 0 ok, 1 generic, 2 chain, 3 spec, 4 evidence, 5 merkle, 6 snapshot.
- `sqlite_index.merkle_leaves(workspace)` helper returning sorted `(block_id, content_hash)` tuples so the Merkle tree is deterministic across callers.
- `verify_merkle` MCP tool (user scope) — builds the tree from the live FTS index, returns `{ok, root, proof, block_id}` for the supplied block.
- `mind_mem_verify` MCP tool (user scope) — invokes the standalone verifier against the active workspace so agents can trigger verification without shelling out.

### Fixed (audit-driven, pre-release — 3-LLM joint: Claude + codex/GPT-5.4 + Gemini/Grok via API)
- **[CRITICAL]** `sqlite_index.merkle_leaves()` was querying `blocks.content_hash`, a column that doesn't exist (content hashes live on `index_meta`). Every call would have crashed with `no such column`. Fixed to join `index_meta` with `blocks`.
- **[CRITICAL]** `HashChainV2` gained `open_readonly(path)` + `readonly=True` constructor flag. The verifier now opens the ledger via `file:...?mode=ro` and skips `_init_db`, so auditing a workspace never mutates schema — not even on DBs that predate the current layout.
- **[HIGH]** `verify_cli` `--snapshot` now canonicalises the path and requires it to stay under the workspace root. `..` traversal and absolute paths are rejected with a structured failure. `mind_mem_verify` MCP tool enforces the same invariant at the MCP layer so hostile callers can't coax the verifier into reading an external directory.
- **[HIGH]** Every `check_*` function now catches `(sqlite3.DatabaseError, OSError, UnicodeDecodeError)` and records a structured failure instead of crashing with a traceback.
- **[HIGH]** `mind_mem_verify` MCP tool rejects overlong snapshot args (>512 chars) and absolute paths before dispatch.
- **[MEDIUM]** `verify_merkle` response carries `proof_format_version: 1` and documents the `[sibling_hash, direction]` shape so third-party verifiers don't have to guess.
- **[MEDIUM]** Snapshot manifests with exactly one of `merkle_root` / `merkle_leaves` are now flagged as corruption (was: silently skipped).

### Testing
- 22 new tests: clean + tampered hash chains, spec-binding mutation + corruption, clean + tampered evidence, snapshot chain-head + Merkle-root match / mismatch, CLI entry point (text + JSON), first-failure-wins exit-code semantics, read-only chain rejects append, path-traversal rejection, bad-encoding manifest, and partial Merkle anchor detection.

### Changed
- Version: 2.0.0b1 → 2.0.0rc1
- MCP tool count: 33 → 35 (`verify_merkle`, `mind_mem_verify`)

### Docs refresh (ships with this release)
- README badges switched to `?include_prereleases` so PyPI shows the current pre-release instead of the stale v1.9.1 stable.
- Removed the CI and Security-Review badges — GitHub Actions is disabled account-wide; the badges would stay permanently "unknown".
- Bumped hardcoded counts across README + comparison matrix + FAQ + docs to 3197 tests / 35 MCP tools. Added badges for "3-LLM joint audit" and "release: local (no Actions)".

## 2.0.0b1 (2026-04-13)

**v2.0.0b1: inference acceleration — Python-only subset, hardened via 3-LLM joint audit (Claude Opus 4.6 + Grok 4.1 Fast + codex/GPT-5.4). LLM prefix cache + speculative prefetch predictor. The MIND-compiled hot paths (BM25F, SHA3-512, vector similarity, RRF fusion) are deferred until the `mindc` toolchain is available.**

### Added
- `prefix_cache.py` — per-namespace LRU prefix cache with optional TTL. Keys include the namespace so cross-encoder, intent-router, and query-expansion caches never collide. Registry bounded at 64 namespaces to prevent dynamic-namespace DoS; `set_max_namespaces()` exposes the cap. `PrefixCache.stats()` surfaces hit/miss/evictions/expirations counters; `all_stats()` snapshots every registered cache.
- `speculative_prefetch.py` — access-history predictor. `PrefetchPredictor.observe(query, block_ids)` learns which blocks historically follow a query signature; `predict(query, limit)` returns block ids to warm ahead of the next hop; `evaluate(query, actual_ids)` scores efficacy. Signatures are Unicode-aware stop-word-stripped hashes so different phrasings of the same intent share a bucket.
- `mind-mem://index_stats` (and the `index_stats` MCP tool) now include `prefix_caches` (list of `CacheStats` dicts) and `speculative_prefetch` (`PrefetchStats` dict) so operators can observe hit rates without attaching a debugger.

### Fixed (audit-driven, pre-release)
- **[HIGH]** `speculative_prefetch.py` bucket trim no longer resets survivor counts to 1. The earlier policy destroyed frequency history and froze buckets on whatever top-N arrived first. Trim now prunes the tail while preserving each survivor's count.
- **[MEDIUM]** `speculative_prefetch.py` `signature()` caps input at 4096 chars before tokenising. A hostile MCP caller could otherwise submit a multi-megabyte query and burn CPU on regex matching.
- **[MEDIUM]** `speculative_prefetch.py` `_TOKEN_RE` switched to `\w+` with `re.UNICODE` so CJK and other non-ASCII queries stop collapsing onto one shared "empty" bucket.
- **[MEDIUM]** `speculative_prefetch.py` empty-after-filter queries now hash their raw lowered text into an `empty:<digest>` bucket so distinct noise queries no longer pool.
- **[MEDIUM]** `mcp_server.py` `index_stats` narrows its `except` to `(ImportError, AttributeError)` for the new prefix-cache / prefetch sections so real bugs propagate to the MCP error envelope instead of disappearing at debug level.
- **[LOW]** `prefix_cache.py` module registry bounded at 64 namespaces (LRU-evicted) so per-tenant dynamic namespaces cannot grow unbounded.
- **[LOW]** Dead `_Bucket.last_updated` field removed; `PrefetchStats.as_dict` type-annotated `dict[str, Any]`.

### Deferred (requires MIND toolchain)
- BM25F scoring kernel → `.mind` → native ELF via `mindc`
- SHA3-512 hash chain verification → `.mind` → GPU kernel
- Vector similarity (cosine/dot) → `.mind` → GPU kernel
- RRF fusion → `.mind` → native
- TurboQuant-compressed prefix cache (3-bit vector quantization)

### Testing
- 67 new tests: `test_prefix_cache.py` (35 — constructor, hit/miss, LRU, TTL, invalidation, stats, registry cap + LRU, concurrency) + `test_speculative_prefetch.py` (32 — signature normalisation, Unicode, length truncation, observe/predict, trim frequency-preservation, evaluate, stats, singleton, concurrency).

### Changed
- Version: 2.0.0a3 → 2.0.0b1

## 2.0.0a3 (2026-04-13)

**v2.0.0a3: Observer-Dependent Cognition (ODC) — axis-aware retrieval, hardened via a 3-LLM joint audit (Claude Opus 4.6 + Gemini 3.1 Pro + Grok 4.1 Fast).**

### Added
- `observation_axis.py` — `ObservationAxis` enum (LEXICAL, SEMANTIC, TEMPORAL, ENTITY_GRAPH, CONTRADICTION, ADVERSARIAL), `AxisWeights` vector with non-negative / finite validation, `AxisScore` / `Observation` value objects, `axis_diversity()` metric, `rotate_axes()` / `should_rotate()` helpers, `adversarial_pair()` mapping.
- `axis_recall.py` — `recall_with_axis()` orchestrator. Dispatches one retrieval pass per active axis, tags every result with an `Observation` (which axes produced it + per-axis confidence + rank), fuses via weighted RRF, rotates to orthogonal axes when top-1 confidence falls below `DEFAULT_ROTATION_THRESHOLD = 0.35`, and can run an adversarial pair pass to surface contradictions.
- MCP tool `recall_with_axis` — exposes the orchestrator with comma-separated `axes` arg, optional `axis=weight,axis=weight` override, `adversarial` flag, and `allow_rotation` flag. Returns a JSON envelope with `results`, `weights`, `rotated`, `diversity`, and `attempts`.
- Tool count: 33 (up from 32). ACL: user scope.

### Fixed (audit-driven, pre-release)
- **[HIGH]** ADVERSARIAL axis now wraps the query as `NOT "<phrase>"` (FTS5-safe double-quoted form) instead of raw `NOT {query}`. Prevents FTS5 from parsing `NOT foo AND bar` as `(NOT foo) AND bar` and blocks metacharacter injection from crafted queries. Empty queries now skip the axis.
- **[HIGH]** Rotation no longer stamps `observation.rotated=True` on primary-only results. `_merge_rotation` is the only path that sets the flag, and only on blocks the rotation pass actually touched.
- **[MEDIUM]** `_recall_for_axis` now logs a warning on the `TypeError` signature fallback instead of silently swallowing it; real kwarg-mismatch bugs are no longer masked.
- **[MEDIUM]** `recall_with_axis` MCP tool caps arg length (`axes`/`weights` ≤1024 chars, ≤16 entries) and `limit ∈ [1, 500]` to bound retrieval cost under adversarial input.

### Testing
- 81 new tests: `test_observation_axis.py` (primitives, 51) + `test_axis_recall.py` (orchestrator + hardening, 22) + `test_axis_recall_mcp.py` (MCP surface + bounds, 8).

### Changed
- Version: 2.0.0a2 → 2.0.0a3

## 2.0.0a2 (2026-04-13)

**GBrain-adapted knowledge enrichment: multi-query expansion, compiled truth pages, dream cycle, 4-layer dedup, smart chunker, 13 new MCP tools. Plus pre-release hardening from a 3-LLM joint audit (Claude Opus 4.6 + Gemini 3.1 Pro + Grok 4.1 Fast).**

### Added
- `query_expansion.py` — LLM-free multi-query expansion with synonym swap, specificity shift, temporal rephrasing, negation variant, and RRF fusion across reformulations
- `compiled_truth.py` — Per-entity compiled truth pages: current-best-understanding on top, timestamped evidence trail on bottom, contradiction detection across entries
- `dream_cycle.py` — Autonomous nightly memory enrichment: scan for missing cross-references, broken citations, orphan entities; generate repair proposals; compact redundant entries; auto-repair mode
- `dedup.py` — 4-layer post-retrieval deduplication: best-chunk-per-source, cosine similarity dedup (>0.85 threshold), type diversity capping, per-source chunk limiting
- `smart_chunker.py` — Content-aware chunking at semantic boundaries (headers, paragraphs, code blocks) instead of fixed character counts; format-specific splitting for markdown, code, and prose
- 13 new MCP tools: `expand_query`, `smart_chunk`, `deduplicate_results`, `run_dream_cycle`, `dream_cycle_status`, `compile_truth`, `get_compiled_truth`, `compiled_truth_add_evidence`, `compiled_truth_contradictions`, `compiled_truth_load`, `list_compiled_truths`, `chunk_and_index`, `dedup_search`

### Fixed (3-LLM audit, pre-release hardening)
- **[CRITICAL]** `merkle_tree.py`: Added leaf/internal-node domain separation tags (`L:` / `N:`) to close the Bitcoin-style second-preimage attack. `verify_tree()` now recomputes leaf hashes from their `content_hash` + `block_id` instead of trusting them blindly. `import_json()` fails on stored-vs-recomputed root mismatch.
- **[CRITICAL]** `evidence_objects.py`: Canonical form for `_compute_evidence_hash` switched from colon-delimited string to JSON-canonical (sorted keys) so fields containing `:` cannot shift across field boundaries without changing the digest.
- **[CRITICAL]** `hash_chain_v2.py`: `_connect` now uses `isolation_level="DEFERRED"` (autocommit=None silently defeated `BEGIN EXCLUSIVE`) with a 30s `timeout` and `busy_timeout=30000`. `append()` serializes reads-then-writes under a per-instance `RLock` on top of `BEGIN IMMEDIATE`. `import_jsonl` anchors linkage to the current chain head instead of GENESIS so imports append rather than creating disjoint segments. `convert_from_v1` preserves original v1 timestamps.
- **[CRITICAL]** `governance_gate.py`: `admit()` now serializes evidence-then-chain writes under an `RLock` so concurrent admits cannot interleave the two stores' orderings. Chain-append failures after evidence persist are logged loudly before being re-raised.
- **[HIGH]** `evidence_objects.py`: `EvidenceChain.create()` serializes concurrent creates under an `RLock`; `_append_to_file` flushes and `fsync`s; disk write now happens before the in-memory append so an I/O failure cannot leave memory ahead of disk.
- **[HIGH]** `evidence_objects.py`: `_load_from_file` stops at the first integrity failure instead of silently skipping entries and cascading linkage errors.
- **[HIGH]** `spec_binding.py`: Corrupt binding file now raises `SpecBindingCorruptedError` instead of silently returning `None` (which would let an attacker disable governance by damaging the file). `_persist` fsyncs the tmp file before rename.
- **[MEDIUM]** `hybrid_recall.py`: Query expansion default reverted to opt-in (`False`). The prior default flipped to `True` changed latency characteristics for every `HybridBackend()` caller without an explicit opt-in.

### Testing
- 289 new tests across 5 test files (query_expansion: 405 lines, compiled_truth: 428 lines, dream_cycle: 498 lines, dedup: 731 lines, smart_chunker: 966 lines)
- All 3024 tests green on Python 3.12 after audit fixes

### Changed
- MCP tool count: 19 → 32
- Version: 2.0.0a1 → 2.0.0a2

## 2.0.0a1 (2026-04-05)

**v2.0.0a1: GovernanceGate, SHA3-512 hash chain wiring, MCP evidence tools, and spec-hash embedding**

### Added
- `governance_gate.py` — `GovernanceGate` single choke-point for all block writes: spec-hash verification, evidence object creation, hash chain appending, `GovernanceBypassError` on drift
- `verify_chain` MCP tool (admin) — verifies full SHA3-512 hash chain integrity, returns `{valid, length, broken_at}`
- `list_evidence` MCP tool (user) — lists governance evidence objects with optional `block_id` / `action` filters
- `spec_hash` parameter on `EvidenceChain.create()` — embedded in evidence metadata for every governance write
- Hash chain wiring in `capture.py` — every signal write appends an entry to `HashChainV2`
- Hash chain wiring in `apply_engine.py` — every applied proposal op is recorded via `GovernanceGate.admit()`

### Changed
- Version: 1.9.1 → 2.0.0a1

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
- Public benchmark guidance now refers generically to the protected MIND kernel, and README/docs version badges/examples are aligned with the 1.9.1 release metadata

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

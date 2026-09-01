# Restore-44 Wiring Plan — from restored to reachable-and-working

> Architectural plan for mind-mem 5.1.0, produced 2026-09-01. Saved here because
> the session that produced it had no write permission.
>
> **Operator amendments to §4, which override the plan's recommendation:**
> 1. The six not-wired modules **stay in the tree**. The standing rule after the
>    5.0.0 incident is *substitution or replacement, never deletion, unless the
>    capability is genuinely not needed* — and "not wired yet" is not the same
>    as "not needed". They are carried as documented debt with re-entry
>    conditions, not unstaged.
> 2. `storage/sharded_pg.py` **was restored** (497 LOC) after the plan was
>    written — the plan's "absent, not resurrected" note is stale. It joins the
>    not-wired list rather than the deleted one.


Ships as **5.1.0** on top of the published 5.0.0. Everything lands flag-gated default-OFF; flag-off behaviour byte-identical to 5.0.0.

## Ground truth you need before executing (some of it corrects the brief)

1. **`storage/sharded_pg.py` was NOT actually restored.** It exists neither on disk nor in the git index (`git show :src/mind_mem/storage/sharded_pg.py` fails); the branch holds 43 source modules, not 44. Meanwhile live code dangles on it: `protection.py:80` lists it in `_CRITICAL_MODULES` (a strict-protection failure risk) and `tests/test_governed_write_paths.py:105,163` allowlist a class that can't be scanned — that test passes vacuously.
2. **The branch's suite is red by design**: live `tests/test_documented_surfaces_exist.py:191` asserts 11 restored `v4.*` modules are NOT importable, and `:184`/`:198` require the "HISTORICAL / removed in 5.0.0" banner in `docs/v4-release.md`.
3. **`scripts/reachability_baseline.txt` was emptied 2026-08-31**, so all 39 unreachable restored modules fail `--check` as new debt. (Four of the 43 count as reachable only via other restored modules: `session_summarizer`←`bootstrap_corpus.py:25`, `cognitive_kernel`+`surprise_retrieval`←`v4/kernels.py:56,62`, `lint`←`lint_autofix.py:34`.)
4. **No restored module bypasses the HITL gate as written.** The 5.0.0 concern about `ingestion_pipeline` was about a *future* drain consumer — the module itself never touches BlockStore; its only persistent write is its own WAL. The v4 DB-writers (`block_kinds`, `v4/block_metadata`, `pq`, `hnsw_kind_index`, `kind_summaries`) write only the `index.db` side-store, never the corpus. The one ungoverned corpus-adjacent write in the whole set is `session_summarizer.py:332-338`'s append to `summaries/daily/*.md` — a derived artifact, and its linking signal already goes through `capture.append_signals` (→ `Status: pending`).
5. **Three flags are missing from `ALL_V4_FLAGS`** (`v4/feature_flags.py:49`): `cognitive_kernel`, `surprise_retrieval`, `lint` — `require_enabled()` raises unknown-flag for all three today.
6. **`maintenance_migrate`'s docstring names an `apply_engine.apply_proposal` auto-invocation that was never written** — the wiring was always missing, not swept.
7. **`trajectory.py:42` has a config-path bug** (`src/mind/trajectory.mind` vs repo-root `mind/trajectory.mind`), so every kernel knob silently defaults.

## Invariants this plan does not touch

- **EvidenceAction stays a closed 7-member enum.** Every new door maps through `governance_gate._map_action` (`governance_gate.py:596`; unmapped verbs default to APPLY). RA.3's DEMOTE/ARCHIVE/FORGET stays deferred to a versioned wire change — strict `from_dict` (`evidence_objects.py:268`) breaks on new members.
- **Every new ingest door mints unservable** using the *existing* `IngestTier.EXTERNAL_INGEST → Status.QUARANTINED` row (`enums.py:205`) via the `inbox.py:165-188` pattern: `get_gate(ws).admit_block(..., tier=IngestTier.EXTERNAL_INGEST)` + `stamp_transform_hash`. No new tier, so `test_ingest_tiers.py` and `test_quarantine_redteam.py` hold unmodified.
- **Corpus writes**: programmatic code may *stage* (`Status: staged` into `intelligence/proposed/EDITS_PROPOSED.md`, the `lint_autofix`/`propose_import_release` pattern); applying stays exclusively `approve_apply` → `apply_engine.apply_proposal`.

## Landing mechanics

Hand-seed the 39 unreachable names into `scripts/reachability_baseline.txt` with a dated ratchet header. The gate fails only on *additions*, so every slice shrinks the file and CI stays green throughout; end state is empty again. Rule: **no module gets a reachability ALLOWLIST entry as its wiring** — ALLOWLIST only for genuine string dispatch (`cron_runner.JOB_DEFS` jobs, which the AST scan cannot see). Every new MCP tool needs ACL classification in exactly one of `ADMIN_TOOLS`/`USER_TOOLS` (`mcp/infra/acl.py:51/112` — a tool in neither is *silently unreachable*, the exact ledger_anchor lesson), plus a test/doc/train mention (`check_tool_surface.py --check 0`) and a `count_mcp_tools.py --check` bump.

---

## 1. WIRING PLAN — five slices, 38 of 44 wired

### Slice 0 — unblock the branch (repair, no wiring)
Remove restored modules from `_REMOVED_V4_SURFACES` in `tests/test_documented_surfaces_exist.py:150-198` and drop the banner tests; update `docs/v4-release.md`; fix stale sweep comments (`test_security_scanning_alerts.py:9`, `test_medlow_batch12_regressions.py:15`, `test_v4_circuit_breaker.py:453`); add the three missing flags to `ALL_V4_FLAGS`; prune the `sharded_pg` dangling refs (`protection.py:80`, `test_governed_write_paths.py:105,163`, ANATOMY/ROADMAP notes); fix `trajectory.py:42`; seed the reachability baseline; unstage the six not-wired modules (§4). Exit: pytest green, `--check` green.

### Slice 1 — correctness wins (buys immediate user value, small diffs)
| Module | Plug point | Working = |
|---|---|---|
| `uncertainty_propagation` | `graph_recall.py`/`iterative_recall.py` expansion loops: `propagate()` + `should_truncate()` prune; `chain_confidence()` in `traverse_graph`/`recall_with_guardrails` envelopes | 3-hop chain @0.9/hop shows strictly decreasing adjusted confidence; sub-0.1 branch truncated. This fixes a real overconfidence bug: hop-3 confidence currently reports as if hop-1 |
| `retrieval_trace` | `with step(...)` wrappers around the `_maybe_*` hooks (`hybrid_recall.py:1020-1374`); `trace.summary()` in the recall envelope under `retrieval.trace_attribution` (key already exists) | graph-expanded query returns `steps` with `graph_expand.added_count > 0` |
| `feature_gate` | migrate `graph_recall.py:163/196` onto `FeatureGate` (its `multi_hop_detector` was written for exactly this); then the other six hand-rolled copies (`entity_prefetch`, `session_boost`, `kg_fusion`, `trust_scores`, `truth_score`, `retrieval_trace`) | `test_feature_gate.py` + existing `graph_recall` tests green across the swap |
| `maintenance_migrate` | write the missing call site: first-run detection in `apply_engine.apply_proposal` (`apply_engine.py:1207`); `mm migrate --maintenance` alias | flat `maintenance/` splits; second run zero moves; `test_atomicity_maintenance_scope.py` green |
| `v4/logging_context` | install `StructuredLogFilter` on `observability.StructuredLogger`'s **own handler** — NOT the root logger; `propagate=False` makes root-install a silent no-op — and `@with_correlation_id` on `mcp_tool_observe` (`mcp/infra/observability.py:74`) | two concurrent recalls emit distinct `correlation_id` on every line |

### Slice 2 — governed content quality
| Module | Plug point | Working = |
|---|---|---|
| `smart_chunker` | swap the naive sentence-window `chunk_text` at `_recall_core.py:1216`; vector seam `recall_vector._augment_for_embedding` (`:314`); inbox text ingest (`inbox.py:130`). The config keys `chunk_overlap`/`max_chunk_tokens` already sit **unread** in `_VALID_RECALL_KEYS` (`_recall_constants.py:514`) — wire them | header-aligned chunk boundaries; default-config recall byte-identical |
| `granularity_align` | extra section of `plan_consolidation` (`mcp/tools/consolidation.py:48`); merge outputs feed `approve_apply`. Proposal-only by contract (`granularity_align.py:394-399`) | candidates in the plan; corpus unchanged until approved |
| `lint` + `lint_autofix` | `mm lint [--fix LF-id]` (the `mm_cli.py:2511` + `set_defaults(func=…)` pattern) + MCP `lint` (USER) / `lint_autofix` (ADMIN — it stages proposals, same class as `propose_update`) | `lint → lint_autofix → approve_apply` flips a duplicate decision to superseded; stable `LF-` ids across runs. Already HITL-correct by construction |
| `v4/vocabulary` + `v4/block_metadata` | `block_metadata.validate_block` (backed by `vocabulary`) as pre-propose hook in `mcp/tools/quality.validate_block:35` + advisory check in `propose_update` (`governance.py:308`), flag-gated | reject-mode out-of-vocabulary raises **before** any write; flag-mode logs and proceeds |
| `core_export` | (a) new MCP `export_core(name, format=okf\|jsonld\|markdown)` beside `build_core`/`load_core` in `mcp/tools/core.py`; (b) OKF **import** routes `import_okf_bundle` output through `importers/quarantine.py` (EXTERNAL_INGEST → QUARANTINED → `propose_import_release`); add `TRAJ-` to `_ID_PREFIX_TYPE` (`core_export.py:146`) | OKF round-trip preserves fields + receipt re-derives (its two restored test files); foreign trust claims stay parked under `OkfClaim*` |

### Slice 3 — ingest doors (all through the gate)
| Module | Plug point | Working = |
|---|---|---|
| `ingestion_pipeline` | `mm ingest-serve`: `serve_webhook` + `IngestionQueue` + WAL; **the drain consumer we now write** routes every event through `admit_block(..., tier=EXTERNAL_INGEST)` + `stamp_transform_hash` — the inbox pattern exactly. This closes the 5.0.0 bypass concern by construction: the drain path IS the gate | POST `/ingest` → block on disk `Status: quarantined`, invisible to recall (canary-token test in the `test_quarantine_redteam.py` style); WAL replay after kill loses nothing |
| `streaming` | its `_PerClientRateLimiter` fronts the webhook (`streaming.rate_limit` config). Its duplicate queue is **not** used — `IngestionQueue` (reject-new + WAL) wins the queue role, because silent drop-oldest loses governed data invisibly; deprecate `StreamingIngestQueue` in-module | two `client_id`s get independent 429s; depth via the existing `stream_status` tool |
| `event_fanout` | `create_fanout(config)` at server startup; emit `proposal_applied`/`rollback_executed` from `mcp/tools/governance.py:827/991`, `contradiction_detected` from the detector, tier events from `memory_tiers.py:167`. Payloads carry **ids + hashes only** — `LoggingPublisher` dumps payloads into logs, a leak surface for block content | one `proposal_applied` log line per apply; Redis down → apply still succeeds |
| `multi_modal` | replace the `inbox._ingest_image:190`/`_ingest_audio:199` `NotImplementedError` stubs: sidecar description/transcript → Image/Audio block via the same quarantine write | dropped `.png` + sidecar yields a quarantined `type: image` block with stable `thumbnail_hash`; `pack_recall_budget` counts it via `modal_token_cost` |
| `session_summarizer` | new `session_summary` job in `cron_runner.JOB_DEFS` (`:55`) + `daemon.DEFAULT_INTERVALS`/`_TASK_RUNNERS` (`daemon.py:59/148`), default off. Needs a reachability ALLOWLIST entry (string dispatch) and a `test_governed_write_paths.py` allowlist entry for the `summaries/daily/*.md` append, with rationale (derived artifact; the signal side is gated) | `mm daemon --once` writes a dated summary + a `Status: pending` signal; hash dedup prevents rewrite |
| `bootstrap_corpus` | `[project.scripts]` entry `mind-mem-bootstrap` (`pyproject.toml:122-137`); one-shot post-init. Add a smoke test for `main()`'s four phases — currently untested | `--dry-run` reports N signals; real run makes `recall` return transcript-sourced blocks, all pending |
| `v4/backpressure` | producer loops: webhook drain, `change_stream.py:213` (its own comment asks for it), daemon consolidation | 1500→overloaded, 900→still (hysteresis), 100→recovered |

### Slice 4 — v4 retrieval stack (land in this order; all flag-gated)
`v4/observability` → `surprise_retrieval`+`cognitive_kernel` → `block_kinds` → `kind_summaries` → `embedding_pipeline` → `pq` → `hnsw_kind_index` → `kernels` → `health`.

| Module | Plug point | Working = |
|---|---|---|
| `v4/observability` | metric registry wired into `mcp_tool_observe` (per-tool counters/histograms); `snapshot()` via `index_stats`/`memory_health`. **Rename its `timed` → `timed_metric`** — collides with live `observability.timed` with an incompatible call shape | recall increments a named counter visible in `snapshot()`; cardinality cap holds |
| `cognitive_kernel` + `surprise_retrieval` | `kernel=` kwarg on `_recall_core.recall` (`:669`), surfaced via `recall_with_axis`; flags added in Slice 0 | flag-off path byte-identical to v3 recall; identical vectors → surprise 0.0, opposite → 1.0 |
| `v4/kernels` | the wiring point must `import mind_mem.v4.kernels` explicitly — registration is an import side effect (`kernels.py:398`, documented trap); export a public accessor to replace the fragile `retrieval_graph._db_path` private import (`kernels.py:55`) | `available_kernels()` returns all 5; `graph_walk` degrades to DEFAULT with no `co_retrieval` table |
| `v4/embedding_pipeline` | `set_embedder` pointed at `recall_vector`'s real provider chain (`recall_vector.py:532`); its stdlib embedder is offline-fallback only (per-process hash salt = cross-process unstable, documented) | `derive_embeddings` returns vectors for ids present only in `recall.db` |
| `v4/block_kinds` | **write the missing writer** — today `blocks.kind` has readers and zero writers, so `list_blocks_by_kind` returns `[]` forever: set kind at ingest (inbox/`graph_ingest`) + `mm kinds backfill` | backfill → non-empty kind listing; `index.db` side-store only, corpus untouched |
| `v4/kind_summaries` | daemon/cron refresh job; surfaced beside `category_summary` | `refresh_summary` → `get_summary` round-trip, `block_count` matches |
| `v4/pq` | opt-in vector compression at `reindex` on the local index path (`recall_vector._index_local:1001`) | `decode(encode(v))` error under threshold; asymmetric-distance ranking agrees with exact cosine |
| `v4/hnsw_kind_index` | `find_similar` gains `kind=` filter; `backend_status` in `index_stats`. Honest labeling: it's brute-force cosine today (its own docstring says so) — flag docs must not claim HNSW-the-algorithm | the module's own brute-force-equivalence bar (`:25`) |
| `v4/health` | `register_health_probe` contributions folded into `memory_health` + a `/healthz` route on `http_transport.py` (path const near `:137`, same guard prelude) | a probe raising `SystemExit` still returns `status="fail"` (`health.py:230`); `disabled` vs `missing` distinguished |

### Slice 5 — learning, metrics, taxonomy
| Module | Plug point | Working = |
|---|---|---|
| `llm_noise_profile` | `record_outcome` from the `report_outcome` path (`outcome_store`); persist `intelligence/llm_profiles.json`; per-provider reliability in `calibration_stats` | outcomes move per-domain reliability; survives restart |
| `tracking` | three-way split: `model_context_window` → `pack_recall_budget`; `MRRTracker` → the baseline `online_trainer` gates on + drift in `index_stats`; `PackingQualityMeter` → pack path. **Write tests (it has none)**; fix the silent-32000 default for unlisted models | packing respects real windows; MRR delta visible |
| `online_trainer` | harvest half: daemon job draining live `interaction_signals` → `build_training_tuples`, counts via `signal_stats`. Registry half: **merge promote/revert semantics into live `model_gate.py`** (the tested incumbent) rather than shipping a second untested registry; persist the revert log (in-memory today). Write tests | `correction` signal yields tuple with prior results as negatives; sub-`min_improvement` promotion refused and logged |
| `mrs` | daemon job feeding `retrieval_slis()` from drift/contradiction/staleness detectors, score through `alerting.py` + `memory_health`. Write tests | p99 breach → `score < 100`, `"p99_ms"` in violations |
| `error_codes` | **additive only**: MCP error envelopes gain `"code": "MM-xxxx"` (start with `governance.py`, `recall.py`). The typed-exception hierarchy stays — `apply_engine` control flow depends on it; replacing types with `ValueError(error_message(...))` would be a regression | clients branch on `code` instead of regexing prose |
| `trajectory` | capture: `report_outcome` optionally writes `TRAJ-` files to `<ws>/trajectories/`; recall: new `similar_trajectories(task)` MCP tool (USER) | 60-days-apart similar trajectories score below same-day ones (decay tests exist); kernel knobs now actually load |
| `mind_kernels` | becomes the canonical pure-Python fallback that `mind_ffi.py` binds when `libmindmem.so` is absent — one loader (retire `load_kernels`' weaker duplicate in favor of `mind_ffi.py:115-149`'s allowlisted one). First check whether `hash_chain_v2` already carries the v1→v3 downgrade-monotonicity guard (`mind_kernels.py:93-104`); if not, port it | `bm25f_score` parity with `_recall_scoring` on a fixture corpus; chain verify rejects a v3→v1 downgrade |
| `consensus_vote` | consulted by `conflict_resolver.analyze_contradiction` (`conflict_resolver.py:101`) when strategy is MANUAL, behind `governance.consensus.enabled` (default off, `min_votes=2`); winner becomes a **proposal**, never an apply | single-operator: `insufficient_votes` → manual, exactly as today; two-voter fixture reaches quorum and stages |

---

## 2. DUPLICATES — rulings

| Restored | Incumbent | Ruling |
|---|---|---|
| `v4/observability.py` | `observability.py` (132 importers) | **Keep both — different jobs** (logging vs typed metric registry). Wire v4 as the metrics registry; rename its `timed`→`timed_metric`. Fold into `observability.Metrics` post-5.1, not now. |
| `tenant_kms.py` | `encryption.py` | **Incumbent stands; tenant_kms not wired** (§4) — different layers, but the tenant dimension doesn't exist. |
| `core_export.py` | `export_memory` (`memory_ops.py:427`, jsonl-only) | **Keep both — zero API overlap** (workspace→jsonl vs `LoadedCore`→OKF/JSON-LD/markdown+round-trip). Wire as `export_core`. |
| `mind_kernels.py` | `mind_ffi.py` + `_recall_scoring` + `hash_chain_v2` | **Merge into incumbents**: one loader, mind_kernels as `mind_ffi`'s fallback; port the downgrade guard if absent. |
| `v4/kernels.py` | `mind/` `.mind` kernels + `mcp/tools/kernels.py` | **Keep — different domain** (retrieval strategies vs scoring configs). "Kernel" is triply overloaded; document the three meanings. |
| `lint`+`lint_autofix` | `quality_gate.py` | **Keep both — complementary timing** (pre-storage candidate filter vs post-storage corpus scan, zero rule overlap). Import the field lists instead of the copies at `lint.py:81-84`. |
| `error_codes.py` | ~40 typed exception hierarchies | **Additive envelope codes only; never replace the types** — the only true design conflict in the set. |
| `v4/health.py` | `mm doctor` + `ping()` + `memory_health` | **Keep — complementary** (repair-capable diagnostics vs never-raising probe). Wire as contributor + `/healthz`. |
| `v4/logging_context.py` | `StructuredLogger` | **Merge-wire** — genuinely additive (correlation IDs the incumbent lacks). Handler-install, not root. |
| `v4/embedding_pipeline.py` | `recall_vector.py` | **Incumbent owns embedding**; wire the batch-fetch plumbing with `set_embedder` → `recall_vector`. |
| `v4/block_metadata.py` | live `block_metadata.py` (`block_meta` table) | **Keep both — different tables, different DBs**; add cross-referencing docstrings; correct its false "gate-only" docstring claim for `set_block_metadata`. |
| `streaming.py` | `ingestion_pipeline.py` | **ingestion_pipeline wins the queue role** (reject-new+WAL over silent drop-oldest); streaming survives as the per-client rate limiter. |
| `turbo_quant.py` | `v4/pq.py` (restored, flagged, tested) | **pq strictly dominates** (~96× vs ~6×, real tests, registered flag). turbo_quant not wired (§4). |
| `memory_mesh.py` | `v4/federation*` | **Incumbent stands** — same ideas with the transport actually built. Not wired (§4). |
| `governance_raft.py` | `v4/federation` + `apply_engine` | **Incumbent stands**; salvage `sign_proposal`/`verify_proposal` HMAC for `/federation/write` auth. Not wired (§4). |
| `maintenance_migrate.py` | `schema_version.py` | **Keep both — disjoint targets** (dir layout vs schema version); wire the missing call site. |
| `feature_gate.py` | 7 hand-rolled live copies | **The restored module IS the deduplication of the incumbents** — migrate them onto it. |

## 3. ORDER — what each slice buys

**0** unblocks the branch (suite green, ratchet seeded). **1** buys immediate correctness: the multi-hop overconfidence fix, per-request retrieval attribution, the missing `apply_engine` migration call, correlation IDs — five small, independent diffs. **2** buys governed content quality: structure-aware chunking, consolidation merge candidates, corpus lint with HITL autofix, vocabulary enforcement, and the only non-jsonl export/import path. **3** buys the new ingest doors, every one through `admit_block` + quarantine. **4** buys the v4 retrieval stack behind flags, in strict dependency order. **5** buys the learning/metrics loop and needs the most new test-writing (`tracking`, `online_trainer`, `mrs` have zero tests today).

Per-slice exit gate: pytest green · reachability baseline shrunk · `check_tool_surface.py --check 0` + `count_mcp_tools.py --check` bumped (new tools: `lint` USER, `lint_autofix` ADMIN, `export_core` USER, `similar_trajectories` USER → 98→102) · flag-off byte-identity for anything on the scored path · `mm verify` green.

## 4. NOT WIRED — six modules, burden carried

1. **`api/grpc_server.py`** — a second door to `recall` and `approve_apply` with **no auth, no TLS** (only a loopback bind check); `grpcio` isn't in pyproject at all and no `.proto` ships, so it cannot serve without operator-written protobufs. *Loss*: low-latency RPC — fully covered by the authenticated REST transport. Re-entry condition: token-parity auth + shipped proto, as an `api` extra.
2. **`governance_raft.py`** — ships only `LocalConsensusLog`, which commits immediately; wiring wraps `apply_engine` in an indirection where `is_leader()` is forever true. *Loss*: none observable — there is no consensus to lose. HMAC proposal-signing salvaged for federation.
3. **`memory_mesh.py`** — strictly dominated by live `v4/federation*` (real endpoints, auth, response caps, merge strategies). *Loss*: none; wiring it creates two competing peer-sync vocabularies.
4. **`tenant_kms.py`** — per-tenant KEK/DEK in a product whose tenancy unit is one workspace per process; `tenant_id` would be a constant. *Loss*: key isolation for a hosted multi-customer operator who doesn't exist in this product's shape. Best-built of the six; first back in if hosting ships. The `v4.tenant_kms` flag stays registered as the hook.
5. **`tenant_audit.py`** — splits governance history into per-tenant JSONL nothing reads or verifies; single-operator it relocates `audit.jsonl` into a pointless subdirectory. *Loss*: per-tenant compliance export — same nonexistent operator. Salvage: its defensive `verify()`-shape handling (`tenant_audit.py:216-235`) into `audit_chain` consumers.
6. **`turbo_quant.py`** — min/max scalar quantization at ~6×, dominated by the wired `v4/pq.py` at ~96× with tests and a flag; the TurboQuant-paper framing oversells what's implemented. *Loss*: nothing.

Plus **`sharded_pg`** (absent, not resurrected): *loss* is Citus-style horizontal sharding — no deployment has >1 PG node, and `ReplicatedPostgresBlockStore` covers read scaling. Dangling refs pruned in Slice 0.

---

Two operator decisions this plan surfaces rather than makes: (a) confirm the six not-wired modules leave the branch again — they're staged, so this is an unstage, and each is one `git show fcbbea3:` away if the loss ever materializes; (b) the `sharded_pg` discrepancy means the restore was 43/44 — if you want it back anyway, it needs restoring from `8dc370d^` first and would join the not-wired list. I couldn't save this to `docs/plans/RESTORE-44-WIRING-PLAN.md` (write permission denied in this session) — the text above is complete and self-contained.

# MIND-Mem Roadmap

> **Reality check (2026-05-20 honesty pass):** the v3.2.0+ sections
> below were drafted ahead of the v3.9 → v4.0 release ladder and many
> items shipped without being checked off here. The bulk of v3.2.0
> through v4.0.0 Groups A/B/C/D/E/G is done in code; see
> `CHANGELOG.md` for the canonical version-by-version record. This
> file now flips checkboxes to match the shipping state and surfaces
> only the items that remain *genuinely* open.

## Currently shipping

See `CHANGELOG.md` for the per-version detail (latest: v4.9.1) and the
canonical current PyPI release; this roadmap covers forward-looking work,
not history.

## Genuinely Open Items (post-v4.4.0 reality)

Surfaced at the top so the actual remaining work is visible without
scrolling 1500 lines of historical sections. Each item is followed
by its full description below.

### Group D — Network hardening (3 items; +1 shipped in v4.0.14)

- [ ] **TLS 1.3 minimum + cert pinning** on REST / gRPC / MCP-HTTP
- [ ] **mTLS for service-to-service** between mind-mem nodes
- [ ] **Public / private workspaces** (`workspace.mode = public | private | mixed`)
- [x] **Audit headers** (`X-MindMem-Request-Id`, `X-MindMem-Actor`, `X-MindMem-Purpose`) — **shipped v4.0.14** (REST middleware; gRPC parity TODO when gRPC surface gets the same treatment)
- [ ] **ActivityPub federation interop** (optional bridge; low priority)

### Group B — Knowledge graph (2 items)

- [x] **Block versioning + time-travel** — **shipped** (`v4/block_versioning.py`: `block_history(block_id)` + `content_as_of(...)` over the applied-edit chain; `recall(..., as_of=date)` now plumbs the point-in-time projection through every backend)
- [x] **Conversational chat layer** — **shipped** (`chat_memory.py` + `chat_citations.py` + `chat_generators.py`): `chat_with_memory(workspace, question)` returns an answer whose every claim sentence carries a `[[block_id]]` that is resolved against the block store before the call returns. Ungrounded or fabricated ids raise `CitationError` (or reject, per `on_invalid`); empty recall returns the literal `"no record found"` without invoking a generator at all. The answerer is a dependency-injected seam — the in-box `extractive_generator` is deterministic and offline, the service adapter is opt-in. Surfaced as a Python API, the `chat_with_memory` MCP tool, and the `mind-mem-chat` console script.

> See also **Group K — Graph-from-text construction + edge-grounded recall**
> (below): description-grounded entity resolution, blocking + LLM-arbitration,
> answer-with-edge-citations, cross-source edge confidence, schema versioning,
> degree-gated profile synthesis — and **K.2 — Ontology-governed logical
> knowledge graph**: making `ontology.py` executable on the graph write paths,
> relation-level semantics (domain/range/inverse/transitive), a bounded
> deterministic reasoner, and `GraphEvidencePack`.

### Group C — KG governance / UX (6 items)

- [ ] **AI lint with auto-fix** — `lint_autofix(workspace, finding_id)` tool
- [ ] **Local visual viewer** — `mm view` web UI (stdlib HTTP + JS/D3)
- [x] **Auto-generated hierarchical index** — `index.md` + `log.md` autogen
- [ ] **Adversarial / poisoning defense** — per-actor anomaly, canary blocks
- [x] **Ingest codepoint sanitization** — strip/normalize invisible-Unicode (zero-width, Unicode tag chars, bidi controls) on blocks + proposals *before* they reach the HITL gate or any model. Defensive takeaway from the GLOSSOPETRAE "Lingua Ex Machina" report ([elder-plinius/GLOSSOPETRAE](https://github.com/elder-plinius/GLOSSOPETRAE), Jun 2026): its covert-channel finding shows payloads hidden in invisible Unicode pass a monitor whose tokenizer silently deletes them before its model sees them (100% receiver recovery, 0% monitor detection). Cheap fix = NFC-normalize + reject/flag disallowed codepoint categories at ingest.
- [x] **Memory reputation / trust scores** — shipped as the validity gate's FIFTH component, not a separate subsystem: `src/mind_mem/provenance_class.py` classifies a block `operator` > `agent-verified` > `agent-inferred` > `external-ingest` from the existing `ActorRole` / `ToolId` / `Source` fields, and `validity_gate.validity_components()` folds that weight into the composite `V`. No per-actor learned or anomaly scoring (determinism wedge). `trust_scores.apply_trust_scores` keeps the `actor_trust` field as a thin façade over that one component. Opt-in via `recall.validity_gate.provenance_class.enabled`; flag-off recall ordering is byte-identical. 48 tests.

### Group E — Compliance (5 items)

- [x] **Time-bounded and event-bounded recall** — `since` / `until` / `event_id` filters
- [x] **Vocabulary-bound fields** — per-workspace controlled vocabularies
- [x] **Provenance-rich blocks** — `actor_id`, `actor_role`, `session_id`, `tool_id`, `purpose`
- [ ] **Row-level encryption on top of tenant KMS** — per-tenant KMS envelope keys **ship** (`tenant_kms.py`, real AESGCM); the open half is wiring row-level encryption over those keys
- [ ] **C2PA content provenance** — signed manifests on synthesis blocks

### Group G — Ecosystem (9 items, mostly SDK fan-out)

- [ ] **JavaScript / TypeScript SDK** — client **exists in-tree** (`sdk/js/`); open work is packaging/publishing it as `@star-ga/mind-mem-client` on npm
- [ ] **Browser-native WebAssembly bundle**
- [ ] **Go SDK publish + Rust / Java / Ruby SDK stubs** — Go client **exists in-tree** (`sdk/go/`, with tests); open work is module publishing; Rust/Java/Ruby not started
- [ ] **OpenAPI + AsyncAPI specs** (single source of truth for SDK generation)
- [ ] **Migration importers** — `mm import --from {chroma|mem0|letta} <dump.json>` **ships** (file-based subset: `src/mind_mem/importers/`, `IMP-` blocks in `memory/IMPORTED.md`, `imported:<system>` provenance, idempotent re-import). Open half is the endpoint-backed systems — pinecone / weaviate / qdrant need a live endpoint + credential and are refused with an explicit deferred message
- [ ] **Model-call token metering** — `mm usage` (per-day token counter + optional daily cap; quota/spending-alert surface deliberately dropped: self-hosted mind-mem has no spend outside model calls)
- [ ] **SLSA build provenance level 3**
- [ ] **Plugin SDK** — stable API for custom rules / block kinds / detectors
- [ ] **Chaos testing harness**

### Group RA — Retrieval accountability (proposed 2026-08-28; **revised after audit**, 5 items)

Full description: **[`docs/ROADMAP-RETRIEVAL-ACCOUNTABILITY.md`](docs/ROADMAP-RETRIEVAL-ACCOUNTABILITY.md)**
(partly superseded — read the banner).
Audit: **[`docs/audit/GROUP-R-AUDIT-2026-08-28.md`](docs/audit/GROUP-R-AUDIT-2026-08-28.md)** — verdict **FAIL** on the original spec.

Renamed from "Group R" to **Group RA**: `## Group R — edge evidence` already
exists further down this file.

The original framing (*rent a fact has to keep earning*) is *wrong for a governed
store* and the repo already says so: `cognitive_forget.py:96-100` refuses to forget
a high-importance rarely-read block because "it's often still load-bearing (e.g. an
ADR)", and `memory_tiers.py:113-121` gives only WORKING a TTL. **Decay must act on
attention, never on existence.** Silence-is-deletion is hostile to constitutional
constraints, security rulings, and contradiction records.

Load-bearing audit finding: **the tier ladder is not connected to recall on any
path** — `block_tier_meta` has no writer outside tests, and four different SQLite
files are in play. The original R.1 would have been a no-op.

- [ ] **RA.0 — wire the ladder onto one DB** *(new; blocks everything)* — consolidate `block_meta` / `block_tiers` / `block_tier_meta` (recommend `.sqlite_index/index.db`, which `tier_recall.py:127` already reads). Repoint `_recall_core.py:868`, `mcp/tools/recall.py:594`, `mcp/tools/governance.py:892`, `compaction.py:261`; register blocks at index build; give `_read_meta` an injected `now`. Acceptance: recall → promotion cycle → boost visible end-to-end.
- [ ] **RA.1 — served-set ledger** *(replaces old R.1)* — content-derived `run_id = sha256(query_hash ‖ served ids in rank order ‖ pipeline_hash ‖ index anchor)` (no clock, no randomness — same preimage discipline as `recall_attestation`), a durable `block_serve_counts` table that survives the 30-day prune, and `report_outcome(run_id=…)` so outcomes join a **run**, not a query string. Two counters with asymmetric power: **served** can buy attention tiers only; **credited** (distinct-actor successes, bounded) writes `confirmations` and buys trust tiers. **Absence of credit must never demote** — that is silence-is-deletion in a smaller hat.
- [ ] **RA.2 — precision + waste ledger** — derived views, never stored scores: precision = credited/served per intent type, waste = `served = 0` over the corpus. Not blocked on tombstones. Measure our own corpus; never quote external figures.
- [ ] **RA.3 — lifecycle deaths in the evidence chain** *(replaces the tombstone table)* — add `DEMOTE`/`ARCHIVE`/`FORGET` to `EvidenceAction` (`evidence_objects.py:68-77`) and `archive_block`/`forget_block` to `audit_chain.VALID_OPERATIONS`. A mutable side table would contradict tamper-evidence. **Blocked on** a governed *apply* path for consolidation plans — only `plan_consolidation` (dry-run) exists today.
- [ ] **RA.4 — retention class + alias-merge + rationale-everywhere** — retention class as a pure function of existing fields (PROTECTED / GOVERNED / EPHEMERAL), no clock in the classification. Governed `entities.merge` that re-points edges (`knowledge_graph.py:310-323` currently splits nodes permanently by insertion order). Require `Rationale` on all `propose_update` types, not just decisions.
- [ ] **RA.5 — dashboard** — after one tier axis is chosen and the data exists. There are currently **three** tier systems; do not render three.

> Determinism constraint for all of Group RA: take an injected `now`, following
> `cognitive_forget.py`. Note the earlier note here was stale — `date_score` is
> already UTC-normalised and injectable (`_recall_scoring.py:120-160`); the real
> naive-clock wart is `compaction.py:53,150,180,226`.

> Explicitly **not** building: outcome-weighted online reranking; any auto-written
> constraints file without HITL (`guardrails.py:43-50` provenance refusal exists
> because a ranker-bypass primitive is an injection vector); LLM-judged salience at
> ingest; a fourth tier system.

### Group H — Evolving memory graph (prior-art-informed, 2026-05-29)

Prior art: recent evolving-memory-graph research models
memory as a heterogeneous graph whose *topology* evolves in three
stages (link-on-write → feedback-driven refinement → long-term
consolidation), gated by a single maturity metric. SOTA reported on
LoCoMo / Mind2Web / GAIA. The mechanism maps almost 1:1 onto our
existing `propose_update → approve → consolidate` governance flow; we
already have the adjacent primitives (`scan` ≈ interference pruning,
`memory_evolution`, contradiction edges). These items formalize them.

**Wedge guardrail (load-bearing):** the source mutates topology
*autonomously* from feedback (non-deterministic learned rewiring),
which conflicts with the mind-mem auditable-provenance / bit-identity
wedge. Every evolution step here MUST route through the existing HITL
`propose_update`/approval gate — the source-of-truth graph never
self-modifies. We adopt the connectivity model, not the autonomy.

- [x] **Typed edge layer over the block store** (shipped v4.4.0) — first-class
      `supports / contradicts / refines / supersedes / derived-from`
      edges; relationship-aware recall instead of flat fusion. Cheapest
      high-leverage add; subsumes the existing contradiction-edge work.
- [ ] **Granularity / abstraction alignment** — a named merge operation
      for the known duplicate-memory pain (cf.
      `docs/block-type-taxonomy-roadmap.md`).
- [ ] **Maturity metric as consolidation gate** — governance signal for
      what graduates ephemeral → consolidated; surfaced in `scan`.
- [x] **Iterative re-compression ("sleep") — fixed-point cluster rewrite**
      — `recompaction.py` + `compressors.py` + `bench/recompaction_bench.py`
      (landed 2026-07-10; `mind_mem.recompaction`, 43 tests across
      `test_recompaction.py` + `test_compressors.py`). Motivating prior art: recent memory-consolidation work
      argues a *single* compression pass over context is lossy and that
      **re-reading one's own summary and re-compressing** recovers what the first
      pass missed (reported large accuracy gains on long-context math for small
      models; unverified at frontier scale — treat the *mechanism*, not the
      number, as the takeaway). The distinct piece vs the existing
      mark→merge→archive sleep cycle (Groups above): this takes a
      `find_similar` **cluster** of related blocks and re-compresses it against
      full sibling context, **looping until the rewrite is byte-identical to its
      input** (a fixed point), then emits the settled text as a `propose_update`
      proposal. Two load-bearing wedge guardrails, both already enforced +
      tested: (1) **fixed point, not fixed count** — the loop stops on
      byte-identity or raises `NonConvergenceError` at the bound, never a silent
      truncation (same discipline as the mic@1 self-host fixed-point gate); a
      "4 sleep loops" hyperparameter would hide non-convergence. (2) **injected
      compressor** — the model call is a `Callable`, so the loop / convergence /
      order-independent cluster digest / retention floor are proven with **zero
      API calls** in CI. Never mutates source of truth; every rewrite carries
      `source_ids` + `input_digest` for HITL provenance. **Follow-ups:**
      - [x] wire a real LLM compressor behind the `Compressor` type —
        `compressors.py` ships `EchoCompressor` (control, returns input verbatim)
        and `OllamaCompressor` (local ollama, `temperature=0` + fixed seed =
        deterministic, so a fixed point is legitimately reachable).
      - [x] a **fact-retention benchmark** — `mind_mem.bench.recompaction_bench`
        (`--model <echo|ollama-tag> --clusters N --db <path>`) prints a
        machine-greppable `recompaction_score: <float>` where score =
        `fact_retention * convergence_rate` over deterministic kNN clusters; an
        autoresearch harness (`config.mind-recompaction.yaml` +
        `program.mind-recompaction.md` + `run_mind_recompaction.sh`, a 4-stage
        gated evaluator: lint/type → test suites → echo-control-must-be-1.0 →
        measure) drives it against the real ~1469-block corpus to optimize the
        loop + compressor prompt.
      - [ ] a `mm recompact` / dream-cycle pass 6 that clusters via `find_similar`
        and routes results through `propose_update` (engine + bench shipped; the
        CLI verb and scheduler wiring are not yet built).
      - [ ] a **before/after recall benchmark on our own corpus** (gate on the
        LoCoMo item below) — the retention bench measures fact preservation, not
        end-to-end recall; do not take the reported accuracy gain on faith,
        measure recall on the governed corpus.
      - [ ] **`mind-mem:4b` retrain decision.** Empirical finding (live bench run
        2026-07-10): 4B **converges** — reaches a fixed point in ~2 passes and
        changes text (not a no-op) — but is **lossy**: it over-compresses and
        trips the retention floor on a meaningful fraction of clusters, so the
        safety gate rejects those rewrites. This is the "converges but degrades →
        case for a narrow retrain" outcome. Retrain target, if pursued: given N
        related blocks, emit one block such that (1) re-feeding its own output
        reproduces it (fixed point) **and** (2) recall over the merged block
        answers every question the source blocks answered. Decide use-as-is vs
        narrow retrain once the recall benchmark above lands.
      **Naestro warning (load-bearing):** do NOT port this to the
      Naestro vault as-is — its append-only + provenance-chain guarantee forbids
      in-place rewrite; a correct port appends a superseding block with a
      back-pointer, i.e. a materialized view, not "sleep".
- [x] **LoCoMo recall benchmark** (harness shipped v4.4.0; full-corpus run measured on a quiet node) — adopt as a standing mind-mem eval so
      recall quality is a number, not a vibe. **Do first** (cheapest,
      gives a baseline for everything else here).

#### H.1 — Graph *integration* (not a new graph) — 2026-07-05

Grounding pass (2026-07-05): a source read confirmed mind-mem already
ships **three** graph subsystems, not zero — so the remaining work is
*integration*, not designing a relation layer. Live graph today: **52
entities / 103 edges** over predicates `authored_by / depends_on /
part_of / related_to` (`graph_stats`, schema v1.0). The three:
`knowledge_graph.py` (typed `entities/aliases/edges` triple store with a
`Predicate` enum + per-edge provenance/confidence/temporal-validity —
*more* rigor than the generic MCP entity/relation/observation memory
pattern), `graph_recall.py` (implicit xref graph auto-built free from
canonical-ID mentions, BFS ≤3 hops, auto-fires on multi-hop queries),
and `causal_graph.py` (directed `depends_on/supersedes/informs` +
staleness propagation).

The gap, confirmed by grep not guessed: `graph_add_edge` /
`KnowledgeGraph(...)` are constructed **only** in the MCP wrapper,
`agent_bridge.py:281`, and `mcp/tools/core.py:62` — **never inside the
`propose_update`/`approve_apply` write path**. The rich typed graph is
populate-on-demand; nothing extracts entities/relations when a block is
written. (Provenance of this framing: a repo the operator floated,
`Memento-Teams/Memento`, turned out to be a CBR `(question,plan,reward)`
case-bank with *no* graph model — nothing to port. The
entity/relation/observation pattern lives in the unrelated
`gannonh/memento-mcp`. We adopt the *pattern's* one real modeling idea
below, no code, no attribution in public artifacts.)

- [ ] **Auto-extract edges on the write path (HITL-gated)** — wire
      lightweight entity/relation extraction (generalize the
      `block_parser.py:60-64` `_ENTITY_ID_RE` beyond canonical IDs to
      named entities) into `propose_update`, so writing a block *proposes*
      typed KG edges. **Extracted edges land as proposals, never
      auto-committed** — same approval gate as blocks, honoring the Group
      H wedge guardrail (source-of-truth graph never self-modifies).
      Closes the populate-on-demand gap; the single highest-leverage item.
- [x] **First-class entity `observations` field** (shipped v4.4.0, flag-gated) — the one real modeling
      gap vs. the generic pattern: mind-mem entities are bare canonicalized
      strings (`EntityRegistry.resolve`, `knowledge_graph.py:190-212`) with
      no accreted per-entity facts. Add `entities.observations` (small
      `ALTER TABLE … ADD COLUMN observations TEXT NOT NULL DEFAULT '[]'`),
      feature-flag-gated, so entity-centric multi-hop questions that never
      mention a block ID have somewhere to aggregate. Composes with the
      v4 `BlockKind.ENTITY`/`CONCEPT` work (`v4/block_kinds.py`).
- [ ] **Fuse `KnowledgeGraph.neighbors()` into `recall()`** — today the
      typed triple store is reachable only via `graph_query`, separate from
      `recall()`; only the *free* xref graph feeds `graph_expand()`. Blend
      typed-predicate neighbor hits into the RRF-scored result set so a
      single `recall()` benefits from typed hops, deterministically and
      logged (no learned re-ranker).
- [ ] **Independently reproducible benchmarks** — the headline numbers
      (NIAH 250/250, LoCoMo vs Memobase/Letta/Mem0) are self-published
      today. Ship a one-command repro harness that pins the exact dataset
      version + commit + config + seeds and writes RAW per-query outputs
      (not just the aggregate) to a checked-in artifact, so a third party
      can rerun and diff byte-for-byte. This is the single biggest lever
      on external credibility — claims should be reproducible, not trusted.
      (Aligns with the determinism wedge: a benchmark that replays
      bit-identically is itself evidence.)

### Group I — Feedback-quality recall scoring (prior-art-informed, 2026-06-20)

Prior art: recent scaling-law research on agent harnesses argues that
agent success scales not with raw compute (tokens, tool calls) but with
**how efficiently a budget is converted into durable, task-sufficient
feedback**. The headline coordinate credits a piece of feedback only
when it is **informative ∧ valid ∧ non-redundant ∧ retained for
subsequent decisions**, and the best predictor *normalizes that quantity
by task demand*. Reported separation is stark: raw tokens / tool calls
predict task outcome at R²≈0.33 / 0.42; the four-criteria coordinate
normalized by task demand reaches R²≈0.99 with an oracle and R²≈0.92 on
mixed real traces (≈0.85 on a prospective holdout); holding cost and
tool calls *fixed*, improving feedback quality alone moves success from
0.27 → 0.90.

**Why this is ours to steal.** mind-mem *is* the retention leg of that
formula — "feedback retained for subsequent decisions" is a one-line
argument for why a governed memory store exists. We already implement
two of the four criteria: **non-redundant** (RRF dedup in
`hybrid_recall` / `union_recall`) and **retained** (the governed block
store + lineage). We do **not** explicitly score **informative** (does
this block reduce uncertainty for the *current* decision?) or **valid**
(is it still true / not contradicted / not stale?). Closing that gap
turns recall from "we returned 8 blocks at score X" into "we returned
*enough durable, valid, non-redundant, on-task* context for this task
class" — a defensible, scaling-law-grounded product metric.

**Wedge guardrail (load-bearing):** the per-hit credit must be a
*deterministic, inspectable* function of existing block fields
(contradiction edges, staleness flags, lineage, dedup membership,
query-overlap) — **not** a learned black-box re-ranker. The score is
evidence, computed the same way on every substrate, auditable in the
retrieval log. No autonomous reweighting; if a credit weight changes it
ships as a versioned config, like any other governed change.

- [x] **Per-hit feedback-quality credit in `retrieval_diagnostics`** — shipped v4.7.0 (`retrieval_graph.feedback_quality_credit`, Stage 3.1, flag-gated; `valid` reuses the v4.6.0 validity gate via the shared `validity_components()` helper). —
      extend `retrieval_graph.retrieval_diagnostics` to emit, per
      returned block, a four-component credit
      `{informative, valid, non_redundant, retained}` instead of only a
      relevance/confidence histogram. `valid` ← contradiction-edge +
      staleness lookup (already in the store); `non_redundant` ← RRF
      dedup membership (already computed); `retained` ← governance state;
      `informative` ← marginal-uncertainty-reduction proxy (top-score
      delta vs. the already-packed set). **Do first** — cheapest, and it
      makes the rest measurable.
- [x] **Recall-sufficiency score (EFC ÷ task-demand analog)** — shipped v4.8.0 (`retrieval_graph.recall_sufficiency`, Stage 3.2; product-mass ÷ `INTENT_DEMAND`, surfaced in `retrieval_diagnostics` + `pack_recall_budget`). Originally: a single
      normalized "did this recall deliver enough on-task durable context
      for this query class" number, surfaced in `retrieval_diagnostics`
      and `pack_recall_budget`. The novel product metric; report it
      instead of (or beside) raw block counts.
- [x] **Validity gate wired into fusion** — shipped in **v4.6.0** as
      `validity_gate.py` (recall Stage 2.65, flag-gated
      `recall.validity_gate.enabled`, default off). Four deterministic criteria
      (corroboration / status / contradiction / staleness) → composite `V`;
      below threshold the hit is *demoted, never dropped*. Routes through the
      existing contradiction-log + `list_staleness_scores` primitives; no
      clock/rand on the scored preimage; annotates `hit["validity"]` for
      `retrieval_diagnostics`. Fable-spec'd, regression-gated
      (`tests/test_validity_gate.py`).
- [x] **Feedback-quality → downstream-success bench** — shipped v4.9.1 (`benchmarks/feedback_success_bench.py`; 48 deterministic episodes, starved 0.00 → sufficient 1.00 at matched budget). **Completes Group I.** Originally: add a standing
      eval that predicts agent task-failure from recall feedback-quality
      coordinates (their headline method), proving mind-mem *improves
      agent success*, not just retrieval scores. Pairs with the LoCoMo /
      reproducible-benchmark items in Group H; the matched-budget
      0.27 → 0.90 framing is the pitch slide.

> Provenance (arxiv id, authors, exact tables) recorded privately in
> `mind-internal`, per the no-public-attribution rule — public artifacts
> say "recent scaling-law research" only.

### Group J — Client-side anticipation cache + tool-output offload (prior-art-informed, 2026-07-03)

Prior art: recent hosted agent-memory tooling ships two client-side
patterns we lack, both aimed at the same goal — **keep bytes out of the
round-trip / out of the context window**. (1) An *anticipation cache*: a
local TTL-scoped bundle store fronted by a cheap BM25 lookup, so likely-
relevant context is served locally at sub-round-trip latency instead of
re-fetched from the store. (2) A *novel-term gate*: a dependency-free
confidence heuristic that suppresses a local-cache hit when the query's
novel-term ratio exceeds a threshold (the local corpus can't answer it →
fall through to the source), with a corpus-size floor so the ratio isn't
dominated by stopwords on a cold cache.

**Why this is ours to steal.** mind-mem already has the *hard* half — the
governed store, the federation transport (U1-served Postgres+Redis), and
the idle machinery (`prefetch` / `speculative_prefetch`, the learned
co-retrieval graph). What we lack is the *consumer* pattern that turns
those idle tools into an actual local hot-path cache, plus the cheap
local-vs-source decision gate. The offload idea generalizes further:
tool/command output (a 50k-line `cargo test` / `pytest` dump) is the
single biggest context sink for coding agents and has no home in the
block store today.

**Wedge guardrail (load-bearing):** the novel-term gate and the offload
summarizer must be **deterministic, LLM-free** functions of existing
signals — a pattern-extraction summary (same input → same summary
bytes), a stem-ratio gate computed the same way on every substrate,
inspectable in the retrieval/offload log. Cache eviction and gate
thresholds ship as versioned config, not autonomous reweighting. Fail
safe: the offload summarizer must never silently drop a failure line;
dropped-line counts are logged.

- [ ] **Anticipation-cache consumer** — wire the existing idle
      `prefetch` / `speculative_prefetch` tools into a local TTL bundle
      cache fronted by BM25, so a recall checks the local bundle before a
      round-trip. Reuses the co-retrieval graph as the "what to prefetch"
      signal; **feed the loop that's currently starving** (`signal_stats`
      = 0, prefetch observations = 0 today). Redis is the push
      transport — do **not** add a second one.
- [ ] **Novel-term gate** — a ~40-line deterministic heuristic: suppress
      a local-cache hit when the query's novel-stem ratio exceeds a
      configured threshold (default ≈0.45) once the cached corpus has
      ≥N stems (default ≈200), else fall through to the store. The cheap
      local-vs-source confidence signal the prefetch layer lacks. **Do
      first** — it's the piece the cache consumer needs to be safe.
- [x] **Tool-output offload store** (v4.2.0, `mind_mem.tool_output`) — a new
      Postgres-backed block kind + `store_and_summarize(text, source,
      exit_code)` path returning `{handle, summary, line_count}` only,
      with `recall_output(handle)` for on-demand full text. Deterministic
      pattern-extraction summary (failures + head/tail + pass/fail
      tallies); a dependency-free `mm-run -- <cmd>` wrapper streams a live
      tail to the user and emits only the summary+handle to the agent.
      Closes the biggest single context sink for the `mind` repo
      (247 test binaries). **The real build** of the three.
- [ ] **One-command federation connect (`mind-mem connect`)** — the only
      onboarding gap vs. hosted "shared context across all CLIs" comps:
      a wrapper that wires a new CLI into the U1-served federation
      (Postgres+Redis DSN) without hand-editing config. We already own
      the shared-context substrate; this is the frictionless join.

> Provenance (source repos, exact heuristics) recorded privately in
> `mind-internal`, per the no-public-attribution rule — public artifacts
> say "recent agent-memory tooling" only.

### Group K — Graph-from-text construction + edge-grounded recall (prior-art-informed, 2026-07-24)

Prior art: a recent structured-knowledge-graph-from-unstructured-text
pipeline builds a queryable entity/relation graph entirely from
schema-constrained LLM calls (no trained NER / relation-classifier),
resolves surface-form variants into canonical nodes via LLM clustering,
and answers multi-hop questions where **every claim cites the specific
edge + source document that supports it**. It maps almost 1:1 onto our
existing graph surface (`block_lineage.add_block_edge`,
`mcp/tools/graph.{graph_add_edge,graph_query,traverse_graph}`,
`retrieval_graph`, `ontology` typed entities) — these items formalize
the construction + grounding half we don't yet have.

**Wedge guardrail (load-bearing):** the source's resolution/merge and
edge-confidence steps are *autonomous* LLM judgments. Every graph
mutation here MUST route through the existing HITL `propose_update` /
approval gate — the source-of-truth graph never self-modifies from an
un-reviewed model call. We adopt the construction mechanics + the
edge-grounded verdict; we keep our auditable-provenance / bit-identity
wedge (the model call is an injected `Callable`, so the loop / digest /
retention floor prove out with zero API calls in CI, same discipline as
Group H recompaction).

- [ ] **Description-grounded entity resolution** — canonicalize
      surface-form variants (nicknames, abbreviations, cross-lingual
      transliterations) that string-similarity dedup misses, using a
      one-line per-entity description as disambiguation context rather
      than edit distance. Extends the existing block-merge / dedup path
      (`capture.py`, `contradiction_detector`); pairs with Group H
      "granularity / abstraction alignment." Two enforced failure modes:
      unmatched name → single-element cluster (never silently dropped);
      over-merge guarded by description mismatch + HITL review.
- [ ] **Blocking + LLM-arbitration hybrid (resolution at scale)** —
      cheap deterministic blocking (inverted index on name tokens /
      embedding neighbors) narrows candidates to 50–100-item blocks; the
      LLM only arbitrates *within* a block. Keeps resolution sublinear
      and keeps the expensive model call off the easy cases (typos,
      casing) that deterministic logic already handles.
- [ ] **Edge-grounded recall / answer-with-citations** — an answer mode
      constrained to extracted edges: serialize the k-hop subgraph (k=2
      default) of the seed entity as triples, answer only from those
      edges, cite the supporting edge + provenance document per claim,
      and explicitly flag what the graph does **not** contain. The
      construction-side counterpart to the existing evidence-chain
      recall — makes "cite the edge" concrete. Feeds the Group B
      `chat_with_memory` `[[block_id]]` citation item.
- [ ] **Edge confidence from cross-source corroboration** — an edge seen
      in N independent source blocks outranks a single-source edge; a
      calibrated-confidence signal per edge so recall + the contradiction
      gate weight evidence instead of treating all edges as equal.
      Converges with the SIGNALS.md conformal-calibration / calibrated-
      confidence direction (same idea, second source). Sidecar only —
      never in the sealed audit-hash preimage.
- [ ] **Schema versioning alongside the graph** — version entity-type /
      predicate / extraction-prompt changes with the graph so blocks
      extracted under an old schema are distinguishable, comparable, and
      re-extractable; a hard prerequisite before scaling ingestion.
      Aligns with the existing spec-hash-binding (I-5) discipline.
- [ ] **Hub-node profile synthesis (degree-gated)** — for high-degree
      nodes only (degree ≥ 3), pool every mention + graph neighborhood
      into a synthesized profile (summary + 3–5 traceable atomic facts +
      structured time range), "resolve contradictions by preferring the
      most specific claim, invent nothing." Reuses the Group H
      recompaction fixed-point + injected-compressor machinery; emits as
      a `propose_update`, never a direct write.

> Provenance (source cookbook / working note + URL) recorded privately
> in `mind-internal`, per the no-public-attribution rule.

#### K.0 — Graph population is the bottleneck, not graph capability (2026-07-27)

Triggered by an external "graph engineering" workflow board (ingest → entity
extraction → graph build → hybrid index → GraphRAG query → agent memory →
multi-agent swarm → autonomous refresh) evaluated against what we actually run.
Seven of its eight stages are already shipped here or in the surrounding stack.
The audit's real finding was not a missing feature — it was that **the graph we
already built is nearly empty**, so the shipped capability is invisible in
practice.

**Verified live 2026-07-27** (`graph_stats` + `graph_query` against the active
workspace, not read from docs):

- `entities: 52`, `edges: 103`, `orphan_entities: 0`.
- Only **four** predicates in use: `depends_on` (61), `related_to` (18),
  `part_of` (13), `authored_by` (11).
- Multi-hop traversal **works**: `graph_query(entity="mind-mem", depth=3,
  direction="both")` returns real 2-hop paths
  (`mind-mem → mind-kg → mindc`, `mind-mem → starga inc → 512-mind`).
- `extraction.enabled: true`, backend `ollama`, model `mind-mem:4b`.

The deeper finding (code audit, 2026-07-27): the graph was not merely
under-populated — it was **unwired**. `knowledge_graph.py` (typed `Predicate`
enum, `valid_from`/`valid_until` temporal columns, `EntityRegistry` alias
resolution, `neighbors()` BFS capped at 8 hops) shipped with no corpus→graph
ingestion path (the only edge writer was the manual `graph_add_edge` tool —
every live edge is hand-curated or structural-scan) and no recall consumer
(the "graph" that recall walks is the free block-xref graph in
`graph_recall.py`, not the typed store; note **`retrieval.multi_hop` governs
that XREF expansion, not the knowledge graph**). Meanwhile
`extraction.enabled: true` bought only per-recall enrichment whose output the
caller discarded — a pure read-path latency tax. "Measure the extraction
pipeline's yield" was therefore the wrong first step: there was no pipeline to
measure. Wiring comes first, with the yield measurement built into the wiring:

- [x] **Corpus → graph ingestion, HITL-gated, with yield metrics built in** —
      `graph_ingest.py` + `llm_extractor.extract_relations` (prompt
      constrained to the `Predicate` vocabulary; extractor = the configured
      extraction model, swappable, never retrained here). Extracted triples
      stage as SIGNALS.md entries (`auto-capture-relation`); the graph is
      written only by operator approval (`mm graph-backfill --approve`),
      which stamps `source_block_id` + `valid_from` + an origin marker.
      `mm graph-backfill` is dry-run by default and prints edges-per-block +
      a predicate histogram — the yield measurement IS the wiring's default
      output, not a separate diagnostic step.
- [x] **Typed-graph fusion into recall** — `kg_fusion.py` +
      `HybridBackend._maybe_kg_expand`: query terms resolve through the
      entity registry (read-only), typed edges walk ≤2 hops, and each edge's
      `source_block_id` pulls its backing block into the result set with a
      decayed score. Gated behind `retrieval.kg_fusion.enabled` (default
      OFF — recall replays byte-identical until the graph is populated and
      the operator opts in).
- [x] **Stop the read-path extraction tax** — per-recall enrichment now also
      requires `extraction.enrich_on_recall` (default false), so
      `extraction.enabled: true` funds write-path ingestion instead of
      discarded per-query calls.
- [x] **Close the HITL bypass** — `graph_add_edge` moved to the admin ACL
      set; user-scope graph mutation now routes through signal staging +
      approval only, making the Group K "every graph mutation routes through
      HITL" guardrail true in code, not just in docs.
- [ ] **Run the backfill over the live corpus** — with the wiring landed,
      run `mm graph-backfill` over the 1469-block corpus, read the yield
      numbers, review/approve the staged edges, then enable
      `retrieval.kg_fusion` once there is a graph worth walking.
- [ ] **Widen the predicate vocabulary beyond repo topology** — the four live
      predicates cannot express the relations our corpus is actually made of
      (person ↔ organization, decision ↔ rationale, commitment ↔ owner,
      claim ↔ evidence). Pairs with the Group K "schema versioning" item:
      version the vocabulary *before* scaling ingestion, so pre-widening
      blocks stay distinguishable and re-extractable.
- [x] **Pin `retrieval.multi_hop` in the live workspace config** — the XREF
      expansion block (`enabled`, `auto_enable`, `max_hops`, `decay`,
      `max_neighbors_per_hop`) is now written explicitly with the previous
      auto-enable defaults (zero behavior change), alongside
      `retrieval.kg_fusion: {enabled: false}`.

**Not adopted from the source board.** Neo4j — our SQLite incidence tables
already carry typed predicates and a real temporal model, and the dependency
buys nothing we lack. Its "agent memory" and "multi-agent swarm" stages are
mind-mem and the Naestro scheduler respectively, both shipped. Its "autonomous
refresh" is `reindex` plus the existing post-merge/post-commit hook — and note
its refresh loop is autonomous by design, which is exactly the property our
HITL gate deliberately refuses.

- **Status:** Wiring landed 2026-07-27 (ingestion, fusion, read-path gate,
  ACL). Open: the live-corpus backfill run + predicate-vocabulary widening,
  both gated on the yield numbers the backfill prints.

### v3.2.x trailing fixes (4 items, deliberately deferred)

- [ ] **Apply engine — text-range ops** — `insert_after_block` / `replace_range` still on raw `open()`; no v3.2.x caller generates them in practice
- [ ] **FastAPI audit attribution** — `current_agent_id` doesn't propagate through anyio threadpool worker; fix via `request.state.agent_id`
- [ ] **`PostgresBlockStore.snapshot(snap_id=…)`** — current signature still requires filesystem path; cross-host PG snapshots blocked
- [ ] **T-004 webhook allowlist + T-001 content-provenance tags + N-08/N-12/N-13/T-007** — minor security-hardening items (see v3.2.0 section)

### v4.0.x federation transport hardening (3 items; +1 shipped in v4.0.14)

- [ ] **Per-peer identity beyond bearer token** (token → agent_id binding, signed-write envelopes)
- [ ] **mTLS + cert pinning on `FederationClient`**
- [x] **Operator-side peer allowlist** (`MIND_MEM_FED_PEERS=10.0.0.5,…`) — **shipped v4.0.14**
- [ ] **Token rotation primitive** (N-of-K active tokens, `mm token rotate`)

### Cross-cutting (deferred infrastructure)

- [ ] **Kubernetes operator + Helm chart**
- [ ] **Byzantine-safe consensus (PBFT)** — opt-in, long-horizon
- [ ] **Edge deployment mode** — PyOxidizer single-binary
- [ ] **Managed-service console** — multi-tenant dashboard
- [ ] **Kafka / NATS event fan-out**

### Pure-MIND Core Port (long-horizon architectural goal — gated on `mindc` library-emit)

- [x] Hot scoring kernels in pure MIND (`mind/*.mind`, bench-gated)
- [ ] Governance / decision / boundary layer in pure MIND
- [ ] Core retrieval engine (index walk, fusion, rerank) in pure MIND
- [ ] I/O adapters via MIND C-ABI / FFI
- [ ] Python reduced to thin shim, then removed

### Advanced Agent Memory Primitives (5 planned block types, future)

- [ ] **`[CAUSAL]` block type** — world-model storage for learned state transitions
- [ ] **`[SKILL]` block type** — named strategy captures with preconditions / effects / success-rate
- [ ] **Cross-domain recall adapter** — surface most-similar `[TRAJECTORY]` / `[SKILL]` blocks across environments
- [ ] **`[VISUAL]` block type** — grid-state / image-state embeddings for perception-grounded memory
- [ ] **Evidence-chain submission format** — tamper-evident per-episode export

### Companion Tools (1 doc item)

- [ ] **GitNexus** documentation in README under "Companion Tools" section

---

**Sizing summary** (genuine remaining work):

- **Small (1–3 day items, ship-this-month):** audit headers, public/private workspaces, peer allowlist, token rotation, time-bounded recall, time-travel/as_of, OpenAPI specs, GitNexus doc, vocabulary-bound fields, T-004/T-001/N-08/N-12/N-13, Group J novel-term gate + `mind-mem connect`
- **Medium (1–3 weeks):** TLS 1.3 + cert pinning, mTLS service-to-service, AI lint with auto-fix, JS/TS SDK, content provenance + provenance-rich blocks, audit attribution ContextVar fix, FastAPI request.state, PostgresBlockStore snapshot snap_id, migration importers, plugin SDK, cost metering, Group I per-hit feedback-quality credit + recall-sufficiency score, Group J tool-output offload store + anticipation-cache consumer
- **Large (multi-month):** local visual viewer (`mm view` web UI), conversational chat layer, Kubernetes operator, managed-service console, Byzantine consensus, Pure-MIND port (gated on `mindc` C-ABI maturity)
- **Long-horizon / research (post-v4):** Pure-MIND port completion, [CAUSAL]/[SKILL]/[VISUAL] block types, ActivityPub interop, edge deployment, Group I validity-gated fusion + feedback-quality→success bench

---

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
- [x] **LoCoMo benchmark with an external LLM judge** — full 10-conversation LLM-as-judge evaluation (1986 questions, 134 min)
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

## v1.9.1 — Released 2026-03-06

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
  export MIND_MEM_WORKSPACE=~/.mind-mem/workspace
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

## v3.2.0 — Production Deployment ✅ Released (2026-04-13, rolled into v3.2.0 → v3.9.0 ladder)

Closed the production-readiness gap. Everything local-first but horizontal-ready. No changes to the retrieval pipeline; all new work is adapters + gateway.

- [x] **Postgres storage adapter** — `src/mind_mem/block_store_postgres.py` implements `BlockStore` protocol
- [x] **Storage factory** — `src/mind_mem/storage/__init__.py` selects adapter from `mind-mem.json`; `ConnectionManager` accepts adapter type + pool size
- [x] **MCP tool-surface consolidated** — recall-family tools unified under `recall` with explicit modes; `propose_update` / `approve_apply` / `rollback_proposal` retained as discrete tools because the multi-phase flow benefits from explicit naming for agents. Tool surface ended at **84** post-v3.9 (vs the ~20 target). Decision: agent context-window cost is dominated by tool *bodies* not names; consolidation is a perpetual evolution, not a one-time cut.
- [x] **REST API layer** — `src/mind_mem/api/rest.py` (FastAPI), endpoints mirror the MCP tool set; `src/mind_mem/api/auth.py` provides OIDC/JWT; `mm serve` + `mm http-serve` CLI commands wired
- [x] **JS/TS SDK + Go SDK** — `clients/js/` and `clients/go/` ship matching the Python surface (typed Pydantic v2 → TypeScript / Go generated)
- [x] **Dockerfile + docker-compose** — `Dockerfile` + `docker-compose.yml` at repo root with mind-mem + pgvector + Ollama; one-command bring-up
- [x] **One-command installer** — `install.sh` at repo root + `mm install-all` CLI
- [x] **Full OIDC / SSO auth** — `src/mind_mem/api/auth.py`; Okta / Auth0 / Google Workspace / Azure AD via OIDC discovery; scope → ACL role mapping
- [x] **Per-agent access control** — `src/mind_mem/namespaces.py` + `mcp/infra/acl.py` enforce per-agent grants; audit chain attributes every read/write to `agent_id`
- [x] **OpenTelemetry traces + SLO dashboards** — `src/mind_mem/v4/observability.py` wraps `recall`/`propose_update`/`scan` with OTel spans; Prometheus exporter configurable
- [x] **Distributed query cache** — Redis adapter shipped; in-process LRU fallback when Redis not configured; invalidated on `propose_update`/`apply`
- [x] **Postgres read replicas** — `storage.replicas` config + read/write routing in `block_store_postgres.py`; read-heavy MCP tools route to replicas
- [x] **Hot/cold tier wire-up** — `v4/tier_memory.py` + `tier_recall.py` + `tiered_memory.py` + `memory_tiers.py` wire WORKING/ARCHIVAL/COLD tiers into the recall path
- [x] **CLI debug visualization** — `mm inspect`, `mm explain`, `mm trace` all shipped (see `mm_cli.py` argparse map)
- [x] **Config schema additions** — `storage.{adapter,url,pool_size,replicas}`, `api.{rest,grpc,auth}`, `observability.{otel_endpoint,prom_port}`, `cache.{redis_url,ttl_seconds}` sections all wired in `mind-mem.json`

### Structural-debt cleanup (from the 2026-04-18 audit)

Four code-health items surfaced by the architectural audit
(`AUDIT_FINDINGS_FOR_CLAUDE.md`). Scoped into v3.2.0 because each is
a prerequisite for the production-deployment work above:

- [x] **Decompose `src/mind_mem/mcp_server.py`** — broken into the
  `src/mind_mem/mcp/` package (`mcp/tools/*.py` per domain +
  `mcp/infra/{acl,observability,workspace,…}.py`). `mcp_server.py`
  remains as a thin dispatcher / argparse entry. Tool surface ended
  at 84 (see v3.2.0 main section).
- [x] **Centralize task-status literals into an `enums.py`** —
  `src/mind_mem/enums.py` ships `TaskStatus(str, Enum)`; all eight
  call sites migrated.
- [x] **Unify validation around `validate_py.py`** — `validate.sh`
  removed; `validate_py.py` is the single enforcement path.
- [x] **Route `apply_engine.py` writes through `BlockStore`** — five
  of seven `_op_*` handlers route through `store.get_by_id` +
  `store.write_block`; the two remaining text-range ops
  (`insert_after_block`, `replace_range`) are now block-aware via
  the markdown writer adapter (no v3.2.0 caller generates these in
  practice; full BlockStore route deferred to text-range refactor).
- [x] **Widen snapshot atomicity scope** — snapshot now covers
  `maintenance/` + `intelligence/applied/` (resolved in v3.7.0).

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

- [x] **N-01 / T-002: Default-on ACL gate** — `MIND_MEM_ACL_DISABLED`
  opt-out flag added; admin tools rejected unless explicit elevation.
- [x] **T-006: Bound `vault_scan` / `vault_sync` filesystem walks** —
  `MIND_MEM_VAULT_ALLOWLIST` enforced; `realpath` symlink-safe match.
- [x] **N-02: REST `rollback_proposal` requires `reason`** —
  `RollbackProposalRequest.reason: str = Field(..., min_length=8)`.
- [x] **N-04: `staged_change` rollback forwards `rationale`** —
  rationale-as-reason wired through dispatcher.
- [x] **T-003: `propose_update` input bounds enforced** —
  `_sanitize_reason_for_markdown` applied at entry.
- [x] **N-03: Rate limiter per-pid bucket** — fallback to
  `pid-{os.getpid()}` when no access token.

#### Should fix in v3.2.0 (low UX cost, real surface area)

- [x] **N-05: `/v1/health` workspace path stripped** when unauth.
- [x] **N-06: `/v1/metrics` Prometheus gated** behind auth / localhost.
- [x] **N-07: OIDC callback omits scopes**; issuer allowlisted.
- [ ] **T-004: Webhook/Slack alerting URL allowlist** — open;
  operator-supplied alert URLs still unconstrained. Tracked.
- [x] **T-005: `--token` CLI arg rejected** — env-only.
- [ ] **T-001 (partial): Content-provenance tags on block writes** —
  `source ∈ {agent, user, external}` frontmatter NOT yet added to
  every write path. Tracked.

#### Nice-to-have in v3.2.x (low priority, low UX cost)

- [ ] **N-08: `decrypt_file` audit trail** (`decrypted_files.jsonl`)
  — open; admin-tool forensic coverage gap. Tracked.
- [x] **N-10: FTS5 token sanitiser Unicode** — `re.UNICODE` applied.
- [x] **N-11: `export_memory` size cap** — `max_blocks` validated.
- [ ] **N-12: REST rate-limit bucket key sha256** — open; not
  practically exploitable (operator controls tokens). Tracked.
- [ ] **N-13: OpenAPI docs gated** when token configured + non-local
  — open. Tracked.
- [ ] **T-007: OS-level append-only audit log** (`chattr +a` /
  `chflags uappnd`) — open. Tracked.
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

## v3.2.1 — Hotfix follow-up ✅ Released (rolled into v3.2.x → v3.7.0 ladder)

Closed the two architectural CRITICALs surfaced by the v3.2.0
self-audit (`docs/review-architecture-v3.2.0.md`). v3.2.0 Postgres
+ REST surfaces promoted from "beta" to GA. The three still-`[ ]`
items below are real, tracked, and deliberately deferred — they
are surfaced in the **Genuinely Open** section at the top of this
file.

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

Baseline: v1.1.0 overall mean 70.54 (external LLM answerer + judge). LoCoMo category breakdown shows where points bleed:

| Category | Baseline | N | Biggest intervention |
|---|---|---|---|
| multi-hop | 51.10 | 321 | graph traversal + decomposition |
| temporal | 65.89 | 96 | half-life decay in scorer |
| single-hop | 68.68 | 282 | query reformulation + RRF |
| open-domain | 70.27 | 841 | conversation-boundary preservation |
| adversarial | 87.22 | 446 | at ceiling |

Projected v3.3.0 overall with Tier-1+2 shipped: **74-76 (same model as answerer + judge)** / **82-85 (stronger answerer + external LLM judge)**.

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
- [ ] **mind-mem-4b v2 retrain — rebase to Qwen3.8-4B + catch up to the current surface** — training recipe + data generators shipped (`docs/mind-mem-4b-v2-training-recipe.md`, `benchmarks/generate_dispatcher_examples.py`, `benchmarks/generate_retrieval_examples.py`). **NOT required by the v4.5.0 recall-noise fix or the v4.6.0 validity gate** — both are backend / flag-gated and add no MCP tool, and the 4b is only the swappable KG-extraction/dispatch model (recall + hybrid *scoring* is the `mind/*.mind` kernels + Python, never the 4b). The refresh is warranted because the current 4b was trained knowing ~84 tools while the live surface is now **90** (the typed-edge KG tools — `propose_edge` / `approve_edge` / `reject_edge` / `list_edge_proposals` / `entity_add_observation` — landed in v4.3/v4.4 *after* the 4b was trained). New model = **full retrain on a Qwen3.8-4B base** (up from Qwen3.5-4B) over the surface + v3.2.x dispatchers + v3.3.0/v4.x retrieval shapes (incl. the validity-gate config, OKF v0.2, and hybrid fusion-provenance fields) + LoCoMo replay. **Sequencing (operator decision 2026-08-18): deferred to AFTER the full Pure-MIND port** — do the Qwen3.8-4B rebase + retrain once, over the *post-port* tool/surface, bundled with that milestone rather than now. **The current 4b keeps working in the meantime with zero functional loss**: it is the swappable KG-extraction/dispatch model (never on the recall-scoring path — that is the `mind/*.mind` kernels + Python), and the ~6 newer tools it was not trained on still execute normally when called; only *proactive* suggestion of them is affected. Runpod H200 kickoff (~$55, 8-12 hr external GPU) runs on operator approval at that time.

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
(tracked internally in the audit report).
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
> of MIND-Mem to a separate agent-layer project** — fetching social
> content is an agent-layer concern, not a memory-layer concern.
> MIND-Mem stays the substrate (blocks + recall + governance); the
> agent layer owns per-platform fetching and writes into MIND-Mem via
> the existing MCP surface. This preserves MIND-Mem's zero-dependency
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

### Social Ingestion — moved to the agent layer (2026-05-02)

The platform fetcher set (HN / Reddit / X / LinkedIn / Instagram /
TikTok / Moltbook / Bluesky / Mastodon / Farcaster) and the
URL-to-block ingestion CLI / MCP tool are no longer scoped to
mind-mem. **Tracked in the agent layer** alongside the existing
chat-platform channels (Discord / Slack / Telegram / Feishu) —
fetching social content is the same shape of work as bridging a chat
platform, and the agent layer already owns that surface. Those
extensions write captured posts into MIND-Mem through the existing
MCP recall / capture tools, so MIND-Mem's role (substrate for blocks
+ recall + governance + contradiction detection) is unchanged.

The split keeps MIND-Mem zero-dependency, avoids inheriting per-
platform auth / anti-bot / ToS maintenance, and preserves the
clean layering: **MIND-Mem stores; agents fetch.**

**Estimated:** ~1500 lines audit (CLI + pickle disassembly + Ed25519 +
load-gate + CI hook) — all shipped in v3.8.1 → v3.8.7. The social
ingestion estimate (~2500 lines fetchers + ~600 lines block
integration) is now an agent-layer concern.

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

## v3.12.0 — Local-model GA, hard quality gate, lineage staleness, red-team CI ✅ Released (B, C, D shipped; A superseded by v4.0 retrain)

Four additive themes. **Themes B, C, D shipped fully.** Theme A
(v3.11.0-fullft `mind-mem-4b` retrain) was superseded by the
**v4.0.0 retrain on the v4 surface** — the v4 weights revision is
the GA model now; the v3.11.0 intermediate is skipped.

### Theme A — `mind-mem-4b` v3.11.0-fullft GA bundle ⊘ SUPERSEDED

Skipped in favour of the v4.0.0 retrain. Tracked here for history.

- [x] ~~v3.11.0-fullft retrain~~ — superseded by v4 retrain
  (`mind-mem-4b` v4 revision shipped with v4.0.0)
- [x] HF upload — `star-ga/mind-mem-4b` v4 revision is the GA pointer

### Theme B — Quality-gate hard mode ✅ shipped

- [x] `mind-mem.json` reads `quality_gate.mode ∈ {off, advisory, strict}`
- [x] `propose_update` invokes `validate_block` pre-write in non-off modes
- [x] Metrics counter wired
- [x] `docs/quality-gate.md` runbook
- [x] Config-honor test for all three modes

### Theme C — Block-lineage staleness propagation wiring ✅ shipped

- [x] `add_block_edge(..., kind="contradicts")` schedules bounded pass
- [x] `block_staleness` table — SQLite + Postgres parity
- [x] Recall reranker reads the penalty; `_explain.staleness_penalty`
- [x] CLI: `mm lineage flag <block-id> --kind contradicts <target>`
- [x] e2e test wired

### Theme D — Petri behavioral audit promoted to advisory CI ✅ shipped

- [x] `.github/workflows/red-team.yml` (tag-push, continue-on-error)
- [x] Skips cleanly when `ANTHROPIC_API_KEY` absent
- [x] Transcripts uploaded as artifacts
- [x] `--limit 5` per seed + sonnet judge
- [x] `docs/red-team-audit.md` references the CI workflow

### Out of scope for v3.12.0

- Networked mesh / federated recall — v4.0
- Streaming consensus mixer — stays in the private agent layer
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

### A. Cognition / model layer ✅ shipped in v4.0.0

- [x] **Surprise-weighted retrieval** — `src/mind_mem/v4/surprise_retrieval.py`, `compute_surprise` + `FallbackPolicy`; opt-in via `retrieval.surprise_weight`.
- [x] **Block-tier tags** — `src/mind_mem/v4/tier_memory.py` + `tier_recall.py` + `tiered_memory.py` + `memory_tiers.py` (hot/warm/cold with per-tier decay).
- [x] **Cognitive Mind Kernel** — `src/mind_mem/v4/cognitive_kernel.py`, `KernelKind` enum + `mind_recall`.
- [x] **Multi-modal blocks** — `block_kinds.py` + multi-label `block_kind_tags` table; embeddings only (raw bytes external).
- [x] **Graph-aware retrieval** — typed-edge graph (`block_lineage`) wired into recall; lineage-walk query expansion ships.
- [x] **`mind-mem-4b` v4.0 retrain** — v4 weights revision shipped on Hugging Face; covers v4 surfaces.
- [x] **`mind-mem-4b` base-model evaluation** — Qwen3.5-4B fine-tune confirmed as GA baseline; reviewed at v4.0.0.

### B. Knowledge graph (mostly shipped — 2 items open)

- [x] **Block kinds** — `block_kinds.py` + `kind ∈ {entity, concept, source, synthesis, image, audio, code, structured}`.
- [x] **Block versioning + time-travel** — `v4/block_versioning.py` ships `block_history(block_id)` + `content_as_of(...)` over the applied-edit chain; `recall(..., as_of=date)` now exposes the point-in-time projection on the recall entrypoint across all backends (sqlite / vector / BM25 scan / multi-hop merge), gated on the `self_editing` surface with a graceful no-op when it is disabled.
- [x] **Content-addressable block IDs** — content-hash + CID-style stable id ship; replication uses it.
- [x] **Long-context recall mode** — `mode="long_context"` ships in the recall API.
- [x] **LLM-driven knowledge fusion** — `propose_fuse` tool ships, hooked into `propose_update → approve_apply`.
- [x] **Streaming recall** — generator-style streaming path ships.
- [x] **Conversational chat layer** — `chat_with_memory(workspace, question)` ships with sentence-level `[[block_id]]` citations, store-backed citation resolution, a no-record abstention path, and a pluggable (default deterministic + offline) answerer.
- [x] **Schema layer for LLM prompts** — `mind-mem.json` `prompts.schema` ships.
- [x] **Schema evolution / migration tooling** — `mm migrate-store` covers schema drift for v4 fields.

### C. Knowledge graph governance / UX (partial — 5 items open)

- [x] **Idle-only background ingest** — `src/mind_mem/daemon.py` + `inbox.py` ship; opt-in via `mind-mem.json` `watch.enabled`; resource-capped.
- [ ] **AI lint with auto-fix** — `lint_autofix(workspace, finding_id)` tool not yet shipped; the underlying `scan` does emit findings. Tracked.
- [x] **Contradiction state machine** — `detected → review_ok → resolved` / `pending_fix` lifecycle ships in `governance` engine.
- [x] **Self-healing index** — `mm doctor` triggers integrity check + repair; background reindex runs in idle windows.
- [ ] **Local visual viewer** — `mm view` web UI not yet shipped. Stack target: stdlib HTTP + minimal JS/D3. Tracked.
- [x] **Auto-generated hierarchical index** — `index.md` / `log.md` autogen not wired. Tracked.
- [x] **Real-time contradiction stream** — webhook stream on contradiction-detection ships under the alerting layer.
- [ ] **Adversarial / poisoning defense** — per-actor anomaly detection + canary blocks not yet shipped. Sigstore-signed manifests partial (release artifacts only). Tracked.
- [x] **Approval workflows for sensitive proposals** — multi-reviewer chain (OPA/Rego-style) ships behind opt-in dep.
- [x] **Memory reputation / trust scores** — provenance class surfaced on recall hits as `actor_trust` (`provenance_class.py`, the validity gate's fifth component), with an opt-in low-provenance demotion re-rank.

### D. Network & multi-agent connectivity (partial — 5 items open, big-ticket items deferred)

The primary frame: agents on different machines, owned by different
parties, share governed memory through mind-mem. Single-host scale
(sharded Postgres / K8s) is a sub-bucket for heavy deployments; the
default story is two laptops talking to each other.

**Shipped:**

- [x] **OAuth2 / OIDC client identity** — `src/mind_mem/api/auth.py` ships pluggable IdP integration.
- [x] **DID + Verifiable Credential agent identity** — W3C VC verification ships behind `[did]` extra.
- [x] **Workspace ACLs** — `mcp/infra/acl.py` ships block-level grants on signed chain.
- [x] **Cross-instance federation protocol** — `src/mind_mem/v4/federation.py` ships signed handshake + three-way merge.
- [x] **End-to-end encryption for sensitive workspaces** — `EncryptedBlockStore` ships ciphertext + hash-indexed.
- [x] **Discovery / WebFinger** — `/.well-known/mind-mem` endpoint advertises capabilities + public keys.
- [x] **Subscriptions / webhooks** — `subscribe(workspace, filter, callback_url)` ships.
- [x] **Per-tenant rate limiting + circuit breakers** — `circuit_breaker.py` + `backpressure.py` ship.
- [x] **Per-tenant routing** — `namespaces.py` routes per-tenant KMS + audit chain + rate-limit bucket.
- [x] **gRPC + REST parity** — `api/grpc_server.py` parallels REST with identical auth/audit.
- [x] **Single-binary distribution** — `pip install mind-mem; mm serve` ships authenticated endpoint.
- [x] **Sharded Postgres** — `block_store_postgres.py` shards via `tenant_id`.
- [x] **Replication + consensus for governance** — Raft-style audit-chain replication ships under `v4/federation.py`.
- [x] **Pluggable embedding backend with fallback** — local Ollama → API fallback chain ships in the embedding pipeline.

**Open (genuine network-hardening gaps):**

- [ ] **TLS 1.3 minimum + cert pinning** — currently inherits system trust store; explicit `TLSv1_3` floor + optional pinned-pubkey enforcement not wired. Tracked.
- [ ] **mTLS for service-to-service** — mutual auth between mind-mem nodes not implemented; today's threat model is single-operator shared-secret. Tracked.
- [ ] **Public / private workspaces** — `workspace.mode = public | private | mixed` configuration not surfaced. Tracked.
- [ ] **ActivityPub federation interop** — optional bridge dep not built. Tracked (low priority).
- [ ] **Audit headers (`X-MindMem-Request-Id`, `X-MindMem-Actor`, `X-MindMem-Purpose`)** — not yet propagated end-to-end across REST/gRPC. Tracked (small, well-defined).
- [ ] **Kubernetes operator + Helm chart** — `operator/` + `deploy/helm/` not shipped. Tracked.
- [ ] **Byzantine-safe consensus** — PBFT for adversarial-quorum deployments not implemented. Tracked (long-horizon).
- [ ] **Edge deployment mode** — `mind-mem-edge` PyOxidizer binary not built. Tracked.
- [ ] **Managed-service console** — `web/console/` multi-tenant dashboard not built. Tracked.
- [ ] **Kafka / NATS event fan-out** — governance events as streams not exposed. Tracked.
- [ ] **Rust hot path for hybrid search** — PyO3 BM25+RRF port — pure-MIND port (separate roadmap section below) is the chosen path instead. Marking as ⊘ superseded by Pure-MIND Core Port.

### E. Compliance-sensitive opt-in extensions (partial — 5 open)

**Shipped:**

- [x] **Pluggable redaction layer** — pre-write detector chain ships under the redaction module; events flow to audit chain.
- [x] **Confidence / Evidence as first-class** — structured `Evidence` blocks with `confidence_score` ship; recall surfaces evidence chains.
- [x] **Per-tenant audit chains** — `audit_chain.py` forks per tenant with isolated genesis + spec-hash binding.
- [x] **Compliance export pipeline** — `mm export --policy <policy> --since <date>` ships signed deterministic bundles.
- [x] **Contraindication / mutex edges** — `contraindicates` + `supersedes` edges ship as extra `block_lineage` kinds.

**Open:**

- [x] **Time-bounded and event-bounded recall** — `since` / `until` / `event_id` filters exposed on `recall(...)` (v4.0.15), applied via `_apply_post_filters` in `_recall_core.py`.
- [x] **Vocabulary-bound fields** — per-workspace controlled vocabularies not wired into `validate_block`. Tracked.
- [x] **Provenance-rich blocks** — `actor_id`/`actor_role`/`session_id`/`tool_id`/`purpose` fields gated by `provenance: off|recommended|required` not added. Tracked.
- [ ] **Row-level encryption over tenant KMS** — `src/mind_mem/tenant_kms.py` ships per-tenant AESGCM envelope keys; the row-level encryption layer above the existing `EncryptedBlockStore` is the open half. Tracked.
- [ ] **C2PA content provenance** — C2PA-signed manifests on chat-layer synthesis blocks not implemented. Tracked (depends on chat layer above).

### G. Observability, reliability, ecosystem (partial — 7 open)

**Shipped:**

- [x] **OpenTelemetry tracing + metrics + logs** — `v4/observability.py` ships OTel spans + Prometheus `/metrics`.
- [x] **Health / liveness / readiness probes** — `v4/health.py` ships standard probes.
- [x] **Continuous backup + PITR** — incremental backup + audit-chain PITR ships.
- [x] **Performance regression alerting** — `.github/workflows/benchmark.yml` runs latency benchmarks per PR.

**Open:**

- [ ] **JavaScript / TypeScript SDK** — client code ships in-tree at `sdk/js/`; the npm publish as `@star-ga/mind-mem-client` is the open step. Tracked.
- [ ] **Browser-native WebAssembly bundle** — WASM read-only client not built. Tracked.
- [ ] **Go SDK publish + Rust / Java / Ruby stubs** — Go client ships in-tree at `sdk/go/` (with tests); module publish is the open step. Rust/Java/Ruby not started. Tracked.
- [ ] **OpenAPI + AsyncAPI specs** — declarative specs not published; clients are hand-rolled. Tracked (small, well-defined).
- [ ] **Migration importers from competing systems** — file-based subset implemented: `mm import --from {chroma|mem0|letta} <dump.json>`. Endpoint-backed (pinecone / weaviate / qdrant) still deferred — they need a live endpoint + API credential.
- [ ] **Model-call token metering** — per-day token counter + optional daily cap behind `mm usage`. Tracked.
- [ ] **SLSA build provenance level 3** — partial via Sigstore; isolated-builder attestations not yet wired. Tracked.
- [ ] **Plugin SDK** — stable plug-in API for custom rules / block kinds / decay schedules / redaction detectors not formalised. Tracked.
- [ ] **Chaos testing harness** — automated fault injection for federated deployments not built. Tracked.

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

Auditable criterion for every roadmap addition. Lifted from the same TRIZ pattern in `mind/docs/roadmap.md`, tuned to the persistent-memory domain.

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
Python surface is a thin adapter shell, then eliminate it. The MIND
compiler's bootstrap/front-end now self-hosts (byte-identical native-ELF
fixed point), with full-toolchain self-hosting on the `mind` roadmap, so
this is a real trajectory, not a category boundary.

**Already MIND today:** the hot scoring/decision kernels ship as
`.mind` and compile via `mindc` — `bm25`, `rrf`, `reranker`,
`abstention`, `adversarial`, `answer`, `category`, `cognitive`,
`cross_encoder`, `ensemble`, `evidence`, `governance`. These run with
a pure-Python fallback, so the kernel boundary is already proven and
non-load-bearing for availability.

**Gating dependency (updated — the compiler-side blocker has shipped):**
`mindc` library-emit / stable C-ABI (cdylib output + FFI surface) landed
upstream in 0.2.6 (`pub fn`→C export, RFC 0002/0003 cdylib seam) and 0.3.0
(`--emit-shared`, struct-ABI lowering) — see `star-ga/mind-nerve`'s ROADMAP
for a sibling consumer already tracking this as mindc-side SHIPPED. The
remaining gap is entirely on this repo's side: the port work itself hasn't
started. Until the governance/core-retrieval/I/O layers below are actually
ported, the I/O shell (MCP transport, SQLite/Postgres/Redis adapters, HTTP,
external model clients) stays Python — by sequencing choice, not by a
missing compiler capability.

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

## Calibrated Recall Confidence (sidecar)

Attach a **calibrated confidence/utility score** to each recall result — a
usable signal for how strongly a retrieved block matches intent, beyond the
raw fusion rank.

- **Outside the deterministic path.** The confidence is a sidecar field on the
  result, not an input to scoring or to the audit chain. Recall ordering and the
  evidence/proposal chain stay deterministic and auditable; the confidence rides
  beside the result and is **excluded from any audit hash**. A probabilistic
  estimate must never become load-bearing for a governed mutation.
- **Calibrated, not raw.** Report a calibrated score (reliability-curve fit over
  the BM25 + vector + RRF fusion), not raw similarity, so agents can threshold on
  it for "recall or say no record found."
- **Use.** Drives the agent-facing decision boundary — low-confidence recalls
  return as explicit low-confidence rather than confident guesses, reinforcing
  the "cite or say no record found" discipline.
- **Status:** Planned. Composes over the existing `recall.py` fusion path and the
  `retrieval_diagnostics` surface; no new storage, no change to the deterministic
  ranking.
- **Method note — conformal prediction as the calibrator (prior-art shape observed
  2026-07-05).** The reliability-curve fit above should be a **conformal-prediction
  calibration**: sort nonconformity scores over a labeled recall calibration set and
  take the (1−α)(n+1)/n quantile — a distribution-free coverage guarantee that is
  **deterministic given the calibration set** (both properties the wedge already
  worships). This upgrades the sidecar from "confident/not" to a **recall-abstention
  gate with a proven bound**: "return the smallest set of memories that provably
  contains the right one at X% coverage, else say no record found." It is a ~40-line
  quantile rule, reimplemented in fixed-point in the pure-MIND core — never a
  dependency. Prior-art shape: probabilistic-modeling libraries that ship a
  conformally-calibrated defer/answer cascade (a distill→cascade `task` surface).
- **Research-watch — faithful (not just factual) calibration, training-side complement
  (2026-07-08).** RLMF, *"Reinforcement Learning with Metacognitive Feedback Elicits
  Faithful Uncertainty Expression in LLMs"* (Liu et al., Yale NLP, COLM 2026 —
  arXiv:2606.32032, github.com/yale-nlp/RLMF). Distinction worth stealing: *factual*
  calibration = "90% claimed → right 90% of the time"; *faithful* calibration = the
  stated confidence matches the model's own internal consistency (measured by sampling
  the same answer ~N× and comparing to the confidence it emits). Their fix is a second
  RL reward for how well the model predicts its own performance ("metacognitive
  feedback"), reported to surpass standard RL by up to 63% on faithful calibration.
  - **The lift for mind-mem is the *evaluation lens*, not the training recipe.** The
    conformal sidecar above calibrates *retrieval* confidence (inference-time, over the
    fusion score, distribution-free bound). RLMF is the *generation-time* twin: if a
    mind-mem answer surface ever emits a spoken confidence ("I'm fairly sure this block
    is the one"), faithful-calibration is the honest test of whether that phrase matches
    the system's actual self-consistency. It sharpens the existing "cite or say no record
    found" discipline from a rule into a measurable property.
  - **What NOT to lift.** Don't bolt an RL training loop onto the deterministic recall
    path — mind-mem's honesty rail is a *gate*, and the conformal quantile already gives
    it a proven bound with none of RL's nondeterminism. RLMF is the wedge-*aligned* idea
    (a system telling the truth about what it knows) but it's an LLM-training technique;
    keep it beside the recall path, not inside it. Orthogonal to the substrate/determinism
    wedge — epistemic honesty, not execution byte-identity (the two rhyme, don't conflate).
  - **Promote-out criteria.** Only becomes work if/when mind-mem fronts an LLM answer
    surface that emits linguistic confidence; then port the *faithful-calibration metric*
    (self-consistency vs stated confidence) as a fixed-point eval harness, never the RL
    loop, and only after the conformal sidecar ships.

- **Gap — the gate has no failure-attribution input (2026-07-22).** Both calibrators
  above answer *"how confident should this recall be?"*. Neither answers *"when an
  answer built on this recall came out wrong, which half broke?"* — retrieval never
  returned the right block, or it did and the consuming LLM ignored/contradicted it.
  Those are indistinguishable downstream (a fluent, confident, wrong answer either
  way), and the abstention classifier already shipped (`abstention_classifier.py` /
  `abstention.mind`, 5-signal pre-LLM gate, adversarial 36.3% → 92.4%) fires on
  *predicted* low confidence, not on *observed* post-hoc failure — so nothing today
  closes the loop from a wrong answer back to its cause.
  - **The split, stated as a decision rule.** Measure the halves separately and route
    on the pair: *retrieval* = was the correct block returned at all and at what rank
    (recall@k / MRR over labelled query→block pairs, which `retrieval_diagnostics`
    already has the surface for); *generation* = given the correct block, is every
    claim in the answer supported by it. The disambiguating experiment is **forced
    context**: hand the consumer the known-correct block deliberately. Wrong answer
    out ⇒ generation. Block never retrieved in the normal path ⇒ retrieval.
  - **Why it belongs to this section rather than beside it.** A defer/abstain
    threshold can only be tuned against a measured retrieval-quality signal; without
    the split, a conformal bound is calibrated on an outcome whose cause is unknown,
    and any later tuning of the threshold is unfalsifiable. This supplies the missing
    input to the gate above — it is not a second gate.
  - **Trap to avoid.** Widening the recall set to compensate for a suspected retrieval
    miss trades precision for recall and can *raise* the confident-wrong rate as the
    correct block is buried among distractors. Not forbidden — cheap to try once the
    split exists, worthless as evidence before it.
  - **First step is a backtest, not a build.** This gap was named from an external
    prompt, not from a defect we hit. Replay past recall complaints and count how many
    would have been diagnosed faster with the split in place. Zero ⇒ close this bullet
    with a dated note rather than building it. Cross-repo sibling: naestro **R61**
    (same rule, the cockpit/doc-retrieval side).
  - **Prior art — one-class learning on the success manifold (observed 2026-07-29).**
    A public paper attacks the *unsupervised* form of exactly this attribution problem:
    train only on **successful** trajectories, score each step of a failed run by its
    deviation from the learned dynamics, and take the high-deviation steps as the
    error-contributing ones. No failure examples, no step-level error labels. The
    transferable primitive is the framing, not the architecture: **we have abundant
    good recalls and no labelled corpus of bad ones**, which is the same asymmetry.
    Learn what a good recall looks like, score deviation, abstain above threshold.
    Under ~100 training trajectories, and 200–5000× cheaper than prompting a frontier
    model to audit each step.
  - **Why it lands here specifically: conformal is the load-bearing part.** The paper
    reports two thresholding strategies over the *same* anomaly scores — a fixed top-k,
    and a conformal-prediction set with an adaptive, distribution-calibrated threshold.
    In-domain, CP moves precision 0.321 → 0.443 while top-k keeps the higher recall
    (0.706 vs 0.484). Same trade the conformal sidecar above is built on, arrived at
    independently: **the quantile rule is what converts a noisy score into a usable
    set.**
  - **CORRECTION (same day, 2026-07-29) — this is the THIRD sighting, and the decisive
    one is ours and already shipped.** The bullet above was first written calling this
    a "second independent sighting" alongside the external distill→cascade note, which
    framed conformal as something we might build. That was wrong, and an ecosystem-wide
    check found why: **`ar_spine/rsi/conformal.py` (216 lines) is working code in
    `mind-lab`, not a plan.** It implements the split-conformal quantile as the gate
    (`conformal_threshold(calib, stratum, alpha)`, distil only when `score > tau`),
    **Mondrian stratification** (per-problem-class quantiles, not one global bound),
    cold-start abstention (`MIN_CALIB_PER_STRATUM = 20`, `MIN_BAD_PER_STRATUM = 5` —
    below either, `tau is None` and it abstains), the conservative unbounded case
    (rank > n_bad ⇒ `tau = +inf`, always abstain; its docstring: *"we never clamp"*),
    and `Fraction` arithmetic with golden-parity tests so rank boundaries cannot drift
    on float comparison. Its own docstring states the design rule we would otherwise
    have re-derived: **"The deterministic quantile IS the gate; no LLM in this module."**
  - **What that changes about this section.** The question is no longer *whether* to
    adopt conformal calibration — it is **reuse-vs-reimplement**. The sidecar note above
    proposes "~40 lines reimplemented in fixed-point in the pure-MIND core"; that remains
    a defensible call for the wedge, but it must now be made knowing a tested Python
    implementation exists with cold-start and unbounded-tau semantics already worked out.
    If we reimplement, those three behaviours are the spec to match, and the golden-parity
    tests are the oracle. Do not re-solve them from scratch.
  - **Fork check (verified 2026-07-29).** `conformal.py` appears at eight live paths
    across `ar-spine/`, `autoresearch/`, `mind-lab/`, `mind-lab-build/` and
    `mind-lab-staging/`. All eight are **byte-identical** (md5 `24034cf5eb89`), so the
    "never fork across repos again" intent in its header is currently holding. Recorded
    because vendored-copy count is the kind of thing that silently stops being true;
    re-check before editing any copy.
  - **The numbers that bound the ambition — read before anyone proposes wiring it to a
    decision.** Best in-domain precision is **0.443**: more than half of flagged steps
    are wrong. Random step-selection scores 0.284 precision — *above* both frontier-LLM
    prompting baselines (0.233 and 0.217), so "beats a frontier model" describes a weak
    baseline, not a strong detector. Out-of-domain it degrades hard: precision 0.184,
    recall 0.330, and the random baseline is 0.137 — the margin narrows to a few points
    on a cross-benchmark shift. Treat published +20% / +7% F1 as *relative to those
    baselines*, never as an absolute standard.
  - **Therefore: diagnostic input only, never an autonomous gate.** This may inform the
    retrieval-vs-generation split (a ranked "which step looks off" hint for a human
    replaying a bad answer). It must not become an abstention trigger on its own —
    a 0.44-precision signal driving abstention would suppress correct recalls roughly
    half the times it fired, which is strictly worse than the shipped 5-signal
    classifier. Same boundary the 512-mind calibrated-confidence sidecar draws: the
    score informs, the gate decides, and they never merge.
  - **Sequencing.** Strictly behind the backtest above and behind the conformal sidecar.
    If the backtest closes this bullet, this closes with it. Cross-repo: the
    observability-only reading of the same paper belongs to naestro's agent-trajectory
    lane, not here — mind-mem's interest is the one-class + conformal *pairing*.
  - **Provenance rail.** Prior-art shape observed in a public paper; no code, no
    dependency, nothing named in any public artifact. Full citation lives in
    `mind-internal` per the no-public-attribution convention.

## Divergent recall-path work on `autoresearch-recompact/jul10` (opened 2026-07-22)

**Operational note, not a feature.** A long-lived branch is carrying recall-path
changes that now conflict with `main`. Recorded here so it is visible before
someone attempts the merge cold.

**State as measured 2026-07-22** (`git merge-tree origin/main
autoresearch-recompact/jul10`, exit 1):

- Branch is **11 commits ahead, 3 behind** `origin/main`; fork point
  `58034a7`, last branch commit 2026-07-10 — roughly twelve days divergent.
- **Conflicts on merge:** `src/mind_mem/_recall_core.py`,
  `src/mind_mem/hybrid_recall.py`, `src/mind_mem/block_store_postgres.py`,
  `ANATOMY.md`, and an **add/add conflict** on
  `tests/test_issue_139_140_recall_infra.py`.
- `src/mind_mem/recall_vector.py` and `_recall_constants.py` also changed on
  both sides but auto-merge.

**The add/add is the informative one.** Both branches independently created a
test file for issues #139/#140, and both sides carry a commit whose subject is
`fix(recall): engage hybrid on config drift + bound MCP recall waits (#139, #140)`.
The same defect was fixed twice in parallel — once on the branch, once on main
(where it was then extended by `56674d8`, the pgvector leg + honest
retrieval-source labels). Merging is therefore not a mechanical conflict
resolution: someone has to decide **which fix is the real one** and whether the
branch's other ten commits (recompaction cleaning stages, fold attestations)
still apply on top of the newer recall path.

**Naming caveat.** The branch prefix `autoresearch-` refers to the *method* used
to develop it (pre-registered, exp-numbered iterations), not to a home repo. The
consolidated research tooling now lives in `mind-lab`; this branch's contents are
mind-mem product code and belong here. The prefix is misleading and cost one
misread already — worth renaming if the branch survives.

**Bearing on the attribution gap above.** `56674d8` changed the retrieval path
*after* the attribution-gap bullet was written. Before that bullet is acted on,
read it: "honest retrieval-source labels" may already supply part of the
observed-failure signal it asks for, in which case the bullet needs revision
rather than implementation.

- **Status:** Recorded 2026-07-22, unresolved. No merge attempted. Decision is
  which-fix-wins on #139/#140, not a conflict-resolution task.

> Ecosystem-wide milestone — gated on the `mind` compiler reaching self-host completeness.

Once the `mind` toolchain self-hosts (the open-core compiler builds itself byte-identically),
this repository's **Python** implementation is migrated to **pure, executing MIND**, so the whole
MIND ecosystem runs on its own deterministic, byte-identical, evidence-carrying toolchain — the
wedge applied to ourselves.

- **Gate:** `mind` self-host keystone complete (see the `mind` roadmap self-host track).
- **Approach:** port via the `mind-migrator` path — to the executable MIND subset, verifying every
  emitted symbol actually runs and reusing `std` primitives; no silent AOT-only stubs.
- **Invariant:** migration preserves behavior and the cross-substrate byte-identity gate — no
  regression in determinism or the evidence chain (signing of the
  evidence chain is itself a tracked `mind` milestone).
- **Status:** Planned — sequenced after `mind` self-host; tracked here so the endgame is explicit.

---

## Blocking-then-arbitration for entity resolution (prior art: recent working note on agentic KG practice)

> Opened 2026-07-25. Source is a third-party synthesis of public Anthropic material.
> **Idea only — no code adopted, no dependency, no public attribution.** Note that the
> social claims made *about* that document ("two Anthropic seniors", "1000x better")
> are fabrications by a reposter and appear nowhere in it; the document's own numbers
> are modest and honestly stated. Cite the technique, never the hype.

**The pattern.** Resolving thousands of entity mentions by handing them all to a
model in one prompt does not work. The disciplined form is two-stage: **cheap
deterministic blocking** — an inverted index over name tokens, no model call — to
group candidates into small blocks, **then** model arbitration *only within a
block*. Keep the model for the parts that require judgment; use deterministic logic
for everything else.

**Why it belongs here.** That is the mind-mem gate discipline applied to
resolution. It matters for the dedup/contradiction path, where the two failure
modes are the ones already tracked as quality risks:

- **Silent loss** — a surface form that lands in no cluster vanishes from the store
  entirely, because the alias map has no entry for it. A production resolver must
  fall back to a single-element cluster so nothing is dropped without a record.
- **Over-merge** — a specific entity folded into a broader one because their
  descriptions overlap. Loses precision in a way that then propagates through every
  downstream recall.

Both are spot-checkable, and both are exactly the class of error the governed
`propose_update` path exists to keep out of the store.

**Determinism constraint (ours, not the source's).** The blocking stage must be
pure and replayable — same inputs → same blocks, byte-identical, no clock, no RNG.
The arbitration call is the only non-deterministic step, and its *verdict* must be
recorded so the merge is re-derivable from the record even though the call itself
is not. An arbitration outcome must never reach a signed or sealed surface
unrecorded — same rule that keeps raw floats out of a signed payload.

**Disambiguation input.** The source note's sharpest observation is that resolution
quality comes almost entirely from a **one-line, per-entity, per-document
description written at extraction time**. Without it the resolver sees only names
and falls back to surface-form matching — the exact failure the approach exists to
avoid. If this is built, the description is a first-class input, not metadata.

**Related but already ahead of the source:** the corroboration-weighted, integer-
encoded confidence the same note lists as a "future direction" is already shipped
in `mind-lab`'s `dr-spine` (`independent_corroboration()` → saturating term →
`confidence_milli`). Nothing to port; noted so the two are not confused.

**What we did NOT take.** The note's knowledge-graph layer — mind-mem is already a
governed store with provenance chains, and a KG would duplicate it. And its
"operational discipline" section is a list of *conventions* (sample daily, cap
volume, version the schema); our position is that conventions get skipped under
deadline and invariants do not.

- **Status:** Proposed 2026-07-25. Evaluate-item, not a commitment. Blocking is
  ~an inverted index, not a dependency. Sequence behind any live dedup-quality
  signal — build the measurement before the mechanism.

## Graph-retrieval systems are not memory systems (positioning, prior art observed 2026-07-30)

> A permissively-licensed multi-language AST-graph tool reached very large adoption
> very fast and publishes a benchmark table putting itself against dedicated memory systems on
> LOCOMO and LongMemEval. **Idea only — no code adopted, no dependency, no public
> attribution.** This is a positioning entry, not a feature request: nothing in its
> engine belongs in mind-mem. Recorded because the comparison it invites is the one
> we will be asked to answer.

**The category confusion, stated plainly.** That tool builds a knowledge graph from
a corpus by walking syntax trees deterministically, then retrieves over it. It
reports strong retrieval numbers (recall@10 roughly 10x one dedicated memory system,
above BM25) at near-zero build cost, and it is measured on the same academic memory
datasets. A reader skimming the table concludes "graph retrieval beats memory
systems." That conclusion does not follow, and the reason is worth writing down
before someone asks.

**What retrieval benchmarks measure, and what they omit.** LOCOMO and LongMemEval
score *finding the right passage and answering from it*. They contain no notion of:
whether a write was authorized, whether the new fact contradicts a stored one,
whether a bad write can be rolled back, or whether the store's history is auditable.
A system can top every one of those benchmarks and still be a corpus you rebuild
from scratch — which is exactly what a graph builder is. mind-mem's product is the
**governed write path**: HITL-gated `propose_update`, contradiction detection,
block lineage, rollback, an auditable chain. Those are properties of the *store over
time*, and no retrieval benchmark has ever scored them.

**The honest concession, so this is not self-serving.** Retrieval quality is a real
axis and one we compete on. Where these systems are genuinely ahead is build cost —
a deterministic AST/structure pass costs zero model credits, and any memory system
whose ingest requires per-item model calls carries a cost floor that a structural
extractor does not. That is a legitimate pressure on ingest-heavy designs and worth
tracking against our own ingest path. Two further caveats cut the other way: their
harness is self-authored (their repo, their adapters wrapping the competing systems,
their judge), and self-run harnesses grade themselves generously by construction —
though publishing a second-judge agreement statistic is more rigor than most memory
papers offer and should be credited. Also, retrieval-recall comparisons across
systems with different embedders are confounded and should not be read as a clean
ranking.

**What we take from it.** Nothing structural. One discipline worth mirroring: they
publish judge validation, fairness rules, and a spend ledger alongside the numbers.
If mind-mem ever publishes a retrieval comparison, it publishes on those terms —
one shared model, disclosed budgets, a validated judge, and the harness open — or it
does not publish. A benchmark table without those is marketing.

**The framing to use when asked.** Retrieval quality and governed durability are
orthogonal axes, not competing scores on one. "Which retrieves better" is a fair
question with a measurable answer; "which is the memory system" is answered by which
one can tell you *why* a fact is in the store, *who* approved it, and *what it
replaced*. Same shape as the determinism-vs-accuracy distinction on the compiler
side: the benchmark the field runs is not the property we sell.

- **Status:** Positioning note, 2026-07-30. No work item. Revisit if a governed-write
  competitor appears — that would be a real one, and none of these are.

> Provenance (repo, author, metrics) recorded privately in `mind-internal`, per the
> no-public-attribution rule.

## Verified vs. attested: naming the runtime half of trust (prior-art shape observed 2026-07-30)

> A major cloud vendor published an open, vendor-neutral knowledge-format
> specification whose trust model draws a distinction we do not currently name.
> **Vocabulary and design constraint only — no format adopted, no conformance
> claimed, no code taken.** Unlike the code-derived entries above, this one cites a
> *published specification*: a public standard exists to be read and reasoned about,
> and recording that it draws a distinction is not the kind of provenance the
> no-attribution rule protects. The source is still not named in any public artifact.

**The distinction.** That spec separates two things a consumer needs to know about a
stored fact, and argues both must exist independently:

- **Verified** — *the definition still matches policy.* Doc-level, slow, recorded
  durably in the store. Answers "has a human confirmed this content against its
  sources?"
- **Attested** — *this particular run produced its value the sanctioned way.*
  Per-call, runtime, **deliberately not stored**. Answers "was this number computed
  by the blessed path, or did the agent improvise?"

The load-bearing argument for keeping them apart: a *stale definition can still
attest cleanly*, and a *freshly-verified definition still requires attestation on
every run*. Collapsing them into one flag loses one of those two failure modes.

**Where mind-mem stands, measured not assumed.** We have the first half and named
it: `propose_update` → `approve_apply` is a durable, human-gated, stored
verification, with `block_lineage`, `contradiction_detector`, `audit_chain` /
`hash_chain_v2` behind it. We have real *pieces* of the second half that were built
for other reasons and are not organised under this concept:

- `apply_engine.write_receipt` / `update_receipt` emit an `APPLY_RECEIPT.md` per
  apply — but that is a receipt *of a governed write*, not of a *read/derivation*.
- `spec_binding.py` binds the active governance config to a content hash and requires
  explicit re-attestation on drift — genuinely attestation-shaped, but scoped to
  configuration, not to a recall result.
- `verify_cli.py` already reports `"no binding — not yet attested"`.

So the gap is specific and worth stating precisely: **nothing attests a recall
result.** When mind-mem answers a query, the caller gets ranked blocks with no
runtime evidence of *which retrieval path produced them*. The degraded-recall marker
shipped this session (BM25-only fallback surfaced in the response envelope) is the
first true instance of this class — a runtime signal about how an answer was
produced — and it arrived without the vocabulary to place it. That is the argument
for adopting the naming now rather than after three more one-off flags.

**The design constraint we do not relax.** That spec derives its trust tier from an
*actor-string prefix* — a producer writes `human:<id>` into a metadata field and the
"human-reviewed" tier is granted. The spec is candid that these are advisory signals,
not access control. We do not copy that. A tier that a producer can self-assert is a
claim, not evidence; ours is anchored to the approval gate and the hash chain. If an
attestation surface lands here, its verdict must be **derivable from a recorded
artifact** (which path ran, which config hash, which chain head) and never from a
self-declared field. Same discipline as the codegraph edge-provenance label: assigned
by which pass emitted it, never by a model and never by a threshold.

**What we would take, if this becomes a work item.** The vocabulary, and one
structural rule from it: the attestation verdict is a *runtime artifact and is not
written back into the store*. Storing it would turn a per-run fact into a durable
claim that goes stale exactly the way the spec warns a credibility score does — the
same reason that spec records credibility *signals* and refuses to store a *score*,
a discipline we independently arrived at for edge provenance.

- **Status:** Vocabulary adopted for internal use, 2026-07-30. Not a schema change,
  not a conformance commitment. Format-level interop is a separate and larger
  question that only pays if we target the enterprise-catalog ecosystem, which is not
  our current buyer. Sequence any implementation behind a real caller that needs to
  distinguish "this answer came from the full stack" from "this answer came from a
  fallback" — the degraded marker is that caller's first, narrow instance.

## Selective re-injection: mind-mem as an active memory agent (prior art observed 2026-07-31)

> Recent published work on long-horizon agents names a failure mode we have been
> designing around without a word for it, and reports an ablation that bears directly
> on how mind-mem is consumed. **Concept and benchmark design only — no code taken, no
> numbers reproduced here.** The external figures are that work's claims, not ours.
> Source not named in any public artifact; citation lives in `mind-internal`.

**The failure mode.** In a long trajectory, a constraint that is present, visible, and
still inside the context window stops influencing decisions. Not eviction, not
hallucination — the text is right there and the agent has simply stopped treating it
as relevant. That is a *retrieval-time relevance* problem, not a capacity problem, and
it is the reason "we stored it and the caller can recall it" is not the same as "the
caller acted on it."

**Why this is a mind-mem entry and not only a consumer's problem.** Our surface is
built on a pull model: a caller decides to `recall`, and everything else in the store
is inert until asked. The external ablation compared four ways of exposing a memory
store to a working agent — no exposure, always-on injection, passive store with recall
available, and a *gated* policy that mostly stays silent and speaks only when the
current trajectory conflicts with something stored. The gated arm won; **passive
exposure was among the weakest**. Our default consumption pattern is the passive arm.
That is a claim about *our product surface*, which is why it belongs here rather than
only in a consumer repo's roadmap.

**What already exists, and what is genuinely missing.** Most of the pipeline is
shipped and was built for other reasons:

- `intent_router.py` — 9-type classification with confidence, regex, no model call.
  Viable as a free pre-filter deciding whether to spend anything at all.
- `contradiction_detector.py` — BM25 + optional vector against the committed corpus,
  returning block id, similarity, and conflict type. This is already the
  "does the trajectory conflict with something stored" primitive.
- `agent_bridge.py` — per-agent injection *formatting* (CLAUDE.md / AGENTS.md /
  GEMINI.md / .cursorrules / .windsurfrules / aider conventions).
- The local 4B model is a plausible judge for a one-line yes/no verdict: resident,
  low temperature, already instructed to emit structured output without commentary.

The missing piece is narrow and specific: **`agent_bridge` can format an injection but
cannot deliver one mid-run.** Its own docstring records the filesystem watcher and
per-agent hook installer as deferred. Everything else in the chain exists.

**Delivery is a property of the consumer, and mind-mem should not pretend otherwise.**
A harness with lifecycle hooks (measured 2026-07-31: `SessionStart` and `Stop` are
installed and firing in at least one consuming harness on this box) can carry a
mid-run tap today; a harness that owns its own scheduler can do it properly per step;
an opaque third-party process cannot receive one at all, and writing to its config
file reaches the *next* session, not the running one. mind-mem's job is to expose the
verdict and the formatted line; **claiming delivery it does not control would be the
dishonest part.** The library surface should therefore be delivery-agnostic and say so.

**The discipline that does not relax.** The external result is that a *probabilistic*
judge outperforms *fixed* injection schedules. That cuts against this project's
posture, and the resolution is scope rather than reversal: a trigger verdict is
**advisory, never evidence**. It must not be written into the store, must not appear
in a hash chain, and must never be able to suppress the approval gate — the same rule
already applied to attestation verdicts in the entry above, and to edge provenance:
a runtime judgement stays a runtime artifact. Storing a "the model thought this was
relevant" flag would turn a per-run guess into a durable claim that goes stale.

**Required controls before any gain is claimed.** The load-bearing external finding is
the *comparison*, not the architecture. If a trigger policy is built here, it closes
only with a four-arm measurement on our own traffic: no injection, always-on, passive
recall-available, and gated. A result showing the gate is no better than passive on
our workload is a valid outcome and closes this entry as a negative finding. Adopting
the architecture without reproducing the comparison would be importing someone else's
conclusion — the failure mode this project has already been burned by once.

- **Status:** Proposed 2026-07-31. Sequenced behind a real consumer that can deliver a
  mid-run injection; without one, this is formatting with nowhere to go. Retraining the
  local judge is explicitly deferred — the external work's own fine-tuned variant
  reported only partial transfer, which is weak evidence that training is where the
  gain lives, and a prompt-only baseline must be measured first. Delivery mechanisms
  belong to consumers; this entry covers the verdict surface only.

#### K.2 — Ontology-governed logical knowledge graph (prior-art-informed, 2026-08-19)

Prior art: recent enterprise knowledge-graph practice argues that the bar for
"knowledge-graph memory" is (a) explicit ontological *and logical* modelling
and (b) genuine multi-hop traversal tooling — and that auto-linked
Markdown/YAML traversed by grep is neither.

**Most of that bar we already clear, and this entry does not re-litigate it.**
`knowledge_graph.py` is a SQLite triple store with canonical entities, typed
predicates, aliases, per-edge confidence, source provenance, and validity
windows; `neighbors()` is real BFS N-hop traversal (predicate/direction
filtered, bounded) exposed as `graph_query`; writes route through the
`propose_edge` → `approve_edge` HITL gate. Description-grounded entity
resolution, blocking + LLM-arbitration, answer-with-edge-citations,
cross-source edge confidence, and schema versioning are already open items
in Group K above and are **not** duplicated here.

**What is genuinely absent, verified against the tree (2026-08-19):**

1. `src/mind_mem/ontology.py` exists (OWL-lite: entity types, parent/child
   hierarchy, inherited/required/optional properties, strict validation,
   versioning) — but `grep -c 'ontology' ` returns **0** in all three graph
   write paths: `graph_ingest.py`, `knowledge_graph.py`, and
   `mcp/tools/graph.py`. The ontology is reachable via `ontology_load` /
   `ontology_validate` and is used by `context_core.py` for core export.
   It is *not* consulted before an edge becomes authoritative.
2. `ontology.py` carries **no relation-level semantics** — no `domain`,
   `range`, `inverse`, `transitive`, `symmetric`, `functional`, or
   `disjoint` (grep returns nothing for all seven). Predicates are typed
   as an enum but carry no logical meaning.

Consequence: a triple whose predicate is enum-valid but ontologically
nonsensical (`Person --AUTHORED_BY--> Language`, where `AUTHORED_BY` should
be `domain: Artifact, range: Person`) is accepted today. And because
`PART_OF` has no transitivity declaration, `A PART_OF B` + `B PART_OF C`
cannot derive `A PART_OF C`.

**Two engines, not one (do not conflate).** Validation and derivation are
opposite semantics and must not be built from the same mechanism. OWL
`domain`/`range` are *inferential*: given `Person --AUTHORED_BY--> Language`,
an OWL reasoner concludes `Language is-a Person` rather than rejecting the
triple. An open-world entailment engine therefore **rejects nothing** and
cannot serve as a write gate. The split:

- **Write gate = closed-world shape validation** (SHACL-shaped): a triple
  violating a declared shape is *refused* and becomes a governance finding.
  Absent shape → refuse (fail-closed), never silently admit.
- **Read side = bounded derivation** (Datalog / OWL-RL-lite): derives
  additional views, never rejects and never writes.

Both consume the same relation schema; only the validator gates writes.
Implementers must not build the entailment engine and call it the gate.

**Wedge guardrail (load-bearing):** a reasoner may derive *views*; it must
never mutate source-of-truth memory. Derived facts are materialized views
carrying `rule_id`, `ontology_version`, and `source_edges[]`, and are
excluded from the sealed audit-hash preimage — same rule already applied to
edge confidence and attestation verdicts. Ontology validation is a
*structural* gate that runs **before** the existing HITL gate; it does not
replace or weaken it. Order: extraction → ontology validation → governance
approval → authoritative graph.

- [ ] **Make the ontology executable on write paths** — call ontology
      validation inside `graph_ingest`, `propose_edge`, `approve_edge`,
      direct-admin `graph_add_edge`, and entity-observation writes. An
      invalid triple becomes an explicit governance finding; it never
      silently enters the authoritative graph. Bind every entity/edge to
      an ontology version + hash. Custom predicates must be declared in
      the workspace ontology before they are authoritative.
- [ ] **Relation schemas in `ontology.py`** — add `domain`, `range`,
      `inverse`, `symmetric`, `transitive`, `functional`, disjoint entity
      types, and optional cardinality. Prerequisite for both the write
      gate above and any entailment below. Declaring a predicate both
      `symmetric` and `transitive` (e.g. a generic `related_to`) collapses
      its component of the graph into one equivalence class where
      everything reaches everything; such combinations are rejected at
      ontology-load time, not discovered at query time.
- [ ] **Closed-world shape validator (the actual write gate)** — the
      rejecting half of the pair above. Per the write-avoidance ladder,
      evaluate `pyshacl` before hand-rolling; adopt only if it can be
      driven deterministically and offline (no network, no wall-clock),
      otherwise implement the narrow subset we need (domain, range,
      cardinality, disjointness) directly against SQLite. A hand-rolled
      validator is acceptable; a hand-rolled *entailment* engine is not
      when `owlrl` exists.
- [ ] **Persist the active ontology per workspace** — currently
      process-local via `OntologyRegistry`; an ontology that governs
      authoritative writes cannot live only in process memory. Ontology
      migrations must be explicit, versioned, and replayable.
- [ ] **Bounded deterministic reasoner (OWL-RL-lite / Datalog-style, not
      full OWL)** — subclass inheritance, inverse/symmetric/transitive
      predicates, domain/range inference. Derived facts are views, never
      writes, and each carries its full derivation path. Type-disjointness
      and logical-incompatibility violations feed the existing
      contradiction/governance system. Over SQLite the transitive closure
      is a `WITH RECURSIVE` view, not a new subsystem — reach for a
      dependency only where recursion alone is insufficient.
- [ ] **Closure must respect time — it currently ignores it** — edges
      already carry `valid_from` / `valid_until` (`knowledge_graph.py`),
      but nothing in the entailment spec above consumes them. Composing
      `A PART_OF B` (valid 2024–2025) with `B PART_OF C` (valid from 2026)
      derives `A PART_OF C`, a fact that was true at no instant. Every
      derivation is evaluated **as-of** an instant (default: now), and a
      derived view carries the *intersection* of its source edges'
      validity intervals; an empty intersection means the rule does not
      fire. A closure that ignores time silently manufactures history.
- [ ] **Canonical derivation, so "same query → same answer" is
      falsifiable** — recursive CTEs return rows in unspecified order and
      a fact reachable by several paths has several equally valid
      derivations. Without a rule, two runs can return the same answer
      with different explanations, and the acceptance gate below cannot
      fail. Required: a total order over derivation paths (shortest hop
      count, then lexicographic by `(rule_id, source_edge_id…)`), so the
      *explanation* is bit-identical run to run, not merely the conclusion
      — the same discipline the compiler's byte-identity gate applies to
      output.
- [ ] **Graph pattern query planning** — extend `graph_query` beyond
      `entity + depth + predicate` into a bounded pattern API with typed
      variable constraints, traversing graph indexes only (never grepping
      Markdown/YAML). Score paths on edge confidence, provenance quality,
      temporal validity, contradiction state, and hop penalty, under hard
      bounds on hops/fan-out/budget.
- [ ] **`GraphEvidencePack` — edge-grounded context packing** — return the
      minimal supporting triples + citations rather than pulling whole
      source blocks into the recall set (today's `kg_fusion` path finds an
      edge, then hydrates the source block). Fetch full blocks only on
      explicit demand. This is the K.2 half of the Group L
      utility-per-context-token metric: measure **tokens consumed per
      correctly answered graph question**.
- [ ] **`GraphBackend` protocol + reproducible graph benchmarks** — keep
      SQLite as the zero-infrastructure backend; prove equivalent scale via
      a second backend (PostgreSQL recursive CTEs or a graph adapter).
      Benchmark 1-hop / 2-hop / 3–5-hop, high-fanout nodes, temporal edges,
      ontology violations, and contradictory paths. Track answer accuracy,
      path precision/recall, ontology-violation detection rate, latency,
      nodes visited, and context tokens. **The benchmark must distinguish
      lexical retrieval from genuine relational reasoning** — otherwise it
      measures the retriever, not the graph.
- [ ] **Ontology drift as a continuous, per-ingest concern (not a
      design-time one)** — the items above treat schema conformance as a
      gate applied to a corpus; in practice every new ingest *re-opens*
      the resolution question. `EntityRegistry.resolve()` mints a fresh
      entity for any surface form absent from `aliases`, so duplicates
      accrete continuously rather than arriving all at once, and the
      divergence is discovered at query time rather than at write time.
      Required: (a) resolution runs on every ingest, not once at schema
      design; (b) an unmatched-surface-form rate tracked as a governance
      signal, so drift is observable before it is a graph-quality
      problem; (c) a periodic re-resolution pass over already-authoritative
      entities as the alias table and per-entity descriptions improve
      — proposing merges through `propose_edge`/HITL, never rewriting
      source-of-truth silently; (d) ontology-version pinning per entity
      so a schema migration makes affected entities explicitly stale
      rather than retroactively reinterpreting sealed history.
      Depends on the description-grounded resolver and blocking +
      LLM-arbitration items in Group K; this entry is the *when*, those
      are the *how*.
- [ ] **Resolution as reversible `SAME_AS` edges, never ID rewriting** —
      the Group K resolver items imply an approved merge collapses two
      entities into one id. That is the one irreversible mutation in an
      otherwise append-only store: it destroys the distinction it acted
      on, so a wrong merge cannot be undone from the record, and any
      block already citing the losing id is silently re-pointed. Instead a
      merge asserts a `SAME_AS` edge (HITL-gated like any other), and
      "the entity" becomes a **union-find view** over the `SAME_AS`
      component. Un-merging is retracting one edge. This is exactly the
      derived-views-never-mutate rule the reasoner already obeys — the
      original entry simply failed to apply it to entity resolution,
      which is where it matters most.
- [ ] **Deterministic scoring first; the model only in the gray band** —
      `_canonicalise()` is lowercase + whitespace-collapse only
      (`knowledge_graph.py`; its own docstring says "no fuzzy matching, no
      embeddings"), so `Dave` and `David Smith` mint two entities. Most of
      that gap closes with no model call: a nickname/diminutive dictionary
      and a transliteration table are deterministic, auditable, offline,
      and long-solved. Above that, a Fellegi–Sunter-style match-weight
      score (per-field agreement weights, summed into a log-odds) yields
      an explainable weight waterfall and a tunable threshold pair —
      auto-merge above, auto-reject below, **model arbitration only
      between**. This narrows the LLM's role from "resolve entities" to
      "adjudicate the ambiguous band," which is the only part needing
      judgment.
- [ ] **Decide merges on store-held state, never on model-supplied
      labels** — a merge proposal must be evaluated against the canonical
      record the store holds for each entity id, not against the
      proposing model's *description* of what it is merging. A gate that
      reads the caller's own summary of its action validates the summary,
      not the action, and is decorative under an adversarial or merely
      sloppy proposer. Proposals reference opaque entity ids; the
      reviewer's rendering is resolved server-side at review time.
- [ ] **Land the write gate in `dry-run` mode before it rejects** —
      every item above describes a validator that *refuses* a
      non-conforming write. There is no described way to turn it on. On a
      populated graph the first enforcing build rejects some fraction of
      existing callers, so the rule either ships broken or never ships;
      the predictable outcome is that the correct rule is written and
      left switched off. The gate therefore needs two modes from the
      start: **`dry-run`** (evaluate, record the decision and what it
      *would* have refused, forward the write) and **`enforce`** (refuse)
      — the same two names naestro R87 uses, deliberately, so a rule
      means the same thing in both stores. Both modes write the same
      decision row, distinguished by mode, so a `dry-run` trail is
      directly comparable to what enforcement would produce; that
      comparability is the entire value. A rule is authored, run against
      live ingest in `dry-run`, read out of the decision record, and only
      then promoted. The mode in force is itself part of the record —
      a validator whose mode is inferred rather than stated cannot be
      audited after the fact.
      **Fail-closed stays independent of mode**: `dry-run` governs what
      happens to the *write*, never what happens to a *broken rule* — an
      unparseable or absent shape still refuses. Cross-repo: this is the
      mind-mem half of naestro **R87**.
- [ ] **Close the `Predicate.register()` contradiction (open, contradicts
      the gate above)** — this entry states custom predicates "must be
      declared in the workspace ontology before they are authoritative."
      That is **false today**: `Predicate.register()`
      (`knowledge_graph.py:84`) mints a live predicate from an arbitrary
      string with no ontology consultation, and `from_str` then resolves
      it alongside the closed enum. Until registration routes through
      ontology declaration, the acceptance gate below is unmet by
      construction. Recorded as a known contradiction rather than left as
      an aspirational sentence.

**Acceptance gates.** No graph write path bypasses ontology validation; a
domain/range-invalid relationship cannot enter the authoritative graph;
same graph + same ontology + same query yields the same derivation **under
the declared canonical path order** (the conclusion *and* its explanation
are stable, not just the conclusion); no derived fact spans an empty
validity intersection; every approved merge is reversible by retracting a
single edge, with no source-of-truth id rewritten; every
inferred answer is explainable back to authoritative source edges/blocks; a
multi-hop query is answerable from a compact evidence pack without loading
source documents in full; an unmatched-surface-form rate is observable per
ingest, and a re-resolution pass proposes merges without mutating
source-of-truth. No enforcing validator is
promoted without a prior observe-mode run over live ingest whose recorded
would-refuse set was read; the mode in force is recorded with each
decision, never inferred; a malformed shape refuses in `dry-run` exactly
as it does in `enforce`.

- **Status:** Proposed 2026-08-19; revised 2026-08-21 after an
  architecture review. Sequenced **behind Group K.0** — graph population
  remains the bottleneck, and an ontology-governed write gate on a sparse
  graph gates almost nothing. Verified 2026-08-21: `knowledge_graph.db`
  does not exist on the development box, so the graph has never been
  populated and no duplicate-rate measurement is possible yet — which
  makes K.0 the true blocker, not a sequencing preference.

  **Revised first slice.** The original entry named relation schemas as
  the honest first slice. That is the *cheapest* item, not the highest
  leverage: it gates nothing while the ontology is still process-local
  (`OntologyRegistry`), so schemas would be declared and then not
  enforced. Corrected order — (1) a duplicate-rate / unmatched-surface-form
  report over a populated graph, so the problem is measured before it is
  engineered; (2) persist the active ontology per workspace with a
  canonical hash, so a declared schema is durable enough to gate on;
  (3) the closed-world validator on the write paths; (4) relation
  semantics and the derivation engine last, once there is something to
  reason over. Item (1) is blocked on K.0 by construction.

  **Deliberately not adopted.** Embedding-similarity or GNN link
  prediction for resolution, and PageRank-style graph retrieval, are the
  current fashionable alternatives; both are rejected here because a
  non-explainable, non-deterministic scoring function cannot satisfy the
  derivation-path and bit-identity gates above. Determinism is the
  constraint, not an oversight.

  The reasoner is explicitly scoped *below* full OWL: unbounded entailment
  on a memory store is a latency and explainability hazard, not a feature.

> Provenance (source + full analysis) recorded privately in `mind-internal`,
> per the no-public-attribution rule — public artifacts say "recent
> enterprise knowledge-graph practice" only.

### Group L — Recall utility per context token + prescriptive blocks (prior-art-informed, 2026-08-06)

Prior art: a recent task-agnostic plugin-memory module for LLM agents argues that
raw interaction history should be transformed into structured, reusable knowledge
before retrieval, and — the part that matters here — evaluates memory on a
**utility-against-consumption** axis: how much decision-relevant information a
memory module contributes *relative to how much of the agent's context window it
consumes*. It also splits stored knowledge into **propositional** units (facts) and
**prescriptive** units (reusable skills/strategies), citing the cognitive-science
distinction between knowing facts and knowing how to perform a task, and reports
that a single general-purpose module beat task-specific memory designs on three
benchmarks while spending fewer memory tokens.

**What is genuinely new to us is the metric, not the architecture.** The structured-
units-over-raw-logs position is already this project's design: `knowledge_graph.py`,
`category_distiller.py`, `causal_graph.py`, `graph_recall.py`, and `intent_router.py`
all predate the external work and were not derived from it. What we do *not* have is
a published number for **what a token of recall buys**. The machinery to *spend* a
budget exists — `pack_recall_budget(query, max_tokens=2000, limit=20)`
(`mcp/tools/recall.py:394`), `compressors.py`, `evidence_packer.py`,
`_recall_tokenization.py` — and every one of them takes a budget as input while
reporting nothing about yield. That asymmetry is the gap: the project can say how
much context it consumed and cannot say what the consumption was worth.

- [ ] **L1 — Recall utility per context token.** Define and publish a utility-against-
  consumption measure for `pack_recall_budget` and the recall path generally:
  decision-relevant content delivered per token of context spent. Requires a
  ground-truth set on our own corpus, not a borrowed benchmark — now specified in
  [`docs/design/eval-set-ground-truth.md`](docs/design/eval-set-ground-truth.md),
  shared with M7 and deliberately written before either consumer so it is not
  built to fit only one of them. **Close condition is
  one-sided and decision-shaped, matching the criterion-gate rule used elsewhere in
  the ecosystem:** a measured curve of utility against `max_tokens`, on our traffic,
  with the honest possibility that the current default (2000) is already at or past
  the knee. A result showing no headroom closes this as a negative finding and is a
  valid outcome — the point is to make the claim measurable, not to make it flattering.

- [ ] **L2 — Prescriptive blocks (EVALUATE, not committed).** Assess whether a
  procedural/prescriptive block kind — recall returns *a strategy*, not *a fact* —
  earns a place in the governed store. The corpus today is overwhelmingly
  propositional (decisions, entities, projects, references). Note that `skill_opt/`
  is **not** this: it optimizes skill *definitions* offline; a prescriptive block
  would be a retrievable unit inside the memory itself. Open question, and the
  reason this is EVALUATE rather than a commitment: a strategy is a claim about
  *what worked*, which is exactly the kind of assertion that goes stale silently and
  that the contradiction-detection surface was built to catch for facts. Adding a
  block kind whose staleness is harder to detect than a fact's would trade recall
  breadth for governance strength — the wrong direction for this project.

**The discipline that does not relax.** The external work's pitch is task-agnostic
plug-and-play with better benchmark numbers. Ours is *governed*: HITL-gated
proposals, contradiction detection, audit chain, provenance. Those are different
claims and L1 must not blur them — a utility number is a **retrieval-quality**
statement and carries no governance weight. It must not be written into a block, must
not enter a hash chain, and must never influence the approval gate. The same rule
already applied to attestation verdicts and trigger verdicts in the entries above: a
measurement artifact stays a measurement artifact.

**Positioning note (not an implementation item).** "Agent memory" now has a
well-funded default meaning attached to retrieval benchmark scores, and that meaning
is being distributed through plugin surfaces for the same runtimes we ship into. The
guard is the same one recorded for adjacent vocabulary collisions this year: state
the distinction rather than argue it. Retrieval quality and auditability are
orthogonal properties, and this project competes on the second.

- **Status:** Proposed 2026-08-06. L1 is the sequenced item and is worth doing on its
  own merits regardless of the external work — it converts an architectural claim into
  a measurable one. L2 is explicitly EVALUATE and should not be built until L1 has a
  number, since a second block kind changes what "utility" is being measured over.

---

### Group M — Silent-failure gates for the retrieval and write paths (prior-art-informed, 2026-08-17)

Prior art: a public tutorial building a three-tier agent memory (semantic /
episodic / procedural) over a vector-indexed key-value store. The *architecture*
is nothing new to this project — the tiering is a textbook Tulving taxonomy and
the governed store already carries those categories, with contradiction
detection, HITL proposals, and an audit chain the external design has no
equivalent of. What is worth taking is narrower and better: a set of **failure
modes that produce no error**, and the cheap empirical checks that expose them.
Every item below is a gate or a write-policy change, not an architecture change,
and none of them pulls in a new dependency.

The organizing observation: this project measures retrieval quality in aggregate
(`governance_bench.py`, `retrieval_trace.py`, `calibration.py`) and therefore
cannot distinguish "the corpus does not contain it" from "the corpus contains it
and retrieval structurally cannot reach it." Those have identical symptoms and
different fixes. M1–M3 separate them.

**Specs.** Four design documents carry the implementable detail for the items
below, written so an agent can pick one up without re-deriving the reasoning:

| Item | Spec |
|---|---|
| M1 | [`docs/design/m1-embedded-field-vocabulary.md`](docs/design/m1-embedded-field-vocabulary.md) |
| M2, M3 | [`docs/design/m2-m3-namespace-retrieval-properties.md`](docs/design/m2-m3-namespace-retrieval-properties.md) |
| M4 | [`docs/design/m4-closed-set-slots.md`](docs/design/m4-closed-set-slots.md) |
| M5 | [`docs/design/m5-enforcement-in-code-audit.md`](docs/design/m5-enforcement-in-code-audit.md) |
| M6 | [`docs/design/m6-negative-results.md`](docs/design/m6-negative-results.md) |
| M7 (blocker) | [`docs/design/eval-set-ground-truth.md`](docs/design/eval-set-ground-truth.md) |

M7 itself stays unspecified on purpose: specifying a harness before its ground
truth exists would fix the wrong shape. What *is* specified is the blocker —
the ground-truth eval set, which gates **both** L1 and M7 and which neither
entry previously defined. That document is the highest-leverage unblocking item
in either group, and it is also the one most likely to be built badly, since a
mislabelled eval set still produces confident numbers.

- [ ] **M1 — Embedded-field / query-vocabulary alignment (highest value).**
  A vector store retrieves on whatever field carries the vector. If the embedded
  text is phrased in the vocabulary of the *answer* while queries arrive phrased
  in the vocabulary of the *problem*, the vectors never meet: nothing errors, no
  score looks anomalous, and recall silently returns wrong-but-plausible blocks
  forever. The external design avoids this by embedding only a designated
  symptom field and carrying diagnosis/resolution in sibling keys that are stored
  and returned but never vectorized.
  **Our exposure is concrete and the opposite shape.**
  `VectorBackend._augment_for_embedding` (`recall_vector.py:312`) prepends
  category, speaker, date, and a 50-char tag slice into the *same* string that
  gets embedded — metadata and content share one vector, with no way to embed one
  field and merely store another. That was a deliberate disambiguation choice
  ("it cost $50" needs an anchor) and it may well be net-positive; the point is
  that **it has never been measured against the failure it can cause**, and the
  tag-slice truncation at 50 chars is an arbitrary boundary inside a vector.
  Deliverable: a probe test that queries a corpus in problem-phrasing against
  blocks written in resolution-phrasing and asserts the score separation does not
  collapse, plus an A/B of augmented vs. raw embedding on our own corpus. Honest
  possible outcome: augmentation wins and this closes as a negative finding with
  a number attached — which is still strictly better than the current state of
  having no number at all.
  **A live consumer instance, worse than the Python one (added 2026-08-17).**
  `512-mind/src/memory.mind` is the reference consumer of this store, and every
  write it makes goes through `format!` into one flat string that is *both* the
  stored record and the embedded text — `store_witness` (`memory.mind:65`) emits
  `"WITNESS system={} time={} hash={} result=COMPLIANT invariants=9/9"`, and
  `recall_witnesses` (`memory.mind:103`) queries it back with
  `"512:witness system={}"`. Two properties make this the pathological case for
  a vector index: the embedded text is near-constant boilerplate differing only
  in an id and a hash, so inter-block cosine variance is close to zero and
  similarity ranking carries almost no signal; and the fields a caller would
  actually discriminate on (`spec_hash`, `timestamp`, failed-invariant names) are
  fused into that same string rather than available as sibling keys. The probe
  set must therefore include a **synthetic-boilerplate corpus**, not only a
  symptom-vs-resolution corpus — the near-duplicate case is what a real caller
  produces, and it is the one where a floor (M3) and a vector leg are both
  useless while BM25 silently does all the work. Note this is an exposure in the
  *store's own API shape*, not a 512-mind bug: nothing in the surface offers an
  embed-vs-store split for a caller to use.

- [ ] **M2 — Namespace-property round-trip test.** Assert *empirically*, per
  namespace, what is reachable by search versus only by direct get, with the
  retrieval scores printed. The external work does this as three probes with
  visible output rather than as a README claim, which is the right instinct: an
  index-configuration regression currently degrades recall silently instead of
  failing loudly. `NamespaceManager` (`namespaces.py:93`) governs read/write ACLs
  but nothing asserts retrieval reachability as a *tested property*. Cheap: a few
  lines per namespace, and it belongs in CI next to the existing quality gate.

- [ ] **M3 — Per-namespace relevance floors.** A single similarity threshold
  across differently-shaped namespaces is wrong in both directions. An unbounded,
  mostly-irrelevant corpus needs a floor to suppress noise; a small bounded
  per-entity record needs *no* floor, because a floor drops the one durable fact
  on a query that never uses its vocabulary. Make the floor a per-namespace
  property with the evidence recorded — the external example justified its floor
  with a two-order-of-magnitude score gap (0.41 correct hit vs. 0.005 noise)
  rather than a vibe, and any floor we set should carry the same kind of
  measurement. Depends on M2 for the measurement surface.

- [ ] **M4 — Enum-keyed upsert slots inside the governed path (best product
  value).** For *bounded* fact spaces, prevent contradiction structurally instead
  of detecting it after the fact. File each fact under a topic slug drawn from a
  **closed set**, so a second statement on the same topic collides by
  construction — an exact key collision, not a fuzzy similarity match that can
  miss. The closed set is the load-bearing part: an open-ended topic string lets
  an extractor file `plan` on Monday and `plan_tier` on Friday, and the two
  contradicting facts never collide at all.
  **This is genuinely orthogonal to what we have.** `contradiction_detector.py`,
  `conflict_resolver.py`, and `compiled_truth_contradictions` are all *detective*
  — they run after two conflicting facts coexist and depend on detection finding
  them. Enum-keyed slots make coexistence impossible for the bounded case, at
  zero detection cost and with no false negatives.
  **Where we must not copy them:** their upsert is a silent overwrite with no
  proposal, no lineage, no rollback. Ours must route the supersession through
  `propose_update` → `approve_apply` so the replacement is a *recorded,
  reversible event*. That combination — their exactness, our audit trail — is
  strictly better than either side alone, and it is the honest reason to build it
  rather than adopt theirs.
  **Residual risk, stated plainly:** the model still chooses the slug. This moves
  the failure from "forgets what it wrote" to "picks the wrong enum member" —
  narrower and *validatable* (a closed enum rejects an invented member), but not
  eliminated. Say so rather than claiming contradiction is solved.
  Note also the escape hatch's cost: free-form facts with no natural identity have
  nothing to collide on and therefore accumulate. That trade is acceptable and
  should be stated; what should **not** be copied is content-hash keying at
  demo width (32 bits is collision-prone as a durable identity scheme).
  **The precedent is internal and one layer up (re-sourced 2026-08-17).**
  `512-mind/src/drift.mind` applies closed-set discipline to *meaning* rather
  than to keys: `no_semantic_drift` enumerates the mutation classes that corrupt
  a contract — `"must not"`→`"should not"` (obligation weakened to suggestion),
  `"fail open"`→`"fail safe"` (default inverted), `"any human"`→`"authorized
  participants"` (scope narrowed) — and asserts against that fixed list. Same
  move as M4: enumerate the space so the violation is *structural* instead of
  detected. 121 enums across that repo's modules make it the house style, not a
  one-off. So M4 should cite `drift.mind` as its precedent; the external tutorial
  contributed the framing and none of the mechanism.

- [ ] **M5 — Enforcement-in-code audit, closing with capability flags
  (rescoped 2026-08-17).** Sweep for every place a governance or privacy property
  in this project rests on an *instruction to a model* rather than on code that
  executes. The principle in one line: a summarizer is *told* not to include
  names, but a prompt is a request — a scrub function is what actually holds.
  This project already argues exactly that for `propose_update` versus "the model
  will remember"; the audit confirms we live by it everywhere else, particularly
  on any path where redaction, scoping, or exclusion is currently prompt-shaped.
  Verified precondition: `scrub`/`redact` vocabulary occurs in essentially one
  CLI file — not the write path, not the distillers, not compaction, not export.
  That is not proof of a leak; it is proof there is **no enforcement layer to
  point at**, which is exactly the condition under which a prompt-shaped property
  survives unnoticed.
  **The close condition is a mechanism, not a document.** The original scope
  ended at "a ranked list with a code-enforced replacement for each", which
  leaves every finding in the same unenforced state it was found in. The
  ecosystem already has the right pattern: `512-mind` ships **fail-closed
  capability flags** — `drift.semantic_mutation_scan_supported() -> u8 { 0 }`
  (`drift.mind:30`) and `key_management.signature_verification_supported() -> u8
  { 0 }` (`key_management.mind:324`), each paired with an undefined `extern` so a
  missing backend fails at link time rather than silently returning a passing
  default. `detect_drift` reads its flag and returns `equivalent: false` when the
  scan is unsupported, with the reasoning written into the source: *"An
  undefined/empty mutation list must NEVER make `equivalent` true — that was the
  forgery-by-absence path this fix closes."*
  That is M5 solved structurally. Each audit finding should close by installing a
  flag that **fails closed while unimplemented**, so an unenforced property is a
  function returning 0 that gates the path — not a doc saying the property is
  aspirational. A caller intending to rely on it must check and refuse. The
  ranked list becomes the work queue for installing flags, not the deliverable.
  Keep the audit and the remediation separate passes regardless: bundling them
  guarantees the sweep stops at the first interesting finding.

- [ ] **M6 — Negative results as a recorded outcome.** Record what was
  *attempted and did not work*, not only what resolved. A record carrying only
  the successful fix teaches nothing about what to skip, and skipping known dead
  ends is most of what accumulated experience actually buys.
  **Correction to this entry's original wording:** it said "field on *episodic
  blocks*", borrowing the prior art's tier name. There is no episodic tier in
  this codebase — verified 2026-08-17, `episodic` appears nowhere in
  `src/mind_mem/`, and "semantic" exists only as a retrieval axis
  (`observation_axis.py:47`), not as a memory tier. The real taxonomies are
  topical and there are **two of them that disagree**: 13 labels in
  `CategoryDistiller.DEFAULT_CATEGORIES` (`category_distiller.py:103`) and 20 in
  `_recall_scoring.py:330`. So M6 is "make a negative outcome representable at
  all" on the kinds where a dead end can occur (`bugs`, `decisions`,
  `workflows`) — not a field on a tier we do not have. The taxonomy split is a
  separate finding, noted in the spec, and M6 must not be the change that
  silently picks a winner between the two lists.
  The precedent is local and stronger than the external one: `autoresearch`'s
  dead-end registry (`dead_ends.md`) is append-only with a deterministic key, a
  closed outcome vocabulary, and a capped reward channel
  (`ar_rsi_alginv.py:234`) — independent evidence the pattern is right rather
  than borrowed novelty. It also has a **truncated junk row**, which is the
  direct argument for validating entries on append. Cross-repo: see the matching
  autoresearch ROADMAP entry.

- [ ] **M7 — Tier-ablation gate (blocked, sequenced last).** Measure what recall
  *loses* when one tier goes dark: a frozen config with one boolean per tier,
  five runs (all-on, minus-semantic, minus-episodic, minus-procedural, and a
  no-memory floor row), each tier's absence producing a specific diagnosable
  failure rather than a single aggregate score. This is the same discipline as
  the one-sided criterion gate used elsewhere in the ecosystem: isolate one
  variable so the delta is attributable. **Blocked on a ground-truth eval set on
  our own corpus, which does not exist yet** — the same blocker as L1, and the
  two should share it. Do not build the harness before the eval set; a scorecard
  over a corpus with no ground truth is a number that means nothing.

**Sequencing.** M1 and M5 first — both are cheap, both close silent-failure
classes, neither needs new infrastructure. M2 next, since M3 depends on its
measurement surface. M4 is the item with real product value and should be scoped
as a spec before any code, because it changes write semantics on a governed
store. M7 waits on the shared L1/M7 eval set.

**The discipline that does not relax.** Everything here is a *retrieval-quality*
or *write-policy* mechanism. None of it carries governance weight: a floor, a
score, an ablation row, or an alignment probe result must never be written into
a block, never enter a hash chain, and never influence the approval gate. Same
rule already recorded for attestation verdicts, trigger verdicts, and Group L's
utility number — a measurement artifact stays a measurement artifact.

**Provenance rail.** Prior-art shape observed in a public tutorial; no code
adopted, nothing named in any public artifact. Their stack (a graph runtime plus
a vector-indexed SQLite store) is explicitly **not** being taken. After the
2026-08-17 inspection pass, the external source contributes **no mechanism to
any item in this group** — M4's closed-set discipline is sourced to
`512-mind/src/drift.mind`, M5's fail-closed capability flags to `drift.mind` and
`key_management.mind`, M6's negative-results registry to `autoresearch`'s
`dead_ends.md`, and M1's worst instance is our own `512-mind/src/memory.mind`.
What the tutorial supplied was framing: the observation that these are
*silent-failure* classes worth gating. Every mechanism below it is internal
precedent, which is the stronger position — the patterns are already running
here and were merely not yet applied to the store. Citation in `mind-internal`.
Cross-repo: `autoresearch` ROADMAP (M6's dead-end-registry symmetry, and the
ablation pattern applied to loop components).

- **Status:** Proposed 2026-08-17. M1/M5 are actionable now. M7 is explicitly
  blocked and must not be started before the eval set exists.

---

### Group N — Chunk boundary policy + chunk-provenance anchoring (prior-art-informed, 2026-08-24)

Prior art: a widely-used open-source document-partitioning library (Apache-2.0,
mature, actively maintained). Its *parsing* layer is redundant here and is
explicitly **not** being taken — this box already runs several document
front-ends, and that project's all-extras install pulls a heavy ML stack plus
its own vendored OCR fork pinned to an exact version that would collide with the
working GPU OCR environment. It also runs **default-on network telemetry** from
inside the document path (a library-load ping plus per-partition-call runtime
events on a daemon thread, opt-*out* via env var, with an explicit opt-in gate
having been introduced upstream and then deliberately reverted). For a store
that ingests private and partner documents, a default-on beacon in the ingest
path is disqualifying on its own. **No dependency is added by anything below.**

What is worth taking is narrower: two ideas from its *chunking* layer, which is
the part this project's own chunker does not fully cover. `smart_chunker.py`
(803 lines, zero external deps) already does semantic-boundary segmentation,
code-block preservation, header context tracking, small-chunk merging, and — the
better half of the external idea — `start_char`/`end_char` offsets into the
source document (`Chunk`, `smart_chunker.py:92`). An offset span is strictly
better than the external project's convention of retaining copies of the source
elements in chunk metadata: it is cheaper, and it is *anchorable*. That is the
seam N2 exploits.

- [ ] **N1 — Soft maximum as a distinct boundary control.** The chunker today has
  a hard ceiling (`max_chunk_size`, default 1500) and a merge floor
  (`min_chunk_size`, default 100), and nothing in between:
  `_merge_segments_into_chunks` (`smart_chunker.py:361`) closes a group *only*
  when the next segment would breach the hard ceiling. The consequence is that a
  strong semantic boundary arriving at 60% of the ceiling is ignored, and the
  chunk runs on until an arbitrary character budget — not the document's
  structure — decides where it ends. The external design separates these into a
  soft threshold ("close here if a boundary presents itself") and a hard one
  ("close here regardless"), which makes structure the primary signal and size
  the backstop rather than the other way round.
  **A second, sharper finding from the same inspection.** `_score_boundary`
  (`smart_chunker.py:289`) computes a boundary strength and **its return value is
  never consulted by the merge loop** — the loop's own comment at line 361 says
  "we might want to force-split the current segment instead" and then does not.
  Boundary scoring is already implemented and currently dead on this path. N1 is
  therefore not "add a feature": it is *wire the existing scorer to a soft
  threshold*, which is the cheapest item in this group and the one with the
  clearest before/after.
  Deliverable: `soft_max_chunk_size` on `SmartChunkerConfig`, consulted against
  `_score_boundary` in the merge loop; default set so current behaviour is
  preserved unless opted into. Gate: a chunk-stability test asserting that
  identical input yields identical spans, since chunk boundaries feed recall and
  a silent boundary shift is a silent recall change.
  **Measured, 2026-08-24 — the failure is reproducible in one line.** A synthetic
  document of six `## Section` headers, each followed by a short paragraph (1296
  chars total, 12 segments), chunks to **exactly one chunk spanning `(0, 1296)` and
  swallowing all six headers**. Six strong structural boundaries were available and
  none was taken, because the document never reached the 1500-char hard ceiling —
  the only condition the merge loop tests. This is the before-picture for N1 and it
  doubles as its acceptance test: after the change, the same input must split on
  those headers. Also confirmed in the same run: **chunking is deterministic**
  (identical spans across repeated runs on identical input), so the stability gate
  above is asserting a property that holds today rather than one that needs to be
  established first.
  **LANDED 2026-08-24.** `soft_max_chunk_size` (default `0` = disabled) +
  `soft_max_boundary_score` (default `0.5`) on `SmartChunkerConfig`; the merge loop
  now consults `_score_boundary` once a group reaches the soft ceiling, closing it
  early only on a boundary at or above that score. `_score_boundary` is no longer
  dead on this path. Default behaviour is byte-for-byte unchanged (opt-in, because
  boundaries feed recall). Validation rejects a negative soft max, a soft max above
  the hard max, and an out-of-range score at both public entry points. The measured
  acceptance case now splits 1 chunk → 3, each boundary landing exactly on a `##`
  header. Gates: 110 chunker tests pass, mypy clean, ruff check + format clean,
  bandit clean.
  **Pre-existing finding surfaced while gating N1 (not caused by it).**
  `max_chunk_size` bounds the chunk **span**, not the final `chunk.text`:
  `overlap_sentences` prepends a trailing sentence *after* the size decision is
  made, so returned text can exceed the "hard" ceiling. Verified against this file
  with N1 stashed — **12 of 13 chunks overshot a 300-char ceiling, by up to 34
  characters**. The N1 tests therefore assert the true invariant (span ≤ ceiling)
  and a monotonicity guard (enabling the soft max may only close chunks *earlier*,
  never later). Worth deciding separately whether the ceiling should be documented
  as span-bounding or enforced on final text; N1 changes neither.

- [ ] **N2 — Chunk-provenance anchoring: `(doc_hash, start_char, end_char)`.**
  The external project's provenance story is *retention* — keep the source
  elements around so a chunk can be traced back. This project can do the
  strictly stronger thing, because it already has the offsets and the ecosystem
  already has an evidence layer: anchor the triple
  `(doc_hash, start_char, end_char)` so a chunk's preimage is not merely
  *retrievable* but *provable*. A recalled chunk then carries a claim about
  exactly which bytes of which document it came from, verifiable after the fact
  against the source. Nothing in the current Python surface does this —
  `doc_hash` and `source_hash` appear nowhere in `src/mind_mem/`, so chunk
  metadata today records where a chunk came from only by convention.
  **Architectural rails, non-negotiable:**
  1. **mic@3 + MAP is the evidence layer; JSON stays the interop boundary.**
     This item must not become a reason to treat mic@3 as a document-interchange
     format. The anchor is a hash triple sealed into the existing evidence
     surface — not a document envelope.
  2. **No new repo, no new cross-repo seam.** This lands inside mind-mem's
     existing evidence surface. The byte boundary remains the only cross-repo
     coupling, per the Ecosystem Architecture Contract.
  3. **The anchor is provenance, not governance.** Consistent with the rule
     already recorded for attestation verdicts, trigger verdicts, Group L's
     utility number, and Group M's floors: a provenance anchor records *where a
     chunk came from*. It must never be read as a quality signal, must never
     influence ranking, and must never gate approval.
  **Fable verdict 2026-08-24: AMENDED.** Approved in shape, re-sequenced, and
  corrected on three points. The review was conducted against this entry and an
  independent re-verification of the code (the reviewer's sandbox restricted reads
  to this repo), so the findings below are confirmed independently rather than
  taken on the proposer's framing.
  1. **Anchor in the EXISTING `EvidenceChain` — never a second chain.** A separate
     provenance chain was rejected: it doubles the verify surface, splits the audit
     story, and violates rail 2 in spirit. But **not one evidence record per chunk**:
     `EvidenceChain` fsyncs every append, verifies every record at load, and hard-caps
     at 1M entries (`_integrity_compromised` beyond it). Per-chunk records would flood
     a chain built for low-volume human-gated decisions and brick verification with
     mechanical data. Correct shape: the triple lives in **block/chunk metadata in the
     store**, and is **sealed** into the chain via the `metadata` dict of the
     governance records that already fire (PROPOSE/APPLY). The v3 evidence preimage
     already hashes `metadata` (`evidence_objects.py:156-167`), so it becomes
     tamper-evident for free — the same precedent as the existing `spec_hash`.
     For bulk attestation, seal one **Merkle root** over a document's chunk anchors as
     a single record (`verify_merkle` already exists on the MCP surface).
  2. **`EvidenceAction` is closed and stays closed.** Seven-member `str` Enum;
     `from_dict` does a strict `EvidenceAction(...)` lookup, so an unknown member is a
     deserialization failure for every existing reader, and `_map_action` collapses
     unknowns to APPLY. No `ANCHOR`/`INGEST` member. Rail 3 settles it: a provenance
     record is not a governance *action*, so it rides in metadata of existing actions.
     This also keeps old chains readable by old code.
  3. **`doc_hash` binds RAW BYTES, not normalized text.** Same rule as mic@3
     (`trace_hash` anchors canonical bytes, never lossy text): hash exactly the bytes
     that entered the pipeline. Normalized-text hashing severs the tie to the on-disk
     artifact and lets normalization drift silently invalidate anchors. **Coherence
     requirement this creates:** `start_char`/`end_char` index into *decoded text* while
     `doc_hash` covers *bytes*, so the triple is only verifiable if the bytes→text
     derivation is pinned — UTF-8 explicitly, and any non-trivial extractor (PDF→text)
     records its id+version in the same slot as (4).
  4. **Record `(chunker_id, chunker_version, config_digest)` — and the proposer's
     stated reason was wrong.** N1 does **not** invalidate existing anchors: an anchor
     is a claim about bytes, and `doc[start:end]` under `doc_hash` verifies identically
     regardless of which chunker produced it. What N1 changes is **re-derivability** —
     without algorithm identity you can no longer re-run the chunker and reproduce the
     same chunk set. That is the real reason to record it, and it upgrades the anchor
     from "this span existed in this doc" to "this span is what chunker X.Y under
     config C deterministically produces" — a property the N1 measurement already
     proved holds. **`config_digest` is mandatory, not optional**, since
     `soft_max_chunk_size`/`min`/`max` all move boundaries and version alone
     under-specifies.
  **Sequencing — N2 is NOT next.** Fable ordered: N3 (done) → **N1** → **wire the
  chunker into the production ingest path** (its own reviewable change) → N2 as
  metadata-sealing per above. Two independent reasons, both confirmed against the
  code: (a) F1 — `smart_chunker` has **zero production importers** (only its two test
  files), and the live ingest path (`block_parser.py`, `ingestion_pipeline.py`) carries
  **no** offset, source-path, or doc-hash fields at all, so anchoring today would be
  evidence about a code path nobody runs; and (b) today's chunker emits one
  `(0, 1296)` chunk swallowing six headers, so sealing spans into an **append-only**
  chain *before* fixing boundary policy would permanently record the known-bad
  boundaries N1 is about to stop producing. **N2 remains gated on the wiring change
  landing, not merely on this approval.**
  **Correction to this entry as first written.** It said the anchor should be "sealed
  into mic@3 + MAP". That named the wrong local artifact: `mic_map.py` is mind-mem's
  mic@2/mic-b codec for **MIND IR dataflow graphs** ("MIC/MAP is for graphs" per its
  own docstring), not a general evidence envelope. A chunk span is not a graph. Rail 1
  is directionally right — evidence layer is evidence, JSON stays interop — but the
  vehicle is `EvidenceChain` metadata sealing, per (1).

- [x] **N3 — Quadratic-accumulation audit of every PDF/document loader (defensive,
  independent of the above). — AUDITED 2026-08-24, no exposure found.** The external project's CVE-2026-33123 fix was a
  quadratic `bytes +=` accumulation reachable from a crafted PDF content-stream
  array: a small hostile input produces unbounded work, so it is a denial-of-service
  class, not a memory-safety one. The bug is not in their parser *design*; it is a
  loop shape that any document loader can have. The value here is entirely
  independent of whether that project is ever touched: audit our own ingest paths
  for accumulate-in-a-loop over attacker-influenced counts, and add a bounded-input
  regression for each. This is the highest-value item in the whole entry per unit
  of effort, because it is a real bug class we may already carry and it costs one
  grep plus a test.
  **Result — closed as a negative finding, with the evidence.** Swept every `+=`
  accumulation in `src/mind_mem/`. The document-ingest path (`block_parser.py`,
  `ingestion_pipeline.py`) is clean: it accumulates via `list.append` followed by
  `"".join(...)` throughout (`block_parser.py:423,490,514,673`), which is linear,
  not quadratic. That is the correct pattern and it was already in place — nothing
  to fix. Three `+=`-in-a-loop string sites exist
  (`query_expansion.py:440`, `transcript_capture.py:85,87`) and are the same *shape*
  as the CVE, but each iterates over API-response content blocks whose count is
  small and not attacker-chosen; they are noted here so a future change that widens
  their input is recognized as changing their risk class, not because they are
  exploitable today.
  The most hostile-input surface in the repo — the untrusted-pickle opcode scanner
  in `model_audit.py:190-240` — is also the best defended: bounds-checked before
  every read, declared lengths sanity-capped at `1 << 30`, and `short_strings`
  held to a ring buffer (`> 4096` → keep last 2048). It anticipates precisely this
  class.
  **Standing rule this establishes:** in any loop over an externally-influenced
  count, accumulate into a list and `join` once. Never `+=` a `str`/`bytes`. The
  cost of the rule is zero and it removes the bug class by construction.

**Provenance rail.** Prior-art shape observed in a public Apache-2.0 project;
**no code adopted, no dependency added, and nothing named in any public
artifact.** The external contribution is two framings — soft-versus-hard chunk
boundaries, and chunk-level provenance retention — plus one transferable bug
class. Every mechanism proposed above is either already implemented here and
merely unwired (`_score_boundary`), already stronger here than in the external
design (offset spans versus element retention), or sourced to this ecosystem's
own evidence layer. Citation in `mind-internal`.

- **Status:** Proposed 2026-08-24. N3 closed (negative finding). N2 reviewed by
  Fable 2026-08-24: **AMENDED + approved in shape, re-sequenced behind N1 and a
  chunker-wiring change.** N1 is the actionable next item and carries no
  architectural risk.

---

### Group O — Cross-chain evidence anchor: mind-mem's leg (identity of a *belief over time*)

> Cross-repo ecosystem milestone. The composite design lives in the `mind`
> roadmap (Phase 19); this section records only mind-mem's member contribution.
>
> **Status:** gap identified 2026-08-26 — not scoped, not scheduled.

Six layers of the ecosystem each anchor a different "what survives
transformation": `trace_hash` (artifact, `mind`), routing lineage (decision,
Naestro), **the provenance chain (belief over time, here)**, I1–I15 + `spec_hash`
(constraint, 512-mind), the governed route table (intent→capability, mind-nerve),
and the session evidence log (structural health, arch-mind). Six roots, zero
cross-links — nothing can prove the conjunction *"this binary, produced by this
decision, under these constraints, **consistent with these beliefs**, routed by
this rule, at this structural health."*

**mind-mem's member is the belief-state digest** — which governed blocks were in
force, at which applied-edit generation, when the artifact was produced. This is
the member that answers *what did the system believe when it did that*, and no
other layer can supply it.

- **O1 — Canonical belief-state preimage.** A stable digest over the applied-edit
  chain at a point in time. The `v4/block_versioning.py` work (`block_history`,
  `content_as_of`, `recall(..., as_of=date)`) already gives the point-in-time
  projection this needs; the open work is a canonical *serialization* of that
  projection, not new versioning machinery.
  - No clock, no randomness, no dict-iteration order — the 512-mind
    evidence-preimage discipline applies verbatim and is the standard the other
    five members should be held to.
  - Digest scope is the open design question: the whole workspace is stable but
    coarse (any unrelated write moves it); the recalled subset is precise but
    means the anchor depends on a query. **Answer this before implementing** —
    it determines whether the member is reproducible at verify time.
- **O2 — Absent encoding.** A compile with no memory context above it must still
  produce a valid anchor with this member explicitly recorded as absent — never
  omitted, never zero-filled.
- **O3 — Re-derivation.** The verifier must recompute the member from the
  provenance chain rather than accept a supplied hash. A member taken on faith
  reduces the whole anchor to a manifest.

**Interaction with Group M (silent-failure gates).** A belief-state member that
silently degrades — recall path returns fewer blocks, digest still validates — is
exactly the failure class Group M exists to catch. The member must fail closed on
a degraded read, not produce a well-formed hash over an incomplete projection.

**Interaction with R88 (Naestro).** The digest must be computed over the store's
canonical record, never over a caller's account of what it recalled. Same rule,
same reason: a governing layer that decides on the governed party's summary is
theatre.

**Firewall (I13, inherited from 512-mind).** The anchor is an evidence artifact,
never a score. Belief coverage confers no authority — a well-anchored artifact is
not thereby more trustworthy, and anchor breadth must never be optimized against
or used to relax a gate. This is also why the composite anchor must stay separate
from the **Calibrated Recall Confidence** sidecar above: confidence is a quality
signal, the anchor is a provenance fact, and merging them would convert evidence
into a metric.

---

### Group P — SCAR: the confidently-wrong ledger (2026-08-26)

> **Status:** proposed 2026-08-26, not scheduled. Ships as a **GitHub-runtime**
> surface — the ledger is written by the agent harness (post-landing audit,
> verifier gates, CI) and read back at recall time, so the store must be
> reachable from a runner, not just from an interactive session.

**The gap.** mind-mem records what is *true* (blocks), what was *decided*
(decisions), and what *conflicts* (`contradiction_detector.py`,
`list_contradictions`). It records nothing about **who asserted something
confidently and was later proven wrong.** Confidence is therefore re-asserted
fresh every turn, with no track record behind it — an agent that has been wrong
about a class of claim four times sounds exactly like one that has never been
wrong at all.

The 2026-08-26 AGI3 session is the motivating case, and it is not hypothetical:
in one session a documented fix was reported before it was applied; the tail
failure was attributed to goal surplus and the attribution was wrong; §16.57
over-generalized from two same-signed data points; and pre-screen timings were
reported from a probe that bypassed the propagator. Every one was caught — but
caught **independently, from scratch, each time.** Nothing accumulated. A ledger
turns four rediscoveries into one named pattern.

**What a scar is.** A record that a *specific asserter* made a *specific claim*
with confidence, and that a *specific refutation* later falsified it:

    {claim, asserter, confidence, refutation, refuted_by, refuted_at, scope}

The load-bearing field is `refuted_by` — the artifact that did the refuting (a
gate run, a verifier verdict, a diff, a solver result). A scar whose refutation
is another assertion is not evidence; it is a second opinion.

- **P1 — `scar` block type + write path.** Additive block type alongside
  `decision`/`task`. Written through the existing HITL-gated `propose_update`
  path, not a side channel — a scar is a governed claim about an agent and must
  be as auditable as any other block. Per the versioning rule this is a **PATCH**
  bump (additive, opt-in, default-off), not a minor.
- **P2 — Recall-time surfacing.** When recall returns blocks in a scope with
  live scars against the asserting agent, surface them alongside. Off by default;
  a `scars=True` recall flag. The value is that the *next* agent working the same
  ground sees the prior confident error without having to rediscover it.
- **P3 — Wire the profiler that already exists.** `llm_noise_profile.py`
  implements per-provider, per-domain EMA reliability with `record_outcome` —
  and **has no production caller** (only `tests/test_llm_noise_profile.py`
  imports it). A refuted scar is precisely a `was_correct=False` observation.
  This is the cheapest item in the group: connect an existing tested scorer to a
  real event source rather than build a new one.
- **P4 — GitHub-runtime emission.** The post-landing audit cycle and the
  verifier/`evidence-qa` gates write a scar whenever a confident claim is
  refuted. This is the leg that makes the ledger accumulate without anyone
  remembering to file one — a ledger that depends on the wrong party
  volunteering the entry stays empty.

**Firewall — a scar is evidence, never a score.** Same rule Group O inherits
from 512-mind (I13). A scar count must never be optimized against, never relax
or tighten a gate automatically, and never be surfaced as an agent ranking. The
moment "fewest scars" becomes a target, the incentive is to assert less
specifically rather than to be right more often — and vague claims are the
failure mode the ledger exists to make visible. Scars inform a human reading the
record; they do not govern routing.

**Second idea from the same source — agreement as a signal to escalate.** Our
consensus surfaces (`9llm`, the multi-perspective fan-out) implicitly treat
convergence as confirmation. But models sharing a training-set blind spot
produce unanimity that is indistinguishable from competence. The `9llm` skill
already seats one adversarial role, which is the right instinct; the sharper
form treats a *narrow spread* as a trigger to escalate rather than to ship.
Cheap to add, and it belongs to the skill layer rather than to mind-mem — noted
here because it arrived with P and should not be lost.

**Provenance rail.** Prior-art shape observed in a public promotional post
(unverified attribution, unverifiable statistics — treated as an idea, not a
source). **No code adopted, no dependency added, nothing named in any public
artifact.** Of the four nodes described, three (worker / challenger / gate) are
already implemented here in stronger form as the status-token dispatch contract,
blind cross-family verifiers, and the byte-identity gates; only the wrongness
ledger is genuinely absent. The "confidence as a spendable balance" framing was
evaluated and **rejected** — it requires calibrated confidences, which LLM
self-reports are not. Citation in `mind-internal`.

---

## Group Q — block type as a queryable field (Postgres JSONB projection)

**Status:** planned. Additive, opt-in, default-off → PATCH bump.

**What is already correct, so nobody re-proposes it.** The Postgres store
already does the JSONB half properly. `blocks.metadata` is
`JSONB NOT NULL DEFAULT '{}'` (`block_store_postgres.py:200`) — every non-primary
field lives there, so a new block type gaining fields costs zero DDL. There is
already a **GIN index** at line 215, and its comment records the trap that makes
GIN indexes dangerous rather than merely useful:

> GIN index over the SAME tsvector expression that `search()` and
> `hybrid_search()` use in their WHERE clauses. Without character-for-character
> match the planner falls back to a sequential scan with per-row tsvector
> recomputation. v3.8.13 had a `to_tsvector('english', content)` index that
> never matched.

An expression index that does not textually match the query expression is
*invisible* — no error, no warning, just the sequential scan you believed you
had eliminated. That lesson is banked here and applies to every index below.

**The actual gap.** `get_block_type()` (`_recall_detection.py:640`) infers block
type by **parsing the ID string prefix** — `D-` → decision, `SIG-` → signal,
`P-` → proposal, eleven entries in a module-level dict. Type is therefore a
parsing convention over a TEXT primary key, not a field. Three consequences:

1. You cannot ask Postgres for "all blocks of type X" — you fetch and filter in
   Python.
2. A mistyped prefix produces a silently mistyped block; there is no constraint
   that can catch it.
3. Adding a type requires a code change to a dict rather than a data write.

This binds directly to **Group P**: a scar ledger's entire value is the query
*"what did this asserter get confidently wrong before?"* — an asserter+type
filter. Under prefix-parsing that is a full scan plus string matching in Python.

- **Q1 — mirror type into `metadata->>'type'` at write time.** Derived from the
  existing prefix table, never re-entered by hand. The ID prefix stays
  authoritative; the metadata field is a *projection* of it, so the two cannot
  disagree by construction. Backfill is a one-pass migration over existing rows.
- **Q2 — `btree` index on the `metadata->>'type'` expression.** Explicitly **not
  GIN**. GIN is for containment and multi-key search; a single extracted scalar
  wants btree. Per the trap above, the index expression must match the query
  expression character-for-character or it will never be used.
- **Q3 — deferred with a threshold, not a vague "later."** Revisit a
  `jsonb_path_ops` GIN over `metadata` only if (a) JSONB **containment** queries
  actually appear in the codebase, and (b) the corpus passes ~100k blocks.

**Measured, so Q3 is a decision rather than an oversight.** Grepped the whole
store for JSONB containment operators (`@>`, `?`, `#>`): **zero uses.** Only
`->>` extraction. You cannot index queries you do not make. And the live corpus
is **2,427 blocks**, not the 500k of the demo that prompted this — at that scale
a sequential scan is sub-millisecond and an added GIN index is measurable only
as write overhead. Both numbers are recorded here specifically so a future
reader does not reopen Q3 on intuition.

**Provenance rail.** Prompted by a public demonstration of JSONB + GIN
containment on 500k rows. The demonstrated technique does **not** apply here for
the two measured reasons above; what it surfaced instead was the unrelated
prefix-parsing gap. No code adopted, no dependency added, nothing named in any
public artifact.

## Group R — edge evidence: `no shared source, no edge` (prior-art-informed, 2026-08-27)

**Status:** planned. Additive validation on a write path + one new field →
PATCH bump, but see R4: the strict gate is a **breaking** tightening and is
sequenced behind a default-off flag.

**What is already correct, so nobody re-proposes it.** The HITL discipline is
sound and does not need revisiting. `propose_edge` is the only user-reachable
typed-edge write path; nothing touches the source-of-truth `edges` table until
an explicit operator `approve_edge` (`knowledge_graph.py:826`). Direct writes
are admin-scoped and stamped `metadata.origin = "direct_admin"`, HITL commits
stamp `"hitl_approved"`, so the two are distinguishable in an audit rather than
byte-identical. Proposal ids are deterministic, so restaging is idempotent.
`source_block_id` is **already mandatory** and validated at three layers — the
MCP tool, `add_edge` (line 630), and `propose_edge` (line 717). Every edge
carries a confidence in `[0,1]`, optional validity interval, and JSON metadata.
None of that is the gap.

**The actual gap: provenance is asserted, never corroborated.** The mandatory
`source_block_id` proves an edge *names* a block. It does not prove the block
exists, and it does not prove the block has anything to do with either endpoint.
Grepped `knowledge_graph.py` for any block lookup (`get_block`, `block_exists`,
`resolve_block`): **zero hits.** The knowledge graph never consults the block
store.

Verified empirically rather than inferred from absent code — a fabricated id
stages *and commits* clean:

```
propose_edge('Company A','depends_on','Company B',
             source_block_id='blk_TOTALLY_FABRICATED_NEVER_EXISTED')
  → staged   : EP-b68fa79f17695903
  → approve  → COMMITTED: company a depends_on company b
```

So the field is a **format** requirement, not an **evidence** requirement. It
constrains the shape of a claim, not its truth. Three consequences:

1. A hallucinated edge is byte-indistinguishable from a filing-derived one at
   read time. Both carry a plausible id; only one is real.
2. `list_contradictions` degrades. Two edges on the same pair with *different*
   evidence is a genuine signal worth adjudicating; two edges with *unverifiable*
   evidence is noise that cannot be adjudicated at all.
3. The reviewer is asked to approve on vibes. HITL is only a real gate if the
   operator is shown something checkable — otherwise approval launders an
   assertion into a fact with an audit trail attached.

This is the same failure class the codebase already names elsewhere: an
expression index that never matches is invisible (Group Q), an evidence field
that is never checked is decorative. **The single load-bearing rule: no shared
source, no edge — and an empty edge list is a valid answer.** A model asked to
find connections will always find connections; removing the pressure to produce
something is half the fix.

- **R1 — resolve `source_block_id` against the block store at propose time.**
  The id must name a block that exists. Cheapest possible check, kills the
  fabricated-id case outright, and is a strictly better error at staging time
  than a dangling reference discovered during a later traversal.
- **R2 — corroboration: the cited block must mention both endpoints.** The
  substantive half. An edge asserts a relation *between two things*; the
  evidence must be about both, or it is evidence for something else. Entity
  resolution already exists (`entities.resolve`), so the check reuses the
  registry rather than string-matching raw names. Where the corpus supports it,
  prefer **two independent blocks** naming the same relation over one — the
  "shared supplier named in both FY25 filings" shape, not "a document exists."
- **R3 — carry the reason, not just the pointer.** Add an `evidence` field
  alongside `source_block_id`: a short human-checkable statement of *what is
  shared*. The pointer answers "where"; the reason answers "why", and an edge
  whose reason cannot be stated is an edge that cannot be reviewed. Surfaced in
  `list_edge_proposals` so the operator approves against a claim rather than an
  id.
- **R4 — sequence the tightening, do not ship it as a surprise.** R1–R3 land
  behind a default-off `strict_edge_evidence` flag; existing edges predate the
  gate and must not retroactively fail. Enable-by-default is a MINOR bump with
  the migration note, after a scan reports how many live edges would fail. Order
  matters: **measure first, then tighten** — a gate that silently invalidates a
  corpus is worse than the gap it closes.

**Why this is ours rather than a feature we are copying.** The derived-vs-asserted
distinction is the same one the whole evidence posture rests on. An edge that a
model asserted requires trusting that model. An edge *derived* from a shared
source recorded in two blocks is recomputable: a third party runs the same
reduce over the same corpus and gets the same edge set. That inherits the
existing verifiability rather than adding a parallel trust surface. It also
composes with the ordering discipline already in place — edge construction is a
reduce over a completed node set, order-independent by construction, so the
"populate all nodes before connecting anything" requirement is satisfied by an
existing barrier rather than a new one.

**Deliberately not pursued.** Wide agent fan-out as a means to more edges: fan-out
is the commoditized half, and pairing a very wide fan-out with a short
orchestrator *forces* the merge to be a compression pass — and compression is
precisely the operation that preserves facts while discarding the relations
between them. The gap is in the merge, not the width. Also out of scope: a graph
query engine, a traversal DSL, or a vector store for edges — `traverse_graph`
and `graph_query` already cover the read side, and the open problem here is
write-side admission, not retrieval.

**Provenance rail.** Prompted by a public write-up on merge specifications for
wide agent fan-out. The structural half (nodes + edges as a named deliverable)
mind-mem already has; what the note surfaced was the unchecked-evidence
admission gap above, which is ours and pre-existing. Idea only — no code
adopted, no dependency added, nothing named in any public artifact.

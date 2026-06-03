# mind-mem v4.0 — Design Rationale

> Status: planning draft.
>
> The **canonical task list** lives at [`../ROADMAP.md`](../ROADMAP.md)
> under `## v4.0.0` (Groups A–H). This doc explains the *why* behind the
> architectural choices.

---

## Vision

mind-mem v4.0 is a network-native, governed-memory substrate for AI
agents to share knowledge across hosts and organizations. The v3.x
substrate stays general-purpose; v4 layers on the connectivity,
knowledge-graph, and compliance primitives that make it usable as
public infrastructure.

Three principles drive the design:

1. **Compounding over re-deriving.** Knowledge is compiled once and
   kept current with structured incremental merges. Naive RAG
   re-derives on every query and fragments relational understanding.
2. **Governing over storing.** Every claim has provenance, every
   contradiction is detected, every change is auditable. A memory
   substrate without governance becomes a pile of poison waiting to
   happen.
3. **Public-default, local-first.** Default config runs on a laptop
   without phoning home. Network connectivity is opt-in, authenticated,
   and audited end-to-end when enabled.

---

## Three concurrent threads

### 1. Cognition — three-tier memory + Cognitive Mind Kernel

The memory store gains tier semantics (hot/warm/cold or
short/long/persistent). Hot is what's relevant *right now*; cold is
what we always know. Decay schedules differ per tier: hot uses
LRU+TTL, warm uses TTL with surprise-weighted re-promotion (high-
surprise reads bump back to hot), cold is indefinite gated by
contradiction-density.

Surprise is computed as semantic distance from the rolling recall
context — a deterministic, retrieval-time signal, NOT a gradient. The
test-time-update governance pattern is what we already ship as
`propose_update → approve_apply`; v4 just routes proposed changes
through tier promotion as a side effect.

The **Cognitive Mind Kernel** exposes retrieval strategies as a
first-class parameter:

```
mind_recall(workspace, query, kernel="surprise_weighted" |
                                     "lineage_first"     |
                                     "recent_first"      |
                                     "contradicts_first" |
                                     "graph_walk")
```

Memory routing becomes composable. Power users compose multi-stage
routes; default callers stay on `recall(query)` with no kernel
parameter.

### 2. Knowledge graph — typed edges + multi-page entities

The v3.11 typed-edge graph (`cites` / `implements` / `refines` /
`contradicts` / `cooccurrence`) is more sophisticated than untyped
bidirectional links: every edge has a kind-aware decay multiplier,
contradictions are detected at write time, and BFS traversal is
bounded with kind-aware penalties.

v4 promotes blocks from flat to multi-page: `kind ∈ {entity, concept,
source, synthesis, image, audio, code, structured}`. An ingest of a
new source extracts entity and concept blocks via the LLM, links them
to the source block via `cites` edges, and merges into existing
entities of the same name via structured proposals.

Two retrieval modes coexist:

- **Chunked top-K** (current default) — fast, low token cost, good
  for "find the doc about X."
- **Long-context union** — returns full entity/concept pages whose
  summaries match. Higher token cost; preserves relational
  understanding across the graph. Good for chat / synthesis.

Caller picks per call. Both modes operate on the same store.

### 3. Network connectivity — public-default authentication + federation

Mind-mem becomes the substrate for agents on different machines,
owned by different parties, to share governed memory. The transport
layer is TLS 1.3 minimum, mTLS for service-to-service, OAuth2/OIDC
for client identity, with W3C DIDs and Verifiable Credentials for
portable agent identity.

Workspaces gain block-level ACLs backed by signed grant chains.
Public/private/mixed visibility is per-workspace; per-block grants
override. Two instances negotiate workspace sync via a signed-chain
federation handshake; three-way merge with proposals when divergent.

Single-binary distribution makes the solo-dev path trivial:
`pip install mind-mem; mm serve` brings up an authenticated public
endpoint. Heavy deployments add sharded Postgres, K8s operator, and
Raft consensus on top — same authentication contract.

Subscriptions / webhooks turn pull-only memory into push-capable
infrastructure: an agent subscribes to "all blocks tagged X" or "any
contradiction in workspace Y" and gets POSTed when matches land.

### Compliance-sensitive opt-in (cross-cutting)

Several primitives — pluggable redaction, time-bounded recall,
controlled vocabularies, provenance fields, structured
evidence/confidence, signed compliance export, tenant KMS,
contraindication edges — generalize concerns common to many regulated
deployments. They ship as opt-in capabilities; domain-specific bits
(redaction packs, vocabulary sets, export policies) ship as separate
optional packages so general-purpose users pay nothing in
dependencies, install size, or attack surface.

The general-purpose memory store stays primary. Compliance is a
concentric ring around it, not a pivot toward any one vertical.

---

## API additions (proposed)

```python
# block kinds
propose_update(workspace, kind="entity", statement="...", ...)

# anti-fragmentation long-context recall
recall(workspace, query, mode="long_context", include_cold=False)

# time-travel / time-bounded recall
recall(workspace, query, since="2026-01-01", until="2026-12-31")
block_history(workspace, block_id) -> list[block_revision]

# LLM-driven knowledge fusion
propose_fuse(workspace, source_block_id) -> list[proposal_id]

# tiered memory
list_blocks(workspace, tier="hot"|"warm"|"cold", limit=N)
promote_block(workspace, block_id, reason="manual"|"surprise")

# lint
lint(workspace) -> list[finding]
lint_autofix(workspace, finding_id) -> proposal_id

# chat layer
chat_with_memory(workspace, question, model="mind-mem:4b") -> stream

# Cognitive Mind Kernel
mind_recall(workspace, query, kernel="surprise_weighted")

# subscriptions
subscribe(workspace, filter, callback_url) -> subscription_id

# block-level ACL
grant(block_id, agent_id, perm="read"|"write"|"audit") -> grant_id

# viewer
mm view --port=8765 --read-only

# compliance
mm export --policy <policy_name> --since <date>
```

---

## v4.1 — Graph discipline extensions

Once v4.0 lands typed edges + multi-page entities, six orthogonal
extensions sharpen the graph without re-touching the kernel surface.
Each is opt-in, additive, and ships against the v4.0 schema with no
migration cost.

### 1. Edge provenance tagging

Every typed edge (`cites` / `implements` / `refines` / `contradicts` /
`cooccurrence`) gains a provenance dimension orthogonal to its kind:

- **EXTRACTED** — derived directly from source text. Reproducible
  from the source byte stream alone; deterministic parser produced it.
- **INFERRED** — produced by an LLM pass over the source. Not present
  verbatim in the source; the model judged the relationship.
- **AMBIGUOUS** — multiple candidate edges survive resolution; the
  edge is recorded with the candidate set rather than collapsed.

Storage cost: one byte per edge (enum field). Operational value: a
recall caller can scope to `EXTRACTED`-only when grounding matters,
or include `INFERRED` when synthesis is acceptable. Contradiction
detection at write time becomes per-provenance: an EXTRACTED edge
contradicting an INFERRED one is far stronger signal than two
INFERRED edges disagreeing.

API surface:

```python
recall(workspace, query, edge_provenance="extracted")
recall(workspace, query, edge_provenance=["extracted", "inferred"])
list_edges(block_id, provenance="ambiguous")  # diagnostic
```

### 2. Community detection + hub-node surfacing

Periodic (opt-in) community detection over the typed-edge graph
surfaces topic clusters and high-degree hub nodes — the blocks
everything else cites or depends on. Output is a diagnostic, not a
retrieval primitive:

- **Communities**: a partition of blocks into clusters where intra-
  community edges dominate inter-community edges. Surfaces "what
  this workspace is *about*" at a coarser grain than per-block
  recall.
- **Hub nodes**: top-N blocks by weighted degree centrality across
  typed edges. Often the canonical sources, definitions, or
  anchor entities everything else references.

Surfaced through `index_stats` extension and an `mm graph` CLI verb:

```python
graph_communities(workspace, resolution=1.0) -> list[Community]
graph_hubs(workspace, top_n=20, edge_kinds=["cites"]) -> list[Hub]
```

Cost discipline: detection runs on demand, not on a schedule (per
anti-pattern §5). Large workspaces use a bounded sample of the
edge graph; full-graph runs gated on explicit confirm above a size
threshold.

### 3. Hash-anchored incremental ingestion

Every source ingested via `inbox/`, `ingest_file`, `convert_to_markdown`,
or a future web-extract path records a SHA256 of the source bytes as
part of block provenance. On re-ingest:

- **Hash match** → skip entirely. No work, no LLM call, no embedding
  refresh. Idempotent ingest is the default behavior.
- **Hash diff** → re-process only the changed file. Other files in
  the same ingestion batch that match prior hashes are skipped.
- **Hash missing** (no prior ingest) → full ingest, hash recorded.

Operational impact: a 500-file repo re-ingested after a 3-file edit
processes 3 files, not 500. Combined with anti-pattern §6 (bulk
re-ingest cost-gating), turns large repeated ingest jobs from
expensive to free.

Schema addition: `block.source_sha256: bytes(32)` on `kind=source`
blocks. Existing v4.0 CAS already hashes block bodies; this extends
the same discipline upstream to source files. No new dependencies.

CLI surface:

```bash
mm ingest --inbox ./repo                    # full ingest, records hashes
mm ingest --inbox ./repo                    # second run: hash-skip, near-zero cost
mm ingest --inbox ./repo --force-rebuild    # ignore cache, re-process all
```

### 4. Standards-vocabulary provenance export

Provenance lives natively in mind-mem v4: source SHA256, typed
edges, evidence-chain entries, ingestion activities, model and
operator attribution. External auditors and federated systems
already speak a standard provenance vocabulary (W3C). Today
they cannot read mind-mem provenance without learning the
internal schema. A one-way export adapter at the boundary
closes that gap without changing the internal format.

Native storage stays in the canonical STARGA interchange
format used across the rest of the codebase (MIC@2 for text,
MIC-B for binary). No JSON or other foreign formats are
introduced into the codebase. The adapter exists only at the
export surface; nothing reads PROV-O back in.

Export surface (Turtle is the W3C-standard serialization for
RDF; PROV-O is defined as RDF):

```bash
mm export-provenance <block-id> --format turtle > lineage.ttl
mm export-provenance <block-id> --depth 3 --format turtle
mm export-provenance --workspace ws --format turtle
```

Mapping (internal → standard vocabulary at export time only):

- block → Entity
- source file (with source_sha256) → Entity, hash carried as
  derivation identifier
- ingestion / propose_update / approve_apply / extract →
  Activity
- LLM model that extracted → Agent (software)
- user / operator who approved → Agent (person)
- typed edges (cites / refines / contradicts / etc.) →
  derivation relations carrying edge-provenance tag from
  section 1 (EXTRACTED / INFERRED / AMBIGUOUS)
- Q16.16 score → quality assessment on the activity
- evidence-chain link → derivation chain element

Roundtrip property: hashes survive export. An auditor can
take exported Turtle, fetch original source bytes named by
the hash, recompute the hash, and confirm byte-identity. The
provenance is independently verifiable, not trust-us-it-
matches.

What this unlocks:

- **Cross-system audit without internal knowledge.** Compliance
  reviewers load the Turtle into any RDF store, run standard
  SPARQL queries, and answer their question. No need to know
  what mind-mem is.
- **Federation with systems that already speak this vocab.**
  Healthcare informatics (FHIR / openEHR), regulated AI
  pipelines, semantic-data partners can consume mind-mem
  evidence chains as first-class citizens of their existing
  audit tooling.
- **Regulatory readiness.** FDA SaMD, EU AI Act Article 12
  logging, HIPAA audit controls, GDPR Article 30 records of
  processing — all consume the standard vocabulary natively.
  Without the adapter, every regulator needs a custom
  integration. With it, they read what their tools already
  parse.

Scope discipline: export-only. No PROV-O import path in v4.1.
If a future partner needs to inject external provenance into
mind-mem, that's a separate adapter built against a concrete
need, not speculation.

Effort estimate: 1–2 weeks. Data already exists; work is the
serializer over the existing v4 graph, the mapping table, CLI
verb + MCP tool, byte-identical roundtrip tests, and docs.

### 5. Contradiction-cluster ranking

The v4.0 surface already detects contradictions at write time and
exposes `list_contradictions(workspace)`. What it doesn't do is
*rank clusters by magnitude*: a workspace with five low-evidence
singleton disagreements and one ten-block contradiction cluster
looks the same to current tooling. Cluster-ranking turns the
contradiction discipline from "flag everything" into "surface the
most load-bearing disagreements first."

A cluster is a connected component over `contradicts` edges. The
ranking is deterministic — no LLM in the loop — so cluster scores
are reproducible across runs and across instances. Signals
combined into the cluster score:

- **Cluster size** — number of blocks participating in the
  disagreement. Larger clusters represent more reasoning surface
  built on contradictory ground.
- **Evidence weight** — sum of `evidence_score` (Q16.16) over
  participating blocks. A cluster of well-evidenced contradictory
  blocks is structurally more urgent than a cluster of weak
  claims.
- **Recency span** — how long the cluster has been open. Older
  unresolved clusters rank higher: they've outlasted their
  evidence-gathering window and the disagreement is now load-
  bearing rather than transient.
- **Citation density** — how many other blocks `cites` into the
  cluster. High-density clusters affect more downstream
  reasoning; resolving them has higher leverage.

Each signal is computed in Q16.16; the combined score is a
weighted sum with weights exposed in `recall.yaml` (defaults
biased toward citation density and evidence weight, since those
correlate with downstream impact rather than just cluster
mechanics).

Surface:

```
list_contradictions(
    workspace,
    rank_by=["size", "evidence", "recency", "citation_density"],
    limit=N,
)
```

Returns ranked clusters with their participating block IDs, top
citation paths into each cluster, and a single Q16.16
`cluster_load_score`. The same ranking surfaces through
`index_stats` as `top_contradiction_clusters: [...]`, so the
diagnostic is visible without needing to call
`list_contradictions` explicitly.

Effort estimate: ~3 days. Cluster computation is a graph
traversal over existing `contradicts` edges; all four ranking
signals are already fields on participating blocks. Work is the
traversal, the score combiner, and surfacing through MCP +
`index_stats`. No schema change.

### 6. Observer-scope on provenance export

Provenance export is a measurement event, not a passive read. Two
exports of the same graph from different observer scopes — different
workspace filters, different `cites`-traversal depth, different
edge-provenance scope (§1: EXTRACTED-only vs INFERRED-inclusive) —
will legitimately differ. Today the Turtle output carries no
metadata about which scope was applied, so two divergent exports
look like contradictions when they are in fact frame-relative views
of the same underlying graph.

Each Turtle export records the observer scope as a metadata header
inside the document, deterministic-seeded so two exports under the
same scope are byte-identical:

```turtle
# observer_scope:
#   workspace: <ws-id>
#   depth: 3
#   edge_provenance: [extracted, inferred]
#   traversal_seed: 0x7c93...
#   exported_at: <ts>
#   exporter_version: mind-mem/4.1
```

Two exports differing only in observer scope are not contradictions.
Two exports under identical observer scopes that differ in content
*are* contradictions and warrant investigation.

Why this matters: nested governance is real. When mind-mem is used
inside an agent that itself audits another mind-mem instance (or
when a partner system queries mind-mem under one policy and an
auditor queries under another), frame-relativity at the export
boundary is the cleanest place to record it. Without scope-tagging,
nested observers produce export pairs that look like the graph
silently mutated between reads.

Effort: ~1 day on top of §4. The observer scope is just the union
of CLI flags already passed to `export-provenance`; serializing
them into the Turtle header as `mindmem:` namespace metadata is the
only new work.

### 7. Tool-routing preselector (intent-classification layer before MCP dispatch)

The MCP surface has crossed 84 tools and is still growing. Today
the calling LLM sees the full tool list in every system prompt and
decides which tool to invoke by reading all of them. Two problems
compound as the surface grows: every additional tool burns prompt
tokens unconditionally, and the LLM's classification accuracy
degrades the more candidates it has to sift through. The naïve fix
("truncate to top-K by some static rule") trades correctness for
context budget. The principled fix is a router model.

**The preselector.** A small intent classifier — under 50M params,
runs on CPU at single-digit-millisecond latency — sits between the
caller and the MCP tool registry. Given the current request, the
router produces:

- **Intent class** — one of a discrete taxonomy (recall, propose,
  search, contradictions, federation, evidence-export, governance,
  introspection, etc.). Taxonomy is registry-defined, extensible.
- **Tool candidates** — top-K MCP tools the router believes apply
  (typical K=3 to 8, configurable per call).
- **Confidence** — per-candidate score the dispatcher can use to
  decide whether to consult the operator before invoking.

The caller sees only the top-K tool descriptions in the system
prompt, not the full 84+. Token cost becomes constant in K, not
linear in the registry size. Tool library growth becomes a
corpus-growth problem (more training data for the router), not a
context-window problem.

**Coordination with the orchestration router.** Same architectural
pattern at a different layer. The orchestration router routes
between LLMs and skill-library entries at the orchestration
substrate; v4.1 §7 routes between MCP tools inside the memory
substrate. The two routers may share training infrastructure but
have distinct taxonomies and dispatch surfaces. An orchestration
session that touches mind-mem fires both routers in sequence: the
orchestration router picks the LLM and the relevant skills,
v4.1 §7 picks the MCP tools the chosen LLM will see.

**Training the router.** Two data sources, both compatible with the
existing rfn-mind v3.1 deterministic FT discipline:

1. **Operational traces.** Every MCP invocation today is logged
   with the request, the tool chosen, and the eventual outcome.
   That trace becomes labeled training data — "this request shape,
   this tool actually applied." Self-improving by construction.
2. **Synthetic intent corpus.** Generate labeled examples
   programmatically across the taxonomy, train the router
   cold-start, refine on operational traces. Faster bootstrap.

**Governance.** The router never invokes tools directly. It
proposes the top-K candidate set; dispatch goes through the
existing `propose_update → approve_apply` discipline for any tool
that writes state. Read-only tools can be invoked without
governance approval (current behaviour); the router does not
change that boundary. The router is a token-saving optimization,
not a policy enforcer.

**Failure mode.** If the router is unavailable or its confidence
falls below a threshold, the dispatcher falls back to current
behaviour (full tool list, LLM decides). The router is a
performance optimization, not a correctness requirement; the
substrate must work without it.

**Why this matters now.** The 84-tool surface is already at the
edge of what models can reliably classify in a system prompt. The
v4.0 network/federation work adds federation-aware tools, the
v4.1 §1-§6 graph-discipline work adds provenance and contradiction
tools — by the time v4.1 ships, the surface will plausibly be in
the 100-150 tool range. Without a router the system prompt cost
becomes the bottleneck, not the inference itself. The preselector
keeps the surface free to grow.

**Phases of delivery.**

1. **§7-0.** Intent taxonomy registry + tool-ID index. No model
   yet — establishes the candidate set and dispatch contract.
   ~3 days.
2. **§7-1.** Operational-trace ingestion pipeline. Routes existing
   MCP invocation logs into a training-ready corpus. ~1 week.
3. **§7-2.** Synthetic intent corpus generator. Produces labeled
   examples across the taxonomy for cold-start training. ~1 week.
4. **§7-3.** Router model cold-start training. Target: ≤50M params,
   ≤10ms p95 inference on CPU. ~2 weeks.
5. **§7-4.** Dispatcher integration. Calling LLM sees top-K tool
   set instead of full list. Falls back to full list on router
   failure. ~3 days.
6. **§7-5.** Refinement loop. Operational traces feed back into
   training corpus, periodic re-train pinned by content hash so
   router versions are auditable. ~1 week.

**Effort.** ~6 weeks for §7-0 through §7-5, with §7-3 (model
training) as the dominant cost. The training infrastructure is
shared with the orchestration router — same approach, different taxonomy and
target — so the marginal cost on the second router (whichever
ships later) is significantly lower than the first.

**Out of scope for v4.1 §7.**

- Routing inside individual MCP tool execution. The router picks
  the tool; what the tool does is unchanged.
- Replacing the governance gate. `propose_update → approve_apply`
  stays intact; the router only narrows the candidate set.
- Federation across mind-mem instances with different routers.
  Each instance trains its own router; cross-instance router
  federation is a v2 concern.

### 8. Token-level late-interaction reranker (gated on dual-encoder)

Today the hybrid stack is BM25 + dense-vector + RRF fusion over
document-level embeddings. A token-level late-interaction reranker
adds a third re-rank pass that scores
`Σ_i max_j <q_i, d_j>` over per-token query and document embeddings —
preserving fine-grained term-level matches that pooled dense vectors
collapse.

This is additive and orthogonal to the v4.0 schema: it operates on
recall results, not on stored blocks. No migration cost.

Hard dependency: a dual-encoder that emits per-token embeddings.
mind-mem ships pooled embeddings today, so this section is *gated*
behind that encoder landing. We hold the slot open and design the
recall API surface to admit a reranker without breaking callers:

```python
recall(workspace, query, rerank="late-interaction")  # opt-in
```

Cost discipline:

- Reranker runs over the top-K from RRF (default `K=50`), never the
  full corpus.
- Per-token embeddings are computed lazily for the candidate set on
  query, or cached at ingest behind an explicit
  `mind_mem.json:embedding.per_token = true` flag (off by default).
- The contraction kernel itself uses tile-resident outer-reduce on
  GPU substrates where available; CPU path uses the BLAS fallback.

What this is NOT:

- Not a replacement for BM25 + dense + RRF — it is a third pass on
  the fused result.
- Not a graph-discipline extension — it stays inside the
  retrieval kernel and never touches typed edges.
- Not a default. Opt-in per recall call.

### 9. Opt-in bring-your-own-key governed enrichment

The default retrieval path is local-only: own encoder, no external
API in the loop, zero marginal cost per query, nothing leaves the
caller's infrastructure. That is the load-bearing privacy/cost
property and stays the default, untouched.

For callers who want to trade locality for the last increment of
recall quality, v4.1 adds an **opt-in** enrichment branch:

```python
recall(workspace, query, enrich="byok")  # off by default
```

- Uses the caller's own API keys and the caller's own budget — no
  hosted credits, no silent egress. The caller makes the
  cost/privacy tradeoff explicitly; it is never made for them.
- Enrichment may rephrase the query and re-rank candidates. It
  **never** writes to the store outside the propose→review→apply
  gate. A synthesised summary becomes a proposal in
  `intelligence/SIGNALS.md`, not a direct write — consolidation
  stays governed.
- Off unless explicitly enabled per call or per workspace; the
  local-only default benchmark number is what ships in the README.

Federation transport stays a first-class core capability, not a
gated add-on: evidence-chain-preserving cross-instance sync is part
of the v4.0 network pivot and remains available without an
enrichment tier.

#### Federation transport as a reusable primitive (consumers + extraction)

The Group-D federation layer (`v4/federation.py` — per-agent version
vectors, explicit conflict log, three-way-merge resolution that routes
through the v3 governance propose/approve gate; `federation_client.py`
+ `http_transport.py` — stdlib HTTP, bearer-token auth, the
`/federation/{vclock,conflicts,write,resolve}` endpoints) is
deliberately **domain-agnostic**. It moves and reconciles state across
nodes; it has no opinion about *what* it syncs — memory blocks,
ratings, anything.

Naestro's federated trust-rating system is the **second real consumer**
of this transport (the first being mind-mem itself). The boundary is:

- **mind-mem owns the transport** — vclock, conflict log,
  propose/approve resolution, HTTP endpoints. Stays generic; no
  rating/badge/reputation semantics ever move in here.
- **naestro owns the rating *system*** — DRD-derived reputation
  scoring, Bronze/Silver/Gold/Government badge tiers, Ed25519 signing,
  Q16.16 deterministic aggregation, consent/governance, evidence-chained
  collective evolution. It *consumes* this transport; it does not
  absorb into it.

Dependency arrow: **naestro federation → mind-mem v4 transport.**

**Deferred extraction (gated on evidence):** if the naestro↔mind-mem
import boundary stays clean — naestro touches only the transport, never
mind-mem internals — extract this layer into a standalone
`mind-federation` package that both products import. A leaky boundary
means it stays here. Decide on the second consumer, not before; the
arrow above is the experiment that answers it. Federation is a
**write/consistency** layer and is NOT observability — the read-side
trust-score dashboards belong in observability, the transport does not.

What this is NOT:

- Not a hosted paid path. No STARGA-operated API, no per-query
  billing surface. BYOK only.
- Not a relaxation of the governed-write model. Enrichment is
  read-side; any write still routes through `/apply`.
- Not the default. The published recall number is the local-only
  path.

### 10. Deferred candidate resolution for dangling references

When a stored memory references an entity or block that does not yet
exist (`[[unresolved-link]]`), the reference is currently dropped or
left broken. Proposal: persist unresolvable references in a dedicated
governed candidate table with ranked resolution suggestions, resolved
on a later pass when the target entity is created — never auto-linked,
always surfaced through the existing propose → review → apply gate.

Rationale: a broken `[[ref]]` is silent knowledge loss; a held
candidate with suggestions is recoverable and auditable. This extends
the no-direct-writes guarantee to link resolution — deferred, not
dropped, not silently guessed.

Status: v4.1 candidate, design-only. Gated behind §5 contradiction-
cluster ranking (shares the candidate-surfacing UX).

### 11. Tiered progressive context loading (L0/L1/L2)

Recall currently returns full memory blocks. On wide recall sets this
spends token budget on blocks the caller may only need to know
*exist*. A tiered loading model returns progressively richer
representations on demand:

- **L0 — anchor (~100 tokens).** Block id, title, one-line compiled
  summary, relevance score. Enough for the caller to decide whether to
  expand.
- **L1 — overview (~500–2k tokens).** The compiled-truth digest of the
  block plus key evidence-chain references.
- **L2 — full block.** Complete content, expanded only on request.

The mechanism composes over the existing compiled-truth layer: the
working/episodic/semantic/procedural tiers already maintain the
summaries L0/L1 draw from, so tiered recall is a retrieval-surface
change, not a new storage primitive. A recall call returns L0 for the
full result set and L1/L2 only for blocks the caller drills into,
cutting token cost on broad recalls without losing the path to full
fidelity.

Coarse-to-fine retrieval falls out of the same surface: L0 ranking
over the full candidate set selects survivors, and L1/L2 expansion
applies only to those — multi-stage retrieval rather than one-shot
block return. Pairs with the late-interaction reranker (§8).

**Surface sketch.** `recall` gains an optional `tier` parameter
(`l0` | `l1` | `l2`, default `l2` for backward compatibility) and an
optional `expand` list of block ids. A typical broad-recall flow:

1. `recall(query, tier="l0")` → ranked anchors for the full result
   set, ~100 tokens each.
2. caller inspects anchors, picks the ids worth the budget.
3. `recall(query, tier="l2", expand=[ids])` → full blocks only for
   the chosen ids; everything else stays at L0.

A token-budget guard can also cap total expansion: return L0 for all,
L1 for the top-k by score, L2 only for the top-n, so a single call
self-throttles to a target context size without a round trip. The
governed-write path is untouched — tiering is read-side only;
`propose_update → review → approve_apply` and the evidence chain are
unaffected.

Status: v4.1 candidate, design-only. No new storage primitives —
reuses compiled-truth. Read-side only; no governance change.

### 12. Drift-resistant iterative-edit workflows

Industry research on semantic drift in iterative LLM workflows
reports that repeated automated editing of a document degrades its
semantic anchor to the original by 25–50% over time — small
cumulative numeric and clausal errors compounded by larger nuance
loss. The document remains formatting-coherent (looks identical at a
glance) but no longer faithfully represents the original.

This is the failure mode mind-mem's governed-write architecture was
designed against. Every memory block update routes through
`propose_update → review → approve_apply`, with the prior state
content-addressed and retained in the evidence chain. Iterative
edits accumulate as proposals, not as silent mutation. The original
remains cryptographically anchored regardless of how many revision
cycles touch the surface representation.

Three concrete v4.1 surfaces that operationalise this property:

- **Drift surface report.** A read-only API that summarises the
  cumulative semantic distance between a block's current state and
  its earliest committed version, with a per-revision contribution
  attribution. Lets operators see drift accumulating before it
  reaches a damaging threshold.
- **Provenance-preserving export.** When mind-mem blocks are
  exported into downstream iterative workflows (LLM editing pipelines,
  document generators), the export packet includes the content-
  addressed original anchor and the evidence-chain ID, so the
  downstream workflow can re-verify against the original after each
  pass.
- **Stable-original retrieval mode.** A retrieval flag that returns
  the earliest committed version of a memory block alongside the
  current, for workflows that need the unmodified reference (legal
  discovery, scientific reproducibility, regulated audit).

Rationale: in regulated industries (legal, medical, scientific,
financial) the ability to demonstrate a provable original becomes a
hard requirement as automation grows. mind-mem's governed-write
guarantee is the structural mechanism that delivers this; §12
surfaces the operator-facing API that lets buyers in those industries
verify the property holds in their workflow.

Status: v4.1 candidate, design-only. No new storage primitives —
all three surfaces compose over the existing evidence chain and
proposal queue. Coordinates with [[512-mind frame-local invariant
work]] for cross-jurisdiction drift attribution.

### 13. Deterministic weighted graph traversal for related-entity resolution

The entity graph mind-mem builds from `[[ref]]` links between blocks
already needs "nearest related block" queries — ranking resolution
suggestions for dangling references (§10), seeding contradiction
clusters (§5), and surfacing hub nodes (§2). These are weighted
shortest-path / traversal problems over the block-and-edge graph.

The opportunity is not a faster shortest-path algorithm — that
frontier is saturated and owned by specialists, and reaching for a
novel bound would be negative-value effort. The fit for mind-mem is a
traversal layer with properties the standard implementations lack:

- **Deterministic ordering.** Equal-weight paths and tied candidates
  resolve by a stable secondary key (content-addressed block id), so
  the same graph state always yields the same ranked result — across
  runs, machines, and index rebuilds. This is the stable-tie-break
  discipline the recall path already requires; traversal inherits it
  rather than reintroducing run-to-run nondeterminism.
- **Reproducible given graph state.** A traversal result carries the
  graph-state hash it was computed against, so a ranking can be
  replayed and audited — "why was this block suggested" is answered
  from the evidence chain, not re-derived.
- **Governed, not auto-applied.** Traversal produces *ranked
  suggestions* surfaced through propose → review → apply (§10), never
  silent auto-links. Read-side proposes; the gate decides.

Rationale: mind-mem's edge over a generic graph library is the same as
everywhere else in the system — not a better abstract algorithm, but
standard algorithms given determinism and auditability they don't ship
with. A suggested link a reviewer can reproduce and trace beats a
faster one they can't.

Status: v4.1 candidate, design-only. Read-side only; composes over the
existing edge set (§1 provenance tags), candidate queue (§10), and
contradiction ranking (§5). No new storage primitives, no governance
change.

### Why v4.1 not v4.0

All sub-sections are additive, schema-compatible, and orthogonal to the
v4.0 network/federation work. Shipping them in v4.0 would dilute
focus on the core network-native pivot. Holding them for v4.1 lets
v4.0 land cleanly, then turns graph discipline into a separate
release with its own retrain cycle if the LLM needs to learn the
provenance tags during extraction.

---

## Anti-patterns we explicitly avoid

(Tracked in ROADMAP §F. This section explains *why* each is
forbidden.)

1. **Always-on background daemon.** A daemon that runs whether
   anyone asked or not is a CPU bomb waiting to fire. The v4 watcher
   is opt-in, idle-only, resource-capped, and exits cleanly when its
   config flag flips off.

2. **Auto-marketplace reinstall.** A package that puts itself back
   after the user removes it disrespects user intent and is a
   security risk. Removal is permanent.

3. **Multi-process worker fan-out without caps.** Multiple processes
   spinning embedding work in parallel is how a memory system
   accidentally consumes 500% CPU. Single supervised process, bounded
   embedding queue.

4. **Inline embedding during user-facing tool calls.** Blocking a
   tool call on embedding latency makes the call unpredictable.
   Embedding runs on a dedicated worker; tool call returns
   immediately, streams updates as work completes.

5. **Background polling that wakes on schedule.** Polling = wasted
   CPU when there's nothing to do. v4 uses inotify on `inbox/`
   (event-driven) or user-triggered. Never timer-based.

6. **Bulk re-ingest of historical transcripts.** A 200MB session log
   that nobody asked to be indexed should not be indexed. Every
   ingest job gates on a pre-flight cost check; over the threshold,
   requires explicit confirm.

7. **Implicit paid-API calls.** Default to local-first. Explicit
   opt-in to API embedding/extraction backends, with a monthly
   budget cap that refuses runs above the cap.

8. **Trust raw input from unauthenticated public sources.** Federated
   sync requires signed provenance; an instance won't accept blocks
   from peers it can't verify.

9. **Embed raw bytes from unauthenticated sources.** Multi-modal
   blocks store hashes + verified-source URLs only — never raw
   binary fetched without authentication.

10. **Telemetry leakage.** No usage data leaves the host without
    explicit opt-in.

---

## Open questions

These are the design decisions still in flight — should be resolved
before implementation begins on each item.

1. **Block kind migration.** v3.x blocks have no `kind`. Backfill
   strategy: heuristic from `category` field plus user batch reclass
   tool. v4 reads both old (`category`) and new (`kind`) for one
   minor version, then deprecates. **Open:** what's the heuristic?
   Direct mapping or LLM-pass to reclassify?

2. **Long-context recall token cost.** Returning whole entity pages
   can blow context budgets. Cap by `recall(..., max_tokens=N)`? Or
   pre-summarize each page when written so we have a "summary tier"?
   **Open:** which strategy default in mind-mem.

3. **Federation conflict resolution.** Three-way merge with proposals
   is the safe path, but UX is bad if two machines have hundreds of
   conflicting writes. **Open:** batch-approve trivial cases; how
   does the UX surface the human-required ones?

4. **Watcher trust model.** Inbox folder is trusted by default —
   anything dropped in gets ingested. **Open:** for shared inboxes
   (federated/multi-user), require signed ingest tokens? Probably
   yes.

5. **Cognitive Kernel as first-class vs. composable.** Is
   `mind_recall(kernel=...)` the right API, or should kernels be
   composable middleware on top of `recall`? Probably composable —
   single tool with strategy parameter. **Open:** does the API
   surface exposed kernels or hide them as implementation detail?

6. **Default retrieval mode.** Chunked vs. long-context vs. hybrid
   (chunked candidates → long-context over candidate pages). Hybrid
   likely beats both. **Open:** which is default — and is the choice
   per-call or per-workspace?

7. **v4.0 model retrain scope.** ~140 new corpus probes across many
   categories (block kinds, fusion, tier semantics, viewer, lint,
   chat, transport, plus all compliance surfaces). Corpus likely
   grows substantially. **Open:** retrain in one shot or stage by
   sub-tier?

8. **Compliance plug-in distribution.** Per-domain redaction packs
   and export policies live as separate packages. **Open:** who owns
   them — STARGA Apache-2.0 alongside core, or third-party with
   stable plug-in API?

9. **Federation protocol baseline.** Build our own signed-chain
   handshake or layer onto an existing federation standard?
   **Open:** trade off interop vs. governance fidelity.

10. **Multi-modal block storage location.** Embeddings in mind-mem,
    raw bytes external. **Open:** which content-addressable store is
    the default — local filesystem, S3-compatible, IPFS, or
    pluggable (caller picks)?

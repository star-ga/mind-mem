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

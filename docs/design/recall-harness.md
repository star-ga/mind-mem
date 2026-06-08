# Design: the recall harness — deterministic working memory for search agents

Status: Draft (2026-06-08) · Owner: mind-mem

## Motivation

Recent agent-memory research makes a sharp argument: a multi-step **search agent**
should not carry its working state inside a growing transcript. Forcing the model to
re-derive "what have I seen, which evidence is useful, which constraints are still open,
which claims are actually verified" from raw context wastes the policy on *recoverable
bookkeeping* the environment can maintain more reliably — and ablations show removing that
externalized state measurably hurts retrieval recall.

mind-mem is already the reliable environment. This design adds the **recall harness**: a
working-memory *session* the agent externalizes its retrieval state into, so the agent keeps
only the semantic decisions (what to search, what to keep/discard, what to verify, when to
stop) and mind-mem keeps the state.

**Why mind-mem and not a transcript or a scratchpad:** the harness here is **deterministic,
governed, and evidence-chained**. The same query trace reconstructs the same working set
byte-for-byte, and every "verified" mark is backed by the cryptographic evidence chain — not a
model's say-so. That is the wedge: reproducible, tamper-evident working memory, not merely
"reliable" bookkeeping.

## Components (each maps to an existing mind-mem primitive)

| Harness component | mind-mem primitive | Gap to close |
|---|---|---|
| candidate pool | `recall` / `hybrid_search` results | hold per-session, not just return |
| importance-tagged curated set | block score + a session keep-set | add keep/drop + importance tag |
| compact evidence links | block-edge graph (`add_block_edge`, `traverse_graph`) | session-scoped link capture |
| **verification records** | **evidence chain (`verify_chain`/`verify_merkle`)** | mark curated items verified→chain anchor |
| dedup'd observations | `find_similar` + contradiction scan | fold near-dupes into the pool |
| budget-aware rendering | **`pack_recall_budget` (exists)** | render curated+verified set, not raw recall |

## Session object (deterministic)

A `RecallSession` is a pure function of its append-only event log:
`open(query, budget) → [search → curate(keep/drop, importance) → link → verify → dedup]* → render`.
Every event is logged; replaying the log reconstructs the session state identically. No wall
clock or RNG in the state transition (same rule as the rest of mind-mem's determinism surface).
Persisted under the session id; recoverable and auditable.

## Proposed MCP surface (additive, behind the session)

- `recall_session_open(query, token_budget, sampling?) -> session_id`
- `recall_session_search(session_id, query) -> candidates[]`  (extends `recall`/`hybrid_search`)
- `recall_session_curate(session_id, block_id, keep: bool, importance?) -> ok`
- `recall_session_link(session_id, from, to, relation) -> ok`  (wraps `add_block_edge`)
- `recall_session_verify(session_id, block_id) -> evidence_ref`  (anchors via `verify_chain`)
- `recall_session_render(session_id) -> packed_context`  (extends `pack_recall_budget`:
  importance-ranked, verified-first, deduped, under `token_budget`)
- `recall_session_close(session_id) -> summary`  (the curated+verified set as a durable block)

All read paths stay BM25+vector+RRF; the harness only *organizes* recall, never bypasses the
governed store. ACL + budgets from the existing MCP infra apply unchanged.

## Determinism + governance invariants

- Session state = deterministic fold over the append-only event log (replayable byte-identically).
- `verify` marks are backed by the evidence chain — a curated item is "verified" iff its chain
  entry validates; the render step orders verified-first and can refuse to render unverified
  claims when asked.
- No new authority over the SoT: the harness reads/curates; `propose_update` remains the only
  write path into long-term memory (the session `close` summary goes through it, HITL-gated).

## Why this matters competitively

Search agents are converging on "externalize working memory into the environment." Whoever
provides the *best* environment-side memory wins that layer. mind-mem's bid: the only one whose
working memory is **deterministic + tamper-evident + governed**. The recall harness is how an
agent plugs into it.

## Phasing

1. `RecallSession` core + event log + deterministic replay (no MCP yet) + tests.
2. `pack_recall_budget` → `recall_session_render` (importance/verified-first/dedup).
3. MCP surface + ACL/budget wiring.
4. Verification-first rendering backed by the evidence chain.

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

## Model dependency — `mind-mem:4b` must be retrained

The harness changes the **I/O contract** the policy model operates over. Today
`mind-mem:4b` (the Qwen3.5-4B fine-tune; generative — query rewrite / expansion /
generative-retrieval, not a vector encoder) is trained against **flat recall**: it sees
raw results and rewrites/expands queries. The harness moves the policy to *semantic
decisions over a structured working set* — `curate(keep/drop, importance)`, `link`,
`verify`, `render`, `stop` — with the environment (mind-mem) holding the state.

A model trained on flat recall will not drive the structured harness well. This is the
load-bearing point of the source research: **the harness is part of what the policy
learns to use** — the model is not the whole system. So shipping the recall harness
**requires a 4B retrain** on harness-structured traces, sequenced AFTER the harness exists:

1. build the deterministic harness (env-side state) + instrument it to log decision traces;
2. generate a corpus of `(working-set state → good curate/verify/stop decision)` traces —
   mirror the existing 4B fine-tune discipline (production-query-anchored corpus +
   paraphrase robustness + zero-contamination, per the prior 4B GO/NO-GO checkpoint);
3. retrain `mind-mem:4b` as the **policy over the harness** (SFT on the traces; RL only if
   a reliable reward — e.g. curated-recall — is wired). Keep the prior 4B as fallback until
   the harness-trained model wins on the production-query spec.

The retrain is gated on the harness being deterministic first, so the traces (and thus the
training corpus) are themselves reproducible.

## Phasing

1. `RecallSession` core + event log + deterministic replay (no MCP yet) + tests.
2. `pack_recall_budget` → `recall_session_render` (importance/verified-first/dedup).
3. MCP surface + ACL/budget wiring.
4. Verification-first rendering backed by the evidence chain.
5. **Trace instrumentation + `mind-mem:4b` retrain on harness-structured decisions**
   (the model dependency above) — the harness is inert as a *policy* interface until the
   4B is retrained to use it; phases 1–4 stand alone as deterministic infrastructure.

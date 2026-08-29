# Group R — Independent Architecture Audit

**Date:** 2026-08-28
**Scope:** `docs/ROADMAP-RETRIEVAL-ACCOUNTABILITY.md` (commit df9b82c) and the
`ROADMAP.md` Group R summary.
**Method:** independent review against the shipping tree; every finding anchored
to `file:line`. Verdict is adversarial by design — the reviewer was instructed
not to protect the committed doc.

**Status:** FAIL. Group R must not be started as specified. Corrected ordering
is section D; the load-bearing findings (B1, B3, B5) were re-verified by hand
before this file landed.

---

**FAIL**

Group R as committed should not be started as specified. The central diagnosis of R.1 describes a feedback loop that does not close in the shipping tree; R.2 tombstones a non-event; two of the "have it" claims in the gap table are hand-waves; and the determinism note cites a wart that has already been fixed while missing four that remain. The corrected design below is smaller than Group R, reuses more, and lands on a roadmap item (`docs/AGENTIC-MEMORY-SOTA.md` P2) that Group R re-invents without citing.

---

## A. VERDICT TABLE

| Item | Verdict | One-line reason |
|---|---|---|
| **R.1** split returned/used, promote on used | **ADOPT WITH CORRECTION** (major) | "Returned ≠ used" is true in principle, but the promotion path never sees `access_count` at all: recall writes `.mind-mem/block_meta.db` (`_recall_core.py:868`), the ladder reads `intelligence/tiers.db` (`compaction.py:261`), and `block_tier_meta` has no production writer. The ladder is dormant; the fix is wiring + a served-set id, not two new columns. |
| **R.2** tombstone table on `_evict` | **SUPERSEDE** | `_evict` deletes only the `block_tiers` row (`memory_tiers.py:381-391`); the block stays in the corpus and `get_tier` reads it as WORKING again (`:196-204`). Nothing dies there, and it never runs in production. Real deletions already have receipts (`deleted_blocks.jsonl`, `audit_chain`). The death record belongs in the evidence chain, not a side table. |
| **R.3** waste ledger + precision | **ADOPT WITH CORRECTION** | The denominator already exists (`retrieval_log.mem_ids`, `retrieval_graph.py:45-57`) in the same DB as `recall_outcome` (`calibration.py:123-137`). Not blocked on R.2 — waste is `served = 0`, tombstones are irrelevant. Blocked on a durable (unwindowed) serve counter and a joinable run id. |
| **R.4** dashboard | **ADOPT WITH CORRECTION** (defer) | Sequencing rule is right. But there are *three* tier systems (`tiered_memory.py`, `memory_tiers.py`, `v4/tier_memory.py`) and none is fully wired; "dying but recoverable" = `ARCHIVED`, which nothing ever writes (consolidation is dry-run only). Pick one axis first. |
| Rejected: "$11 vs $500K" | **REJECT** (agree) | Marketing. |
| Rejected: "no embeddings" | **REJECT** (agree) | Unmeasured usage is the problem; the vector leg is not. |
| Rejected: WRITER "fact + reason" as *have it* | **ADOPT WITH CORRECTION** | `extractor.py:365-460` emits `{type, content, speaker, date, source_id, confidence}` — no reason. Only `propose_update` for `decision` blocks requires a rationale (`mcp/tools/governance.py:62-67`). Every automatic write path lacks it. |
| Rejected: constraints file = `guardrails.py` | **ADOPT WITH CORRECTION** | Read side is *better* than a constraints file (trigger-fired, ranker bypass). Write side is absent: guardrails are operator-only, `propose_update` cannot mint one (`guardrails.py:38-42`), TRAJ `Lessons:` has no consumer, SessionStart hook is `mm status` (`hook_installer.py:343`). Group P (SCAR) is the write-side design and is unscheduled. |
| Rejected: graph "nodes first, then edges" | **ADOPT WITH CORRECTION** | `EntityRegistry.resolve` mints an entity on first sight (`knowledge_graph.py:271-293`); `add_alias` inserts an alias row and does **not** re-point edges already written under the alias's own id (`:310-323`). Insertion order therefore splits nodes permanently. The fix is a governed alias-merge, not a two-phase ritual. |
| Rejected: Skills triad → skills-hub | **REJECT** (agree) | Correctly out of scope for mind-mem. |

---

## B. CORRECTIONS (with evidence)

**B1. The R.1 loop cannot close — four disconnected paths.**
- Recall increments `access_count` through `BlockMetadataManager(".mind-mem/block_meta.db")` — `_recall_core.py:865-871`, `:1791-1794` → `block_metadata.py:81-96`.
- The MCP tools open a *second* copy at `memory/block_meta.db` — `mcp/tools/recall.py:594`, `mcp/tools/governance.py:892`.
- `TierManager` promotion reads `block_meta JOIN block_tier_meta` (`memory_tiers.py:168-176`) from `intelligence/tiers.db` (`compaction.py:261`). `_read_meta` returns `None` unless **both** rows exist (`:457-462`), so `check_promotion` is always `None`.
- `block_tier_meta` has **no writer anywhere** outside tests: the only inserts in the tree are `INSERT INTO block_tiers` (`memory_tiers.py:406`, `:437`); `tests/test_memory_tiers.py:36-79` seeds both tables by hand. The docstring's "extended by callers (e.g. contradiction detectors)" (`memory_tiers.py:10-13`) describes a caller that does not exist.
- `_all_tracked_ids` reads `block_tiers` (`:503-509`); `_register_block` says "used by tests and ingestion" (`:400`) but has no ingestion caller. Promotion and decay cycles iterate an empty set. `_evict` has never fired in production.
- `tier_recall` boosts read `block_tiers` from `.sqlite_index/index.db` (`tier_recall.py:127`, `:146`) — a *third* DB that the promotion cycle never writes.
- The whole cycle runs only from the `mind-mem-compact` CLI (`pyproject.toml:116`, `compaction.py:351`); no daemon/cron invoker found.

Consequence: R.1 as spec'd (new columns on `block_meta`, gate `_satisfies_policy` on `used_count`) is byte-identical to today with the flag **on** as well as off. The doc's acceptance test ("returned N times with zero outcomes never promotes") already passes — because nothing promotes.

**B2. The ladder already separates access from trust — the doc missed it.** `default_policies()` needs `min_access_count` only for SHARED (3) and LONG_TERM (10); LONG_TERM additionally needs `min_confirmations=2`, VERIFIED needs `min_confirmations=5` + `min_confidence=0.9` (`memory_tiers.py:124-137`). Access can only ever buy the *attention* tiers; trust tiers require confirmations. The defect is that `confirmations` has no writer, not that `access_count` is the wrong counter.

**B3. `_evict` is not death; death already has receipts.** `_evict` → `DELETE FROM block_tiers` (`memory_tiers.py:381-391`); content untouched, still indexed, still recallable, implicitly WORKING. Real removal: `block_store.delete_block` appends to `memory/deleted_blocks.jsonl` (`block_store.py:390-396`, `:754-779`); Postgres has `deleted_blocks` (`block_store_postgres.py:273`, `:1223`); `audit_chain` has a `delete_block` operation (`audit_chain.py:39-52`). Cognitive forgetting only *plans* (`mcp/tools/consolidation.py:30-38` "dry-run"; `should_forget` is "governance-gated at the caller", `cognitive_forget.py:124-141`); `v4/eviction.py:27-30` says "Eviction does not mean delete". There is no apply path for ARCHIVED/FORGOTTEN in the tree — R.2's "dying but recoverable" state is never entered.

**B4. "Returned" data already exists; the join key is what's missing.** `retrieval_log(query_hash, mem_ids, scores, credits, timestamp)` is written after every recall (`_recall_core.py:1817-1824` → `retrieval_graph.py:133-178`), pruned at 30 days every 100th call (`:196-202`). `recall_outcome` is in the same `.mind-mem-index/recall.db`. But `make_query_id` = `cal-<sha12>-<epoch_ms>` (`calibration.py:158-166`, wall-clock) while `retrieval_log.query_hash` = `sha256[:16]` (`retrieval_graph.py:162`) — a prefix-match join that cannot distinguish two runs of the same query. `recall_attestation` is by design never persisted (rail 2, `recall_attestation.py`). The missing primitive is a **persisted, content-derived run id**.

**B5. Determinism note is stale and incomplete.** `_recall_scoring.py:120-140` now has `_utc_now` / `_as_utc` and `date_score(block, *, now=None)` (`:153`) — UTC-normalised, injectable. "Reads local timezone" is false today. The remaining warts the doc should name instead:
- `memory_tiers._read_meta` `:477` reads `datetime.now` non-injectably (age gate) while `run_decay_cycle` `:324` in the same file is injectable.
- `block_metadata.update_importance` `:142` reads `datetime.now`, and the stored result multiplies recall scores at `_recall_core.py:1236-1237`.
- `compaction.py:53,150,180,226` use **naive** `datetime.now()` — this is the real local-timezone wart.
- `retrieval_diagnostics` filters with SQL `datetime('now', ?)` (`retrieval_graph.py:549-551`).

**B6. Gap table overstates "Have it".** WRITER (B-table row 7), RENEWER (dormant, B1), and the constraints-file mapping (row 8) are each half-true. `ROADMAP.md` summary "we already hold three of its four nodes" should read: DECAY yes (planning only, no apply); WRITER partial (HITL decisions only); RENEWER designed but disconnected; GRAVE partial (deletion receipts exist, lifecycle receipts do not).

**B7. Unacknowledged overlap.** `docs/AGENTIC-MEMORY-SOTA.md` P2 "Attestation→Outcome Credit Ledger" is the same problem with the correct answer (exact served-set × outcome join, evidence not actuation). Group R is a weaker restatement. `ROADMAP.md:3684` Group P (SCAR) is the write side of the constraints-file idea. Neither is referenced.

---

## Answers to the six questions

**1. R.1 diagnosis.** Correct in principle, wrong about what is live (B1, B2). Gating promotion on `report_outcome` *does* introduce a bias: only instrumented workflows (CI, harness gates) report outcomes; interactive use never does. With `max_idle_hours` demotion (`memory_tiers.py:127,132,137`) those blocks would also demote — survivorship toward CI-visible knowledge. Cold start: a new workspace has zero outcomes for weeks. Correct signal: two counters with asymmetric power — **served** (from `retrieval_log`) can earn attention tiers only; **credited** (distinct-actor success outcomes, bounded exactly like `outcome_store._OUTCOME_COUNT_CAP`) writes `confirmations` and can earn trust tiers. Fallback when nothing is reported: today's behaviour, capped at SHARED. Absence of credit must never demote — that is the "silence is deletion" trap in a smaller hat.

**2. Rent/decay for a governed store.** Wrong for ours, and the repo already says so: `should_mark` requires low importance **and** staleness, with the docstring "a high-importance block that's rarely read is often still load-bearing (e.g., an ADR)" (`cognitive_forget.py:96-100`); only WORKING has a TTL, "top tiers never auto-delete" (`memory_tiers.py:113-121`); the maturity gate holds anything on a live contradiction (`consolidation_maturity_gate.py:13-18`). The model that beats both extremes: **decay acts on attention, never on existence.** Retention class is a pure function of existing fields — PROTECTED (guardrails, active decisions, contradiction records, scars, any live-contradiction endpoint, `operator` provenance class), GOVERNED (default; mark/archive by plan, forget only via `approve_apply`), EPHEMERAL (`lifecycle: ephemeral`, external-ingest, tool-output; auto-archive allowed, forget still gated). No clock in the classification; decay may move a block between ranking tiers freely and across the retention boundary only by proposal.

**3. GRAVE.** A side table is mutable and duplicates three receipts that exist. The death record should be an **evidence-chain entry**: add `DEMOTE` / `ARCHIVE` / `FORGET` to `EvidenceAction` (`evidence_objects.py:68-77`, currently PROPOSE/APPLY/ROLLBACK/CONTRADICT/DRIFT/RESOLVE/VERIFY) and `archive_block`/`forget_block` to `audit_chain.VALID_OPERATIONS` (`:39-52`), payload `{block_id, content_hash, retention_class, served, credited, implicated, reason, plan_id, proposal_id}`. What theirs structurally cannot do: our grave is hash-chained to the proposal that authorised it and the outcome ledger that justified it; `verify_chain` proves the grave was not edited. R.2's "bounded retention" contradicts tamper-evidence — resolve by keeping chain entries fixed-width and permanent, and letting `deleted_blocks.jsonl` compact under a Merkle root (`merkle_tree.py`, `ledger_anchor.py` exist).

**4. Dismissed too fast.** (a) WRITER: yes — automatic writes carry no "why" (B-table row 7). Deterministic fix: require `Rationale` on all `propose_update` block types, not just decisions, and have `capture.py` fill `Sources:` + the matched pattern as the why. Never an LLM-judged salience (wedge item C). (b) Constraints file: yes on the write side (row 8). The correct form is Group P's SCAR emitted by verifier gates, HITL-approved into a guardrail or `[SCAR]` block, surfaced by the existing trigger match — never an auto-written file. (c) Graph: yes — alias-split nodes are real and order-dependent (row 9); `kg_expand` walks neighbours of resolved query entities (`kg_fusion.py:87-111`), so a split node halves the neighbourhood. Fix = governed `entities.merge(loser→winner)` that re-points `edges.subject/object`, proposed via `propose_edge`'s HITL path.

**5. SOTA move** — section C.

**6. Ordering** — section D. R.1 first is wrong: it patches a path that is not connected.

---

## C. THE SOTA MOVE — Served-Set Ledger (exact credit assignment, governed actuation)

Neither post can do this because neither has a per-run attestation or a governed write path. Every piece below exists except the join key and one counter table.

1. **Run id, content-derived.** At `log_retrieval` (already called after every recall), compute `run_id = sha256(TAG_v1: query_hash ‖ served ids in rank order ‖ pipeline_hash ‖ index anchor)` — the same preimage discipline as `recall_attestation._attestation_preimage` (`:267`), no clock, no randomness. Store it as a new `retrieval_log.run_id` column (idempotent `ALTER TABLE`, pattern at `retrieval_graph.py:107-121`). Return it in the recall envelope beside `query_id` (`mcp/tools/recall.py:352-359`).
2. **Durable serve counts.** `block_serve_counts(block_id PK, served INTEGER, first_run_id, last_run_id)` upserted in the same transaction as the `retrieval_log` insert. This survives the 30-day prune and is the denominator R.3 needs. It replaces R.1's `returned_count` column (which would live in the wrong DB).
3. **Outcome joins a run, not a query text.** `report_outcome(run_id=...)`: `block_ids` defaults to that run's served set; a subset means "these were used". Idempotency and bounded influence are unchanged (`outcome_attribution.py` payload-hash id, `_OUTCOME_COUNT_CAP`, per-actor vote).
4. **Credit is a derived view, never a stored score.** `served(b)`, `credited(b)` (distinct actor × run successes), `implicated(b)` computed on read in `outcome_stats` / `retrieval_diagnostics`. Precision = credited/served per intent type; waste = `served = 0` over the corpus (`index_stats`). This is the honest version of the 71/90 number, measured on our data.
5. **Actuation through the ladder, deterministically, opt-in.** Each distinct-actor success writes `block_tier_meta.confirmations += 1` (bounded); `served` feeds `min_access_count`. Tier boosts stay behind `retrieval.tier_boost` (`tier_recall.py:204-211`). No score reweighting from outcomes — that is wedge-incompatible item A in `AGENTIC-MEMORY-SOTA.md`.
6. **Repair is a proposal.** A failed run → `outcome_proposal` (exists, `outcome_attribution.py`) drafts demote/supersede with the run id attached; operator approves. Lifecycle transitions emit the evidence-chain entries from §3 above.
7. **Retention class** (Q2) gates what any plan may touch; computed from `Status`, `lifecycle`, provenance class, block type, and live-contradiction membership.

Claim that falls out: *the only memory that can prove which blocks it served into which run, whether that run succeeded, and that its forgetting was authorised* — served/credited/buried are all verifiable from the chain, and no node ever carries a mutable credibility number.

---

## D. REVISED ORDERING

0. **R.0 — Wire the ladder (new; blocks everything).** One DB for `block_meta` / `block_tiers` / `block_tier_meta` — recommend `.sqlite_index/index.db` (already has an unused `block_meta` table at `sqlite_index.py:248` and is where `tier_recall` reads). Point `_recall_core.py:868`, `mcp/tools/recall.py:594`, `governance.py:892`, `compaction.py:261` at it; read-merge the old paths for migration. Register blocks into `block_tiers` at index build. Make `_read_meta` take injected `now`. End-to-end test: recall → promotion cycle → boost visible. Additive but touches a path → still PATCH if old paths are read-merged; say so in the changelog.
1. **R.1′ — Served-Set Ledger** (C.1–C.4): run id, serve counts, run-joined outcomes, confirmations writer.
2. **R.3′ — precision + waste** from C.4. Not blocked on R.2.
3. **R.2′ — evidence-chain lifecycle actions.** Blocked on something not in the repo: an *apply* path for consolidation plans (only `plan_consolidation` dry-run exists). Build the governed apply first (plan → proposal → `approve_apply` → transition + chain entry), then the entries are free.
4. **Retention class + alias-merge + Rationale-everywhere** — small, independent, can interleave.
5. **R.4′ — dashboard**, after one tier axis is chosen and data exists.

Blockers named: consolidation apply path (3), tier-axis decision (5), SCAR write side (constraints-file loop lives in Group P, not here).

---

## E. WHAT NOT TO BUILD

- `returned_count` / `used_count` columns on `block_meta` — wrong DB, duplicates `retrieval_log`, and the promotion path cannot see them.
- A tombstone on `_evict` — records a non-event that never fires.
- Promotion gated *only* on outcomes, or demotion for the absence of outcomes — survivorship bias toward instrumented workflows; silence-is-deletion in disguise.
- Any decay that removes existence automatically — `cognitive_forget.py:96-100` already refuses this; keep it.
- A mutable, bounded-retention grave — contradicts tamper-evidence; put deaths in the chain.
- LLM-judged "reason it mattered" at ingest (wedge item C); use a required deterministic `Rationale`.
- Auto-written constraints files without HITL — the guardrail provenance-refusal rule (`guardrails.py:43-50`) exists because a ranker-bypass primitive is an injection vector.
- A two-phase "materialise all nodes first" ingest — the real fix is alias-merge; the ritual just hides the split.
- A fourth tier system, or a dashboard that renders three.
- Obsidian export as a memory feature — `export_memory` exists; fine as a formatter, not roadmap.
- Online outcome-weighted reranking of any kind.

---

**Recap.** I read the committed doc, both ROADMAP entries, and every source file the brief cited plus the callers and DB paths around them. The load-bearing finding is that the tier ladder Group R proposes to fix is not connected to recall on any path (four different SQLite files, no writer for `block_tier_meta`, no ingestion registration, no scheduled invoker), so R.1 as written changes nothing and R.2 tombstones a state transition that never occurs. The corrected design is a served-set run id + durable serve counts + run-joined outcomes feeding the existing `confirmations` gate, with lifecycle deaths recorded in the evidence chain — which is the P2 item already in `docs/AGENTIC-MEMORY-SOTA.md`. Next step for the operator: replace `docs/ROADMAP-RETRIEVAL-ACCOUNTABILITY.md` with the R.0 → R.1′ → R.3′ → R.2′ → R.4′ order above, and file the DB-path split (B1) as a bug independent of any roadmap work; I have not modified any repo files or written to mind-mem — the B1 finding is worth a `propose_update` once you confirm which DB path should win.

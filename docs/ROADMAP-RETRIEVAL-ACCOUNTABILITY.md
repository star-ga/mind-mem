> **⚠ SUPERSEDED IN PART — audit 2026-08-28. Do not implement R.1/R.2 as written.**
>
> An independent audit (`docs/audit/GROUP-R-AUDIT-2026-08-28.md`) returned **FAIL**.
> Three findings were re-verified by hand and are load-bearing:
>
> 1. **The tier ladder is not connected to recall.** `block_tier_meta` is created
>    and INNER-JOINed (`memory_tiers.py:157,173`) but has **no writer anywhere
>    outside `tests/test_memory_tiers.py:79`** — so `_read_meta` always returns
>    `None` and `check_promotion` never fires. Recall writes
>    `.mind-mem/block_meta.db` (`_recall_core.py:868`), the MCP tools write
>    `memory/block_meta.db` (`mcp/tools/recall.py:594`), the ladder reads
>    `intelligence/tiers.db` (`compaction.py:261`), and `tier_recall` reads
>    `.sqlite_index/index.db` (`tier_recall.py:127`). **R.1 as specified would be
>    a no-op with the flag on**, because it patches a path nothing traverses.
> 2. **`_evict` is not death.** It deletes only the `block_tiers` row
>    (`memory_tiers.py:381-391`); the block stays in the corpus and reads back as
>    WORKING. R.2 would tombstone a state transition that never occurs. Real
>    deletions already have receipts (`deleted_blocks.jsonl`, `audit_chain`), so
>    the death record belongs in the **evidence chain**, not a side table.
> 3. **The determinism note below is stale.** `date_score` is already UTC-normalised
>    and takes an injected `now` (`_recall_scoring.py:120-160`). The real naive-clock
>    wart is `compaction.py:53,150,180,226`.
>
> Corrected dependency order is **R.0 → R.1′ → R.3′ → R.2′ → R.4′** (audit §D).
> R.0 (wire the ladder onto one DB) is new and blocks everything else.
> The text below is kept for the diagnosis of *intent*, not as an implementation spec.

# Group R — Retrieval Accountability (memory as rent, not storage)

> **Status:** proposed 2026-08-28. Not started. Every code reference below
> was verified against the tree at the time of writing; re-check before
> implementing.

## Origin

Two external write-ups (X/@0xWast3, Aug 2026) describe an agent-memory
architecture built on four nodes — WRITER / DECAY / RENEWER / GRAVE — and a
companion piece on "memory engineering" around large context windows. The
marketing numbers in both (a \$11/mo layer vs a \$500K vector DB; 90,000
stored facts of which 71,000 were never retrieved) are **unverified thread
framing and must not be cited**. The mechanism is the substantive part, and
a gap analysis against our tree shows we already have three of the four
nodes — in most cases in a stronger form.

What survives the analysis is **one missing subsystem and one real
measurement bug**, plus a metric we cannot currently compute. That is what
this group covers.

## Gap analysis vs. shipping code

| External node | mind-mem today | Verdict |
|---|---|---|
| WRITER — store a fact plus the reason it mattered | `extractor.py` + block schema with provenance/evidence fields | **Have it** |
| DECAY — age memories down; silence deletes | `memory_tiers.py:313 run_decay_cycle` + `cognitive_forget.py` 4-stage mark → merge → archive → forget | **Have it, stronger** — ours is reversible with a grace window; theirs is one-way |
| RENEWER — re-lift only what was actually used | `block_metadata.py:81 record_access`, called from `_recall_core.py:1793` | **Have it, but measuring the wrong thing** — see R.1 |
| GRAVE — hold what died, and why nobody reached for it | nothing; `memory_tiers.py:381 _evict` is a bare `DELETE FROM block_tiers` | **Missing** — see R.2 |

Explicitly **not** adopted: "no embeddings." Our hybrid BM25 + vector + RRF
retrieval is not the problem being described — unmeasured usage is — and
dropping the vector leg would not address it.

From the second article, the Skills / constraints-file / context-graph
triad maps onto surfaces we already own (`~/.agents/skills-hub` +
mind-nerve routing; `guardrails.py` / `guardrail_patterns.py`;
`knowledge_graph.py` / `retrieval_graph.py` / `graph_recall.py`). No new
work is proposed from it. The one transferable framing is that a
constraints file is *memory of corrections* — which is what
`outcome_attribution.py` records and R.1 finally puts to use.

---

### R.1 — Split `returned_count` from `used_count` (correctness fix)

**The defect.** `_recall_core.py:1793` calls
`meta_mgr.record_access(returned_ids, ...)` where `returned_ids` is every
block in the result set. `block_metadata.py:81 record_access` then does
`access_count = access_count + 1` for each. So `access_count` counts
blocks **returned by recall**, not blocks the caller actually used.

`memory_tiers.py:492` gates promotion on
`meta["access_count"] >= policy.min_access_count`, with thresholds of 3
(`memory_tiers.py:125`) and 10 (`:130`). The result is a feedback loop:
retrieval promotes its own output, a block returned 400 times and ignored
400 times is indistinguishable from one that was acted on, and the tier
ladder reinforces the ranker regardless of whether the ranker was right.

**The fix is mostly wiring, not new infrastructure.**
`outcome_attribution.py:311 report_outcome(workspace, block_ids, outcome,
...)` already records `success` / `failure` / `neutral` per block. It just
never reaches the tier-promotion path.

- Add `returned_count` (incremented where `access_count` is today) and
  `used_count` (incremented from `report_outcome` on `success`) to the
  `block_meta` schema, following the existing `_PROVENANCE_COLUMNS`
  `ALTER TABLE` migration pattern in `block_metadata.py`.
- Keep `access_count` as a read-only alias of `returned_count` so nothing
  downstream breaks.
- Promote on `used_count`; keep `returned_count` for the R.3 metrics.
- Behind a config flag, default off. Flag-off tier behaviour must be
  byte-identical to today.

**Acceptance:** a block returned N times with zero reported outcomes never
promotes; a block returned once and reported `success` three times does.
Existing tier tests pass unchanged with the flag off.

---

### R.2 — Tombstone store on eviction (the GRAVE node)

**The defect.** `memory_tiers.py:381 _evict` executes
`DELETE FROM block_tiers WHERE id = ?` and returns. Nothing records that
the block existed, what it claimed, how often it was pulled, or why it was
never reached for. That discards the single most useful dataset the system
produces — the external write-up's 71,000-of-90,000 figure is exactly this
data, and we throw it away on every decay cycle.

**Proposal.** A `block_tombstones` table written inside `_evict` before the
delete, in the same transaction:

```
block_id, content_hash, died_at, tier_at_death,
returned_count, used_count, death_reason, workspace
```

`death_reason` reuses the existing `DemotionReason` vocabulary
(`memory_tiers.py:74`) plus an `EVICTED` member.

This buys two things we cannot do today:

1. **Resurrection evidence** — if a forgotten fact turns out to have been
   needed, there is a record that it existed and why it was dropped.
2. **An auditable death record.** This is where our version should be
   better than the source: their GRAVE is a design choice; ours is an
   evidence requirement, consistent with the rest of the governance story.

Retention is bounded (configurable; tombstones are small and fixed-width,
but they must not grow without limit).

**Acceptance:** every `_evict` produces exactly one tombstone row; a decay
cycle over N evicted blocks yields N tombstones with correct counts;
tombstones survive a reindex.

---

### R.3 — Retrieval-waste ledger + precision metric

Blocked on R.1 and R.2; trivial once both land.

- `used_count / returned_count` per block and per workspace = **retrieval
  precision** — the share of pulled rows that were actually used.
- Count of blocks with `returned_count > 0 AND used_count = 0` = the
  **waste ledger**: rows pulled and never used.
- Corpus-wide share of blocks with `used_count = 0` — the "how much of what
  the agent knows has it never once used" number. We can compute this on
  our own corpus once R.1 has run for a while; **do not quote the external
  71/90 figure**, measure ours.

Surface through the existing `retrieval_diagnostics` and `index_stats`
tools rather than adding new ones.

---

### R.4 — Memory dashboard (visualization)

A reference visualization exists (Kimi "Engineering Desk") showing tier
shelves, a store graph, a write-path pipeline strip, retrieval-precision
and eviction-rate charts, and a context wall. Roughly 70% of its panels are
computable from mind-mem today:

- **Tier shelves** → `MemoryTier` WORKING / SHARED / LONG_TERM / VERIFIED
  (`memory_tiers.py:65`).
- **Store graph** (nodes = blocks, edges = shared keys, ring = time to
  eviction) → `retrieval_graph.py` + `graph_stats` + tier + `last_accessed`.
- **Write-path pipeline** → `pipeline_status`.
- **Cost of a hit** (which tier answered, at what latency) →
  `retrieval_diagnostics`.

The two panels that are **not** computable are retrieval precision and the
waste ledger — the same gap R.1 and R.2 close.

**Sequencing is load-bearing: schema first, visualization second.**
Building the dashboard before R.1/R.2 produces two panels displaying
numbers the schema cannot support, which is the false-green failure mode
this project has been bitten by before.

One panel we can render that the reference cannot: because
`cognitive_forget.py` is a reversible 4-stage pipeline with a grace window,
our graph can show **dying but recoverable** as a distinct state. Design it
in from the start.

---

## Determinism constraint (applies to all of Group R)

`_recall_scoring.py:105` reads wall-clock time and local timezone — a known
determinism wart on the scoring path, pre-existing and tracked separately.
**Do not follow that pattern.** Any new decay, tombstone, or dashboard
logic must take an injected `now`, as `cognitive_forget.py` already does
(`should_mark(..., now: datetime)`, `plan_consolidation(..., now:
Optional[datetime] = None)` — the docstring names it "clock injection point
for deterministic tests").

## Suggested order

1. **R.1** — correctness fix, unblocks everything, smallest diff.
2. **R.2** — independent of R.1, also small.
3. **R.3** — falls out of 1 + 2.
4. **R.4** — only after 1–3 are landed and have produced real data.

## Versioning note

R.1 and R.2 are additive, opt-in, and default-off, so per the project's
versioning rule they are **PATCH** bumps, not minor.

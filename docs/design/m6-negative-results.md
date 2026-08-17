# Design: M6 — negative results as a first-class recorded outcome

Status: Draft (2026-08-17) · Owner: mind-mem · Roadmap: Group M, item M6

## The claim

A memory that records only what worked teaches nothing about what to skip.
Most of what accumulated experience actually buys is **not repeating a known
dead end**, and that information is currently discarded at write time: the
resolution is kept, the four things tried before it are not.

## Correction to the roadmap entry

The Group M roadmap text calls this "negative-results field on **episodic
blocks**." That phrasing was taken from the prior art and is wrong for this
codebase. **There is no episodic tier here.** Verified 2026-08-17:

- `grep -ri "episodic" src/mind_mem/` returns nothing. The word "semantic"
  appears only as a *retrieval axis* (`observation_axis.py:47`,
  `ObservationAxis.SEMANTIC` — vector similarity), never as a memory tier.
- The actual taxonomies are **topical, and there are two of them, and they
  disagree**: `CategoryDistiller.DEFAULT_CATEGORIES` (`category_distiller.py:103`)
  carries 13 labels (`architecture`, `decisions`, `people`, `preferences`,
  `workflows`, `bugs`, `credentials`, `integrations`, `goals`, `constraints`,
  `configuration`, `memory`, `governance`); `_recall_scoring.py:330` carries a
  separate 20-label set (`IDENTITY`, `PREFERENCE`, `EVENT`, `RELATION`, …).
  Neither is a Tulving tier and neither is a superset of the other.

So M6 cannot be "add a field to the episodic tier." It is: **make a negative
outcome representable at all**, on the block kinds where a dead end is a thing
that can happen — in practice `bugs`, `decisions`, and `workflows`.

The taxonomy split above is a real finding and out of scope here. It is noted
so a future item can decide whether the two lists should reconcile; M6 must not
be the change that silently picks a winner between them.

## The local precedent is better than the external one

The prior art records a `tried: list[str]` on each episode. This project's
sibling repo already runs a stronger version of the same mechanism, in
production, with a test: `autoresearch/dead_ends.md`.

Its properties, read from the artifact and `ar_rsi_alginv.py`:

- **Append-only, stated in the file header** — "Never edit existing rows."
- **Deterministic key** — the iteration counter, not a timestamp.
- **Three columns** — `iter | outcome | direction`. The outcome vocabulary is
  closed in practice (`discard`, `crash (metric missing)`,
  `finding (harness-unreachable)`), which is the same closed-set discipline M4
  argues for on the write key.
- **It feeds a reward channel** — `coverage_from_node_stats(...,
  dead_end_delta=...)` (`ar_rsi_alginv.py:234`) grants a coverage bonus capped
  at `COVERAGE_DEAD_END_CAP`. Recording a dead end is worth something, and the
  cap stops it from being farmable.
- **It has a known corruption mode** — one row in the live file is a truncated
  fragment ("` line and `dead_ends.md`/results entry once the MEASURE stage
  reports."). An append-only registry with no row validator eventually
  accumulates junk rows. M6 must not reproduce that.

This is independent evidence the pattern is right rather than borrowed novelty,
and it gives us a schema to match rather than invent. **Nothing is copied from
the external tutorial for this item.**

## What M6 adds

A `negative_results` field on the block metadata: an ordered list of attempts
that did **not** resolve the block's subject.

Each entry carries:

| Field | Meaning | Constraint |
|---|---|---|
| `attempt` | what was tried, one plain statement | required, non-empty after strip |
| `outcome` | why it did not work | required, from a closed set (below) |
| `evidence` | pointer to the run/commit/block that shows it | optional |

The `outcome` vocabulary is closed, mirroring `dead_ends.md`'s de-facto set and
M4's closed-set argument:

- `refuted` — measured and shown not to work
- `regressed` — made a measured thing worse
- `blocked` — could not be completed (missing dep, permission, environment)
- `inconclusive` — ran, produced no signal either way

`inconclusive` is the escape hatch and carries the same honest cost M4's does:
it is the bucket that will absorb sloppy entries. It is included because
forcing a false `refuted` is worse than admitting no signal, and its share of
entries is worth watching as a quality indicator.

## Write policy

**Append-only within a block, matching the registry precedent.** A
`negative_results` entry is never edited and never removed by an ordinary
write; a later attempt appends. The list is part of the block, so it travels
through `propose_update` → `approve_apply` like any other content change and
inherits lineage and rollback for free — no new governance surface.

Explicitly **not** a separate store. A dead end is a property of the thing it
was an attempt at, and splitting it into its own namespace would recreate the
join problem the block model exists to avoid.

**Bounded.** Cap the list per block (proposed: 12). On overflow, the proposal
fails loudly and asks for consolidation rather than silently dropping the
oldest entry. Silent truncation in a record whose entire purpose is "do not
retry this" is the one failure that makes the feature actively harmful. Note
that this is the same class of decision as the external work's cap-at-6 on its
always-injected tier, but for the opposite reason: theirs is a token-budget
cap on a tier injected every turn, ours is a corruption guard on a stored list
that is never auto-injected.

**Row validation on append** — non-empty `attempt`, `outcome` in the closed
set. This is the direct lesson from the truncated row in `dead_ends.md`: an
append-only artifact with no validator accumulates junk it can never remove.

## Retrieval policy, and its interaction with M1

Negative results are stored and returned. **Whether they are embedded is a
question M1 must answer first, and M6 must not pre-empt it.**

The tension is real and worth stating rather than resolving by assertion:

- Embedding them helps a query like *"has anyone tried X"* — arguably the single
  highest-value query this feature enables.
- Embedding them also injects failure vocabulary into the block's vector, which
  is exactly the `_augment_for_embedding` (`recall_vector.py:312`) blending
  problem M1 exists to measure. A block about a working `foo` that lists four
  failed `bar` attempts starts matching `bar` queries.

**Sequencing: M6 ships with `negative_results` stored-but-not-embedded.** That
is the conservative default — it cannot degrade existing recall, because it
changes no existing vector. Embedding them becomes a candidate change *after*
M1 has produced its numbers, evaluated on M1's own probe.

## Deliverable

1. `negative_results` on block metadata; closed-set `outcome`; append-only;
   capped at 12 with a loud failure on overflow; row validation on append.
2. Round-trip test: an entry survives write → read → export → re-import with
   field order and content intact.
3. A test asserting the cap fails loudly rather than truncating.
4. A test asserting the field is **not** embedded (pairs directly with M2's
   namespace-property round-trip test — same "assert the storage property
   empirically" discipline).
5. Cross-repo note in `autoresearch` that the registry is the precedent.

## Non-goals

- **No auto-extraction.** Nothing infers "this was a dead end" from block text.
  Entries are written explicitly. An extractor guessing negative results would
  manufacture false dead ends, and a false dead end is worse than a missing one:
  it stops future work on something that was never actually refuted.
- **No governance weight.** A `negative_results` entry is content. It must never
  enter a hash chain as a distinct evidence kind, never influence the approval
  gate, and never affect a contradiction verdict. Same rule as the rest of
  Group M.
- **No reward channel.** `autoresearch` grants coverage credit for recording a
  dead end because it runs an optimizing loop that needs one. mind-mem has no
  such loop and must not grow a metric that rewards writing more entries.

## Open question for the implementer

Does a negative result on a superseded block survive supersession? If block B
supersedes block A, A's dead ends are still true — the attempts still failed.
Proposed answer: they survive, carried forward on supersession, because the
information is about the *subject*, not about the *claim that was replaced*.
Flagged rather than decided, because it interacts with M4's keyed-upsert
supersession semantics and should be settled once, in one place, for both.

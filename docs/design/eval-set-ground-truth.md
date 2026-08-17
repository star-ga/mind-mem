# Design: the ground-truth eval set — the shared blocker for L1 and M7

Status: Draft (2026-08-17) · Owner: mind-mem · Roadmap: blocks Group L (L1) and Group M (M7)

## Why this document exists

Two roadmap items are blocked on the same missing artifact:

- **L1** (recall utility per context token) needs to know, for a given query,
  which blocks were the right answer — otherwise "utility delivered" has no
  denominator.
- **M7** (tier-ablation gate) needs the same thing — otherwise turning a tier
  off produces a different number with no way to say whether it is worse.

Neither can start without it, and the roadmap says so in both places without
saying what the artifact *is*. This document specifies it. It is deliberately
written before either consumer, because if the eval set is built to fit one of
them it will not serve the other.

**This is the highest-leverage unblocking item in Group L or M.** It is also the
one most likely to be built badly, because a bad eval set still produces
confident numbers.

## The failure this must not become

An eval set assembled by running current recall and labelling what comes back
measures **agreement with today's ranker**, not correctness. Every subsequent
change then scores as a regression against the incumbent, and the ablation gate
becomes a machine for rejecting improvements. This is the single most common way
retrieval evaluation goes wrong and it is silent: the numbers look fine.

Guard: **queries are authored against the corpus, answers are labelled by
reading the corpus — never by reading recall output.** The construction
procedure below enforces this by ordering.

## Shape

A frozen, versioned file: an ordered list of cases.

```
case_id        stable, deterministic, never reused
query          the question, in the phrasing a caller would actually use
relevant       block ids that genuinely answer it, each with a grade
irrelevant     block ids that a plausible ranker would wrongly return (optional)
notes          why these blocks and not others — the labelling rationale
```

Grades are three-valued, not binary:

- `2` — answers the query on its own
- `1` — contributes but is insufficient alone
- `0` — related but does not answer it

Binary relevance would make L1 unmeasurable. "Utility per token" is a question
about *how much* of the answer a token bought; a grade-1 block that costs 80
tokens and supplies a third of the answer is exactly the case the metric exists
to price.

`irrelevant` is the deliberate-distractor column and is what makes M7's
per-tier failure signatures diagnosable: a tier going dark should change *which
distractors surface*, not merely lower a score.

## Construction procedure

The ordering is the method. Doing these steps out of order produces the
incumbent-agreement failure above.

1. **Sample queries from real traffic, not imagination.** Recall calls made
   during actual sessions are the population. Invented queries drift toward
   whatever the system already does well. If traffic logs are unavailable, this
   step is the first thing to build — do not substitute synthetic queries and
   proceed.
2. **Freeze a corpus snapshot** by commit or export hash. A moving corpus makes
   every number incomparable across time. Record the hash in the file header.
3. **Label by reading the corpus**, one query at a time, without running recall.
   Slow and unavoidable. The labeller's job is "which blocks in this corpus
   answer this," not "is this result any good."
4. **Only then run recall**, to harvest distractors for the `irrelevant` column
   — the one step where looking at ranker output is legitimate, because it is
   being used adversarially.
5. **Second-pass review** of any case where labelling was uncertain, by someone
   who did not do the first pass. Cases that survive disagreement get a note
   recording it.

## Size

**Target 60–100 cases. Do not scale up before the first consumer runs.**

Fewer than ~50 and per-category slices are single-digit, so a category-specific
regression is indistinguishable from noise. Beyond ~100 the labelling cost grows
faster than the statistical return, and — more importantly — a large set built
before either consumer has run is a large investment in possibly the wrong
shape. Build 60, run L1 against it, then decide whether more cases or better
cases is the constraint.

Coverage: spread across the corpus's real category distribution, not uniformly.
An eval set uniform across categories measures a corpus we do not have.

## Determinism

The eval set is a **fixture, not a generated artifact**. It is checked in, hand-
maintained, and changes only by explicit edit with a version bump. It carries no
timestamps and no wall-clock or RNG dependence in anything derived from it, so a
scorecard is reproducible from `(corpus_hash, evalset_version)` alone. This is
the same determinism rule the rest of the project runs under and it is what makes
a number from three months ago comparable to one from today.

Append-only is **not** the right discipline here, unlike the dead-end registry in
[`m6-negative-results.md`](m6-negative-results.md): a mislabelled case is a bug
and must be fixable. But a corrected case bumps the version, and any published
number states the version it was measured against.

## Governance boundary

**The eval set carries no governance weight, and neither does anything computed
from it.** No score, grade, curve, or ablation row may be written into a block,
enter a hash chain, or influence the approval gate. The `relevant` column is a
judgement about retrieval quality, not a claim about truth — a block can be
correctly retrieved and factually wrong, and conflating those would corrupt the
governance surface with a retrieval opinion.

Same rule already recorded for attestation verdicts, trigger verdicts, Group L's
utility number, and all of Group M.

## What this unblocks, and what it does not

**Unblocks:** L1 (utility curve against `max_tokens`, with the honest possible
outcome that 2000 is already past the knee), and M7 (five-row ablation scorecard
with per-tier failure signatures).

**Does not unblock M1.** M1's probes are self-contained — they construct their
own problem-phrased/resolution-phrased pairs and measure score separation
directly. M1 remains actionable now and should not wait for this. Stated
explicitly because "we need an eval set first" is an easy way to stall the one
Group M item that needs nothing.

## Close condition

The eval set is done when 60+ cases exist, labelled by the procedure above,
against a hash-pinned corpus snapshot, with second-pass review on the uncertain
ones — and when running current recall against it produces a baseline score that
a reviewer agrees is *plausibly wrong in the places recall is known to be weak*.
An eval set on which the incumbent scores near-perfect has almost certainly been
labelled from recall output, and should be rejected and rebuilt.

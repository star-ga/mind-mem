# 100% across every dimension — what that means per dimension, measured

The target is total dominance: 100% wherever a number can be produced. This
file translates that into per-dimension targets against measured reality, and
separates the dimensions where 100% is a **property you can prove** from those
where it is a **score you approach** — because the second kind has ceilings that
are not ours to move, and knowing which is which is what makes the target
actionable rather than aspirational.

Measured 2026-09-07 on the full 470-question LongMemEval-S eligible set.

## Class A — dimensions where 100% is a provable property

These are gates, not benchmarks. A single counter-example fails them, and no
amount of tuning is involved. This is where mind-mem's architecture already
differs in kind from a vector store, and where "100%" is the correct and
achievable target.

| dimension | 100% means | state |
|---|---|---|
| Provenance | every block carries a source and an evidence-chain entry | chain verifies; `verify_chain` is a gate |
| Deterministic replay | same input, same state, byte-identical result | the floor arm reproduces its committed values to every digit, run to run |
| Ontology consistency | no edge can contradict its predicate's domain/range | **landed 2026-09-07**, off by default; needs the entity-typing rollout to enforce |
| Entity resolution | one canonical id per real-world entity, aliases resolve to it | registry + aliases ship; resolution quality unmeasured |
| Contradiction handling | every contradiction detected and surfaced, none silently overwritten | detector ships; no measured recall figure |
| Governance | no ungoverned write path exists | `add_edge` is the single choke point, enforced by `require_admission` |

The honest gap in Class A is not capability, it is **measurement**: several of
these ship and none has a published coverage number. A 100% claim needs a
falsifiable test per row, not a feature per row.

## Class B — dimensions that are statistical scores, with real ceilings

### `recall_any@5` — 100% is reachable

Current hybrid: **0.9787**. Ten questions of 470 have no gold document in the
top five.

Five of those ten are *ranking* misses (the gold sits at rank 6–10) and five are
*retrieval* misses (absent from the top ten entirely). More usefully: **seven of
the ten are already retrieved at 5 by an arm we already run** — the zero-dep
floor rescues six, `no_expansion` five, BM25F four. Only **three questions** are
missed by every arm.

So the path to 1.0000 is not new retrieval. It is fusion over arms that already
exist, worth ~0.9936 on its own, and then three genuinely hard questions.

### `recall_all@5` — 100% is IMPOSSIBLE, and the ceiling is 0.99362

`n_gold` across the 470: `{1: 170, 2: 229, 3: 39, 4: 18, 5: 11, 6: 3}`.

**Three questions have six gold documents. `k=5` has five slots.** No retrieval
system can place six documents in five positions, so `recall_all@5` cannot
exceed **467/470 = 0.99362** for anyone, ever, on this benchmark.

Two of those three already score `all@10 = 1` — we *do* retrieve all six. The
metric's `k` cannot express it. That is a limit of the ruler, not the system.

**Therefore the 100% target for completeness belongs on `recall_all@10`**, which
measures the same property without an artificial slot limit. Current hybrid
`all@5` is 0.8601 against a 0.99362 ceiling, so there is 0.1335 of genuine
headroom before the ceiling is even the binding constraint.

Eleven further questions have exactly five gold documents, so they demand a
*perfect* top five — every slot correct, nothing else admitted. Those are real
and winnable, and they are where the remaining `all@5` work actually lives.

### MRR — 100% is reachable in principle and brutal in practice

MRR 1.0000 requires the first gold document at rank 1 on all 470 questions.
Current hybrid: **0.9014**, against a zero-dependency BM25 floor of 0.9081 —
i.e. we are not yet ahead of a forty-line baseline on ordering, which is the
sharpest single statement about where the retrieval work remains.

### Judge-scored benchmarks — 100% is not a meaningful target

LoCoMo and the LongMemEval answer protocols score through an LLM judge. A judge
is stochastic and some gold labels are contested; one project withdrew a 100%
LoCoMo claim itself, noting the last fraction of a point came from inspecting
wrong answers. Chasing 100% there optimises the judge, not the memory.

The correct target on those benchmarks is to **beat every published figure under
a stated protocol with a committed artifact** — which no competitor currently
does, and which is a harder and more defensible claim than a number.

## The corrected metrics, and the finite list standing between us and 1.0000

`all@5` was the wrong ruler — three questions carry six gold documents against
five slots. Measured on the metric that *can* represent the capability:

| arm | recall_any@10 | recall_all@10 |
|---|---|---|
| floor (`bm25_baseline`) | 0.9809 | 0.9021 |
| BM25F head | 0.9872 | 0.9213 |
| **hybrid** | **0.9894** | **0.9489** |

The lead over the floor is *larger* here than at k=5 (+0.047 against +0.036) —
fixing the ruler did not lower the bar, it revealed more of the capability.
`max(n_gold)` is 6, so any k ≥ 6 can express full completeness and **1.0000 is a
legitimate target on both**.

What stands between here and 1.0000 is now a finite, nameable list:

* **`any@10`: 5 questions.**
* **`all@10`: 24 questions**, of which another existing arm completes 4 — so
  **20 need capability we do not currently have.**

### Those 24 are a graph problem, not a retrieval problem

| property | value |
|---|---|
| retrieved a full 10 results | 23 of 24 — the slots were never the limit |
| `n_gold = 2`, found exactly 1 | **11 of 24** |
| found ≥ half the gold | 18 of 24 |
| found none of it | 5 of 24 |
| mean fraction of gold retrieved | 0.481 |
| dominant types | multi-session (11), temporal-reasoning (10) |

The dominant failure is **"found one piece of evidence and missed its
sibling"**, concentrated on the two question types where evidence is
*relational*.

No better BM25 and no better embedder retrieves that second document, because
the reason it belongs is not that it resembles the query — it is that it is
**linked** to a document already retrieved: same entity, adjacent in time, same
session thread. A lexical arm scores it low because it shares few query terms; a
dense arm scores it low because it is semantically about something adjacent.

So the last 5% of completeness is exactly the layer that distinguishes this
architecture from a vector store. **Sibling-expansion over the knowledge graph —
retrieve, then walk one hop along entity and temporal edges and re-admit — is
the mechanism, and these 20 questions are its acceptance test.** That is a
measurable target with a named question list, not a research direction.

## What this changes about the plan

Nothing about the ambition. It changes where the effort goes:

1. **Class A is the moat and it is under-measured.** Six dimensions where 100%
   is provable, and not one has a published coverage number. A falsifiable test
   per row is worth more than any retrieval point, because no competitor in the
   surveyed field can produce those numbers at all.
2. **`any@5` to 1.0000 is a fusion problem**, not a retrieval problem — seven of
   ten remaining misses are already solved by an arm we run.
3. **Retire `all@5` as a 100% target** in favour of `all@10`. The k=5 ceiling is
   0.99362 and three questions make it unreachable by construction.
4. **Ordering is the honest weak point.** MRR 0.9014 against a floor of 0.9081.
   Everything else is ahead of the floor; this is not.

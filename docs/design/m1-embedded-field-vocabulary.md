# Design: M1 — the embedded field must speak the query's vocabulary

Status: Draft (2026-08-17) · Owner: mind-mem · Roadmap: Group M / M1

## The failure this closes

A vector store can only match a query against whatever text was actually
embedded. If the embedded text is written in the vocabulary of the *answer*
while queries arrive in the vocabulary of the *problem*, the vectors never
meet. Nothing errors. No score is suspiciously low, because the low score
*is* the retrieval — you simply get the wrong blocks, forever, and the
output is indistinguishable from "the corpus does not contain it."

That indistinguishability is the whole reason this needs a gate rather than
a code review. We already carry one live instance of the symptom shape: the
stale recall corpus (pg ingest halted 2026-07-13) presents identically —
empty or irrelevant recall with no diagnostic distinguishing "absent" from
"unreachable". A silent-failure class with no probe is a class we cannot
claim to have ruled out.

## What the code does today

`VectorBackend._augment_for_embedding` (`src/mind_mem/recall_vector.py:312`)
builds the embedded string as:

```
[Category] [Speaker] [Date] [Tags[:50]] <searchable text>
```

Two facts follow, and both are verified against the source rather than
assumed:

1. **There is no embed/store split.** No mechanism exists anywhere in the
   package to embed one field while merely storing another — the whole
   augmented string is what gets vectorised, and the same string is what a
   query is compared against. The external prior art keeps diagnosis and
   resolution in sibling keys precisely so only the symptom carries a
   vector; we do the structural opposite.
2. **The metadata prefix is unconditional.** Category, speaker, date, and a
   50-character tag slice ride inside every vector. The docstring's
   justification is real and probably net-positive — a chunk saying "it cost
   $50" is ambiguous without an anchor. But `tags[:50]` is an arbitrary
   boundary sitting *inside a vector*: a block whose 51st tag character
   would have mattered is silently truncated, and a block with long tags has
   proportionally less of its actual content driving the match.

Neither of these is asserted here to be wrong. The claim is narrower and
harder to argue with: **the augmentation has never been measured against the
failure it is capable of causing.**

## The gate

A probe test, not a refactor. Three parts, each of which must produce a
number rather than a pass/fail feeling.

### 1. Vocabulary-divergence probe

Construct a fixture corpus where the embedded text is written in
resolution language and the queries in problem language — the canonical
shape being a block whose content describes *why a threshold was raised*
queried by *the symptom the raised threshold caused*. Assert the score gap
between the correct block and a known-irrelevant block.

The pass condition is a **separation ratio**, not an absolute score. On a
healthy pairing, the correct hit should stand at least an order of magnitude
above the noise floor. Prior art reports 0.4100 against 0.0054 on a
two-item corpus — roughly two orders — which is what a *working* pairing
looks like and gives the shape of the expected result, not a target to
tune toward. If our separation collapses toward 1.0 on the divergence
fixture, the vocabulary mismatch is demonstrated.

### 2. Prefix-contribution measurement

Embed the same corpus twice, once with `_augment_for_embedding` and once
with the raw text, and report recall@k for both across the fixture query
set. This answers a question we currently cannot answer at all: does the
metadata prefix help, hurt, or wash out? Three outcomes, all publishable
internally:

- **Helps** — the prefix earns its place, and the truncation width becomes
  a tuning question with evidence behind it.
- **Washes** — the prefix is a token cost with no retrieval return; drop it
  or make it opt-in per namespace.
- **Hurts** — the prefix is diluting content signal, and M1 has found a
  real regression that predates the gate.

A negative finding closes this item exactly as validly as a positive one.
The point is a measurement, not a predetermined fix.

### 3. Tag-truncation boundary probe

Two blocks identical except that one's distinguishing tag falls after
character 50. Assert whether they are separable by query. If they are not,
the truncation is load-bearing in a way nobody chose deliberately.

## The rule this establishes

Written down so it survives the specific fix:

> **The embedded field must be in the vocabulary of the query, not the
> vocabulary of the answer.** Diagnosis, resolution, and provenance ride in
> sibling keys that are stored and returned but not vectorised.

Adopting the rule fully requires an embed/store split that does not exist
today. That split is **out of scope for M1** — M1 establishes whether we
need it, with numbers. Proposing the split before the measurement would be
the same mistake in the other direction.

## Non-goals

- No change to the embedding model, dimensionality, or backend.
- No change to `_augment_for_embedding` inside this item. M1 delivers the
  probe and the measurement; any change to the augmentation is a follow-on
  justified by what the probe reports.
- No governance weight. A separation ratio, a recall@k delta, and a
  truncation verdict are measurement artifacts: they must never be written
  into a block, never enter a hash chain, and never influence the approval
  gate.

## Done when

- The three probes exist as tests and run without an API key (they exercise
  the storage and retrieval path, not an extractor).
- Each reports a number, and the numbers are recorded with the commit.
- The vocabulary rule is stated in the retrieval documentation.
- If the measurement is negative — the prefix is fine, the vocabularies do
  meet — that is written down explicitly and the item closes. A ruled-out
  failure class is the deliverable.

## Provenance rail

Prior-art shape observed in a public tutorial; no code adopted, nothing
named in any public artifact. The separation figures above are cited as the
*shape of a working pairing* observed externally, not as our measurement and
not as a target. Citation in `mind-internal`.

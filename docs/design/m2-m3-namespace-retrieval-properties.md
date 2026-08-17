# Design: M2/M3 — namespace retrieval properties, asserted and tuned

Status: Draft (2026-08-17) · Owner: mind-mem · Roadmap: Group M / M2, M3

Two items in one document because M3 cannot be evaluated without M2's
measurement surface, and specifying them apart invites building the floor
policy before there is any way to tell whether a floor helps.

---

## M2 — assert each namespace's reachability empirically

### The failure this closes

Namespaces differ in how they are meant to be reached. Some are searched.
Some are meant to be fetched directly and never surfaced by search at all —
an always-injected namespace that leaks into search results pollutes every
query with content that was never meant to compete for a slot.

Today that distinction lives in configuration and in intent. Nothing
asserts it. An index-configuration regression would not fail a test; it
would quietly change what recall returns, and the change would look like
ordinary ranking drift.

The prior art demonstrates the check in three lines, and the demonstration
is the part worth taking — not the storage engine. They write to an
unindexed namespace and then probe it three ways:

- a keyword-matching search → **0 hits**
- an unfiltered search → **0 hits**
- a direct `get` by key → **returns the value**

That is a *falsifiable test of a storage-tier property*, as opposed to a
claim in a README. An unindexed item is invisible to vector search — not
ranked last, not returned with a null score, simply absent.

### The gate

For each namespace, declare its intended reachability and assert it:

| Property | Meaning | Assertion |
|---|---|---|
| `searchable` | reachable by similarity query | a matching query returns it above the noise floor |
| `direct-only` | reachable only by key | matching query returns 0; unfiltered search returns 0; direct fetch returns the value |
| `always-injected` | delivered unconditionally | present in the assembled context without any query; **and** `direct-only` holds |

Two details make this a real test rather than a tautology:

- **Print the scores.** A round-trip that asserts only `0 hits` passes
  vacuously if retrieval is broken everywhere. Recording the score of the
  intended hit alongside the zero proves the probe itself is live.
- **Probe both search forms.** Query-matched and unfiltered. An item can be
  absent from a keyword query for the boring reason that the words differ,
  while still being reachable by an unfiltered listing. Only both together
  demonstrate index exclusion.

### Cost of the always-injected tier

The prior art is explicit about a constraint worth adopting as a stated
rule: an always-injected namespace is **a fixed token tax on every single
turn**. That is why theirs is capped (a small maximum rule count) and why
it holds *behaviour* only — never diagnosis, never facts. Content that
merely might be relevant does not belong in a tier that is paid for
unconditionally.

Any always-on tier here needs the same two properties declared explicitly
in `pack_recall_budget`: a hard cap, and a stated content type it is
allowed to hold. Without the cap the tier grows until it crowds out the
retrieved content it was meant to supplement.

### Done when

- Every namespace has a declared reachability property.
- Each property is asserted by a test that prints its scores.
- Any always-injected namespace has a declared cap and content type.
- The tests run without an API key — they exercise storage and retrieval,
  not extraction.

---

## M3 — relevance floors as a per-namespace property

### The observation

A single similarity floor applied across all namespaces is wrong in both
directions at once, and the reason is a property of the corpora rather than
of the tuning.

The prior art applies **asymmetric floors** and justifies each:

- The episodic corpus is **unbounded and mostly irrelevant to any given
  query** — most stored episodes have nothing to do with the ticket in
  hand. It gets a floor, because without one the top-k is padded with noise
  that outranks nothing but still consumes slots.
- The per-customer semantic corpus is **small and bounded** — every fact in
  it is about the entity being asked about. It gets **no floor**, because a
  floor would drop a relevant fact whenever the query does not happen to
  share its vocabulary. Their example is exact: a ticket that never says the
  word "plan" should still retrieve the plan fact.

The generalisation: **a floor is correct when the corpus is large and mostly
irrelevant, and harmful when the corpus is small and mostly relevant.** That
is a property of the namespace, not a global tuning constant.

### Why the number needs evidence, not a guess

Their floor is defensible because they show the separation it sits between:
a correct hit at 0.4100 against noise at 0.0054 on a small corpus — roughly
two orders of magnitude. A floor placed in that gap is obviously safe. A
floor chosen without knowing the gap is a guess that will silently drop
relevant results the day the distribution shifts.

So M3 depends on M2: the per-namespace score distributions that M2's probes
print are the evidence a floor policy requires. Setting floors first and
measuring later reverses the dependency and produces numbers nobody can
defend.

### Specification

- Floor becomes a **declared per-namespace property**, with three settings:
  a numeric floor, explicit `none`, or `inherit-global`.
- `none` is a **first-class, documented choice** — not an omission and not
  a zero. A bounded namespace where every item is relevant should say so.
- Every numeric floor must cite the separation measurement that justifies
  it, recorded next to the declaration.
- Changing a floor is a change with a measurable effect and must report the
  recall delta on the fixture set, not merely the new value.

### The open dependency

A floor policy that is *tuned* rather than merely *declared* wants a
ground-truth eval set on our own corpus — the same blocker as L1 and M7.
M3 is deliberately scoped below that line: declaring floors per namespace
and recording the separation evidence for each is achievable now and is
already strictly better than one global constant. Optimising the values
waits for the eval set.

### Done when

- Floor is a per-namespace declared property with `none` as a valid setting.
- Every numeric floor cites its separation measurement.
- The rule — floor when large-and-mostly-irrelevant, none when
  small-and-mostly-relevant — is documented as the decision criterion.

---

## Non-goals (both items)

- No new storage engine, index, or backend.
- No change to the embedding model. M1 owns the embedded-field question;
  these items assume whatever M1 concludes.
- No governance weight. Reachability assertions, score distributions, and
  floor values are measurement and configuration artifacts: they never
  enter a block, a hash chain, or the approval gate.

## Provenance rail

Prior-art shape observed in a public tutorial: the index-exclusion
round-trip probe, the always-injected token-tax constraint, and the
asymmetric-floor justification. The cited figures are the *shape of a
working separation* observed externally — not our measurements and not
targets. No code adopted, nothing named in any public artifact. Citation in
`mind-internal`.

# Design: M4 — closed-set slots, structural contradiction prevention

Status: Draft (2026-08-17) · Owner: mind-mem · Roadmap: Group M / M4

**This item changes write semantics on a governed store. It is specified
before any code is written, and the spec is the deliverable.**

## The two ways to keep memory from contradicting itself

mind-mem prevents contradiction **detectively**. Two conflicting facts are
written, and machinery afterwards notices: `list_contradictions`,
`compiled_truth_contradictions`, the `propose_update` → `approve_apply`
gate. This works on open-ended knowledge, produces lineage, and is
reversible. Its cost is that it runs *after* the conflict exists and
depends on similarity detection actually finding it — a fuzzy operation
with false negatives.

The prior art prevents contradiction **structurally**. Every fact about a
bounded attribute is filed under a slug drawn from a *closed set*, so the
storage key is the attribute itself. A second statement about the same
attribute lands on the same key by construction and replaces its
predecessor. A key collision is exact. It cannot miss.

The load-bearing detail is the *closed* part, and the prior art is explicit
about why: an open-ended topic string lets a model file `plan` on Monday and
`plan_tier` on Friday, and you are back to two facts that contradict each
other and never collide. A free-text key is not a key.

These two mechanisms are **orthogonal, not competing.**

| | Structural (closed-set key) | Detective (governed proposal) |
|---|---|---|
| Wins on | bounded attributes: status, tier, role, owner | open-ended knowledge |
| Detection cost | zero — key equality | similarity search + review |
| False negatives | none | possible |
| Audit trail | **none — silent overwrite** | full lineage, rollback, evidence |

Each covers the other's weakness exactly.

## The synthesis

**Enum-keyed slots inside the governed path.**

For a namespace declared as *slotted*, the write is a keyed upsert — the
collision is structural and exact — and the supersession is still routed
through the proposal gate, so replacing a value is a **recorded, reversible
event** rather than a destructive overwrite.

This is strictly better than either mechanism alone. The prior art's
semantic write destroys the old value with no proposal, no lineage, no
rollback, and no contradiction record; that is precisely the tier holding
the facts, and it is the unguarded one in their design. We get their
exactness without adopting their data loss.

## Specification

### Slot declaration

A namespace may declare a **slot set**: a closed, versioned enumeration of
attribute names, each with a stated meaning. Declaration is configuration,
not inference — a slot set is authored and reviewed, never grown at write
time by a model.

Two properties are required of the enumeration:

- **Closed.** A write naming a member outside the set is *rejected*, not
  coerced and not silently filed elsewhere. Rejection is the mechanism; a
  slot set that accepts unknown members is a free-text key wearing a
  costume.
- **Versioned.** Adding, removing, or renaming a member is a schema change
  with its own migration, because it changes what collides with what.
  Renaming a slot silently orphans every fact filed under the old name.

### Write semantics

Within a slotted namespace, a write carries `(slot, value)`:

1. **No existing occupant** — the write proceeds as a normal governed
   proposal for a new block.
2. **Existing occupant, value materially unchanged** — a no-op, recorded
   as a re-assertion rather than a supersession. Restating a known fact
   must not manufacture churn in the lineage.
3. **Existing occupant, value differs** — a **supersession proposal**. The
   prior block is not deleted. On approval it is marked superseded, the new
   block records the block it supersedes, and lineage links the two. The
   contradiction is resolved *by construction* rather than detected, but
   the resolution is still an auditable, rollback-able event.

Case 3 is the entire point. The structural collision determines *that*
these two facts conflict; the governed path determines *what is recorded
about the resolution*.

### The escape hatch and its honest cost

Not every fact in a bounded namespace has a natural slot. The prior art
provides an `other` member keyed by a content hash, which means such facts
accumulate rather than supersede, and states that trade plainly. The trade
is acceptable and should be stated the same way here: **fixed slots
supersede, free-form facts pile up.**

What must *not* be copied is the key width. A truncated 32-bit content hash
is collision-prone as a durable identity scheme — fine in a demo, not in a
governed store where a collision silently merges two unrelated facts. Any
content-derived key here uses a full-width digest.

Better still, unslotted facts in a slotted namespace should simply route to
the ordinary detective path. They are the open-ended case; that is what the
detective path is for. The escape hatch does not need to invent a third
mechanism.

### The residual failure, stated plainly

**The model still picks the slug.** This design moves the failure from
"the model forgets what it wrote last week" to "the model picks the wrong
slot member." That is narrower and it is *validatable* — a closed
enumeration rejects an invented member, which a free-text key cannot do —
but it is not eliminated. A fact filed under the wrong valid slot both
overwrites something it should not have and fails to supersede something it
should have.

Two mitigations, neither of which fully closes it:

- Reject-on-unknown catches invention, which is the common case.
- Supersession-as-proposal means a mis-slotted write is *reviewable and
  reversible* rather than a silent destruction — which is exactly the
  property the prior art lacks.

This paragraph exists so the claim made externally is the honest one:
contradiction is prevented for bounded attributes, not solved in general.

## Migration

A namespace becoming slotted is a migration, not a flag flip. Existing
blocks in that namespace pre-date the slot set and are unslotted by
definition. The migration must decide, per block, whether it maps to a slot
— and mapping is exactly the "model picks the slug" problem applied
retroactively across the whole corpus at once.

Therefore: **slotting applies to new writes; back-filling existing blocks is
a separate, individually-reviewed operation.** A bulk automated back-fill
would perform thousands of unreviewed supersessions, which is the failure
mode this design exists to prevent, executed at scale.

## Non-goals

- Not a replacement for the detective path. Open-ended knowledge keeps
  using contradiction detection; this is an addition for bounded attributes.
- No new storage engine, index, or backend.
- No governance weight beyond what the existing proposal gate already
  carries. A slot is a key, not a claim about truth: slot membership must
  never influence confidence, scoring, or the approval decision itself.
- No automatic slot-set inference from the corpus. A slot set is authored.

## Done when

*(Spec-stage completion. Implementation is a separate item gated on this
document being accepted.)*

- The slot-declaration format is specified, including versioning and the
  rejection rule.
- The three write cases are specified with their lineage effects.
- Supersession is specified as a proposal, never a destructive overwrite.
- The migration boundary is specified: new writes only.
- The residual mis-slotting failure is documented in the user-facing
  description, not only here.

## Provenance rail

Prior-art shape observed in a public tutorial: closed-set slug as upsert
key, and the escape-hatch trade. No code adopted, nothing named in any
public artifact. The synthesis — keyed upsert *inside* the governed
proposal path, supersession recorded rather than overwritten — is ours and
is the part that differs from the source.

**Internal precedent, established 2026-08-17.** The closed-set discipline
is already house style one layer up, applied to *meaning* rather than to
keys. `512-mind/src/drift.mind` enumerates the mutation classes that
corrupt a contract — `"must not"`→`"should not"` weakens an obligation to a
suggestion, `"fail open"`→`"fail safe"` inverts a default, `"any human"`→
`"authorized participants"` narrows scope — and `no_semantic_drift` asserts
against that fixed list. Same structural move as this item: enumerate the
space so a violation collides by construction rather than needing to be
noticed. 121 enums across that repo make it a convention, not a one-off.
Cite `drift.mind` as the precedent for closed-set-as-mechanism; the
external source contributed the specific application to storage keys and
nothing else. Citation in `mind-internal`.

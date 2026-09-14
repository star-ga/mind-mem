# M4 closed-set slots: shipped write contract

M4 is opt-in per workspace. The declaration lives in `mind-mem.json`:

```json
{
  "closed_slots": {
    "version": 1,
    "namespaces": {
      "profile": {
        "version": 1,
        "slots": ["status", "tier"]
      }
    }
  }
}
```

Both the declaration version and each namespace version are positive integers.
Namespace and member names are lowercase identifiers (`a-z`, digits, `_`, `-`,
and `.`), beginning with a letter. The list is closed: `propose_slot_update`
returns a structured error for an undeclared namespace or member. A malformed
section also fails closed. A workspace without `closed_slots` keeps the normal
free-form `propose_update` path.

`propose_slot_update(namespace, slot, value, rationale)` writes only a staged
proposal in `intelligence/proposed/EDITS_PROPOSED.md`. A reviewer must call the
existing `approve_apply(proposal_id, dry_run=False)` gate. The proposal appends
a decision block when the slot has no active occupant. A differing value uses
the existing atomic, snapshot-backed `supersede_decision` operation: the old
block remains with `Status: superseded` and `SupersededBy`, and the successor
carries `Supersedes` plus the same namespace, member, and declaration version.
A materially identical value returns `reasserted` without changing the source
corpus or creating lineage churn. Repeated identical staged requests return
the existing proposal; a different update for that slot is refused while one
is awaiting review.

Only blocks written after a declaration exists carry slot metadata. Existing
free-form blocks are not backfilled or reinterpreted. Unslotted facts in a
slotted namespace continue through the ordinary detective path. The full
SHA-256 digest in `SlotValueDigest` is an audit binding for the stored value;
it is not a key and does not change confidence or ranking.

The valid-slot check is a structural boundary, not a truth oracle: a caller
can still choose the wrong *valid* member. Review, reversal, and the existing
proposal evidence remain necessary.

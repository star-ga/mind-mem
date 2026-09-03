# Governed writes

`UngatedWriteError`, `UngatedDeleteError` and `UngatedRestoreError` point here. This is
that page: the three ways content can leave or enter the store, and the receipt each one
requires.

## The rule

**No block reaches the store without a gate receipt.**

Every `BlockStore.write_block` begins with `require_admission(block_id)`. A write with no
open admission, or with a receipt that does not cover the block being written, raises
`UngatedWriteError`. It is raised, not logged and not degraded — an ungated write *is* a
governance bypass, and the apply path already lets that class propagate to abort.

## Why it is enforced rather than agreed

The claim "every write goes through `propose_update` → `approve_apply`" used to be a
code-review convention. It was false in the tree: `GovernanceGate.admit()` had two callers
while `write_block` had thirteen. Drop-folder ingest and agent messaging wrote blocks
stamped `Status: active` — indexed, immediately recallable, with the hash chain, the
evidence chain and the audit chain all absent.

A convention that four doors already ignored is not an invariant. This one is enforced at
the store, so a new door cannot forget.

## How to write a block

Open an admission through the gate. There are three, and which one you may use is decided
by your tier, not by you:

| opener | covers | who uses it |
|---|---|---|
| `admit_block(action, block_id, content, …)` | exactly that id | a single authored block |
| `admit_batch(action, batch_id, block_ids, content, …)` | the named set | bulk ingest, migration, re-extraction |
| `admit_proposal(proposal_id, content, …)` | any block, for the scope | `apply_engine` applying an approved proposal — **the only path to `ACTIVE`** |

```python
from mind_mem.governance_gate import get_gate

with get_gate(workspace).admit_block(
    action="MESSAGE", block_id=block_id, content=text, actor="agent",
):
    store.write_block(block)
```

The receipt is published on a `ContextVar` by the opener and read by `write_block`. You do
not pass it, and you cannot construct one — that is deliberate: a receipt you could hand
in is a receipt you could forge.

## Tiers decide status, callers do not

`INITIAL_STATUS` is the single place an initial `Status` is chosen. `admit_block` and
`admit_batch` refuse any tier whose row is servable, so **only an approved proposal can
mint `ACTIVE`**. A source with no `IngestTier` cannot obtain a receipt at all, so a new
ingest door fails closed rather than landing recallable blocks.

Untrusted sources — the inbox drop folder, importers, agent messages — arrive
`QUARANTINED` with an external-ingest tier and stay invisible to recall until a governance
proposal releases them.

### Confined tiers

One tier is neither an input door nor a proposal apply. `DETECTOR_FINDING` records the
integrity scanner's own findings — contradictions and drift signals — and mints
`Status: open`, which recall recognises. It is allowed to because it is **confined**:
`enums.TIER_ID_PREFIXES` names the block-id prefixes it may write (`C-`, `DREF-`), and
`require_admission` refuses its receipt for any other prefix and for any status but the
one row it mints. Two corpora, one status, no choice.

Confinement narrows *where* a tier may write; it never widens *what* it may mint —
`SERVABLE` is still `{ACTIVE}` and `admit_block`/`admit_batch` still refuse every tier that
reaches it. A confined tier's row is also excluded from `admissibility.UNADMITTED`
(`enums.mints_quarantine`), because `open` is the lifecycle state of one corpus rather than
a quarantine marker — folding it in would withhold every `open` block in the product.
`tests/test_quarantine_redteam.py` pins the confined set to exactly one tier, so a second
one has to be argued for rather than added.

## Deletes

**No block leaves the store without a delete receipt.** Every `BlockStore.delete_block`
begins with `require_delete_admission(block_id)`, exactly as every write begins with
`require_admission`. A WRITE receipt is not transferable to a delete: the two are different
operations and a receipt names one of them. The openers are `admit_delete(block_id, …)` for
one id and `admit_delete_batch(batch_id, block_ids, …)` for a named set; a proposal-scoped
receipt is refused for a delete outright, because "any block, for the scope" is ambient
authority to destroy anything.

The scope records what it destroyed, not what it was asked to destroy: the store calls
`receipt.record_removal(block_id, content)` for each block it actually removed, and the
close record carries a Merkle root over those removals. A delete that removed nothing
writes no removal record — there is nothing to claim — while the admission row that opened
the scope stays in the chain, so "asked and found nothing" is still auditable.

## Restores

**A restore is the third door, and it is admitted at the seam too.** A restore withdraws
every block written since the snapshot and reinstates the versions under it — the most
destructive operation the product has — and through 5.0.1 it was the one mutation held up
by convention: every sanctioned caller went through `apply_engine.restore_snapshot`, and
nothing made a caller that did not fail. Measured on 5.0.2 with the write and delete gates
already closed: a governed block died to a bare `store.restore(snap)` and neither ledger
moved.

Every `BlockStore.restore` now begins with `require_restore_admission(snap_dir)`, before
the snapshot is read, so an ungated caller fails by authorisation and cannot learn whether
a snapshot exists. The receipt it accepts has four properties, each refusing a receipt that
would otherwise be silently transferable into a restore:

| property | must be | why |
|---|---|---|
| `operation` | `WRITE` | a restore re-writes content; a DELETE receipt is not transferable to it |
| `kind` | `BATCH` | a restore reinstates a set and withdraws another, so it needs a receipt naming both — a block receipt covers one id, a proposal receipt covers whatever it is asked about |
| `tier` | `RESTAMP` | minted for a re-stamp of already-governed content, never for an ingest |
| `chain_verified` | true | the gate read the admission back out of the durable chain |

`apply_engine.restore_snapshot` is the opener: it hashes the snapshot manifest, computes the
ids it will reinstate and — from the live tree, before anything moves — the ids it will
withdraw, and opens `admit_batch(action="RESTORE", tier=RESTAMP, block_ids=<both sets>, …)`
around the store call. The load-bearing refusal is the `kind` row: the apply engine rolls
back from *inside* an open `admit_proposal`, and without it the proposal's ambient receipt
would authorise a bare `store.restore()` on that path.

Tests that exercise restore mechanics open the same scope through one shared helper,
`tests/_restore_scope.py::restoring(workspace, batch_id=…, block_ids=…)`, which mints a real
receipt through the gate — never a stand-in — so a test using it fails if the gate stops
minting the shape the seam accepts. Each such test file carries a positive control that an
ungated restore still raises and reverts nothing.

## Threads

`contextvars` do not cross a new `threading.Thread`. That is fail-closed and correct: a
background writer must open its own admission scope inside its thread. Do not work around
it by hoisting the receipt.

## Testing code that writes blocks

Tests that legitimately call `write_block` open a **real** admission through the gate — see
the `admitted` fixture in `tests/conftest.py`. There is deliberately no test-only
constructor, no wildcard receipt and no environment escape hatch, and the fixture is not
autouse: a suite-wide admission would silently defeat the invariant everywhere.

## What keeps this true

`tests/test_governed_write_paths.py` walks the AST for every `write_block` call and every
direct corpus append, and fails on any caller outside an allowlist that must carry a
written justification. It is built not to pass vacuously — a corpus floor, a positive
control on a known call site, and a negative control run against synthetic rogue source.
Without that test the invariant would decay at the next feature.

`tests/test_governed_restore_seam.py` does the same for the third door: it pins
`apply_engine.restore_snapshot` as the only opener of a `.restore(` call in `src/`, checks
that every `restore` implementation calls `require_restore_admission` (an import alone does
not count, and the matcher is itself tested against source that lacks the call), and runs
the ungated restore against a live block to show it raises and the block survives.

# Governed writes

`UngatedWriteError` points here. This is that page.

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

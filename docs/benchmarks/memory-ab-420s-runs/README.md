# A/B run receipts — path-sanitized public copies

Every `*.json` here is a **public copy** of a run receipt, and each one says so
in its own `provenance` block. Read that block before quoting anything.

## What was changed, and what was not

One field: `agent.argv[0]`. It held the absolute path of the agent binary on
the machine that ran the suite. It now holds the command name.

Nothing else. Scores, counts, spend, the summary and every per-task row are
byte-for-byte the original. No figure was recomputed, re-run, re-ranked,
dropped or added.

## Why the `digest` field does not recompute

`digest` is a sha256 over the whole scored record, and the record includes the
`agent` block. Changing `argv[0]` therefore changes the digest's input, so the
stored value **no longer recomputes over this file**. That is stated in each
receipt rather than hidden.

The digest was deliberately **not** regenerated. A regenerated digest would
make a rewritten receipt look like one that was produced that way, which is a
stronger claim than "these are the same numbers" and is not true. Each file
records `original_digest` and `original_bytes_sha256` so the unmodified
receipt can be identified if it is ever produced.

The unmodified receipts are retained privately by the publisher. They are
self-consistent — all 24 recompute to their stored digest — and they are the
artifact a verifier should ask for if a byte-level check is needed.

## What a reader can rely on here

* the measured results, unchanged
* the harness configuration, budget and task set, unchanged
* the identity of the command that was run, without where it lived

## What a reader must not infer

* that `argv` is literally the command line that executed — it is the same
  command with its location removed
* that `digest` verifies this file — it verifies the original bytes

# Append-Only Audit Logs — Operator Runbook

> Closes audit item **T-007** (roadmap v4.0.15, deferred from the
> 2026-04-28 STRIDE audit pass). The mind-mem audit chain already
> detects tampering via hash-chain verification; this runbook makes
> tampering **physically impossible without root** by applying the
> OS-level append-only attribute to the on-disk audit files.

## What gets append-only

mind-mem writes three forensic audit trails as JSON-lines files:

| File                                 | Purpose                                  | Added in |
|--------------------------------------|------------------------------------------|----------|
| `memory/deleted_blocks.jsonl`        | `delete_memory_item` deletion receipts   | v3.0.x   |
| `memory/decrypted_files.jsonl`       | `decrypt_file` admin-tool forensic trail | v4.0.15  |
| `memory/audit.log` (chained)         | Hash-chained governance write events     | v2.x     |

All three follow the same append-only pattern: `O_APPEND` writes,
no in-place edits, JSON-per-line so partial-write resumption is
trivial. Applying the OS attribute is the **second** layer of
defence (hash chain is the first).

## Linux — `chattr +a`

```bash
# As root, after the workspace is initialised (the file must exist
# before the attribute is set; mind-mem touches each file on first
# write):
sudo chattr +a /path/to/workspace/memory/deleted_blocks.jsonl
sudo chattr +a /path/to/workspace/memory/decrypted_files.jsonl
sudo chattr +a /path/to/workspace/memory/audit.log
```

After this:

* `O_APPEND` writes from mind-mem keep working.
* Any process attempting an in-place rewrite (truncate, seek + write,
  open with `O_WRONLY` without `O_APPEND`) gets `EPERM` — even root.
* `unlink()` fails with `EPERM`. The file can only be removed after
  the operator does `chattr -a`.

**Filesystem requirements:** ext2/3/4, btrfs, xfs (with chattr
support). Network filesystems (NFS, SMB) and tmpfs typically do not
honor `+a` — for those, terminate the chain at a host-local volume
mount.

**Verify:**

```bash
lsattr /path/to/workspace/memory/decrypted_files.jsonl
# Should show ``-----a-------e------- /path/...`` (the ``a`` flag).
```

## macOS — `chflags uappnd`

```bash
# User-level immutable + append-only flag (USR_APPEND). Survives
# normal writes via O_APPEND; blocks every other mutation including
# unlink unless the operator clears the flag.
sudo chflags uappnd /path/to/workspace/memory/deleted_blocks.jsonl
sudo chflags uappnd /path/to/workspace/memory/decrypted_files.jsonl
sudo chflags uappnd /path/to/workspace/memory/audit.log
```

To verify: `ls -lO /path/to/workspace/memory/` — the `uappnd` flag
appears in the output.

**System Integrity Protection note:** SIP-protected paths are not
required; user-volume application of `uappnd` is sufficient and
honored by all standard `open(2)` calls.

## Windows

Windows lacks a direct equivalent. Two options:

1. **NTFS ACLs:** grant `FILE_APPEND_DATA` only, deny `FILE_WRITE_DATA`
   on the audit files. `mind-mem` opens with `O_APPEND` which maps to
   `FILE_APPEND_DATA`; the deny on `FILE_WRITE_DATA` blocks every
   other mutation. Requires `icacls /grant` + `/deny` setup as
   Administrator.

2. **Forward to a WORM store:** redirect the audit JSONL files to a
   Windows-side WORM volume (e.g., a write-once SMB share, an
   Object Lock S3-compatible target via `rclone mount`, or a
   commercial WORM file system). This is the recommended pattern
   for compliance deployments.

`icacls` example (Administrator PowerShell):

```powershell
$path = "C:\path\to\workspace\memory\decrypted_files.jsonl"
icacls $path /inheritance:r
icacls $path /grant:r "SYSTEM:(F)"
icacls $path /grant:r "mind-mem-service:(WD,AD)"   # write + append data
icacls $path /deny    "mind-mem-service:(WD)"      # deny in-place write
```

## When to apply

Apply **after** the workspace has been initialised and you have
verified the audit chain integrity at least once (`mm scan
--verify-chain`). Re-applying after a chain repair requires
`chattr -a` / `chflags nouappnd` first.

## Rotation / log-bomb concern

The audit files grow unbounded. For long-lived deployments,
**rotate** rather than truncate:

```bash
# Periodic (e.g. weekly cron) rotation.
sudo chattr -a /path/to/workspace/memory/decrypted_files.jsonl
mv /path/to/workspace/memory/decrypted_files.jsonl \
   /path/to/workspace/memory/decrypted_files.jsonl.$(date +%Y%m%d)
touch /path/to/workspace/memory/decrypted_files.jsonl
sudo chattr +a /path/to/workspace/memory/decrypted_files.jsonl
# Optionally re-apply +a to the rotated file so historical records
# are also tamper-evident:
sudo chattr +a /path/to/workspace/memory/decrypted_files.jsonl.$(date +%Y%m%d)
```

Truncating the live file is intentionally blocked; rotate-and-touch
is the supported path. Same pattern on macOS with `chflags`.

**Chain-continuity caveat (hash-chained ledger).** Rotate-and-touch
restarts the *live* hash-chained ledger from a fresh genesis: an empty
ledger seeds the next entry with `prev_hash = GENESIS` and `seq = 1`.
`verify_chain` walks from genesis over whatever file it is given, so it
validates each rotated segment **independently** — it does **not**
cryptographically bind a rotated segment to its successor. Consequently,
deletion of an *entire* rotated segment is not detectable by the hash
chain (the live chain still verifies clean); that gap is closed only by
the OS append-only attribute plus your retention of the rotated files.
Keep every rotated `*.YYYYMMDD` segment (and re-apply its `+a` /
`uappnd` flag) if you need the ledger to be tamper-evident end-to-end,
and verify each segment separately.

## Verification suite

After applying, run:

```bash
mm doctor --verify-audit-immutability   # roadmap v4.1.0 — not yet shipped
```

Today (v4.0.15), verify manually:

```bash
# Linux: ``a`` flag should be present.
lsattr /path/to/workspace/memory/*.jsonl

# Try an in-place write — must fail with EPERM.
echo "tamper" > /path/to/workspace/memory/decrypted_files.jsonl
# bash: ...: Operation not permitted   ← correct behaviour
```

## Threat model alignment

This runbook closes the post-compromise tampering vector: a malicious
process that gains the mind-mem service's UID can write new audit
records (via `O_APPEND`) but cannot erase or rewrite historical ones.
It does **not** stop the service itself from writing falsified
records into the chain; that's the hash-chain layer's job. The two
layers are independent.

## Deletion vs. tamper-evidence — the tombstone ledger

An append-only ledger and a right-to-forget request pull in opposite
directions: the ledger must never lose a record, the request demands
that content stop existing. `v4.redactable_tombstones` resolves this
by splitting the block into the part that must die (its content) and
the part that must survive (its hash).

| File | Purpose | Added in |
|------|---------|----------|
| `.mind-mem-audit/tombstones.jsonl` | Redaction records — one per redacted block | v4.2 |

With the flag **off** (the default) `delete_memory_item` behaves as it
always has: the block leaves the corpus and a full copy lands in
`memory/deleted_blocks.jsonl`, i.e. the deletion is recoverable.

With the flag **on**, deletion becomes redaction:

* the content is destroyed — corpus text, index rows, FTS terms (the
  FTS5 index is rebuilt, because an FTS delete is only logical), cached
  embeddings, the SQLite WAL and free pages (`secure_delete` + `VACUUM`),
  and any pre-existing plaintext receipt in `deleted_blocks.jsonl`;
* the block's **Merkle leaf is preserved** in the tombstone ledger and
  re-supplied to the tree, so the root does not move and inclusion
  proofs issued before the redaction still verify;
* the deletion event is chained three ways — an evidence record and a
  SHA3-512 hash-chain entry through the governance gate, plus a
  `delete_block` entry in the SHA-256 audit chain — each carrying the
  actor and the reason. The tombstone binds all three receipts and
  back-links to the previous tombstone.

A redaction therefore leaves a block that is *visibly redacted* rather
than one that never existed: `get_block` returns `tombstoned: true`
with the actor, the reason and the preserved leaf, and re-deleting is
idempotent (no second chain entry is written).

Enable it per workspace:

```json
{ "v4": { "redactable_tombstones": { "enabled": true } } }
```

`delete_memory_item(block_id, actor=..., reason=...)` then requires a
reason — it is chained as the justification for destroying content.

### Append-only interaction (read before running `chattr +a`)

* **Do** apply the append-only attribute to
  `.mind-mem-audit/tombstones.jsonl`: it is written with `O_APPEND` and
  is never rewritten. Removing a tombstone is exactly how a redacted
  block would be made to look like a block that never existed, and
  `mind-mem-verify` reports it as a chain failure (`previous_hash
  mismatch`).
* **Do not** apply it to `memory/deleted_blocks.jsonl` while redaction
  is enabled: scrubbing an old plaintext receipt is an in-place
  rewrite, which `+a` forbids. mind-mem logs
  `tombstone_scrub_write_failed` and continues — the block is redacted
  everywhere else, but that one journal line must then be cleared by
  hand. Once redaction is on, no new plaintext is ever written there.

### Verify

```bash
mind-mem-verify /path/to/workspace     # includes a `tombstones` check
```

The MCP `verify_chain` tool reports the same under a `tombstones` key.
Neither surface changes shape for a workspace that has never redacted.

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

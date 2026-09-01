# Security Model

## Overview

MIND-Mem operates on local filesystem data with no network dependencies. Security considerations focus on data integrity, access control, and audit trails.

## Data Integrity

### Proposal System
All memory mutations go through the proposal system:
1. `propose_update` creates a staged change
2. Human reviews the proposal
3. `approve_apply` commits the change
4. Full audit trail maintained

### Contradiction Detection
The `scan` tool detects contradictions between blocks, preventing inconsistent memory state.

### Rollback Support
Any applied proposal can be rolled back via `rollback_proposal`, providing recovery from incorrect changes.

## Access Control

### Agent ID Filtering
Recall supports `agent_id` parameter for namespace-based access control. Each agent sees only its authorized blocks.

### Workspace Isolation
Each workspace is a self-contained directory. Multiple workspaces can run independently without interference.

## Audit Trail

### Block History
`memory_evolution` tracks the full history of block changes including edits, supersedes, and deletions.

### Proposal Log
All proposals are logged with timestamps, reasons, and outcomes.

## File Security

- All data stored as plain text markdown (auditable)
- Optional at-rest encryption (opt-in `encrypted` backend: HMAC-SHA256 keystream + encrypt-then-MAC over block files; the FTS5/sqlite-vec recall index is not encrypted). The zero-config default stores plain-text markdown and relies on filesystem permissions
- No network connections (zero external dependencies)
- No credential storage

### At-rest encryption: two modes, and what each one actually is

Be precise about these, because only one of them is authenticated encryption.

**Default (`encrypted` backend).** An HMAC-SHA256 keystream with
encrypt-then-MAC over block files. It is *not* an AEAD, and it is not AES.
It is honest confidentiality plus integrity for a local file, and the name in
the code says exactly that. The FTS5 / sqlite-vec recall index is not
encrypted in either mode.

**Opt-in envelope mode (AES-256-GCM).** Real AEAD, provided by `tenant_kms`.
New records are written as `MMKMS1 | nonce(12) | AES-256-GCM(ct||tag)` under a
data key that `tenant_kms` mints and wraps with an operator-supplied key. The
wrapped key blob lives beside the store in the existing `0700`
`.mind-mem-keys/` directory. It runs single-tenant (`tenant_id="default"`)
today; the multi-tenant surface is already there and tested behind it.

Three gates must all open, cheapest first, so an unopted install pays nothing:

1. `MIND_MEM_KMS_MASTER_KEY_B64` is set (one dict lookup),
2. the `v4.tenant_kms` feature flag is enabled,
3. the `cryptography` package is importable.

If `cryptography` is missing the store logs a warning and falls back to the
default path — it never crashes, and mind-mem's default install stays
zero-dependency. Reads route on the record magic, so a workspace containing
both formats opens both; there is no migration step and no flag day.

Two behaviours fail closed on purpose: a wrapped key blob that will not unwrap
is never overwritten (the same discipline as a corrupt salt), and `rotate_key`
refuses outright in envelope mode rather than silently rewriting AES-GCM
records with the keystream.

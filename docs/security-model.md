# Security Model

## Overview

mind-mem operates on local filesystem data with no network dependencies. Security considerations focus on data integrity, access control, and audit trails.

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
- No encryption at rest (rely on filesystem permissions)
- No network connections (zero external dependencies)
- No credential storage

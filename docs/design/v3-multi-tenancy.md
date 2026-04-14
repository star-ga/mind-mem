# v3.0 Design: Multi-Tenancy Foundation

Status: **draft**
GH issue: [#505](https://github.com/star-ga/mind-mem/issues/505)

## Problem

mind-mem v2.x assumes a **single-tenant workspace**: one user, one
machine, one set of Markdown corpora under `./decisions/`, etc. The
v3.0 REST-API roadmap item (planned for v3.1) requires
multi-tenancy: multiple orgs sharing one deployment, each with their
own isolated memory, auth, and rate limits.

The current architecture has no org/user/tenant concept. Bolting it
on naively (e.g. prefixing every table with a tenant_id) creates
subtle cross-tenant data leakage bugs when queries forget the
predicate. We need a foundation that makes cross-tenant leakage
*structurally impossible*.

## Proposal

Three-layer tenancy model:

```
Organization  →  Workspace(s)  →  Namespace(s)
```

- **Organization**: billing + RBAC root. Typically 1 per company.
- **Workspace**: physical storage boundary. Files on disk, one SQLite
  DB pool, one encryption key. One or more per organization.
- **Namespace**: logical partition *inside* a workspace (exists in
  v2.x via `NamespaceManager`). Unchanged.

### Identity model

```python
@dataclass(frozen=True)
class Actor:
    org_id: str           # organization UUID
    user_id: str          # user UUID within the org
    roles: tuple[str, ...]  # e.g. ("admin", "auditor")
    session_id: str       # current session UUID
```

Every mutation path takes an `Actor` and persists it into the audit
chain + evidence record. No anonymous writes.

### Storage partitioning

Per-tenant isolation at the **physical** layer, not a query filter:

```
/var/lib/mind-mem/
├── orgs/
│   ├── {org_id}/
│   │   ├── workspaces/
│   │   │   ├── {workspace_id}/
│   │   │   │   ├── decisions/
│   │   │   │   ├── intelligence/
│   │   │   │   ├── .mind-mem-audit/
│   │   │   │   └── mind-mem.json
│   ├── keys/
│   │   └── {workspace_id}.enc    # per-workspace encryption key
│   └── org.json                  # org metadata + billing
```

A workspace never shares a SQLite file with another workspace; a
forgotten WHERE clause can't leak data.

### AuthZ

Per-tool scope matrix layered on the existing ADMIN_TOOLS /
USER_TOOLS split:

| Scope | Tools permitted |
|---|---|
| `read` | recall, get_block, list_contradictions, verify_chain |
| `write` | all read + staged_change(propose), observe_signal |
| `review` | all write + staged_change(approve) |
| `admin` | all review + encrypt_file, reindex, compact, rollback_proposal |
| `owner` | all admin + workspace creation/deletion, RBAC changes |

Role binding happens per (org_id, user_id, workspace_id) triple in a
new `rbac` SQLite DB.

### Rate limiting

Per-org + per-user sliding-window limits. Existing
`SlidingWindowRateLimiter` is already LRU-bounded from v2.9.0; we
re-key it on `(org_id, user_id)` instead of `client_id`.

### Encryption key management

One key per workspace, stored at
`/var/lib/mind-mem/orgs/{org_id}/keys/{workspace_id}.enc`. The key
file itself is encrypted with a KEK derived from an operator
passphrase (from env var or KMS). Key rotation per workspace doesn't
affect other tenants.

## Migration from v2.x

Existing single-tenant installs become a one-org, one-workspace
deployment:

```bash
mm migrate v2-to-v3 --workspace ~/.openclaw/workspace
# → creates default org, assigns current user as owner,
#   links existing workspace
```

## API surface

New MCP tools (on top of the consolidated set from #501):

| Tool | Scope | Purpose |
|---|---|---|
| `workspace_list` | read | List workspaces the current user can access |
| `workspace_create` | owner | Create a new workspace in the org |
| `workspace_delete` | owner | Soft-delete (30-day grace) |
| `user_list` | read | List users in the org |
| `user_invite` | admin | Issue an invite token |
| `rbac_grant` | owner | Grant a role binding |
| `rbac_revoke` | owner | Revoke a role binding |

Every *existing* tool gains an implicit `workspace_id` parameter
(defaulted from the session context). Public schemas keep the
`_schema_version` field so old clients still parse responses.

## Open questions

1. Do we ship user/org management via mind-mem itself, or delegate
   to an external IdP (OIDC) and only store role bindings locally?
   Local is simpler but means operators maintain a user DB.

2. How do we surface multi-workspace queries? An LLM might want to
   search across the user's 3 workspaces in one call. Options:
   (a) explicit `workspace_ids: list[str]` param on `recall`,
   (b) a separate `federated_recall` tool,
   (c) no support in v3.0 (defer to v3.1).

3. Billing hooks: the REST API will need usage metering (tokens,
   storage, requests). Where does that live?

## Plan

| Step | Owner | Deliverable |
|---|---|---|
| Design doc review | STARGA | This file |
| Identity + storage layout | mind-mem | orgs/ tree, Actor dataclass |
| RBAC DB schema | mind-mem | `rbac.db` + `scope_check()` helper |
| Workspace + user MCP tools | mind-mem | 7 new tools |
| Migration CLI | mind-mem | `mm migrate v2-to-v3` |
| REST API (v3.1) | mind-mem | FastAPI wrapper around MCP |
| Docs | mind-mem | multi-tenancy.md, auth.md |

Estimated effort: **3 weeks for foundation + 2 weeks for REST API**.

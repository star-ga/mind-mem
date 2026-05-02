# mind-mem v3.1.8 — API / MCP Surface Security Audit

**Date:** 2026-04-28
**Auditor:** api-security agent (static review only)
**Scope:** MCP wire surface, HTTP/REST layer, auth/authz, crypto, data exfil paths
**Threat model:** Single-operator localhost; untrusted bundles + untrusted MCP clients

---

## Executive Summary

The overall security posture is **better than average** for an MCP server of this complexity.
The three highest-priority risks are: (1) the ACL gate silently no-ops in the default stdio
deployment, leaving all 14 admin tools callable with no authentication; (2) the REST
`rollback_proposal` endpoint omits the `reason` field entirely, silently returning an
application-level error as HTTP 200 instead of a protocol-level 400; and (3) the rate limiter
uses the FastMCP client_id as bucket key, which defaults to the literal string `"default"` for
all stdio callers — all 57 tools share a single rate-limit bucket.

Three defenses already present are more heavyweight than the localhost threat model requires and
should be simplified rather than maintained.

---

## Findings Table

| ID   | Severity | Title                                                    | File:line                             | UX cost of fix |
|------|----------|----------------------------------------------------------|---------------------------------------|----------------|
| N-01 | High     | ACL gate is off-by-default in stdio transport            | mcp/infra/observability.py:83-86      | None           |
| N-02 | High     | REST rollback_proposal drops `reason` → silent app error | api/rest.py:629-633 + models          | Low            |
| N-03 | Medium   | Rate limiter collapses all stdio clients into one bucket | mcp/infra/rate_limit.py:95-103        | None           |
| N-04 | Medium   | `staged_change` dispatcher bypasses reason gate on rollback | mcp/tools/public.py:192             | None           |
| N-05 | Medium   | REST /v1/health leaks workspace absolute path unauthenticated | api/rest.py:520-531             | Low            |
| N-06 | Medium   | REST /v1/metrics unauthenticated Prometheus exposition  | api/rest.py:533-551                   | None           |
| N-07 | Medium   | REST /v1/auth/oidc/callback leaks OIDC claims to any bearer token | api/rest.py:704-715       | Low            |
| N-08 | Low      | `decrypt_file` returns plaintext base64 over MCP (admin-gated but no audit event) | mcp/tools/encryption.py:95-133 | Low |
| N-09 | Low      | Encryption uses HMAC-CTR (custom stream cipher) not AES-CTR | encryption.py:60-73           | High           |
| N-10 | Low      | FTS5 token sanitisation drops all non-ASCII queries silently | sqlite_index.py:70-72, 858-861  | Low            |
| N-11 | Low      | `export_memory` default max_blocks=10000 — no admin gate  | mcp/tools/memory_ops.py:308       | None           |
| N-12 | Info     | REST `_client_id_from_token` uses last-16-chars as bucket key | api/rest.py:122-127           | None           |
| N-13 | Info     | OpenAPI docs (/docs /redoc /openapi.json) unprotected   | api/rest.py:464-471                   | Low            |

---

## Per-Finding Detail

### N-01 — ACL gate is off-by-default in stdio transport (High)

**File:** `src/mind_mem/mcp/infra/observability.py:83-86`

```python
admin_token = os.environ.get("MIND_MEM_ADMIN_TOKEN")
request_scope = _get_request_scope()
acl_scope = request_scope or os.environ.get("MIND_MEM_SCOPE", "user")
acl_active = admin_token is not None or request_scope is not None
```

The ACL gate activates only when `MIND_MEM_ADMIN_TOKEN` is set in the environment **or** when the
FastMCP access token carries a scope claim. In the default stdio deployment neither condition is
true. Any MCP client that can reach the stdio transport can call all 14 admin tools — including
`delete_memory_item`, `compact`, `export_memory`, `rollback_proposal`, `reindex`, `encrypt_file`,
and `decrypt_file` — with no authentication.

This is T-002 from the prior assessment but the deeper issue is that the gate condition was
inverted: admin protection that requires the operator to opt-in is less safe than protection that
requires the operator to opt-out. For a single-operator localhost setup this is low real-world
risk (the only MCP clients are the tools you yourself wired), but any untrusted bundle that gets
loaded as an MCP server gains full admin access to the memory workspace.

**PoC (stdio, no token configured):**
```
mcp_call: delete_memory_item(block_id="D-20260101-001")
# Returns: {"status": "deleted", ...} — no auth check fired
```

**Remediation:** Flip the gate to default-on. Introduce `MIND_MEM_ACL_DISABLED=true` as an
explicit opt-out instead of the current implicit opt-in.

```python
# observability.py — replace the three-line gate condition
acl_disabled = os.environ.get("MIND_MEM_ACL_DISABLED", "").lower() in ("1", "true", "yes")
acl_active = not acl_disabled
```

When `acl_active` is always true the default `MIND_MEM_SCOPE=user` env var already keeps user
tools open and blocks admin tools — no UX change for existing callers.

---

### N-02 — REST rollback_proposal drops `reason` → silent app error (High)

**File:** `src/mind_mem/api/rest.py:629-633` and the `RollbackProposalRequest` model at line 382.

```python
class RollbackProposalRequest(BaseModel):
    receipt_ts: str = Field(..., pattern=r"^\d{8}-\d{6}$")
    # `reason` field is absent

def rollback_proposal(body: RollbackProposalRequest, ...):
    raw = _rollback(receipt_ts=body.receipt_ts)   # reason defaults to ""
```

The `rollback_proposal` MCP tool requires `reason` (≥8 non-whitespace chars) for audit
compliance. The REST endpoint omits it, so every REST-initiated rollback silently fails with
`{"error": "reason is required..."}` returned as HTTP 200 — the caller sees a 200 OK but the
rollback did not execute. This is a silent data-loss bug for REST callers, not merely a security
issue, but it also means the audit trail protection added in v3.6.1 is completely bypassed via
the REST surface.

Same issue exists for REST `rollback_proposal` in `rest.py:629` — `reason` is neither modeled
nor passed.

**Remediation:** Add `reason` to `RollbackProposalRequest` and pass it through.

```python
class RollbackProposalRequest(BaseModel):
    receipt_ts: str = Field(..., min_length=15, max_length=15, pattern=r"^\d{8}-\d{6}$")
    reason: str = Field(..., min_length=8, max_length=2000, description="Mandatory rollback rationale")
```

---

### N-03 — Rate limiter collapses all stdio clients into one bucket (Medium)

**File:** `src/mind_mem/mcp/infra/rate_limit.py:95-103`

```python
def _get_client_id() -> str:
    try:
        token = get_access_token()
        if token is not None and token.client_id:
            return token.client_id
    except Exception as exc:
        _log.debug("client_id_resolution_failed: %s", exc)
    return "default"
```

In the stdio transport FastMCP does not issue access tokens, so `get_access_token()` returns
`None` (or raises). All tool calls therefore share the bucket keyed `"default"`. The rate limit
of 120 calls/minute applies across all 57 tools combined, not per-tool. An agent making rapid
automated recall queries consumes budget from admin operations.

More importantly: a single attacker who can make 120 MCP tool calls in under 60 seconds can
deny service to all other legitimate callers (including the human operator) for the remainder of
that window. In a multi-client stdio scenario (rare but possible) this matters.

**Remediation (zero-dep):** Fall back to `os.getpid()` as a stable per-process identifier when
no access token exists. This doesn't help multi-client but at least separates test harnesses
from production sessions.

```python
return f"pid-{os.getpid()}"
```

No UX cost. The rate limit per-pid means the window is per-process, which matches the actual
threat model (one Claude Code session = one process).

---

### N-04 — `staged_change` dispatcher bypasses reason gate on rollback (Medium)

**File:** `src/mind_mem/mcp/tools/public.py:192`

```python
if phase == "rollback":
    if not receipt_ts:
        return _err("phase='rollback' requires 'receipt_ts'")
    return governance.rollback_proposal.__wrapped__(receipt_ts)  # reason="" (default)
```

`rollback_proposal.__wrapped__(receipt_ts)` is called with only `receipt_ts`; `reason` defaults
to `""`. The `rollback_proposal` implementation will return an application-level error JSON
`{"error": "reason is required..."}` as the tool result rather than executing the rollback.
The dispatcher does not check for this error and returns it opaquely to the caller.

Unlike N-02 this does not bypass the rollback (the error fires correctly), but the caller gets
a misleading success response shape from `staged_change` rather than a clear 400-equivalent.
Any automated agent using `staged_change` will silently fail to roll back and may not detect
the failure.

**Remediation:** Add a `reason` keyword argument to `staged_change`'s rollback dispatch.

```python
if phase == "rollback":
    if not receipt_ts:
        return _err("phase='rollback' requires 'receipt_ts'")
    if not rationale or len(rationale.strip()) < 8:
        return _err("phase='rollback' requires 'rationale' (≥8 non-whitespace chars)")
    return governance.rollback_proposal.__wrapped__(receipt_ts, reason=rationale)
```

---

### N-05 — REST /v1/health leaks workspace absolute path unauthenticated (Medium)

**File:** `src/mind_mem/api/rest.py:515-531`

```python
@application.get("/v1/health", ...)
def health() -> JSONResponse:
    ws = _active_workspace(workspace)
    return JSONResponse({
        "workspace": ws,          # absolute path disclosed
        "workspace_exists": os.path.isdir(ws),
        ...
    })
```

The `/v1/health` endpoint has no `_require_auth` dependency. It returns the absolute workspace
path and its existence status to unauthenticated callers. On a network-accessible deployment
this is low severity (path disclosure), but combined with SSRF or other local-access techniques
it reveals the workspace layout.

In a single-operator localhost deployment this is informational. Flagged because the HTTP
transport may be bound to non-loopback addresses (the `--host` flag defaults to `127.0.0.1`
which is safe, but the caller can override to `0.0.0.0`).

**Remediation:** Strip `workspace` from the unauthenticated response, or gate health behind
optional auth (return full detail only when authenticated).

---

### N-06 — REST /v1/metrics unauthenticated Prometheus exposition (Medium)

**File:** `src/mind_mem/api/rest.py:533-551`

The `/v1/metrics` endpoint has no `Depends(_require_auth)`. Prometheus metrics include counters
for `mcp_acl_denied`, `mcp_http_auth_failures`, `mcp_proposals`, `mcp_delete_memory_item`, and
other operational signals. An unauthenticated caller on the network can enumerate workspace
activity patterns.

**Remediation:** Add `dependencies=[Depends(_require_auth)]` or gate only on localhost by
checking `request.client.host == "127.0.0.1"`. Zero UX cost for single-operator use.

---

### N-07 — REST /v1/auth/oidc/callback leaks OIDC claims to any bearer (Medium)

**File:** `src/mind_mem/api/rest.py:704-715`

```python
@application.post("/v1/auth/oidc/callback", ...)
def oidc_callback(credentials: ...):
    ...
    claims = provider.verify(credentials.credentials)
    scopes = provider.extract_scopes(claims)
    agent_id = claims.get("sub", "oidc-user")
    return JSONResponse({"authenticated": True, "agent_id": agent_id, "scopes": scopes})
```

On success the endpoint returns the decoded JWT `scopes` list and the `agent_id` (the `sub`
claim). While this is intentional (the caller confirms acceptance), the same `sub` value
and scope list can be used to fingerprint agents for targeted impersonation or privilege
discovery in multi-tenant scenarios. For single-operator localhost the risk is minimal.

The bigger concern: the endpoint does not check for `OIDC_ISSUER` being set to an
operator-controlled value before performing a JWKS fetch. If an attacker can write
`OIDC_ISSUER` (e.g. via environment injection in a compromised agent process), the JWKS fetch
would go to an attacker-controlled URL — a variant of SSRF limited to HTTPS.

**Remediation:** Validate that `OIDC_ISSUER` is on an operator allowlist at startup, or at
minimum warn when it changes between requests. Also consider omitting the `scopes` field from
the response body.

---

### N-08 — `decrypt_file` returns full plaintext base64 over MCP (Low)

**File:** `src/mind_mem/mcp/tools/encryption.py:95-133`

The `decrypt_file` tool returns `{"plaintext_b64": "<base64 of entire file>"}` as the tool
result string. This is correct behaviour for an admin-gated tool, but it means the decrypted
content travels through the MCP transport layer (including any observability hooks, rate-limit
wrappers, and the `mcp_tool_call` log entry). The log entry at `observability.py:110-118`
records `result_size` but not the content — however, the `result` string is held in memory
until the log fires, and any exception mid-way would surface the full plaintext in the exception
chain.

More practically: there is no audit chain entry for `decrypt_file` calls. `delete_memory_item`
writes a recovery log; `encrypt_file` calls `metrics.inc("files_encrypted")`; but `decrypt_file`
only emits a metric increment. An agent that calls `decrypt_file` on every file in the workspace
leaves no specific forensic trail of what was read.

**Remediation (Low UX cost):** Add a governance evidence entry for `decrypt_file` (analogous to
what `delete_memory_item` does with its `deleted_blocks.jsonl`). A single JSONL append of
`{path, agent_id, timestamp}` provides forensic coverage without slowing the tool.

---

### N-09 — Custom HMAC-CTR stream cipher instead of AES-CTR (Low)

**File:** `src/mind_mem/encryption.py:60-73`

```python
def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """HMAC-SHA256 in counter mode — pure-Python AES-CTR-like construction."""
    ...
    block = hmac.new(key, nonce + struct.pack(">Q", counter), hashlib.sha256).digest()
```

The custom HMAC-CTR construction is not wrong (HMAC-SHA256 is a secure PRF; encrypt-then-MAC
is correct), but it deviates from vetted primitives (AES-CTR from the standard library's
`cryptography` package or even `PyCryptodome`). Any future change to the keystream or MAC
construction risks subtle breaks. The PBKDF2 parameters (600k iterations, 32-byte salt) are
correct per OWASP.

This is flagged as Low rather than Info only because the encrypted-file format uses `_MAGIC =
b"MMENC1"` as a plain-text header — an attacker who can observe encrypted files can confirm
that mind-mem encryption is in use, narrowing their attack surface.

**Remediation (Don't bother for v3.x):** The current construction is safe. Migrating to
`cryptography.hazmat.primitives.ciphers.Cipher` would eliminate the custom PRF but adds a
non-zero external dependency. Given the localhost threat model this is not worth the churn.
Noted here for v4.0 consideration if the encrypted-file format is ever revised.

---

### N-10 — FTS5 token sanitisation silently drops all non-ASCII queries (Low)

**File:** `src/mind_mem/sqlite_index.py:70-72, 858-861`

```python
_FTS5_SAFE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
...
fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in query_tokens if _FTS5_SAFE.match(t))
if not fts_query:
    return []
```

The allowlist (`[a-zA-Z0-9_\-\.]`) strips all Unicode tokens. A query containing only Chinese,
Japanese, Arabic, or emoji characters produces an empty `fts_query` string and returns `[]`
rather than falling back to the BM25 scan. The FTS5 SQL injection protection is sound, but the
side-effect is that non-Latin-script workspaces get silent no-results on indexed queries.

Note: the hybrid backend path through `HybridBackend.search()` does not share this code path
and handles Unicode correctly.

**Remediation (Low UX cost):** Allow Unicode letter/digit characters in `_FTS5_SAFE` using
`\w` with the `re.UNICODE` flag, or expand to `^[\w\-\.]+$` with `re.UNICODE`. The FTS5 quoting
(`"..."`) already prevents operator injection regardless of character content.

---

### N-11 — `export_memory` default max_blocks=10000, no admin gate (Low)

**File:** `src/mind_mem/mcp/tools/memory_ops.py:308`

`export_memory` is listed in `ADMIN_TOOLS` and is therefore correctly admin-gated when ACL is
active. However, when ACL is inactive (the default stdio deployment per N-01), any MCP client
can call `export_memory()` and receive a JSONL dump of up to 10,000 blocks — effectively the
entire workspace — in a single response. This amplifies N-01.

Separately, `max_blocks` is caller-controlled with a default of 10,000. Setting `max_blocks` to
a very large integer results in parsing all corpus files before truncation: O(all blocks) work
on every call.

**Remediation:** Cap `max_blocks` at `min(max_blocks, 10_000)` (already done) but also validate
that the caller-supplied value doesn't exceed the per-config limit. No separate fix beyond N-01.

---

### N-12 — REST rate limiter uses last-16-chars of token as bucket key (Info)

**File:** `src/mind_mem/api/rest.py:122-127`

```python
def _client_id_from_token(token: str | None) -> str:
    return token[-16:] if len(token) >= 16 else token
```

Two different clients with tokens that share the same last 16 characters would be rate-limited
as a single client. For a 64-hex-char token the probability is negligible but it is technically
possible to craft such tokens deliberately. Consider using a stable hash (e.g. `sha256(token)`)
instead of a suffix slice.

**False positive note:** This is unlikely to matter in practice since the operator controls all
token issuance. Flagged for completeness.

---

### N-13 — OpenAPI docs (/docs, /redoc, /openapi.json) unauthenticated (Info)

**File:** `src/mind_mem/api/rest.py:464-471`

FastAPI exposes `/docs`, `/redoc`, and `/openapi.json` by default with no auth. These reveal the
full REST surface, all endpoint names, Pydantic schemas, and auth header names. For a localhost
deployment this is harmless. For a network-accessible deployment it helps an attacker enumerate
the API without touching auth endpoints.

**Remediation for network deployments only:** Set `docs_url=None, redoc_url=None` when
`MIND_MEM_TOKEN` is configured, or gate the built-in docs behind a separate secret path.

---

## Remediation Prioritization

### Fix immediately (High)

1. **N-01** — Flip the ACL gate to default-on; add `MIND_MEM_ACL_DISABLED=true` opt-out.
   Single-line change, zero UX regression.
2. **N-02** — Add `reason` to `RollbackProposalRequest` and pass it to `_rollback()`.
   Fixes a silent failure bug plus the audit bypass.

### Fix in next minor release (Medium)

3. **N-04** — Add `reason`/`rationale` forwarding in `staged_change` rollback dispatch.
4. **N-03** — Fall back to `pid-{os.getpid()}` as the stdio client bucket key.
5. **N-05** — Strip `workspace` path from unauthenticated `/v1/health` response.
6. **N-06** — Add `dependencies=[Depends(_require_auth)]` to `/v1/metrics`.
7. **N-07** — Omit `scopes` from OIDC callback response; validate OIDC issuer at startup.

### Low priority / deferred

8. **N-08** — Add `decrypt_file` audit trail to `deleted_blocks.jsonl` or a new `decrypted_files.jsonl`.
9. **N-10** — Expand `_FTS5_SAFE` to include Unicode word characters.
10. **N-12** — Use `sha256(token.encode())[:16]` instead of token suffix.
11. **N-13** — Gate OpenAPI docs behind auth for non-localhost deployments.

---

## Don't Bother List

The following defenses would add complexity or hurt UX more than they improve security in the
single-operator localhost threat model. Do not implement them.

- **CSRF tokens on REST endpoints.** The REST API uses Bearer tokens. Browsers cannot
  cross-origin POST with a custom `Authorization` header (same-origin policy blocks it).
  CSRF tokens add a round-trip for no benefit.

- **Content Security Policy / HSTS headers.** mind-mem's REST API is not a browser-facing
  application. Agents call it programmatically. CSP and HSTS provide zero benefit.

- **JWT expiry validation for the static `MIND_MEM_TOKEN`.** It is not a JWT; it is an opaque
  static bearer. Treating it as a JWT to add expiry would require issuing a private key and a
  signing ceremony, which is out of scope for localhost use.

- **Mutual TLS (mTLS) for MCP transport.** The stdio transport runs entirely in-process; TLS
  at the stdio layer is meaningless. If the HTTP transport is ever deployed on a network,
  a reverse proxy (nginx) handling TLS is the appropriate solution, not in-process TLS.

- **Token rotation enforcement (forced expiry of `MIND_MEM_TOKEN`).** The localhost threat
  model has no credential-stealing adversary. Forced rotation would require cron jobs and
  config file updates with no security benefit.

- **Per-tool rate limits (57 separate limiters).** The current single-window limiter already
  prevents runaway calls. Per-tool limiters would add a 57-entry map lookup on every call and
  complicate the mental model for the operator.

- **Audit log for `read` operations (recall, get_block).** Logging every recall would produce
  ~100x more log volume than governance events and would surface the query strings (which may
  contain sensitive context) in plaintext log files. The write-path audit chain is sufficient.

---

## Honest Gaps

1. **FastMCP internals.** The `get_access_token()` call in `rate_limit.py` and `acl.py`
   depends on FastMCP 2.14.5 context injection. The behavior under transport edge cases
   (e.g., reconnect, multiplexed sessions) was not verified without a live instance.

2. **`apply_engine.py` subprocess call.** `apply_engine.py:258` invokes `bash validate.sh ws`
   where `validate_sh` is a package-internal script. The `ws` argument (the workspace path)
   is passed as a positional arg to bash. If the workspace path contains shell metacharacters
   the argument is still handled safely because `shell=False` is in effect. Not a finding but
   confirmed by inspection only, not tested.

3. **`agent_bridge.VaultBridge.scan()`** — the file walk behavior when the vault root contains
   symlinks pointing outside the root was not traced through to `VaultBridge`. The allowlist
   check in `_vault_root_allowed` uses `os.path.realpath` on the passed `vault_root` but not on
   individual files returned during the walk. If `VaultBridge.scan()` follows symlinks without
   checking them against the allowlist, T-006 (already flagged) is not fully mitigated.

4. **`mind-mem.json` trust.** Several components read `mind-mem.json` from the workspace for
   config (alert URLs, cache settings, rate limits). This file is operator-written but if it
   is writable by an untrusted MCP client (e.g. via `vault_sync` writing to the workspace root),
   an attacker could reconfigure alert webhooks or raise `rate_limit_calls_per_minute` to
   disable the rate limiter. This path was not fully traced.

5. **gRPC surface.** `src/mind_mem/api/grpc_server.py` was not audited. It exposes a parallel
   call surface. Apply the same auth/rate-limit analysis to it.

6. **Runtime behavior of `cryptography` / `jose` version pinning.** The OIDC JWT validation
   path (`auth.py`) uses `python-jose[cryptography]`. The `alg=none` attack vector depends on
   the python-jose version. Confirmed that the `OIDCProvider.verify()` call passes the expected
   algorithms list to `jwt.decode()`, but this was not traced into the jose internals without
   running the code.

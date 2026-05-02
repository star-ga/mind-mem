# mind-mem API Security Review — 2026-04-28

**Scope:** MCP API surface — tool input schemas, ACL scope logic, HTTP transport, DoS/amplification paths, error leakage, webhook SSRF, audit tampering via API.
**Methodology:** OWASP API Security Top 10 (2023); ATT&CK mapping; static code review of the v3.2.0 decomposed module tree.
**Prior art:** STRIDE threat model `threat-model-2026-04-28.md` (T-001 through T-011) reviewed; findings below are additive, non-duplicated.

**Verdict (as-is):** Not defensible even for single-user localhost at default config. Four independent Tier 1 paths exist.
**Verdict (after all Tier 1 fixes — new + T-002/T-003/T-006):** Defensible for single-user stdio. HTTP transport stays risky until A-003 is addressed. All Tier 2 items are real but unexploitable without a compromised/malicious agent.

---

## New Findings

---

### A-001: MIND_MEM_SCOPE env var fully bypasses the ACL gate on stdio
**File:** `mcp/infra/observability.py:83-88`
**Tier: 1**
**OWASP API:** API5:2023 Broken Function Level Authorization
**ATT&CK:** T1548 Abuse Elevation Control Mechanism

The ACL scope for stdio requests is resolved at line 85:
```python
acl_scope = request_scope or os.environ.get("MIND_MEM_SCOPE", "user")
```
`request_scope` is `None` for stdio (no FastMCP access token). Therefore `acl_scope` is read directly from the process environment. `acl_active` at line 86 is `True` whenever `MIND_MEM_ADMIN_TOKEN` is set, which is exactly what T-002's fix mandates. After T-002 lands, any subprocess or shell that has `MIND_MEM_SCOPE=admin` in its environment inherits full admin scope with no token challenge. This includes: a malicious MCP client spawned from a shell with that var set, a compromised shell script wrapping the server, or a process that calls `os.putenv("MIND_MEM_SCOPE", "admin")` in the same Python process.

The hint text in `acl.py:113` documents this as the designed mechanism, which is fine for explicit single-user config, but it means T-002's token requirement is defeatable without knowing the token.

**Minimal fix:** At startup in `main()`, when `MIND_MEM_ADMIN_TOKEN` is set, emit a `CRITICAL` log warning if `MIND_MEM_SCOPE=admin` is also present in the environment, and clear it: `os.environ.pop("MIND_MEM_SCOPE", None)`. This preserves single-user convenience (explicit `MIND_MEM_SCOPE=admin` with no admin token still works as before) while preventing the token-bypass once a token is configured. One line + one log statement.

---

### A-002: `graph_add_edge` is a persistent-write tool in USER_TOOLS
**File:** `mcp/infra/acl.py:54`; `mcp/tools/graph.py:31-82`
**Tier: 1**
**OWASP API:** API5:2023 Broken Function Level Authorization / API3:2023 Broken Object Property Level Authorization
**ATT&CK:** T1565.001 Stored Data Manipulation

`graph_add_edge` writes a typed, persistent edge into `knowledge_graph.db`. It is listed in `USER_TOOLS` (acl.py:54), so any authenticated user token can call it. The `source_block_id` parameter is accepted without verification that the caller owns or created that block — a user-scope agent can attach KG edges to any block ID including blocks created by other sessions.

The KG feeds `vault_sync`'s `include_links` path (agent.py:168) and all `traverse_graph`/impact-analysis paths. Poisoned edges shape future impact analyses and `vault_sync` outputs without any governance gate. Unlike `propose_update`, graph writes are immediate and irreversible via normal user-scope tools.

**Minimal fix:** Move `graph_add_edge` from `USER_TOOLS` to `ADMIN_TOOLS` in `acl.py`. Single-character change to the frozenset. Agents that need to write graph edges must hold admin scope — consistent with the principle that persistent, governance-bypassing writes require elevation.

---

### A-003: HTTP transport binds to all interfaces; no fail-secure on missing auth
**File:** `mcp/server.py:177-191`
**Tier: 1**
**OWASP API:** API8:2023 Security Misconfiguration / API2:2023 Broken Authentication
**ATT&CK:** T1190 Exploit Public-Facing Application

`mcp.run(transport="sse", port=args.port)` at line 189 passes no `host` argument. FastMCP/Starlette defaults to `0.0.0.0`, making the server reachable from the local network or any Docker bridge.

When no tokens are configured (`auth_tokens` empty at line 179), the code emits a warning and continues rather than refusing to start. On the LAN-accessible interface this results in a fully unauthenticated MCP surface. The ACL gate is also disabled (because `admin_token is None` and `request_scope is None`, making `acl_active = False`), so all 57 tools including `delete_memory_item`, `export_memory`, and `decrypt_file` are executable by any host on the network.

There is no explicit CORS middleware or `Origin` header check in the code path. Browser-originated cross-site `fetch()` requests to the SSE endpoint may succeed depending on FastMCP's version-specific CORS defaults.

**Minimal fix:** Two changes in `server.py`:
1. Add `--host` argument defaulting to `127.0.0.1` and pass it to `mcp.run()`.
2. At startup, if `args.transport == "http"` and `not _check_token() and not os.environ.get("MIND_MEM_ADMIN_TOKEN")`: call `raise SystemExit(...)` rather than warn-and-continue.

---

### A-004: `reject_proposal` omits proposal_id format validation present in `approve_apply`
**File:** `mcp/tools/governance.py:334`
**Tier: 2**
**OWASP API:** API3:2023 Broken Object Property Level Authorization
**ATT&CK:** T1565.001 Stored Data Manipulation

`approve_apply` validates proposal_id with `re.match(r"^P-\d{8}-\d{3}$", ...)` (governance.py:246). `reject_proposal` only checks `not proposal_id or not proposal_id.strip()` (governance.py:334). The `reason` parameter is passed to `_mark_proposal_status` which appends it verbatim to the proposal file. If `reason` contains unescaped markdown, this is an injection path into the proposal file — the same root cause as T-003 but at the rejection write boundary, which the STRIDE pass did not inspect.

**Minimal fix:** One line — add `if not re.match(r"^P-\d{8}-\d{3}$", proposal_id):` guard mirroring `approve_apply`. Apply `_sanitize_reason_for_markdown` to `reason` before the `_mark_proposal_status` write (same fix pattern as T-003).

---

### A-005: `alerting.py` WebhookSink accepts localhost/RFC-1918/IMDS as webhook destinations
**File:** `alerting.py:141-143`
**Tier: 2**
**OWASP API:** API7:2023 Server Side Request Forgery
**ATT&CK:** T1071.001 Application Layer Protocol

`WebhookSink.__init__` validates only the URL scheme:
```python
if not url.startswith(("http://", "https://")):
    raise ValueError(...)
```
No host or port validation. `http://169.254.169.254/latest/meta-data/` (AWS IMDSv1), `http://127.0.0.1:8765/` (self-SSRF back into MCP), and `http://10.0.0.1/admin` (RFC-1918) all pass the check. The POST body includes `alert.workspace` (absolute host path), `alert.payload` (block excerpt content), and `alert.event`. If `mind-mem.json` is tampered by a compromised agent, governance events automatically exfiltrate workspace content and host paths to an attacker-controlled or IMDS endpoint.

`SlackSink.__init__` only warns (not rejects) for non-`hooks.slack.com` URLs (line 183).

This confirms T-004 with the specific SSRF mechanism identified. T-004 recommended a URL allowlist; this finding specifies the denylist classes needed.

**Minimal fix:** Add a 15-line `_reject_ssrf_target(url)` helper in `alerting.py` called from both `WebhookSink.__init__` and `SlackSink.__init__`. Resolves the hostname and rejects: loopback (127.0.0.0/8, ::1), link-local (169.254.0.0/16), and RFC-1918 (10/8, 172.16/12, 192.168/16). Does not require an allowlist — denylist is sufficient and does not affect legitimate Slack/PagerDuty endpoints.

---

### A-006: `export_memory` max_blocks has no server-side upper bound — amplification + bulk exfiltration
**File:** `mcp/tools/memory_ops.py:308-368`
**Tier: 2**
**OWASP API:** API4:2023 Unrestricted Resource Consumption
**ATT&CK:** T1530 Data from Local System (bulk extraction)

`export_memory(max_blocks: int = 10000)` accepts any integer from the caller. The function builds the entire block list in memory before truncating (lines 323-349). A caller passing `max_blocks=2_000_000_000` causes Python to load all corpus files into a list without any page-limit before the slice. On a large workspace this can exhaust RAM and block the event loop for the duration.

`export_memory` is correctly in `ADMIN_TOOLS` (acl.py:36), limiting this to admin-scope callers. However, a compromised or prompt-injected admin agent is the exact threat model for this product — bulk-export with no anomaly signal is a silent data drain.

**Minimal fix:** Two lines:
```python
_MAX_EXPORT_BLOCKS = 50_000
max_blocks = min(max(1, max_blocks), _MAX_EXPORT_BLOCKS)
```
Add an alerting event via `get_alert_router(ws).fire(severity="warning", event="bulk_export", payload={"block_count": len(all_blocks)})` when `block_count > 500`. Does not degrade normal admin export use.

---

### A-007: `compact` tool can age-off audit logs without integrity exclusion
**File:** `mcp/tools/memory_ops.py:651`
**Tier: 3**
**OWASP API:** API5:2023 Broken Function Level Authorization (scope boundary gap)
**ATT&CK:** T1070.002 Clear Linux or Mac System Logs

`compact` calls `cleanup_daily_logs(ws, days=180, dry_run=dry_run)`. There is no explicit exclusion of `audit.log` or the `audit/` directory from this sweep. T-007 identifies that the audit chain has no OS-level append-only flag; A-007 adds that a legitimate admin call to `compact(dry_run=False)` with the default `180`-day window can remove audit entries — and a caller can pass `archive_days=1` to accelerate this. No alert is fired on log removal.

Since `compact` requires admin scope this is Tier 3 — already-privileged access is required. But the combination of T-007 (no `chattr +a`) and A-007 (compact sweeps logs) means audit log erasure requires only one compromised admin session and one tool call.

**Minimal fix:** One guard in `cleanup_daily_logs`: skip any file path containing `audit`. Does not affect compaction of block archives, snapshots, or signals.

---

## Confirmed by Second Look

**T-002 (ACL gate off by default):** Confirmed with line-level evidence. `observability.py:86`: `acl_active = admin_token is not None or request_scope is not None`. When both are `None` (default stdio, no tokens), the entire `if acl_active` block (lines 87-95) is skipped. All tools including admin tools execute unconditionally.

**T-006 (unbounded vault_scan):** Confirmed. `agent.py:37-41` — `_vault_allowlist()` returns `[]` on unset `MIND_MEM_VAULT_ALLOWLIST`. `_vault_root_allowed()` returns `(True, "")` on empty allowlist regardless of the path passed. `vault_scan` and `vault_sync` are in `USER_TOOLS`, so no admin elevation is needed.

---

## Unsurfaced Gaps

1. **`public.py` consolidated dispatchers** (`mcp/tools/public.py`) — registered last in `server.py:110`, shadows the `recall` name, and adds `staged_change`, `memory_verify`, `graph`, `core`, `kernels`, `compiled_truth` dispatchers. Finding 2 in the prior report (staged_change silently drops `reason`) came from this file. The ACL re-check behavior for each dispatch `mode` was not fully traced. If any mode routes to an admin tool without re-entering `mcp_tool_observe`, it is a BFLA bypass. Recommend a focused review before v3.2.0 ships.

2. **`VaultBridge.write()` path traversal within allowlisted root** — `vault_sync` validates that `vault_root` is allowlisted, but `relative_path` is passed to `VaultBridge.write()` without a secondary `os.path.realpath` containment check visible in the MCP layer. If `VaultBridge.write()` constructs the final path as `os.path.join(vault_root, relative_path)` without resolving symlinks, a `relative_path` of `../../outside-vault/file.md` escapes the allowlisted root. Could not confirm without reading `agent_bridge.py`.

3. **FastMCP version-specific CORS posture** — CORS behavior on the SSE transport depends on the installed FastMCP/Starlette version. Empirical test needed: `curl -H "Origin: https://evil.example.com" http://127.0.0.1:8765/sse` — if `Access-Control-Allow-Origin: *` is returned, the DNS-rebinding attack surface is real for any machine on the LAN.

4. **`online_trainer.py` (carried from T-009)** — Not reviewed in this pass. If trainer fetches from URLs derived from proposal content, it is a second SSRF vector orthogonal to A-005.

---

## Friction-Tax Check

None of the seven recommendations above touch `recall`, `hybrid_search`, `find_similar`, `intent_classify`, `prefetch`, `pack_recall_budget`, or `retrieval_diagnostics`. The hot recall path is unaffected.

The one user-visible friction item: `graph_add_edge` promotion to admin scope (A-002) requires agents that currently write graph edges under user tokens to be updated. This is an agent-configuration change, not a latency change, and is the correct tradeoff given that persistent KG writes are semantically equivalent to governance events.

---

## Remediation Summary

| ID | File:Line | Tier | Effort |
|----|-----------|------|--------|
| A-001 | `mcp/infra/observability.py:85` | 1 | Warn + clear `MIND_MEM_SCOPE=admin` when `MIND_MEM_ADMIN_TOKEN` is set |
| A-002 | `mcp/infra/acl.py:54` | 1 | Move `graph_add_edge` to `ADMIN_TOOLS` |
| A-003 | `mcp/server.py:189` | 1 | Default `--host 127.0.0.1`; fail-secure on HTTP with no tokens |
| A-004 | `mcp/tools/governance.py:334` | 2 | Add `P-\d{8}-\d{3}` regex gate to `reject_proposal`; sanitize `reason` |
| A-005 | `alerting.py:141` | 2 | SSRF denylist (loopback, link-local, RFC-1918) in WebhookSink + SlackSink |
| A-006 | `mcp/tools/memory_ops.py:308` | 2 | Clamp `max_blocks` to 50,000; alert on exports over 500 blocks |
| A-007 | `mcp/tools/memory_ops.py:651` | 3 | Exclude audit files from `cleanup_daily_logs` sweep |

---

**Report metadata**
- Generated: 2026-04-28
- Methodology: OWASP API Top 10 (2023) + ATT&CK
- Agent: `api-security-reviewer` (STARGA Inc.)
- Pre-read: `security/threat-model-2026-04-28.md` (T-001 through T-011)
- Next review: after v3.2.0 Tier 1 fixes land; review `public.py` dispatchers and `VaultBridge.write()` path containment.

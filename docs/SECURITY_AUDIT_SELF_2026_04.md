# mind-mem v3.2.0 — Self-Audit Plan (Post-Release Deliverable)

**Status:** deferred to v3.2.1 per release gating decision 2026-04-20.
**Author:** STARGA, Inc.
**Scope:** the v3.2.0 release candidate tagged at commit `f0d3bdb` (plus any hotfixes).

This document is the **plan**, not the audit itself. The audit runs
in the week after the v3.2.0 PyPI release. The baseline scanner
outputs at `docs/security-baselines/bandit-v3.2.0-baseline.json`
are the "as-released" reference; post-release runs compare against
this snapshot to isolate regressions from pre-existing issues.

---

## 1. Release-gate summary

v3.2.0 ships with these security artefacts **already in-tree**:

| Artefact                                      | Status                           |
| --------------------------------------------- | -------------------------------- |
| `SECURITY.md` — disclosure policy             | v3.2.x supported (bumped 2026-04-20) |
| `docs/security-audit-sow.md` — external-audit SoW | Published                      |
| `SECURITY_AUDIT_2026-04.md` — internal audit  | 3 findings, all fixed (v2 addendum in commit `edc15e4` shipped 3 more) |
| `.github/workflows/security.yml`              | CodeQL + bandit + pip-audit + gitleaks + trivy + SBOM |
| `.github/workflows/release.yml`               | Sigstore-signed wheels + tarballs, SBOM attached |
| `.pre-commit-config.yaml`                     | ruff + bandit + detect-secrets   |
| `docs/supply-chain-security.md`               | cosign-verify snippets           |
| Token hardening                               | 4096-byte cap, 32-char min       |
| Query-length cap                              | 8192-char cap on recall          |
| Schema-name allowlist (Postgres DDL)          | `[A-Za-z_][A-Za-z0-9_$]{0,62}` regex |
| Docker compose secrets                        | Required env-var substitution    |

**Ship status:** the release **is** publishable. The self-audit
below is for the v3.2.x service lifetime, not a v3.2.0 tagging
gate.

---

## 2. STRIDE threat model — mind-mem v3.2.0

### 2.1 Actors

| Principal       | Capability                                                 |
| --------------- | ---------------------------------------------------------- |
| **Operator**    | Runs the server, holds `MIND_MEM_ADMIN_TOKEN`. Trusted.    |
| **Agent user**  | Holds `MIND_MEM_TOKEN` OR a `mmk_*` API key. Scoped.       |
| **LLM**         | Untrusted. Inputs arrive through MCP / REST from operator or agent. |
| **Network**     | TLS-trusted when HTTP transport is used. Assumed hostile.  |

### 2.2 STRIDE grid

| Surface             | Spoofing   | Tampering  | Repudiation | InfoDisc    | DoS        | ElevPriv   |
| ------------------- | ---------- | ---------- | ----------- | ----------- | ---------- | ---------- |
| MCP stdio           | N/A        | Proposal forgery (audit chain) | Chain covers | — | JSON bomb | ACL escape |
| MCP HTTP            | Token replay | Same + TLS | Same       | Logs / metrics | Rate limit | Admin path |
| REST                | JWT replay | Same       | Chain + agent_id | Response body | Request body size | Scope escalation |
| SQLite index        | —          | WAL tamper | File perms  | File perms  | DB lock    | —          |
| Postgres            | DSN leak   | SQL inject | DB audit    | Row-level sec | Pool exhaust | GRANT abuse |
| Snapshots           | Manifest tamper | Content tamper | `agent_id` in receipt | Snapshot read | Snapshot dir size | —        |
| Installer           | HTTPS MITM | Script replacement | Install log | — | Disk fill  | root install |
| CI                  | OIDC federation | Workflow tamper | Actions log | Secrets exfil | Runner DoS | Write to main |

### 2.3 Mitigations already in-tree vs gaps

**Mitigated:**

- Spoofing — constant-time token compare, OIDC iss/aud/exp checks,
  per-agent API keys with SHA-256 storage.
- Tampering — SHA3-512 governance chain, Merkle proofs, SQLCipher
  (opt-in), schema-name allowlist.
- Repudiation — `agent_id` attribution in every audit record.
- DoS — sliding-window rate limit, token/query/path-length caps,
  snapshot-size implicit cap via `SNAPSHOT_EXCLUDE_DIRS`.
- Elevation — ACL gate on every MCP tool, `_require_admin` on
  REST admin endpoints.

**Open gaps flagged for the post-release audit:**

1. OIDC is not wired into `_require_admin` (arch review §5 — HIGH).
2. REST workspace scoping via `os.environ` mutation (arch review
   §3 — CRITICAL for multi-worker deployments).
3. Apply engine bypasses BlockStore for op execution (arch review
   §2 — CRITICAL for Postgres backend).
4. JWKS cache has no TTL (sec review INFO-1).
5. Go SDK query params not URL-encoded (sec review LOW-1).
6. One CycloneDX GHA action pinned to `@v2` tag instead of SHA
   (sec review INFO-2).

---

## 3. Phase 1 — Automated scans (ran pre-release, baselined)

### 3.1 Bandit
**Command:** `bandit -r src/mind_mem -ll -f json`
**Result:** `docs/security-baselines/bandit-v3.2.0-baseline.json`
**Counts:** LOC 43,291 | HIGH 0 | MEDIUM 25 | LOW 56

MEDIUMs are predominantly B608 (hardcoded SQL) false positives on
parameterised queries, B404 (subprocess use) on the internal
`exec` bridge, and B110 (try/except/pass) in defensive paths where
the exception is truly ignorable. Post-release audit will triage
and either suppress with `# nosec` (with reason) or fix.

### 3.2 pip-audit
**Command:** `pip-audit` against the installed venv.
**Result:** `No known vulnerabilities found` at release.
**Recurring gate:** CI runs this weekly (`.github/workflows/security.yml`).

### 3.3 Scanners still to run (post-release)
- Semgrep (pending — not installed in this session).
- OSV-scanner on the lockfile (pending).
- Atheris fuzz harness on ingest + recall (new — writes to `tests/fuzz/`).

---

## 4. Phase 2 — Manual review (ran pre-release)

The v2 security-reviewer agent sweep (committed at `7fa80fd` +
`edc15e4`) covered:

- OIDC JWT handling (RS256 only; HS256+none excluded; exp/iss/aud all enforced).
- API key storage (SHA-256 only, constant-time compare, `secrets.token_hex(32)`).
- PostgresBlockStore SQL parameterisation + schema allowlist.
- Redis cache poisoning surface (transparent; no server-side side channels).
- REST admin endpoint authz (fixed HIGH-1 coverage hole).
- Installer `set -euo pipefail` + TLS fetch assumptions.
- CI workflow secrets scope + OIDC federation + sigstore pinning.
- SDK client token handling + retry-after parsing + timeout enforcement.

**Result:** 2 HIGH + 1 MEDIUM fixed in `edc15e4`. 1 LOW + 2 INFO
open.

## 5. Phase 3 — Manual review (pending post-release)

The surfaces below are on the post-release agenda:

1. 57 MCP tool auth boundaries — per-tool trace of the rate-limit
   + ACL + scope check. Already green in unit tests but a fresh
   pair-review pass is valuable.
2. SQLCipher configuration — KDF iteration count, cipher mode,
   key-file permissions, backup/restore key rotation.
3. Merkle preimage construction — byte-for-byte repro from a fresh
   workspace to catch any `bytes` vs `str` corner.
4. Hook installer path traversal — every `os.path.join` in
   `hook_installer/`.
5. Webhook SSRF allowlist — all outbound HTTP calls from the
   server routed through the allowlist.
6. REST + OIDC end-to-end flow against real Okta / Auth0 / Google /
   Azure AD test tenants.

## 6. Phase 4 — Triage + remediation

Cadence: post-release week 1 produces the audit report + fix PRs.
Anything CRITICAL/HIGH lands in v3.2.1 hotfix. MEDIUM+LOW batch
into v3.3.0.

## 7. Phase 5 — Public report

**Deliverable:** a public post-audit summary at
`docs/security-audit-v3.2.x.md` once the audit completes, linked
from `SECURITY.md`. All CRITICAL/HIGH fixes shipped before the
report is published.

## 8. Timeline

| Phase                           | Window                  |
| ------------------------------- | ----------------------- |
| Release v3.2.0                  | 2026-04-20 (today)      |
| Phase 1 (automated scans)       | W+1 (2026-04-21..27)    |
| Phase 2 + 3 (manual review)     | W+2..W+3                |
| Phase 4 (fixes shipped v3.2.1)  | W+3..W+4                |
| Phase 5 (public report)         | W+4 (2026-05-18)        |

## 9. Re-audit cadence

After v3.2.x:
- Every minor release (v3.3, v3.4, ...) re-runs Phases 1-2.
- Every major release (v4.0) triggers a full external audit via
  the SoW in `docs/security-audit-sow.md`.

---

## 10. What would block a v3.2.0 RC → GA re-tag

If post-release triage surfaces a CRITICAL that's exploitable in
the wild, the response is:

1. File a GHSA advisory (private).
2. Ship a v3.2.1 hotfix within 72 hours.
3. 90-day public disclosure per `SECURITY.md`.

Nothing in the pre-release review pointed at such an issue. The
2 architectural CRITICALs (apply-engine-through-BlockStore, REST
workspace scoping) are roadmap items for v3.2.1 — they would only
be user-impacting on Postgres + multi-worker deployments that are
documented as "beta" in the release notes.

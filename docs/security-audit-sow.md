# mind-mem — External Security Audit Statement of Work (SoW)

**Version:** 1.0
**Date:** 2026-04-20
**Author:** STARGA, Inc.
**Contact:** security@star.ga
**Target codebase:** [star-ga/mind-mem](https://github.com/star-ga/mind-mem), v3.2.0 release candidate
**Budget:** request-for-quote (see §8)
**Timeline:** 4–6 weeks from kickoff to final report

This document describes the scope of work for a third-party security
audit of mind-mem, a persistent memory layer for AI agents. It is
intended to give qualified auditing firms the context they need to
scope an engagement without first requesting NDA-gated access to the
codebase — everything auditable is already public under Apache-2.0.

---

## 1. System under test

mind-mem is a file-first persistent memory system used by AI coding
agents (Claude Code, Claude Desktop, Codex, Gemini CLI, Cursor,
Windsurf, OpenClaw, etc.). It exposes:

- **57-tool MCP server** over stdio + optional HTTP with bearer token
  auth (`FastMCP` + `StaticTokenVerifier`).
- **CLI** (`mm`) for workspace lifecycle.
- **Python SDK** + **JS/TS SDK** (v3.2.0) + forthcoming **Go SDK**.
- **REST API** (v3.2.0, FastAPI; OIDC/JWT auth).

Persistence layers under audit:

| Surface                           | Implementation                                                                |
| --------------------------------- | ----------------------------------------------------------------------------- |
| Markdown corpus                   | `decisions/*.md`, `tasks/*.md`, `entities/*.md`, `intelligence/*.md`          |
| Block index                       | SQLite (FTS5 + optional sqlite-vec)                                           |
| Knowledge graph                   | SQLite with typed edges                                                       |
| Governance chain                  | Append-only SHA3-512 hash chain + evidence objects                            |
| Snapshots / rollback              | `intelligence/applied/<ts>/` trees + MANIFEST.json + SHA-256 content hashes   |
| At-rest encryption (opt-in v3.0+) | SQLCipher (AES-256) for indexes, ChaCha20-Poly1305 for block content files    |
| Postgres backend (v3.2.0, opt-in) | `blocks` / `snapshots` / `snapshot_blocks` / `workspace_lock` tables + LISTEN/NOTIFY |

Releases are published to PyPI with OIDC trusted publishing +
sigstore signing (SLSA provenance, v3.2.0+).

---

## 2. Threat model

Assume a three-actor world:

| Principal         | Capability                                                                   |
| ----------------- | ---------------------------------------------------------------------------- |
| **Operator**      | Runs the mind-mem server; holds admin token. Trusted baseline.               |
| **Agent user**    | Non-admin user or external client (REST / MCP) with user-scope token.        |
| **Untrusted input** | Block content from user prompts, vault files, external LLM outputs, MCP proposals. |

Out-of-scope assumption: the **LLM itself is adversarial and may
craft inputs to exfiltrate, corrupt, or bypass governance**. A
malicious LLM must not be able to:

- Escape the workspace sandbox to write outside it.
- Forge proposals that apply without a human-gated review.
- Corrupt the hash chain undetectably.
- Leak admin-scope tools via a user token.
- Exhaust the rate-limited HTTP surface enough to deny service.

---

## 3. Scope — in

The audit covers every path an LLM-generated input can take to reach
durable state or network egress.

### 3.1 Authentication + authorization

- Bearer-token auth (`Authorization: Bearer`, `X-MindMem-Token`).
- `hmac.compare_digest` usage for constant-time comparison.
- `MIND_MEM_ADMIN_TOKEN` vs `MIND_MEM_TOKEN` scope separation.
- Per-tool ACL enforcement
  (`src/mind_mem/mcp/infra/acl.py`).
- FastMCP `StaticTokenVerifier` token metadata.
- Rate-limit bypass via reconnection / client-id rotation
  (`src/mind_mem/mcp/infra/rate_limit.py`).
- OIDC/JWT flow end-to-end (v3.2.0 REST API, when present).

### 3.2 Input validation + injection

- **Path traversal** — every user-supplied path validated against
  workspace root (`_safe_resolve`, `_validate_path`,
  `_safe_vault_path`, `_vault_root_allowed`).
- **SQL injection** — every `cursor.execute(...)` call. Audit for
  f-string / `%`-format / concatenated queries.
- **Command injection** — `subprocess.*` call sites, shell=True usage,
  argument escaping in scripts (`install.sh`, `install-bootstrap.sh`,
  `scripts/**`).
- **YAML deserialization** — `yaml.load` vs `yaml.safe_load` (block
  frontmatter, config).
- **JSON bombs / regex DoS / zip bombs** on MCP tool inputs.
- **Block-header injection** in block serializer
  (`src/mind_mem/block_store.py::_render_block` — newline + `[`
  bigram handling).

### 3.3 Cryptographic correctness

- **SHA3-512 hash chain** — tamper resistance, preimage ordering,
  TAG_v1 NUL-separated composition (v2.10.0+), Q16.16 fixed-point
  score canonicalization.
- **Merkle tree** — proof verification, leaf ordering, root
  reproducibility (`src/mind_mem/merkle_tree.py`).
- **SQLCipher** configuration — KDF iterations, cipher mode,
  key derivation from `MIND_MEM_ENCRYPTION_PASSPHRASE`.
- **ChaCha20-Poly1305 at-rest** — nonce handling, key rotation,
  integrity tag verification.
- **Evidence chain** — `EvidenceObject` serialization determinism,
  cross-version compatibility.

### 3.4 Storage atomicity

- **Snapshot scope** — `SNAPSHOT_DIRS` / `SNAPSHOT_EXCLUDE_DIRS`
  correctness post-v3.2.0 §2.2 refactor
  (`src/mind_mem/corpus_registry.py`).
- **Rollback integrity** — manifest-based restore, orphan cleanup,
  `_cleanup_orphans_from_manifest` exclusion honoring.
- **File lock correctness** (`FileLock`) — held across fork? released
  on crash? double-apply prevention.
- **TOCTOU** — every `os.path.exists` + `open` pair.
- **Windows file locking** — SQLite handle retention during rollback
  (`apply_engine.py::_cleanup_orphan_files` deferred-cleanup path).

### 3.5 Transport + network

- **MCP stdio transport** — stdin/stdout handling under malformed
  JSON-RPC.
- **HTTP transport** — TLS termination assumptions, CORS, CSRF on any
  state-changing endpoints.
- **SSE / WebSocket** (FastMCP) — connection lifecycle, timeout
  handling.

### 3.6 Supply chain

- `pyproject.toml` dependency pins — transitive CVEs.
- Docker image base (`python:3.12-slim`) + installed layers.
- `install-bootstrap.sh` — `curl | bash` surface; absence of checksum
  verification; TLS-only fetch.
- GitHub Actions workflows (`.github/workflows/**`) — secrets in
  env, artifact uploads, OIDC federation config.
- PyPI publishing via OIDC trusted publishing (environment `pypi`).

### 3.7 Operational surfaces

- Log lines — no secret leakage (tokens, passphrases, workspace
  secrets) in structured logs.
- Error envelopes — no stack traces or file paths reaching untrusted
  clients.
- Prometheus metrics endpoint — auth required? any cardinality
  explosion via user-supplied labels?
- OpenTelemetry spans — no PII / secret attributes.

---

## 4. Scope — out

The following are *not* in scope for this engagement (either covered
separately or by deliberate threat-model choice):

- The **LLM itself**. Prompt injection mitigation is the caller's
  responsibility; mind-mem's governance layer contains the blast
  radius but does not try to detect malicious prompts.
- **Physical access** to the host running mind-mem.
- **Cloud provider** security (RDS, S3 backups) — mind-mem is
  local-first; cloud deployments are a user choice.
- **Ollama** / model weights. mind-mem talks to Ollama over an
  HTTP-local socket; auditing Ollama is Ollama's concern.
- **Third-party MCP clients** (Claude Desktop, Cursor, Windsurf,
  etc.) — audit the mind-mem-side protocol, not the client-side
  implementations.

---

## 5. Deliverables expected from the auditor

1. **Written report** (Markdown or PDF) covering:
   - Executive summary (1 page, non-technical audience).
   - Methodology — tools used (Semgrep, CodeQL, manual review,
     fuzzing, …), time allocation, expertise credentials.
   - **Findings**, each with:
     - Title, severity (CVSS 3.1 + CWE), affected file:line.
     - Reproduction steps.
     - Exploit scenario.
     - Recommended fix.
     - Commit-ready patch if feasible.
   - Positive observations (things mind-mem is doing well — useful
     for release notes).
   - Audit trail — files reviewed, tools run, percentage of
     surface covered.
2. **SARIF** file of all findings for GitHub Code Scanning upload.
3. **Remediation validation** — auditor re-tests after STARGA ships
   fixes; issues a "fixes verified" addendum.
4. **Public attestation letter** (optional, at auditor's discretion)
   suitable for inclusion in mind-mem release notes and PyPI README.

---

## 6. Audit-firm expectations

Qualified firms:

- Have published work on Python / database / cryptographic-governance
  systems.
- Can execute dynamic testing (running the MCP server, fuzzing HTTP
  endpoints, poking the hash chain).
- Maintain reproducible reports — every finding must have a precise
  repro recipe, not just "this looks bad."
- Commit to **90-day coordinated disclosure** on any findings STARGA
  cannot fix within the audit window.

Firms we welcome quotes from (non-exhaustive): Trail of Bits,
NCC Group, Latacora, Doyensec, Include Security, X41 D-Sec,
Gotham Digital Science. Firms we will not engage: any firm with an
active conflict against the Apache Foundation, STARGA itself, or the
Claude / Anthropic ecosystem.

---

## 7. Timeline

| Week | Activity                                                         |
| ---- | ---------------------------------------------------------------- |
| 0    | Kickoff call, NDA if required, repo read-through                 |
| 1–2  | Static analysis + manual review, internal report draft           |
| 3    | Dynamic testing — run server, fuzz MCP + HTTP, test encryption   |
| 4    | Draft report + remediation-ready patches                         |
| 5    | STARGA fixes CRITICAL/HIGH findings                              |
| 6    | Auditor re-test + final report + public attestation (if granted) |

Fixed-fee preferred; time-and-materials considered for scope
extensions. We do not use bug bounties to substitute for a paid
audit.

---

## 8. Known prior findings (good-faith disclosure)

Findings already resolved that new auditors should be aware of:

| Year      | Finding                                                   | Disposition                  |
| --------- | --------------------------------------------------------- | ---------------------------- |
| 2026-02   | Mock/prod divergence in integration tests                 | Fixed (real-DB gate added)   |
| 2026-03   | `_cleanup_orphans_from_manifest` didn't honor `SNAPSHOT_EXCLUDE_DIRS` | Fixed in v3.2.0 §2.2 PR-4   |
| 2026-04   | `maintenance/` wholesale snapshot exclusion               | Fixed via tracked/append-only split (v3.2.0 §2.2) |
| 2026-04   | `mcp_server.py` 4.6KLOC god-object                         | Refactored into 14 tool modules + 7 infra modules (v3.2.0 §1.2) |

Findings open at the time of engagement will be listed in
`SECURITY_AUDIT_2026-04.md` once the internal audit completes.

---

## 9. First-party audit baseline

Before an external audit starts, STARGA provides:

- **Internal audit report** at `SECURITY_AUDIT_2026-04.md` (landing
  in v3.2.0 via a security-reviewer sweep).
- **SAST baseline** — Bandit + CodeQL run in CI, results in
  `.github/workflows/security.yml` artifacts.
- **Dependency audit** — `pip-audit` + `trivy` run in CI.
- **Secrets audit** — `gitleaks` in CI + pre-commit hook.
- **SBOM** — CycloneDX JSON attached to every tagged release.
- **SLSA-adjacent provenance** — sigstore-signed wheels + tarballs.

These should **reduce** the auditor's time; nothing here replaces
the deep manual review and dynamic testing described above.

---

## 10. How to engage

Email **security@star.ga** with:

- Firm name + primary contact.
- Expertise statement (prior similar engagements, published work).
- Proposed scope — you may accept the SoW as-is or propose
  additions/exclusions with rationale.
- Proposed timeline + fee structure.
- Sample redacted report from a past engagement.

STARGA will respond within 7 business days with either a scope-match
confirmation + contract draft, or a decline with our reasoning.

---

## 11. License + confidentiality

- mind-mem code is **Apache-2.0**; auditors do not need an NDA to
  read the source.
- Auditor reports remain the auditor's IP unless otherwise
  negotiated. STARGA will ask for permission before quoting them in
  marketing material.
- STARGA retains the right to self-publish a summary of findings
  after the 90-day coordinated-disclosure window, regardless of the
  auditor's publication choices.

---

## 12. Contacts

- **Security lead:** Nikolai (security@star.ga)
- **Legal + contracting:** legal@star.ga
- **Repo:** https://github.com/star-ga/mind-mem
- **PyPI:** https://pypi.org/project/mind-mem/
- **Disclosure policy:** `SECURITY.md` in the repo root
- **Public audit status tracker:** GitHub issues tagged
  [`security-audit`](https://github.com/star-ga/mind-mem/issues?q=is%3Aissue+label%3Asecurity-audit)

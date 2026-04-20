# Security Audit — mind-mem v3.1.9 (April 2026)

## Executive Summary

mind-mem has a strong security baseline: no CRITICAL findings, all SQL
parameterized, no eval/exec/pickle in data paths, path traversal covered
by `_safe_resolve`/`_validate_path` guards, and constant-time token
comparison in place. Three findings were identified — two HIGH (query
length bomb in recall surface, hardcoded credentials in docker-compose)
and one MEDIUM (missing startup warning for weak tokens) — all three
fixed in this session.

---

## Findings

### HIGH — H-01: Unbounded Query Length in recall/hybrid_search/intent_classify

| Attribute | Value |
|-----------|-------|
| File | `src/mind_mem/mcp/tools/recall.py` — `_recall_impl()`, `intent_classify()` |
| CWE | CWE-400 Uncontrolled Resource Consumption |
| CVSS (estimate) | 7.5 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H) |
| Status | **Fixed** |

**Description.** The `recall`, `hybrid_search`, `recall_with_axis`, and
`intent_classify` MCP tools passed `query` directly to BM25/hybrid
search engines with no upper bound on string length. A caller could
submit a multi-megabyte query string, causing unbounded tokenization,
FTS5 query compilation, and BM25 scoring work.

**Exploit scenario.** An MCP client (even authenticated) sends
`recall(query="A" * 50_000_000)`. The BM25 tokenizer iterates over
every character and the FTS5 query plan is compiled without budget.
Depending on hardware, this causes a multi-second hang or OOM kill,
effectively DoS-ing the MCP server process.

**Fix.** Added `_MAX_QUERY_LEN = 8192` constant at the top of
`recall.py` and early-exit checks in `_recall_impl()` and
`intent_classify()`. The existing `pack_recall_budget` and
`recall_with_axis` tools already delegated through `_recall_impl`, so
they are covered. The `signal.py` `update_recall_signals` tool already
had its own 8192-char cap for query inputs.

---

### HIGH — H-02: Hardcoded Credentials in docker-compose.yml

| Attribute | Value |
|-----------|-------|
| File | `deploy/docker-compose.yml` lines 32, 49-50 |
| CWE | CWE-798 Use of Hard-coded Credentials |
| Status | **Fixed** |

**Description.** The Compose file contained inline credentials
(`POSTGRES_PASSWORD: mindmem`, `DATABASE_URL: ...mindmem:mindmem@...`)
that would be used verbatim by anyone running `docker compose up`
without customisation. These defaults appear in version history and in
any container inspection (`docker inspect`).

**Exploit scenario.** An operator runs `docker compose up` on a
cloud VM with public port 5432 forwarded; the Postgres database is
reachable with the default `mindmem`/`mindmem` credential.

**Fix.** Replaced all credential literals with required environment
variable references using Docker Compose variable-substitution syntax
(`${VAR:?error-if-missing}`). `docker compose up` now fails fast with
an actionable message if `POSTGRES_PASSWORD`, `MIND_MEM_TOKEN`, or
`MIND_MEM_ADMIN_TOKEN` are not set, preventing silent use of weak
defaults. A `.env` file is the recommended approach.

---

### MEDIUM — M-01: No Startup Warning for Short Bearer Tokens

| Attribute | Value |
|-----------|-------|
| File | `src/mind_mem/mcp/infra/http_auth.py`, `src/mind_mem/mcp/server.py` |
| CWE | CWE-521 Weak Password Requirements |
| Status | **Fixed** |

**Description.** The HTTP auth layer accepted tokens of any length
(including single-character tokens) without emitting any warning. An
operator who set `MIND_MEM_TOKEN=abc` would receive no feedback that
their token is trivially brute-forceable.

**Fix.** Added `check_token_strength()` in `http_auth.py` that emits
structured log warnings if either token is shorter than 32 characters.
Called once from `main()` in `server.py` at startup. Tokens of any
length continue to work — the change is advisory only, preserving
backward compatibility and existing test tokens. An oversized-header
DoS guard was also added: tokens longer than 4096 bytes are rejected
before the constant-time compare.

---

## What Was Checked — Clean

| Check | Result |
|-------|--------|
| SQL injection — all `execute()` calls reviewed | Clean — `f"...IN ({placeholders})"` patterns use only `","join("?" …)` — no user data interpolated into SQL |
| `PRAGMA busy_timeout={self._busy_timeout}` in `connection_manager.py` | Clean — value is an `int` default, never user-supplied |
| Path traversal — MCP tools (`_validate_path`, `_safe_vault_path`, `_safe_resolve`) | Clean — `os.path.realpath` prefix check enforced on all user-supplied paths |
| Kernel name in `get_mind_kernel` | Clean — `^[a-zA-Z0-9_-]{1,64}$` regex gating before `os.path.join` |
| `core.py` filename sanitization | Clean — `any(ch in out_name for ch in "/\\")` guard present |
| Tar extraction (`backup_restore.py`) | Clean — `_is_safe_tar_member()` guards absolute paths, `..`, symlinks, hardlinks, device files |
| SSRF — `llm_extractor.py` / `query_expansion.py` outbound HTTP | Clean — all URLs are either hardcoded localhost (`11434`/`8000`) or drawn from env vars; no user-supplied URL passed to `urlopen` in data-path tools |
| `recall_vector.py` llama.cpp URL | Clean — `urlparse` hostname whitelist (`localhost`, `127.0.0.1`, `::1`) before any HTTP call |
| Hardcoded secrets in Python source | Clean — `hook_installer.py` patterns are a redaction list, not credentials |
| `yaml.load` | N/A — no `import yaml` in production source (`src/mind_mem/`) |
| `pickle` / `eval` / `exec` | Clean — none present in production paths |
| `subprocess` with `shell=True` | Clean — `cron_runner.py` uses list-form `subprocess.run` without shell |
| Constant-time token compare | Clean — `hmac.compare_digest` used in all token comparison paths |
| Docker image — root user | Clean — `USER mindmem` (uid 1000), non-root |
| Docker image — `ADD http://` | Clean — not present |
| Docker image — secrets in ENV | Clean — only workspace path and transport mode baked in |
| Docker image — healthcheck | Clean — present, non-privileged Python import check |
| install-bootstrap.sh | Clean — `set -euo pipefail`; no `curl | bash` of untrusted content; all arguments sanitised via case/shift parsing |
| Block ID validation (xref graphs) | Clean — `len(sid) < 100` guard in traversal loop |
| Rate limiting | Clean — `SlidingWindowRateLimiter` per client ID with LRU cap at 1024 entries |
| Concurrent write safety | Clean — WAL mode + `threading.Lock` write serialisation in `ConnectionManager` |
| Merkle / hash chain | Clean — SHA3-512, Q16.16 preimages, NUL-separated composition |
| MCP token auth bypass | Clean — `verify_token` called on every request; constant-time compare |
| ReDoS in MCP input regex | Clean — patterns are bounded (`r"^P-\d{8}-\d{3}$"`, `r"^[A-Z]+-…"`) with no catastrophic backtracking |

---

## Dependency Audit

`pip-audit` found 64 known vulnerabilities across 25 packages in the
environment. The findings below are scoped to packages directly or
transitively required by mind-mem.

### Transitive via fastmcp — requires attention

| Package | Installed | CVE | Fix | Severity |
|---------|-----------|-----|-----|----------|
| `authlib` | 1.6.8 | CVE-2026-27962 | 1.6.9 | HIGH — JWK Header Injection allows forging arbitrary JWTs |
| `authlib` | 1.6.8 | CVE-2026-28490 | 1.6.9 | HIGH — RSA PKCS#1 v1.5 padding oracle |
| `authlib` | 1.6.8 | GHSA-jj8c-mmj3-mmgv | 1.6.11 | MEDIUM — CSRF on cache feature in Starlette/FastAPI integrations |
| `aiohttp` | 3.13.3 | CVE-2026-34519 | 3.13.4 | HIGH — response `reason` parameter header injection |
| `aiohttp` | 3.13.3 | CVE-2026-34520 | 3.13.4 | HIGH — null byte / control char acceptance in response headers |
| `aiohttp` | 3.13.3 | CVE-2026-34513 | 3.13.4 | MEDIUM — unbounded DNS cache DoS |
| `aiohttp` | 3.13.3 | CVE-2026-34516 | 3.13.4 | MEDIUM — excessive multipart header memory |
| `aiohttp` | 3.13.3 | CVE-2026-34517 | 3.13.4 | MEDIUM — large multipart field read into memory before size check |

**Recommendation.** mind-mem does not directly invoke `authlib` JWT
verification with `key=None`, and does not expose `aiohttp` to
untrusted HTTP inputs in the default stdio transport. However:

- Upgrade `fastmcp` to a version that pins `authlib>=1.6.9` and
  `aiohttp>=3.13.4` when such a release is available.
- For deployments using the HTTP transport (`--transport http`), the
  `aiohttp` CVEs are in the network stack used by `uvicorn`/`fastmcp`.
  Upgrading the host environment's `aiohttp` (`pip install aiohttp>=3.13.4`)
  is safe independently of `fastmcp` version and is recommended
  immediately for internet-facing deployments.
- The `authlib` PKCS#1 oracle (CVE-2026-28490) and JWK injection
  (CVE-2026-27962) are only reachable if `fastmcp`'s OAuth2 provider
  parsing is active — this is not the case in mind-mem's
  `StaticTokenVerifier` configuration. Upgrading is still recommended.

**Do not bump pinned versions in `pyproject.toml` without re-running
the CI matrix** — `aiohttp` and `authlib` are transitive, not direct
dependencies, and are not pinned in `pyproject.toml`.

---

## Audit Scope

Reviewed per the task specification:

- `src/mind_mem/**/*.py` — all production modules (excluding
  `apply_engine.py` and `block_store.py` per instruction)
- `mcp_server.py` (top-level shim)
- `install-bootstrap.sh`, `install.sh`, `scripts/`
- `deploy/docker/Dockerfile`, `deploy/docker-compose.yml`

Tools used: manual code review, `pip-audit`, `ruff`, grep pattern scans
for secrets, injection, and unsafe API usage.

Audited by: security-reviewer agent, 2026-04-19.

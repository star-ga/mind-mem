# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.2.x   | Yes — current stable |
| 3.1.x   | Security fixes only |
| 3.0.x   | Security fixes only |
| 2.x     | Security fixes only |
| < 2.0   | No |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

### Reporting channel

Email: security@star.ga

Include in your report:
1. A clear description of the vulnerability
2. Affected component (module name, function, version)
3. Steps to reproduce (minimal proof-of-concept preferred)
4. Impact assessment — what an attacker could achieve
5. Suggested fix if you have one

### Response timeline

| Milestone | Target |
|-----------|--------|
| Acknowledgement | 48 hours |
| Severity assessment | 5 business days |
| Fix for CRITICAL | 7 days |
| Fix for HIGH | 14 days |
| Fix for MEDIUM/LOW | Next scheduled release |
| Public disclosure | 90 days from initial report |

We follow responsible disclosure. If you need to publish before 90 days
due to active exploitation, please notify us — we will prioritise the fix.

---

## Scope

### In scope

- `src/mind_mem/**` — all production Python modules
- `mcp_server.py` — MCP server entry point
- MCP tool handlers — especially recall, propose_update, encrypt_file, audit
- HTTP transport auth (`MIND_MEM_TOKEN` / `MIND_MEM_ADMIN_TOKEN`)
- `install-bootstrap.sh` / `install.sh` — installer scripts
- `deploy/docker/Dockerfile`, `deploy/docker-compose.yml`
- Cryptographic primitives — hash chain, encryption, Merkle proofs

### Out of scope

- Issues in transitive dependencies (report to the upstream maintainer)
- Denial of service via workspace files that the attacker already controls
- Issues requiring physical access to the machine running mind-mem
- The `tests/`, `benchmarks/`, `train/`, and `examples/` directories

---

## Security Model

mind-mem is a **local-first** library that operates entirely on the
user's filesystem. It has no network listeners in its default
configuration (stdio MCP transport). The optional HTTP transport binds
to `127.0.0.1` by default.

### Threat Model

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Path traversal via block IDs or file paths | `_safe_resolve()` rejects `..` components and symlink escapes | Active |
| Tar archive extraction (zip-slip) | `_is_safe_tar_member()` rejects absolute paths, `..`, symlinks, hardlinks, device files | Active |
| SQL injection via FTS5 queries | All SQLite queries use parameterized bindings (`?` placeholders); zero string interpolation in SQL with user data | Active |
| Query length bomb (DoS) | `_MAX_QUERY_LEN = 8192` cap in `_recall_impl()` and `intent_classify()` | Active (v3.2.0+) |
| Arbitrary code execution via LLM extraction | Extraction output treated as plain text; never evaluated as code | Active |
| File lock starvation / race conditions | Cross-platform advisory locking via `fcntl`/`msvcrt`/atomic create with stale PID cleanup | Active |
| MCP token auth bypass (HTTP mode) | Bearer token validation on every request; constant-time comparison via `hmac.compare_digest`; oversized-token DoS guard (4096 byte cap) | Active |
| Weak bearer token (brute force) | Startup warning emitted if token is shorter than 32 characters | Active (v3.2.0+) |
| Denial of service via large workspaces | Configurable `top_k` limits, knee cutoff truncation, proposal budget caps (`per_run`, `per_day`, `backlog_limit`) | Active |
| Concurrent SQLite write corruption | WAL journal mode, `busy_timeout=3000`, `timeout=5` on all connections, serialised writer via `threading.Lock` | Active |
| Hardcoded credentials in Docker deployment | `docker-compose.yml` uses required env var references (`${VAR:?…}`) — fails fast if secrets not set | Active (v3.2.0+) |
| Kernel name path escape (`get_mind_kernel`) | Regex `^[a-zA-Z0-9_-]{1,64}$` gating before `os.path.join` | Active |

### Dependencies

- **Zero external dependencies in core** — the recall engine, governance
  pipeline, and all core modules use only Python 3.10+ stdlib.
- **Optional dependencies** are clearly documented and isolated:
  `sentence-transformers` (vector search), `onnxruntime` (ONNX
  embeddings), `fastmcp` (MCP server). None are required for core
  functionality.
- No dependency on `eval()`, `exec()`, `pickle`, `subprocess` with
  `shell=True`, or any code execution primitives in the data path.
- Dependabot monitors for known vulnerabilities in optional extras.

### Input Validation

All external inputs are validated at system boundaries:

- **File paths** — `_safe_resolve()` in `apply_engine.py` and
  `_validate_path()` in `mcp/infra/workspace.py` resolve paths within
  the workspace and reject any that escape via `..` or symlinks.
- **Tar extraction** — `_is_safe_tar_member()` in `backup_restore.py`
  validates every tar member before extraction.
- **Block IDs** — validated against `[A-Z]+-[A-Za-z0-9-]+` pattern.
- **SQL queries** — FTS5 queries use parameterized statements.
- **Query strings** — capped at 8192 characters before entering any
  search engine (BM25, hybrid, FTS5, intent router).
- **MCP tool inputs** — validated by the FastMCP schema layer plus
  per-tool guards (length caps, regex patterns, range checks).

### Concurrency Safety

- **Advisory file locks** — `MindFileLock` provides cross-platform
  locking using `fcntl.flock()` on Unix and `msvcrt.locking()` on
  Windows.
- **SQLite WAL mode** — all connections use `PRAGMA journal_mode=WAL`,
  `PRAGMA busy_timeout=3000`, and a single serialised writer protected
  by `threading.Lock`.
- **Atomic writes** — apply engine writes to temp files then renames,
  preventing partial writes on crash.

### Safe Defaults

- Governance mode defaults to `detect_only` (read-only analysis)
- HTTP transport binds to `127.0.0.1` only
- Token auth enforced when `MIND_MEM_TOKEN` is set
- Proposal budget limits: 3 per run, 6 per day, 30 backlog max
- File watcher debounce at 2 seconds

---

## Security Audit Checklist

This project has been audited (April 2026) against the following:

- [x] OWASP Top 10 for LLM Applications (2025)
- [x] No `eval()`/`exec()`/`pickle` in data paths
- [x] No `shell=True` subprocess calls
- [x] All SQL queries parameterized
- [x] All file paths validated against traversal
- [x] All tar/archive extraction validated against zip-slip
- [x] Query length caps on all search entry points
- [x] Concurrent access protected (file locks + SQLite WAL)
- [x] No hardcoded credentials in source or Compose defaults
- [x] Token auth on HTTP transport with constant-time compare
- [x] Oversized token header DoS guard
- [x] Startup warning for weak tokens (< 32 chars)
- [x] Rate limiting via per-client sliding window + proposal budgets
- [x] Error messages do not leak internal paths or stack traces to callers
- [x] Dependency audit clean for direct dependencies (indirect: see audit)

Full audit report: `SECURITY_AUDIT_2026-04.md`

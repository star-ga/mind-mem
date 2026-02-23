# Security Policy

## Reporting Vulnerabilities

Report security issues to **security@star.ga**. We will respond within 48 hours and aim to release a fix within 7 days for critical issues.

Do **not** open public GitHub issues for security vulnerabilities.

---

## Security Model

mind-mem is a **local-first** library that operates entirely on the user's filesystem. It has no network listeners in its default configuration (stdio MCP transport). The optional HTTP transport binds to `127.0.0.1` by default.

### Threat Model

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Path traversal via block IDs or file paths | `_safe_resolve()` rejects `..` components and symlink escapes (`apply_engine.py:98`) | Active |
| Tar archive extraction (zip-slip) | `_is_safe_tar_member()` rejects absolute paths, `..`, symlinks, hardlinks, device files (`backup_restore.py:269`) | Active |
| SQL injection via FTS5 queries | All SQLite queries use parameterized bindings (`?` placeholders); zero string interpolation in SQL | Active |
| Arbitrary code execution via LLM extraction | Extraction output is treated as plain text; never evaluated as code | Active |
| File lock starvation / race conditions | Cross-platform advisory locking via `fcntl`/`msvcrt`/atomic create with stale PID cleanup (`mind_filelock.py`) | Active |
| MCP token auth bypass (HTTP mode) | Bearer token validation on every request; constant-time comparison; token required when `MIND_MEM_TOKEN` is set | Active |
| Denial of service via large workspaces | Configurable `top_k` limits, knee cutoff truncation, proposal budget caps (`per_run`, `per_day`, `backlog_limit`) | Active |
| Concurrent SQLite write corruption | WAL journal mode, `busy_timeout=3000`, `timeout=5` on all connections, `try/finally` cleanup | Active |

### Input Validation

All external inputs are validated at system boundaries:

- **File paths**: `_safe_resolve()` in `apply_engine.py` resolves paths within the workspace and rejects any that escape via `..` or symlinks. Used by the governance engine before writing any file.
- **Tar extraction**: `_is_safe_tar_member()` in `backup_restore.py` validates every tar member before extraction — rejects absolute paths, directory traversal, symlinks, hardlinks, and device files.
- **Block IDs**: Validated against `[A-Z]+-[A-Za-z0-9-]+` pattern. IDs containing path separators or special characters are rejected.
- **SQL queries**: All FTS5 and SQLite queries use parameterized bindings. No user input is ever interpolated into SQL strings.
- **MCP tool inputs**: All tool parameters are validated by the FastMCP schema layer before reaching handler functions.

### Dependency Policy

- **Zero core dependencies** — the recall engine, governance pipeline, and all core modules use only Python 3.10+ stdlib (`sqlite3`, `json`, `hashlib`, `os`, `re`, `argparse`, `fcntl`/`msvcrt`).
- **Optional dependencies** are isolated: `sentence-transformers` (vector search), `onnxruntime` (ONNX embeddings), `fastmcp` (MCP server). None are required for core functionality.
- No dependency on `eval()`, `exec()`, `pickle`, `subprocess` with `shell=True`, or any code execution primitives in the data path.

### Concurrency Safety

- **Advisory file locks**: `MindFileLock` (`mind_filelock.py`) provides cross-platform locking using `fcntl.flock()` on Unix and `msvcrt.locking()` on Windows. Stale locks are detected via PID liveness checks.
- **SQLite WAL mode**: All database connections use `PRAGMA journal_mode=WAL` for concurrent readers, `PRAGMA busy_timeout=3000` for writer contention, and `timeout=5` on `sqlite3.connect()`.
- **Atomic writes**: Apply engine writes to temp files then renames, preventing partial writes on crash.

### Safe Defaults

- Governance mode defaults to `detect_only` (read-only analysis, no automatic changes)
- HTTP transport binds to `127.0.0.1` only (no external network exposure)
- Token auth is enforced when `MIND_MEM_TOKEN` is set; unauthenticated requests are rejected
- Proposal budget limits prevent runaway automation: 3 proposals per run, 6 per day, 30 backlog max
- File watcher debounces at 2 seconds to prevent resource exhaustion from rapid file changes

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.7.x   | Yes      |
| 1.6.x   | Yes      |
| < 1.6   | No       |

---

## Security Audit Checklist

This project has been self-audited against the following:

- [x] OWASP Top 10 for LLM Applications (2025)
- [x] No `eval()`/`exec()`/`pickle` in data paths
- [x] No `shell=True` subprocess calls
- [x] All SQL queries parameterized
- [x] All file paths validated against traversal
- [x] All tar/archive extraction validated against zip-slip
- [x] Concurrent access protected (file locks + SQLite WAL)
- [x] No hardcoded credentials or secrets
- [x] Token auth on HTTP transport
- [x] Rate limiting via proposal budgets

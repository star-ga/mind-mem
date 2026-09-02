# Troubleshooting

Common issues and solutions for mind-mem.

## Installation Issues

### ImportError: No module named 'mind_mem'

**Cause:** MIND-Mem is not installed in editable mode.

**Fix:**
```bash
pip install -e ".[dev]"
```

### sqlite3.OperationalError: no such module: fts5

**Cause:** Your Python's SQLite build does not include the FTS5 extension.

**Fix:** Install a newer Python build or use the system SQLite:
```bash
# Ubuntu/Debian
sudo apt-get install libsqlite3-dev
# Rebuild Python or use pyenv with system sqlite
```

## Runtime Issues

### Recall returns no results

**Possible causes:**
1. Workspace not initialized — run `init_workspace()`
2. No blocks in workspace — check `decisions/` and `tasks/` directories
3. Index is stale — run `reindex` MCP tool
4. Query too specific — try broader search terms

### FileLock timeout

**Cause:** Another process holds the workspace lock.

**Fix:**
```bash
# Check for stale lock files
find /path/to/workspace -name "*.lock" -mmin +5 -delete
```

### Memory usage grows over time

**Cause:** Large workspace with many blocks and no compaction.

**Fix:**
```bash
# Run compaction to merge old blocks
python3 -m mind_mem.compaction /path/to/workspace
```

## MCP Server Issues

### MCP tools not responding

**Possible causes:**
1. Server not running — check process list
2. Wrong workspace path in config
3. Port conflict

**Fix:**
```bash
# Restart MCP server
python3 mcp_server.py --workspace /path/to/workspace
```

### propose_update rejected

**Cause:** Proposal validation failed.

**Fix:** Check that:
- Block IDs follow the `[TYPE-NNN]` format
- Required fields (Type, Statement) are present
- Content hash is valid

## Performance Issues

### Slow recall queries (>500ms)

**Possible causes:**
1. Large workspace (>10K blocks) without index
2. Stale FTS5 index
3. Missing vector index

**Fix:**
```bash
# Rebuild indexes
python3 -c "from mind_mem.sqlite_index import rebuild_index; rebuild_index('/path/to/workspace')"
```

### High disk usage

**Cause:** SQLite WAL files not checkpointed.

**Fix:**
```bash
# Force WAL checkpoint
python3 -c "
import sqlite3
conn = sqlite3.connect('/path/to/workspace/.mind-mem/index.db')
conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
conn.close()
"
```

## CI/CD Issues

### Tests fail on Windows

**Common causes:**
1. Path separator differences — use `os.path.join()` not string concatenation
2. File locking differences — Windows locks files more aggressively
3. Temp directory cleanup — use `ignore_cleanup_errors=True`

### Tests fail on Python 3.14

**Cause:** Python 3.14 is a released version and a first-class CI row — it is
not forgiven. The known 3.14-specific failure is an `anyio`/`TestClient`
teardown segfault, which is why 3.14 is still absent from the `pyproject.toml`
classifiers even though every 3.14 row must pass.

**Fix:** There is no `allow-failure` or `continue-on-error` carve-out to look
for — it was removed because a job that runs red while its run concludes
`success` is a red row reading as a pass. Reproduce with
`python3.14 -m pytest tests/ --ignore=tests/integration -m "not stress"` and fix
the failure, or open an issue; the matrix goes red until one of those happens.

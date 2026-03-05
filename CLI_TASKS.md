# mind-mem Remaining Tasks — CLI Instructions

Repo: `/home/n/mind-mem` (branch: `main`)
Current: v1.8.1, 2,027 tests passing, PyPI published

## Task 1: Fix `sys.path.insert` Pattern (37 occurrences)

**Problem:** 37 files use `sys.path.insert(0, ...)` instead of proper relative imports.
**Fix:** Replace all `sys.path.insert` hacks with standard Python relative imports using the `src/mind_mem/` package layout (established in v1.8.0 PR #467).

```bash
cd /home/n/mind-mem
grep -rn "sys.path.insert" src/ scripts/ tests/ --include="*.py"
```

For each file:
1. Remove the `sys.path.insert(0, ...)` line and the `import sys` if unused elsewhere
2. Replace the subsequent imports with proper relative or absolute imports from `mind_mem.*`
3. Run `python -m pytest` after each batch to confirm nothing breaks

Commit: `Remove sys.path.insert hacks, use proper package imports`

## Task 2: Deduplicate `recall` / `hybrid_search` MCP Tools

**Problem:** `recall` and `hybrid_search` overlap in functionality. Users are confused about which to use.
**Location:** `src/mind_mem/mcp_server.py` — look for `@mcp.tool` decorated functions `recall` and `hybrid_search`.

**Fix options (pick one):**
- **Option A (recommended):** Merge `hybrid_search` into `recall` as a `mode` parameter. Add `mode: str = "auto"` param to `recall()`. When `mode="hybrid"`, use the hybrid pipeline. Deprecate `hybrid_search` with a wrapper that calls `recall(mode="hybrid")`.
- **Option B:** Keep both but clearly differentiate in docstrings and add a `DeprecationWarning` to `hybrid_search`.

After fixing, update tool count in README (from 19 to 18 if merging).

Commit: `Merge hybrid_search into recall with mode parameter`

## Task 3: Incremental FTS Index Rebuild

**Problem:** Index rebuild drops and recreates the full FTS table. Slow on large corpora.
**Location:** `src/mind_mem/sqlite_index.py` — find the `rebuild` or `reindex` function.

**Fix:**
1. Instead of `DROP TABLE IF EXISTS ... CREATE VIRTUAL TABLE`, use `INSERT OR REPLACE` for changed/new blocks only
2. Track which blocks changed since last index using mtime or content hash
3. Only delete removed blocks, only insert/update changed blocks
4. Keep the full rebuild as a `--force` fallback

Commit: `Implement incremental FTS index rebuild`

## Task 4: Cross-Encoder Batching

**Problem:** Cross-encoder reranking processes one candidate at a time. Slow for large result sets.
**Location:** `src/mind_mem/recall/` — find the cross-encoder reranking step (likely in a submodule).

**Fix:**
1. Collect all (query, candidate) pairs into a batch
2. Send the full batch to the cross-encoder model in one call
3. Respect a configurable `batch_size` (default 32) to avoid OOM
4. Fallback to sequential if batch fails

Commit: `Add batched cross-encoder reranking`

## Task 5: Integration Tests in CI

**Problem:** No integration tests run in CI. Only unit tests.
**Location:** `.github/workflows/` — find the test workflow.

**Fix:**
1. Create `tests/integration/` directory
2. Write integration tests that exercise the full pipeline: ingest → index → recall → propose_update → approve
3. Add a CI job step that runs `pytest tests/integration/ -v` after unit tests pass
4. Use a temp directory for the test corpus (no fixtures checked in)

Commit: `Add integration test suite and CI job`

## Task 6: Re-run Benchmarks (STALE since Feb 16)

**Problem:** Benchmark numbers are outdated. v1.8.0 architecture changes may have affected scores.

```bash
cd /home/n/mind-mem
# LoCoMo benchmark
python benchmarks/locomo_judge.py --compress --output results/locomo_$(date +%Y%m%d).json

# LongMemEval benchmark
python benchmarks/longmemeval.py --output results/longmemeval_$(date +%Y%m%d).json
```

Update README with new numbers. If scores dropped, investigate before publishing.

Commit: `Update benchmark results to v1.8.1`

---

## Git Policy (MANDATORY)
- Author: `STARGA Inc <noreply@star.ga>`
- NO co-author lines
- NO AI mentions in commits
- Run full test suite before each commit: `python -m pytest`
- Create GitHub issues for each task before starting work

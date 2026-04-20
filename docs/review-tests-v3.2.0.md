# Test Review — mind-mem v3.2.0
Date: 2026-04-19  
Baseline: 3642 passing, 7 skipped, 5 deselected (70 s)

---

## Coverage — new v3.2.0 modules

| Module | Coverage | Status |
|--------|----------|--------|
| `api/__init__.py` | 100% | |
| `api/api_keys.py` | 99% | |
| `api/auth.py` | 97% | |
| `api/rest.py` | 82% | |
| `block_store_postgres.py` | **12%** | CRITICAL — all tests skip without live PG DSN |
| `block_store_postgres_replica.py` | 93% | |
| `mcp/tools/public.py` | 73% | below threshold |
| `recall_cache.py` | 76% | below threshold |
| `telemetry.py` | 78% | borderline |
| `tier_recall.py` | 93% | |

`block_store_postgres.py` is the largest gap (317 lines, 12%). Every test in `test_postgres_block_store.py` skips when `MIND_MEM_TEST_PG_DSN` is absent (the entire module is skipped via `pytest.importorskip`). The pure-logic helpers `_validate_schema_name` and `_block_to_row` were completely untested in CI.

---

## Anti-patterns found

### 1. `except Exception: pass` swallowing failures in test teardown — `test_postgres_block_store.py:76-81`

```python
try:
    conn = psycopg.connect(dsn)
    ...
    conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
except Exception:
    pass  # <-- hides real teardown failures
```

Teardown failures silently leave orphan schemas in the test database, contaminating subsequent runs. The broad catch also hides connection errors that indicate misconfigured test infrastructure.

**Fix:** Log the exception at WARNING level instead of silencing it. Teardown failures should be visible even when they are non-fatal.

### 2. `_block_to_row` and `_validate_schema_name` were untested without live Postgres

Both pure functions contain the SQL-injection allowlist and the metadata serialisation logic. Because the entire `test_postgres_block_store.py` file skips when psycopg is absent, these functions had 0% coverage in standard CI (no live DB). A malformed schema name that bypasses `_validate_schema_name` would reach a DDL f-string verbatim.

**Fix:** Pure-unit tests that import only the module-level helpers, requiring no database. Added in `tests/test_v320_gaps.py`.

### 3. `mcp/tools/public.py` — error-path branches (invalid mode/action) fully uncovered

The `_err(...)` return paths in every consolidated dispatcher (`recall`, `staged_change`, `memory_verify`, `graph`, `kernels`, `compiled_truth`) were at 0% coverage. These branches are exercised at agent boundary — they produce the error envelope the agent reads when it sends a bad `mode=` argument. Silent coverage of those paths means contract-breaking regressions (e.g., changing the `valid_modes` field name) would not be caught.

**Fix:** `TestPublicDispatcherErrorEnvelopes` in `tests/test_v320_gaps.py` covers all error branches for `recall`, `staged_change`, `memory_verify`, and `kernels` (8 tests).

---

## Missing test categories per new module

| Module | Missing |
|--------|---------|
| `block_store_postgres` | Unit tests for `_validate_schema_name`, `_block_to_row`, `_row_to_block`; property-based round-trip for JSON metadata serialisation |
| `recall_cache` | Concurrent `set` safety; `max_entries=1` boundary; metrics counter dispatch |
| `block_store_postgres_replica` | Circuit-breaker cooldown expiry (time-based recovery); `record_success` reset |
| `mcp/tools/public` | Error envelopes for every `_err(...)` path |
| `telemetry` | `_fire_metric` dispatch for non-`recall` spans |
| `api/rest.py` | Rate-limit 429 path; `/v1/metrics` Prometheus endpoint; malformed JSON body |

No integration test exists that exercises `recall_cache` in combination with the real recall pipeline (only unit + mock-based). The `tier_recall` module has no property-based test verifying that `apply_tier_boosts` is idempotent (calling it twice must not double-multiply scores).

---

## Flakiness risk (top 5)

1. `test_llm_extractor.py::test_returns_list_when_unavailable` — 4.4 s, uses a real `asyncio` event loop with timeout-based fallback logic.
2. `test_recall_cross_encoder.py::TestCrossEncoderRerankerUnit::test_empty_candidates` — 3.1 s, uses `time.perf_counter` implicitly via a real model warmup.
3. `test_knowledge_graph.py::TestConcurrency::test_concurrent_add_edge_no_corruption` — 2.7 s, 20 threads against an in-memory graph; any GIL-unlocked extension could race.
4. `test_interaction_signals.py::TestSignalStore::test_concurrent_observe_no_loss` — 1.4 s, 50 concurrent observers, relies on thread-safe SQLite WAL.
5. `test_baseline_snapshot.py::TestDetectDrift::test_detects_distribution_shift` — 1.3 s, chi-squared threshold is a floating-point comparison without tolerance.

---

## Top-5 slowest tests

| Test | Time |
|------|------|
| `test_llm_extractor.py::test_returns_list_when_unavailable` | 4.40 s |
| `test_recall_cross_encoder.py::test_empty_candidates` | 3.13 s |
| `test_knowledge_graph.py::test_concurrent_add_edge_no_corruption` | 2.69 s |
| `test_interaction_signals.py::test_concurrent_observe_no_loss` | 1.40 s |
| `test_baseline_snapshot.py::test_detects_distribution_shift` | 1.26 s |

The top two are model-load tests; moving warmup into a session-scoped fixture would reduce total suite time by ~5 s.

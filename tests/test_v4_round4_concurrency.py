"""Concurrency + adversarial-input tests for round-4 v4 modules.

Covers seven surfaces:
    backpressure      thread-safe depth updates, hysteresis stability,
                      monotonic pause, timeout behaviour, singleton safety
    health            probe contention, BaseException swallowing, reset
    logging_context   contextvar isolation across 64 threads, LIFO nesting,
                      correlation-id inheritance
    block_metadata    50-thread distinct writes, same-key last-writer-wins,
                      SQL injection impossibility, limit=0, recursive validator
    observability     200-thread cardinality race, overflow sentinel identity,
                      reset under concurrent creation
    eviction          policy torn-read safety, debug_plan large plan timing,
                      custom policy under contention
    surprise          all FallbackPolicy paths + RAISE reasons + coercion

Each test is designed so that reintroducing the bug it guards will cause it
to fail.
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

import mind_mem.v4.backpressure as bp_mod
import mind_mem.v4.block_metadata as bm_mod
import mind_mem.v4.eviction as ev_mod
from mind_mem.v4.backpressure import FLAG as BP_FLAG
from mind_mem.v4.backpressure import (
    BackpressureController,
    controller,
)
from mind_mem.v4.block_metadata import FLAG as BM_FLAG
from mind_mem.v4.block_metadata import (
    SchemaValidationResult,
    ensure_metadata_schema,
    get_block_metadata,
    list_blocks_by_tag,
    register_schema_validator,
    set_block_metadata,
    validate_block,
)
from mind_mem.v4.eviction import FLAG as EVICT_FLAG
from mind_mem.v4.eviction import (
    EvictionPlan,
    EvictionPolicy,
    active_policy,
    register_policy,
    set_active_policy,
)
from mind_mem.v4.health import (
    health_check,
    register_health_probe,
    reset_custom_probes_for_tests,
)
from mind_mem.v4.logging_context import (
    LogContext,
    current_context,
    with_context,
    with_correlation_id,
)
from mind_mem.v4.observability import FLAG as OBS_FLAG
from mind_mem.v4.observability import (
    MAX_CARDINALITY,
    counter,
    gauge,
    histogram,
    reset_for_tests,
    snapshot,
)
from mind_mem.v4.surprise_retrieval import (
    EmbeddingFailureError,
    FallbackPolicy,
    compute_surprise,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: object) -> Path:
    """Write a mind-mem.json with the given flag config and set MIND_MEM_CONFIG."""
    block: dict = {}
    for k, v in flags.items():
        if isinstance(v, bool):
            block[k] = {"enabled": v}
        else:
            # caller passed a full sub-dict (e.g. for watermarks)
            block[k] = v
    (tmp_path / "mind-mem.json").write_text(
        json.dumps({"v4": block}), encoding="utf-8"
    )
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_global_state() -> object:
    """Reset all module-level singletons before and after every test."""
    reset_for_tests()
    reset_custom_probes_for_tests()
    bp_mod.reset_for_tests()
    # Restore eviction active policy to the default
    ev_mod._active_policy = EvictionPolicy.LRU
    # Clear any custom validators from block_metadata
    with bm_mod._validator_lock:
        bm_mod._validators.clear()
    yield
    reset_for_tests()
    reset_custom_probes_for_tests()
    bp_mod.reset_for_tests()
    ev_mod._active_policy = EvictionPolicy.LRU
    with bm_mod._validator_lock:
        bm_mod._validators.clear()


# ===========================================================================
# BACKPRESSURE
# ===========================================================================


@pytest.mark.unit
def test_bp_100_threads_concurrent_set_depth_consistent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """100 threads writing random depths must leave the controller in a
    consistent (non-corrupted) state — depth >= 0 and overloaded flag
    agrees with the current depth and watermarks."""
    _cfg(tmp_path, monkeypatch, **{BP_FLAG: True})
    ctrl = BackpressureController(high_watermark=500, low_watermark=100)
    barrier = threading.Barrier(100)

    def worker() -> None:
        barrier.wait()
        ctrl.set_depth(random.randint(0, 1000))

    with ThreadPoolExecutor(max_workers=100) as ex:
        futs = [ex.submit(worker) for _ in range(100)]
        for f in as_completed(futs):
            f.result()

    # After all writes: internal state must be self-consistent.
    d = ctrl.depth()
    overloaded = ctrl.is_overloaded()
    assert d >= 0
    # If depth is above high watermark the controller MUST be overloaded,
    # and if below low watermark it MUST NOT be overloaded.
    # (Between the watermarks the hysteresis allows either state.)
    if d >= ctrl.high_watermark:
        assert overloaded is True
    if d <= ctrl.low_watermark:
        assert overloaded is False


@pytest.mark.unit
def test_bp_hysteresis_no_flapping_under_128_boundary_swings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth thrashing exactly at the boundary should not cause unbounded
    state flipping — once overloaded it stays until depth falls past
    low_watermark."""
    _cfg(tmp_path, monkeypatch, **{BP_FLAG: True})
    ctrl = BackpressureController(high_watermark=100, low_watermark=50)

    # First push into overloaded territory.
    ctrl.set_depth(101)
    assert ctrl.is_overloaded() is True

    # Now thrash 128 times at the HIGH watermark boundary (100-101).
    # Should never leave overloaded until depth drops below low_watermark.
    for _ in range(128):
        ctrl.set_depth(100)
        assert ctrl.is_overloaded() is True, "hysteresis must hold at high boundary"
        ctrl.set_depth(101)
        assert ctrl.is_overloaded() is True

    # Only a drop below low_watermark clears the flag.
    ctrl.set_depth(49)
    assert ctrl.is_overloaded() is False


@pytest.mark.unit
def test_bp_recommended_pause_monotonic_during_sustained_overload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """recommended_pause() must be non-decreasing across successive calls
    while the controller stays overloaded (exponential backoff contract)."""
    _cfg(tmp_path, monkeypatch, **{BP_FLAG: True})
    ctrl = BackpressureController(
        high_watermark=100, low_watermark=20, max_pause_seconds=4.0
    )
    ctrl.set_depth(200)

    pauses = [ctrl.recommended_pause() for _ in range(20)]

    for i in range(1, len(pauses)):
        assert pauses[i] >= pauses[i - 1], (
            f"pause regressed at index {i}: {pauses[i]} < {pauses[i-1]}"
        )
    # Final value must not exceed the configured cap.
    assert pauses[-1] <= 4.0


@pytest.mark.unit
def test_bp_wait_until_clear_returns_false_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wait_until_clear() must return False when depth stays above the
    high watermark and timeout expires."""
    _cfg(tmp_path, monkeypatch, **{BP_FLAG: True})
    ctrl = BackpressureController(
        high_watermark=100, low_watermark=20, max_pause_seconds=1.0
    )
    ctrl.set_depth(500)

    t0 = time.monotonic()
    result = ctrl.wait_until_clear(timeout=0.2, poll=0.05)
    elapsed = time.monotonic() - t0

    assert result is False
    # Must not have returned far too early or blocked far too long.
    assert 0.15 <= elapsed <= 0.8


@pytest.mark.unit
def test_bp_post_init_rejects_low_greater_than_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BackpressureController.__post_init__ must raise ValueError when
    low_watermark > high_watermark — without this guard the controller
    would enter an impossible state."""
    _cfg(tmp_path, monkeypatch, **{BP_FLAG: True})
    with pytest.raises(ValueError, match="low_watermark must be <="):
        BackpressureController(high_watermark=100, low_watermark=200)


@pytest.mark.unit
def test_bp_singleton_identity_across_50_concurrent_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller() called concurrently from 50 threads must return the
    *same* object identity every time — the double-checked locking must
    not create duplicate instances."""
    _cfg(tmp_path, monkeypatch, **{BP_FLAG: True})
    bp_mod.reset_for_tests()
    barrier = threading.Barrier(50)
    instances: list[BackpressureController] = []
    lock = threading.Lock()

    def fetch() -> None:
        barrier.wait()
        inst = controller()
        with lock:
            instances.append(inst)

    with ThreadPoolExecutor(max_workers=50) as ex:
        futs = [ex.submit(fetch) for _ in range(50)]
        for f in as_completed(futs):
            f.result()

    assert len(instances) == 50
    first = instances[0]
    for inst in instances[1:]:
        assert inst is first, "controller() returned more than one instance"


@pytest.mark.unit
def test_bp_config_watermarks_from_mind_mem_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller() must honour high_watermark / low_watermark from
    mind-mem.json rather than silently using defaults."""
    block = {
        BP_FLAG: {
            "enabled": True,
            "high_watermark": 8888,
            "low_watermark": 444,
            "max_pause_seconds": 7.0,
        }
    }
    (tmp_path / "mind-mem.json").write_text(
        json.dumps({"v4": block}), encoding="utf-8"
    )
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    bp_mod.reset_for_tests()

    ctrl = controller()

    assert ctrl.high_watermark == 8888
    assert ctrl.low_watermark == 444
    assert ctrl.max_pause_seconds == 7.0


# ===========================================================================
# HEALTH
# ===========================================================================


@pytest.mark.unit
def test_health_register_probe_under_32_thread_contention_last_write_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """32 threads each registering under the SAME probe name must result
    in exactly one probe registered (last-write-wins) — no stacking or
    corruption of the _custom_probes list."""
    _cfg(tmp_path, monkeypatch)  # health_check is never flag-gated
    barrier = threading.Barrier(32)
    call_counts: list[int] = []
    call_lock = threading.Lock()

    def make_probe(idx: int):
        def probe(_ws: Path) -> str:
            with call_lock:
                call_counts.append(idx)
            return "ok"
        return probe

    def register(idx: int) -> None:
        barrier.wait()
        register_health_probe("shared_probe", make_probe(idx))

    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(register, i) for i in range(32)]
        for f in as_completed(futs):
            f.result()

    # After all threads finish exactly ONE probe named shared_probe exists.
    from mind_mem.v4.health import _custom_probes
    names = [n for n, _ in _custom_probes]
    assert names.count("shared_probe") == 1, (
        f"expected 1 registration, got {names.count('shared_probe')}"
    )


@pytest.mark.unit
def test_health_check_survives_probe_raising_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that raises a BaseException subclass (not Exception) must
    not crash health_check — the result should degrade, not propagate."""
    _cfg(tmp_path, monkeypatch)

    class _Keyboard(BaseException):
        pass

    def evil_probe(_ws: Path) -> str:
        raise _Keyboard("simulated signal")

    register_health_probe("evil", evil_probe)
    # This must not raise even though the probe raises BaseException.
    result = health_check(tmp_path)

    # The probe raised something, so status is either "fail" or the
    # module entry contains "error:". Crucially health_check returned.
    assert "status" in result
    assert "evil" in result["modules"]


@pytest.mark.unit
def test_health_check_latency_bounded_under_heavy_probe_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """health_check must complete within 3 seconds even when 200 custom
    probes are registered (each doing trivial work). This guards against
    O(n) sequential probes growing unboundedly."""
    _cfg(tmp_path, monkeypatch)

    for i in range(200):
        register_health_probe(f"probe_{i}", lambda _ws: "ok")

    t0 = time.perf_counter()
    result = health_check(tmp_path)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"health_check took {elapsed:.2f}s — too slow"
    assert result["latency_ms"] >= 0.0


@pytest.mark.unit
def test_health_reset_custom_probes_clears_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reset_custom_probes_for_tests() must remove every registered custom
    probe — leaving the list empty."""
    _cfg(tmp_path, monkeypatch)
    for i in range(10):
        register_health_probe(f"p{i}", lambda _ws: "ok")

    reset_custom_probes_for_tests()

    from mind_mem.v4.health import _custom_probes
    assert _custom_probes == [], "custom probes not cleared"


# ===========================================================================
# LOGGING CONTEXT
# ===========================================================================


@pytest.mark.unit
def test_logging_context_64_threads_no_cross_contamination() -> None:
    """64 threads each push their own correlation_id and verify they
    see only their own value — contextvar isolation must be total."""
    barrier = threading.Barrier(64)
    errors: list[str] = []
    errors_lock = threading.Lock()

    def worker(tid: int) -> None:
        my_cid = f"cid-thread-{tid}"
        barrier.wait()
        token = LogContext.push(correlation_id=my_cid)
        # Simulate work while other threads are pushing their own values.
        time.sleep(0.001)
        ctx = current_context()
        if ctx.get("correlation_id") != my_cid:
            with errors_lock:
                errors.append(
                    f"thread {tid} saw {ctx.get('correlation_id')!r} "
                    f"instead of {my_cid!r}"
                )
        LogContext.pop(token)

    with ThreadPoolExecutor(max_workers=64) as ex:
        futs = [ex.submit(worker, i) for i in range(64)]
        for f in as_completed(futs):
            f.result()

    assert errors == [], "contextvar leak detected:\n" + "\n".join(errors)


@pytest.mark.unit
def test_logging_context_nested_with_context_lifo_discipline() -> None:
    """Nested with_context must restore the outer frame when the inner
    block exits (LIFO). A frame pushed inside must not bleed out."""
    outer_key = "outer_val"
    inner_key = "inner_val"

    with with_context(key=outer_key) as ctx_outer:
        assert ctx_outer["key"] == outer_key

        with with_context(key=inner_key) as ctx_inner:
            assert ctx_inner["key"] == inner_key

        # After inner block exits the outer frame must be restored.
        restored = current_context()
        assert restored["key"] == outer_key, (
            f"LIFO violation: expected {outer_key!r}, got {restored['key']!r}"
        )

    # After outer block exits the stack must be empty.
    empty = current_context()
    assert "key" not in empty


@pytest.mark.unit
def test_logging_context_with_correlation_id_preserves_outer_id() -> None:
    """@with_correlation_id must NOT overwrite an already-present
    correlation_id from an outer context — inner calls should inherit
    the parent's trace ID."""
    outer_cid = "outer-trace-abc"

    @with_correlation_id
    def inner_fn() -> str:
        return current_context().get("correlation_id", "")

    with with_context(correlation_id=outer_cid):
        result = inner_fn()

    assert result == outer_cid, (
        f"inner decorator clobbered outer correlation_id: {result!r}"
    )


@pytest.mark.unit
def test_logging_context_with_correlation_id_creates_id_when_absent() -> None:
    """@with_correlation_id must inject a fresh UUID4 when no
    correlation_id exists in the current context."""

    @with_correlation_id
    def fn() -> str:
        return current_context().get("correlation_id", "")

    result = fn()

    assert result != "", "no correlation_id was injected"
    # UUID4 format: 8-4-4-4-12 hex
    import re
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        result,
    ), f"injected value is not a UUID4: {result!r}"


# ===========================================================================
# BLOCK METADATA
# ===========================================================================


@pytest.mark.unit
def test_block_metadata_50_concurrent_distinct_writes_all_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """50 threads each writing to a DIFFERENT block_id must all be readable
    after completion — no writes must be silently lost to SQLite contention."""
    _cfg(tmp_path, monkeypatch, **{BM_FLAG: True})
    ws = tmp_path / "workspace"
    ws.mkdir()
    ensure_metadata_schema(ws)

    barrier = threading.Barrier(50)

    def write(idx: int) -> None:
        barrier.wait()
        set_block_metadata(ws, f"block-{idx}", {"seq": str(idx)})

    with ThreadPoolExecutor(max_workers=50) as ex:
        futs = [ex.submit(write, i) for i in range(50)]
        for f in as_completed(futs):
            f.result()

    missing = []
    for i in range(50):
        meta = get_block_metadata(ws, f"block-{i}")
        if meta is None or meta.tags.get("seq") != str(i):
            missing.append(i)

    assert missing == [], f"blocks missing or corrupted after concurrent writes: {missing}"


@pytest.mark.unit
def test_block_metadata_50_concurrent_same_key_no_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """50 threads writing to the SAME block_id must produce exactly one
    readable record — no DB corruption, no lost-write assertion error."""
    _cfg(tmp_path, monkeypatch, **{BM_FLAG: True})
    ws = tmp_path / "workspace"
    ws.mkdir()
    ensure_metadata_schema(ws)

    barrier = threading.Barrier(50)

    def write(idx: int) -> None:
        barrier.wait()
        set_block_metadata(ws, "shared-block", {"writer": str(idx)})

    with ThreadPoolExecutor(max_workers=50) as ex:
        futs = [ex.submit(write, i) for i in range(50)]
        for f in as_completed(futs):
            f.result()

    meta = get_block_metadata(ws, "shared-block")
    assert meta is not None, "shared block was lost after concurrent upserts"
    assert "writer" in meta.tags, "tags dict is corrupt"
    # The writer value must be one of the 50 thread IDs — no garbage.
    assert meta.tags["writer"] in {str(i) for i in range(50)}


@pytest.mark.unit
def test_block_metadata_sql_injection_via_tag_key_is_impossible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tag key containing SQL injection payload must be stored verbatim
    and never cause the block_metadata table to be dropped."""
    _cfg(tmp_path, monkeypatch, **{BM_FLAG: True})
    ws = tmp_path / "workspace"
    ws.mkdir()
    injection = "'; DROP TABLE block_metadata;--"

    set_block_metadata(ws, "victim", {injection: "evil"})

    # The table must still exist.
    db = ws / "index.db"
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='block_metadata'"
        ).fetchone()
    assert row is not None, "block_metadata table was dropped — SQL injection succeeded"

    # The data must be retrievable.
    meta = get_block_metadata(ws, "victim")
    assert meta is not None
    assert meta.tags.get(injection) == "evil"


@pytest.mark.unit
def test_block_metadata_list_blocks_by_tag_limit_zero_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """list_blocks_by_tag with limit=0 must return [] regardless of how
    many matching rows exist — guarding the non-positive limit early-exit."""
    _cfg(tmp_path, monkeypatch, **{BM_FLAG: True})
    ws = tmp_path / "workspace"
    ws.mkdir()
    for i in range(5):
        set_block_metadata(ws, f"b{i}", {"env": "prod"})

    result = list_blocks_by_tag(ws, "env", "prod", limit=0)

    assert result == [], f"expected [], got {result}"


@pytest.mark.unit
def test_block_metadata_recursive_validator_does_not_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema validator that calls validate_block recursively on a
    different kind must complete without deadlock (the _validator_lock
    must not be held re-entrantly)."""
    _cfg(tmp_path, monkeypatch, **{BM_FLAG: True})
    ws = tmp_path / "workspace"
    ws.mkdir()

    def child_validator(payload: dict) -> SchemaValidationResult:
        return SchemaValidationResult(ok=True, reason="child_ok")

    def parent_validator(payload: dict) -> SchemaValidationResult:
        # Re-enters validate_block under a DIFFERENT kind.
        inner = validate_block("child_kind", payload)
        return SchemaValidationResult(ok=inner.ok, reason="parent_ok")

    register_schema_validator("child_kind", child_validator)
    register_schema_validator("parent_kind", parent_validator)

    # This must complete; if the lock is not re-entrant and is held across
    # the dispatch this will deadlock (and the test will time out).
    result = validate_block("parent_kind", {"x": 1})

    assert result.ok is True


# ===========================================================================
# OBSERVABILITY CARDINALITY
# ===========================================================================


@pytest.mark.unit
def test_observability_200_threads_past_cap_drop_counter_equals_overflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """200 threads racing to create counters past MAX_CARDINALITY must
    yield a v4.cardinality.dropped_counter value equal to the total
    number of overflow attempts."""
    _cfg(tmp_path, monkeypatch, **{OBS_FLAG: True})
    reset_for_tests()

    # Pre-fill registry to exactly MAX_CARDINALITY - 1 so the next
    # unique name triggers the cap.
    for i in range(MAX_CARDINALITY - 1):
        counter(f"fill_{i}")

    # Now one name fills the last slot; all others overflow.
    counter("last_slot")

    barrier = threading.Barrier(200)
    overflow_count = 0
    overflow_lock = threading.Lock()

    def create_overflow(idx: int) -> None:
        nonlocal overflow_count
        barrier.wait()
        c = counter(f"overflow_{idx}")
        if c is not None:  # always true; sentinel is a Counter
            with overflow_lock:
                overflow_count += 1

    with ThreadPoolExecutor(max_workers=200) as ex:
        futs = [ex.submit(create_overflow, i) for i in range(200)]
        for f in as_completed(futs):
            f.result()

    snap = snapshot()
    recorded_drops = snap.get("v4.cardinality.dropped_counter", 0)
    # Every one of the 200 threads should have triggered an overflow.
    assert recorded_drops == 200, (
        f"drop counter is {recorded_drops}, expected 200"
    )


@pytest.mark.unit
def test_observability_overflow_sentinels_are_shared_not_per_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All names past the cardinality cap must return the SAME sentinel
    instance — if each overflow created a new object, memory would still
    blow up despite the guard."""
    from mind_mem.v4.observability import _OVERFLOW_COUNTER, _OVERFLOW_GAUGE, _OVERFLOW_HISTOGRAM

    _cfg(tmp_path, monkeypatch, **{OBS_FLAG: True})
    reset_for_tests()

    for i in range(MAX_CARDINALITY):
        counter(f"fill_{i}")

    c1 = counter("overflow_a")
    c2 = counter("overflow_b")
    assert c1 is c2 is _OVERFLOW_COUNTER, "counter overflow returned different sentinel objects"

    reset_for_tests()
    for i in range(MAX_CARDINALITY):
        gauge(f"fg_{i}")
    g1 = gauge("og_a")
    g2 = gauge("og_b")
    assert g1 is g2 is _OVERFLOW_GAUGE, "gauge overflow returned different sentinel objects"

    reset_for_tests()
    for i in range(MAX_CARDINALITY):
        histogram(f"fh_{i}")
    h1 = histogram("oh_a")
    h2 = histogram("oh_b")
    assert h1 is h2 is _OVERFLOW_HISTOGRAM, "histogram overflow returned different sentinel objects"


@pytest.mark.unit
def test_observability_reset_under_concurrent_counter_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reset_for_tests() called while threads are creating counters must
    not corrupt the registry (no KeyError, no AttributeError, no crash).
    After reset completes, counter() must work normally."""
    _cfg(tmp_path, monkeypatch, **{OBS_FLAG: True})
    errors: list[Exception] = []
    errors_lock = threading.Lock()
    stop = threading.Event()

    def create_loop(idx: int) -> None:
        i = 0
        while not stop.is_set():
            try:
                counter(f"t{idx}_c{i}").inc()
            except Exception as e:
                with errors_lock:
                    errors.append(e)
            i += 1

    futures = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for t in range(10):
            futures.append(ex.submit(create_loop, t))
        # Let threads run then reset.
        time.sleep(0.05)
        reset_for_tests()
        time.sleep(0.05)
        stop.set()
        for f in futures:
            f.result()

    assert errors == [], f"errors during concurrent reset: {errors}"

    # Registry must be in a usable state after reset.
    c = counter("post_reset")
    c.inc()
    assert snapshot().get("post_reset") == 1


# ===========================================================================
# EVICTION
# ===========================================================================


@pytest.mark.unit
def test_eviction_32_threads_policy_swap_no_torn_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """32 threads alternating set_active_policy between LRU and
    LOW_SURPRISE must never observe a torn (garbage) policy value."""
    _cfg(tmp_path, monkeypatch, **{EVICT_FLAG: True})
    valid_policies = {EvictionPolicy.LRU, EvictionPolicy.LOW_SURPRISE}
    invalid_reads: list[str] = []
    invalid_lock = threading.Lock()
    barrier = threading.Barrier(32)
    stop = threading.Event()

    def writer(idx: int) -> None:
        barrier.wait()
        policies = [EvictionPolicy.LRU, EvictionPolicy.LOW_SURPRISE]
        while not stop.is_set():
            set_active_policy(policies[idx % 2])
            time.sleep(0)

    def reader() -> None:
        barrier.wait()
        while not stop.is_set():
            p = active_policy()
            if p not in valid_policies:
                with invalid_lock:
                    invalid_reads.append(repr(p))

    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(writer, i) for i in range(16)]
        futs += [ex.submit(reader) for _ in range(16)]
        time.sleep(0.2)
        stop.set()
        for f in as_completed(futs):
            f.result()

    assert invalid_reads == [], f"torn policy reads: {invalid_reads}"


@pytest.mark.unit
def test_eviction_debug_plan_10k_entries_completes_under_100ms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """debug_plan() on a 10,000-entry EvictionPlan must finish under
    100ms — it must not be O(n²) or do any I/O."""
    _cfg(tmp_path, monkeypatch, **{EVICT_FLAG: True})
    candidates = [(f"block_{i}", f"lru:last_seen=2026-01-0{i%9+1}T00:00:00Z") for i in range(10000)]
    plan = EvictionPlan(policy=EvictionPolicy.LRU, candidates=candidates)

    t0 = time.perf_counter()
    grouped = plan.debug_plan()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 100.0, f"debug_plan took {elapsed_ms:.1f}ms — too slow"
    assert "lru" in grouped
    assert len(grouped["lru"]) == 10000


@pytest.mark.unit
def test_eviction_custom_policy_register_then_set_active_under_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """register_policy then set_active_policy for a custom name must work
    when called from multiple threads — the policy must be activatable."""
    _cfg(tmp_path, monkeypatch, **{EVICT_FLAG: True})
    barrier = threading.Barrier(32)
    errors: list[Exception] = []
    errors_lock = threading.Lock()

    def custom_fn(workspace, **_):
        return [("custom_block", "custom:test")]

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            register_policy("custom_policy_X", custom_fn)
            set_active_policy("custom_policy_X")
            p = active_policy()
            if p != "custom_policy_X":
                with errors_lock:
                    errors.append(ValueError(f"unexpected policy: {p!r}"))
        except Exception as e:
            with errors_lock:
                errors.append(e)

    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(worker, i) for i in range(32)]
        for f in as_completed(futs):
            f.result()

    assert errors == [], f"errors in concurrent policy registration: {errors}"
    assert active_policy() == "custom_policy_X"


# ===========================================================================
# SURPRISE RETRIEVAL — FallbackPolicy
# ===========================================================================


@pytest.mark.unit
def test_surprise_fallback_neutral_returns_half_on_missing_context() -> None:
    """NEUTRAL policy must return 0.5 when context is None — this is
    the safe 'I don't know' value preserved from prior behaviour."""
    result = compute_surprise([1.0, 0.0], None, fallback_policy=FallbackPolicy.NEUTRAL)
    assert result == 0.5


@pytest.mark.unit
def test_surprise_fallback_promote_returns_one_on_missing_context() -> None:
    """PROMOTE policy must return 1.0 (max surprise) when context is
    None — biases toward tier promotion on embedder failure."""
    result = compute_surprise([1.0, 0.0], None, fallback_policy=FallbackPolicy.PROMOTE)
    assert result == 1.0


@pytest.mark.unit
def test_surprise_fallback_demote_returns_zero_on_missing_context() -> None:
    """DEMOTE policy must return 0.0 (no surprise) when context is
    None — biases toward COLD aging on embedder failure."""
    result = compute_surprise([1.0, 0.0], None, fallback_policy=FallbackPolicy.DEMOTE)
    assert result == 0.0


@pytest.mark.unit
def test_surprise_fallback_raise_on_missing_context() -> None:
    """RAISE policy must raise EmbeddingFailureError with reason='missing'
    when context is None — surfaces the failure for retry logic."""
    with pytest.raises(EmbeddingFailureError) as exc_info:
        compute_surprise([1.0, 0.0], None, fallback_policy=FallbackPolicy.RAISE)
    assert exc_info.value.reason == "missing"


@pytest.mark.unit
def test_surprise_fallback_raise_on_length_mismatch() -> None:
    """RAISE policy must raise EmbeddingFailureError with reason='length_mismatch'
    when candidate and context have different dimensions."""
    with pytest.raises(EmbeddingFailureError) as exc_info:
        compute_surprise(
            [1.0, 0.0],
            [1.0, 0.0, 0.0],
            fallback_policy=FallbackPolicy.RAISE,
        )
    assert exc_info.value.reason == "length_mismatch"


@pytest.mark.unit
def test_surprise_fallback_raise_on_zero_norm() -> None:
    """RAISE policy must raise EmbeddingFailureError with reason='zero_norm'
    when either vector is all-zeros (cosine similarity undefined)."""
    with pytest.raises(EmbeddingFailureError) as exc_info:
        compute_surprise(
            [0.0, 0.0],
            [1.0, 0.0],
            fallback_policy=FallbackPolicy.RAISE,
        )
    assert exc_info.value.reason == "zero_norm"


@pytest.mark.unit
def test_surprise_fallback_none_uses_configured_default() -> None:
    """fallback_policy=None must fall through to the module default
    (NEUTRAL → 0.5 on missing context), never raise."""
    result = compute_surprise([1.0], None, fallback_policy=None)
    assert result == 0.5


@pytest.mark.unit
def test_surprise_fallback_string_coercion_valid() -> None:
    """String values 'neutral', 'promote', 'demote', 'raise' must be
    accepted and coerced to their enum counterpart."""
    assert compute_surprise([1.0], None, fallback_policy="neutral") == 0.5
    assert compute_surprise([1.0], None, fallback_policy="promote") == 1.0
    assert compute_surprise([1.0], None, fallback_policy="demote") == 0.0

    with pytest.raises(EmbeddingFailureError):
        compute_surprise([1.0], None, fallback_policy="raise")


@pytest.mark.unit
def test_surprise_fallback_invalid_string_falls_back_to_default() -> None:
    """An unrecognised string policy must not raise — it must silently
    fall back to the module default (NEUTRAL)."""
    result = compute_surprise([1.0], None, fallback_policy="completely_invalid")
    assert result == 0.5


@pytest.mark.unit
def test_surprise_identical_vectors_return_zero() -> None:
    """Identical candidate and context vectors must yield surprise 0.0
    (cosine similarity 1.0 → distance 0.0)."""
    v = [1.0, 2.0, 3.0]
    result = compute_surprise(v, v, fallback_policy=FallbackPolicy.RAISE)
    assert abs(result) < 1e-9


@pytest.mark.unit
def test_surprise_opposite_vectors_return_one() -> None:
    """Opposite-direction vectors must yield surprise 1.0 (cosine
    similarity -1.0 → distance 1.0)."""
    result = compute_surprise(
        [1.0, 0.0], [-1.0, 0.0], fallback_policy=FallbackPolicy.RAISE
    )
    assert abs(result - 1.0) < 1e-9


@pytest.mark.unit
def test_surprise_orthogonal_vectors_return_half() -> None:
    """Orthogonal vectors must yield surprise 0.5 (cosine similarity 0
    → distance 0.5)."""
    result = compute_surprise(
        [1.0, 0.0], [0.0, 1.0], fallback_policy=FallbackPolicy.RAISE
    )
    assert abs(result - 0.5) < 1e-9

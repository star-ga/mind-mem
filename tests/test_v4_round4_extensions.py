"""Tests for round-4 audit extensions.

Covers six surfaces shipped against the round-4 multi-LLM audit:

    backpressure         hysteresis controller, recommended_pause,
                         wait_until_clear
    health               7-probe health_check + custom probes + status
                         degradation
    logging_context      contextvar stack + with_correlation_id +
                         StructuredLogFilter
    block_metadata       tags + ttl + list_blocks_by_tag + schema
                         validators
    observability        cardinality cap returns overflow sentinel +
                         drop counter
    eviction             debug_plan grouping + set_active_policy /
                         active_policy
    surprise_retrieval   FallbackPolicy NEUTRAL/PROMOTE/DEMOTE/RAISE
                         on bad embeddings
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

import pytest
from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4 import backpressure as bp_mod
from mind_mem.v4.backpressure import (
    DEFAULT_HIGH_WATERMARK,
    DEFAULT_LOW_WATERMARK,
    DEFAULT_MAX_PAUSE_S,
    BackpressureController,
    controller,
)
from mind_mem.v4.backpressure import FLAG as BP_FLAG
from mind_mem.v4.block_metadata import FLAG as BM_FLAG
from mind_mem.v4.block_metadata import (
    BlockMetadata,
    SchemaValidationResult,
    available_validators,
    delete_block_metadata,
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
    StructuredLogFilter,
    current_context,
    with_context,
    with_correlation_id,
)
from mind_mem.v4.observability import FLAG as OBS_FLAG
from mind_mem.v4.observability import (
    MAX_CARDINALITY,
    counter,
    gauge,
    reset_for_tests,
    snapshot,
)
from mind_mem.v4.surprise_retrieval import (
    DEFAULT_FALLBACK_POLICY,
    EmbeddingFailureError,
    FallbackPolicy,
    compute_surprise,
)


def _cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: bool) -> Path:
    block = {k: {"enabled": v} for k, v in flags.items()}
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


def _cfg_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block: dict) -> Path:
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_obs() -> None:
    reset_for_tests()
    reset_custom_probes_for_tests()
    yield
    reset_for_tests()
    reset_custom_probes_for_tests()


# ===========================================================================
# backpressure.py
# ===========================================================================


@pytest.fixture
def bp_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{BP_FLAG: True})


@pytest.fixture
def bp_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{BP_FLAG: False})


@pytest.mark.unit
def test_bp_flag_off_blocks_controller(bp_off: Path) -> None:
    bp_mod.reset_for_tests()
    with pytest.raises(FeatureDisabledError):
        controller()


@pytest.mark.unit
def test_bp_hysteresis_high_then_low(bp_on: Path) -> None:
    ctrl = BackpressureController(high_watermark=100, low_watermark=20)
    assert ctrl.is_overloaded() is False
    ctrl.set_depth(150)
    assert ctrl.is_overloaded() is True
    # Still overloaded between watermarks (hysteresis).
    ctrl.set_depth(50)
    assert ctrl.is_overloaded() is True
    # Below low watermark — clears.
    ctrl.set_depth(10)
    assert ctrl.is_overloaded() is False


@pytest.mark.unit
def test_bp_recommended_pause_scales_with_depth(bp_on: Path) -> None:
    ctrl = BackpressureController(high_watermark=100, low_watermark=20, max_pause_seconds=2.0)
    ctrl.set_depth(50)  # below high → no pause yet
    assert ctrl.recommended_pause() == 0.0
    ctrl.set_depth(100)  # at high → some pause
    p1 = ctrl.recommended_pause()
    ctrl.set_depth(200)  # above high → larger pause, capped at max
    p2 = ctrl.recommended_pause()
    assert p1 >= 0.0
    assert p2 >= p1
    assert p2 <= 2.0


@pytest.mark.unit
def test_bp_wait_until_clear_returns_immediately_when_clear(bp_on: Path) -> None:
    ctrl = BackpressureController(high_watermark=100, low_watermark=20)
    t0 = time.perf_counter()
    ctrl.wait_until_clear(timeout=0.5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2


@pytest.mark.unit
def test_bp_defaults_match_documented(bp_on: Path) -> None:
    ctrl = BackpressureController()
    assert ctrl.high_watermark == DEFAULT_HIGH_WATERMARK
    assert ctrl.low_watermark == DEFAULT_LOW_WATERMARK
    assert ctrl.max_pause_seconds == DEFAULT_MAX_PAUSE_S


# ===========================================================================
# health.py
# ===========================================================================


@pytest.fixture
def health_ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Health check itself isn't flag-gated (it surveys other flags),
    # but to exercise the per-module checks we enable several flags.
    return _cfg(
        tmp_path,
        monkeypatch,
        **{
            "tier_memory": True,
            "block_kinds": True,
            "federation": True,
            "self_editing": True,
            "kind_summaries": True,
            "block_metadata": True,
            OBS_FLAG: True,
            EVICT_FLAG: True,
        },
    )


@pytest.mark.unit
def test_health_returns_status_modules_and_latency(health_ws: Path) -> None:
    out = health_check(health_ws)
    assert "status" in out
    assert out["status"] in {"ok", "degraded", "fail"}
    assert "modules" in out
    assert isinstance(out["modules"], dict)
    # Each module reports a ModuleStatus string ("ok", "missing",
    # "disabled", "error: ..."). Flat string, not nested dict.
    for name, value in out["modules"].items():
        assert isinstance(value, str), f"module {name!r} should be a str status"
    assert "latency_ms" in out
    assert isinstance(out["latency_ms"], (int, float))
    assert out["latency_ms"] >= 0
    assert "checked_at" in out


@pytest.mark.unit
def test_health_custom_probe_returning_error_marks_aggregate_fail(
    health_ws: Path,
) -> None:
    register_health_probe("custom_fail_r4", lambda _ws: "error: synthetic")
    out = health_check(health_ws)
    assert out["modules"]["custom_fail_r4"] == "error: synthetic"
    assert out["status"] == "fail"


@pytest.mark.unit
def test_health_custom_probe_ok_does_not_degrade(health_ws: Path) -> None:
    register_health_probe("custom_ok_r4", lambda _ws: "ok")
    out = health_check(health_ws)
    assert out["modules"]["custom_ok_r4"] == "ok"


@pytest.mark.unit
def test_health_probe_exception_marks_aggregate_fail(health_ws: Path) -> None:
    def boom(_ws: Path) -> str:
        raise RuntimeError("synthetic blow-up")

    register_health_probe("custom_boom_r4", boom)
    out = health_check(health_ws)
    assert out["modules"]["custom_boom_r4"].startswith("error:")
    assert out["status"] == "fail"


@pytest.mark.unit
def test_health_custom_probe_returning_missing_marks_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a workspace with no DB so built-in probes also report
    # "disabled"/"missing" rather than "ok". Then add a custom
    # missing-only probe and assert aggregate becomes "degraded" not
    # "fail" (no error: prefix).
    _cfg(tmp_path, monkeypatch)
    register_health_probe("custom_missing_r4", lambda _ws: "missing")
    out = health_check(tmp_path)
    assert out["modules"]["custom_missing_r4"] == "missing"
    assert out["status"] == "degraded"


# ===========================================================================
# logging_context.py
# ===========================================================================


@pytest.mark.unit
def test_logging_context_stack_push_pop_round_trip() -> None:
    assert current_context() == {}
    token = LogContext.push(workspace="/tmp/x", agent_id="A")
    assert current_context() == {"workspace": "/tmp/x", "agent_id": "A"}
    LogContext.pop(token)
    assert current_context() == {}


@pytest.mark.unit
def test_with_context_manager_pops_on_exit() -> None:
    with with_context(stage="ingest"):
        assert current_context()["stage"] == "ingest"
    assert "stage" not in current_context()


@pytest.mark.unit
def test_with_context_inner_overrides_outer_for_same_key() -> None:
    with with_context(workspace="/outer"):
        with with_context(workspace="/inner"):
            assert current_context()["workspace"] == "/inner"
        assert current_context()["workspace"] == "/outer"


@pytest.mark.unit
def test_with_correlation_id_attaches_uuid() -> None:
    captured: dict[str, str] = {}

    @with_correlation_id
    def fn() -> None:
        captured.update(current_context())

    fn()
    assert "correlation_id" in captured
    assert len(captured["correlation_id"]) > 0


@pytest.mark.unit
def test_with_correlation_id_inherits_existing_id() -> None:
    captured: list[str] = []

    @with_correlation_id
    def inner() -> None:
        captured.append(current_context()["correlation_id"])

    @with_correlation_id
    def outer() -> None:
        captured.append(current_context()["correlation_id"])
        inner()

    outer()
    assert len(captured) == 2
    assert captured[0] == captured[1]


@pytest.mark.unit
def test_structured_log_filter_attaches_ctx() -> None:
    f = StructuredLogFilter()
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="m",
        args=None,
        exc_info=None,
    )
    with with_context(workspace="/tmp"):
        assert f.filter(record) is True
        assert getattr(record, "ctx") == {"workspace": "/tmp"}


# ===========================================================================
# block_metadata.py
# ===========================================================================


@pytest.fixture
def bm_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{BM_FLAG: True})


@pytest.fixture
def bm_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{BM_FLAG: False})


@pytest.mark.unit
def test_bm_flag_off_blocks_set(bm_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        set_block_metadata(bm_off, "b1", tags={"k": "v"})


@pytest.mark.unit
def test_bm_set_get_delete_round_trip(bm_on: Path) -> None:
    md = set_block_metadata(bm_on, "b1", tags={"author": "alice"}, ttl_seconds=60)
    assert isinstance(md, BlockMetadata)
    fetched = get_block_metadata(bm_on, "b1")
    assert fetched is not None
    assert fetched.block_id == "b1"
    assert fetched.tags == {"author": "alice"}
    assert fetched.ttl_seconds == 60
    assert delete_block_metadata(bm_on, "b1") is True
    assert get_block_metadata(bm_on, "b1") is None


@pytest.mark.unit
def test_bm_get_returns_none_when_db_missing(bm_on: Path) -> None:
    # No prior writes — schema absent until first write.
    assert get_block_metadata(bm_on, "missing") is None


@pytest.mark.unit
def test_bm_list_blocks_by_tag(bm_on: Path) -> None:
    set_block_metadata(bm_on, "b1", tags={"env": "prod"})
    set_block_metadata(bm_on, "b2", tags={"env": "prod"})
    set_block_metadata(bm_on, "b3", tags={"env": "dev"})
    out = list_blocks_by_tag(bm_on, "env", "prod", limit=10)
    assert set(out) == {"b1", "b2"}


@pytest.mark.unit
def test_bm_list_blocks_by_tag_empty_when_no_match(bm_on: Path) -> None:
    set_block_metadata(bm_on, "b1", tags={"env": "prod"})
    assert list_blocks_by_tag(bm_on, "env", "missing", limit=10) == []


@pytest.mark.unit
def test_bm_validator_open_by_default(bm_on: Path) -> None:
    result = validate_block("anykind", {"foo": 1})
    assert result.ok is True
    assert "no_validator" in result.reason


@pytest.mark.unit
def test_bm_validator_can_reject(bm_on: Path) -> None:
    register_schema_validator(
        "claim",
        lambda payload: SchemaValidationResult(ok="text" in payload, reason="text required"),
    )
    assert validate_block("claim", {"text": "hi"}).ok is True
    assert validate_block("claim", {}).ok is False
    assert "claim" in available_validators()


@pytest.mark.unit
def test_bm_validator_exception_marks_invalid(bm_on: Path) -> None:
    def boom(_payload: dict) -> SchemaValidationResult:
        raise ValueError("validator broken")

    register_schema_validator("bad", boom)
    result = validate_block("bad", {})
    assert result.ok is False
    assert "validator_raised" in result.reason


@pytest.mark.unit
def test_bm_validator_wrong_return_type_rejected(bm_on: Path) -> None:
    register_schema_validator("wrong", lambda _payload: True)  # type: ignore[arg-type,return-value]
    result = validate_block("wrong", {})
    assert result.ok is False
    assert "wrong_type" in result.reason


@pytest.mark.unit
def test_bm_ensure_schema_idempotent(bm_on: Path) -> None:
    ensure_metadata_schema(bm_on)
    ensure_metadata_schema(bm_on)
    db = bm_on / "index.db"
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='block_metadata'").fetchone()
    assert row is not None


# ===========================================================================
# observability cardinality guard
# ===========================================================================


@pytest.fixture
def obs_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{OBS_FLAG: True})


@pytest.mark.unit
def test_observability_cardinality_returns_overflow_sentinel(obs_on: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shrink cap so we can hit it cheaply.
    monkeypatch.setattr("mind_mem.v4.observability.MAX_CARDINALITY", 5)
    for i in range(5):
        c = counter(f"v4.test.cnt.{i}")
        c.inc()
    # The 6th counter should be the overflow sentinel.
    overflow = counter("v4.test.cnt.6")
    overflow.inc()
    snap = snapshot()
    assert "v4.cardinality.dropped_counter" in snap
    assert snap["v4.cardinality.dropped_counter"] >= 1


@pytest.mark.unit
def test_observability_cardinality_default_is_large(obs_on: Path) -> None:
    assert MAX_CARDINALITY >= 1000


@pytest.mark.unit
def test_observability_cap_separate_per_kind(obs_on: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mind_mem.v4.observability.MAX_CARDINALITY", 3)
    counter("a")
    counter("b")
    counter("c")
    counter("d")  # over cap → overflow
    # Gauges have their own cap — should not be affected by counter cap.
    gauge("g1")
    gauge("g2")
    snap = snapshot()
    assert "v4.cardinality.dropped_counter" in snap
    assert "g1" in snap and "g2" in snap


# ===========================================================================
# eviction debug_plan + active_policy
# ===========================================================================


@pytest.fixture
def evict_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _cfg(tmp_path, monkeypatch, **{EVICT_FLAG: True})


@pytest.mark.unit
def test_eviction_debug_plan_groups_by_tag(evict_on: Path) -> None:
    plan = EvictionPlan(
        policy=EvictionPolicy.COMPOSITE,
        candidates=[
            ("b1", "lru:last_seen=2026-01-01"),
            ("b2", "lru:last_seen=2026-02-01"),
            ("b3", "low_surprise:s=0.05"),
        ],
    )
    grouped = plan.debug_plan()
    assert set(grouped["lru"]) == {"b1", "b2"}
    assert grouped["low_surprise"] == ["b3"]


@pytest.mark.unit
def test_eviction_debug_plan_handles_missing_colon(evict_on: Path) -> None:
    plan = EvictionPlan(policy=EvictionPolicy.LRU, candidates=[("b1", "no_colon_reason")])
    grouped = plan.debug_plan()
    assert "no_colon_reason" in grouped


@pytest.mark.unit
def test_eviction_set_active_policy_round_trip(evict_on: Path) -> None:
    set_active_policy(EvictionPolicy.AGE)
    assert active_policy() == EvictionPolicy.AGE
    set_active_policy("low_surprise")
    assert active_policy() == EvictionPolicy.LOW_SURPRISE
    # Restore default for subsequent tests.
    set_active_policy(EvictionPolicy.LRU)


@pytest.mark.unit
def test_eviction_set_active_policy_rejects_unknown(evict_on: Path) -> None:
    with pytest.raises(ValueError):
        set_active_policy("never_registered_policy")


@pytest.mark.unit
def test_eviction_set_active_policy_accepts_custom_after_register(
    evict_on: Path,
) -> None:
    register_policy("custom_round4", lambda *_a, **_k: [("b1", "custom:test")])
    set_active_policy("custom_round4")
    assert active_policy() == "custom_round4"
    set_active_policy(EvictionPolicy.LRU)


@pytest.mark.unit
def test_eviction_active_policy_flag_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(tmp_path, monkeypatch, **{EVICT_FLAG: False})
    with pytest.raises(FeatureDisabledError):
        active_policy()


# ===========================================================================
# surprise_retrieval FallbackPolicy
# ===========================================================================


@pytest.mark.unit
def test_surprise_default_fallback_is_neutral() -> None:
    assert DEFAULT_FALLBACK_POLICY is FallbackPolicy.NEUTRAL
    # Empty inputs → 0.5 with default policy
    assert compute_surprise([], [1.0]) == 0.5


@pytest.mark.unit
def test_surprise_fallback_promote_returns_max() -> None:
    assert compute_surprise([], [1.0], fallback_policy=FallbackPolicy.PROMOTE) == 1.0


@pytest.mark.unit
def test_surprise_fallback_demote_returns_zero() -> None:
    assert compute_surprise([], [1.0], fallback_policy=FallbackPolicy.DEMOTE) == 0.0


@pytest.mark.unit
def test_surprise_fallback_raise_on_missing() -> None:
    with pytest.raises(EmbeddingFailureError) as exc_info:
        compute_surprise([], [1.0], fallback_policy=FallbackPolicy.RAISE)
    assert exc_info.value.reason == "missing"


@pytest.mark.unit
def test_surprise_fallback_raise_on_length_mismatch() -> None:
    with pytest.raises(EmbeddingFailureError) as exc_info:
        compute_surprise([1.0, 2.0], [1.0], fallback_policy=FallbackPolicy.RAISE)
    assert exc_info.value.reason == "length_mismatch"


@pytest.mark.unit
def test_surprise_fallback_raise_on_zero_norm() -> None:
    with pytest.raises(EmbeddingFailureError) as exc_info:
        compute_surprise([0.0, 0.0], [1.0, 2.0], fallback_policy=FallbackPolicy.RAISE)
    assert exc_info.value.reason == "zero_norm"


@pytest.mark.unit
def test_surprise_fallback_string_coerced_to_enum() -> None:
    assert compute_surprise([], [1.0], fallback_policy="promote") == 1.0
    assert compute_surprise([], [1.0], fallback_policy="demote") == 0.0


@pytest.mark.unit
def test_surprise_fallback_unknown_string_falls_back_to_neutral() -> None:
    assert compute_surprise([], [1.0], fallback_policy="bogus") == 0.5


@pytest.mark.unit
def test_surprise_config_supplies_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg_raw(
        tmp_path,
        monkeypatch,
        {"surprise_retrieval": {"enabled": True, "fallback_policy": "promote"}},
    )
    # No explicit policy → reads from flag config
    assert compute_surprise([], [1.0]) == 1.0


@pytest.mark.unit
def test_surprise_normal_path_unaffected_by_fallback() -> None:
    # Identical vectors → 0.0 surprise (cos_sim = 1)
    s = compute_surprise([1.0, 0.0], [1.0, 0.0], fallback_policy=FallbackPolicy.RAISE)
    assert s == 0.0
    # Opposite vectors → 1.0 surprise (cos_sim = -1)
    s = compute_surprise([1.0, 0.0], [-1.0, 0.0], fallback_policy=FallbackPolicy.RAISE)
    assert s == 1.0

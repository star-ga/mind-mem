"""v3.2.0 gap tests — regression and edge-case coverage for new modules.

Covers:
- block_store_postgres: _validate_schema_name injection guard + _block_to_row edge cases
- block_store_postgres_replica: circuit-breaker cooldown expiry recovery
- recall_cache: LRU update-existing-key path, concurrent set safety, max_entries=1 edge
- mcp/tools/public: error envelopes for all invalid-dispatch paths
- telemetry: _fire_metric for propose_update / approve_apply / rollback spans
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# 1. block_store_postgres — pure-unit tests (no live DB needed)
# ===========================================================================


class TestValidateSchemaName:
    """_validate_schema_name must block SQL-injection candidates."""

    def test_valid_name_passes(self) -> None:
        from mind_mem.block_store_postgres import _validate_schema_name

        assert _validate_schema_name("mind_mem") == "mind_mem"
        assert _validate_schema_name("mm_test_abc123") == "mm_test_abc123"
        assert _validate_schema_name("A") == "A"

    def test_semicolon_in_name_raises(self) -> None:
        from mind_mem.block_store_postgres import _validate_schema_name

        with pytest.raises(ValueError, match="Unsafe Postgres schema name"):
            _validate_schema_name("mind_mem; DROP TABLE blocks--")

    def test_space_in_name_raises(self) -> None:
        from mind_mem.block_store_postgres import _validate_schema_name

        with pytest.raises(ValueError, match="Unsafe Postgres schema name"):
            _validate_schema_name("bad name")

    def test_empty_name_raises(self) -> None:
        from mind_mem.block_store_postgres import _validate_schema_name

        with pytest.raises(ValueError):
            _validate_schema_name("")

    def test_name_starting_with_digit_raises(self) -> None:
        from mind_mem.block_store_postgres import _validate_schema_name

        with pytest.raises(ValueError):
            _validate_schema_name("1bad")

    def test_single_quote_injection_raises(self) -> None:
        from mind_mem.block_store_postgres import _validate_schema_name

        with pytest.raises(ValueError):
            _validate_schema_name("x' OR '1'='1")


class TestBlockToRow:
    """_block_to_row edge cases — missing _id, empty Statement."""

    def test_missing_id_raises(self) -> None:
        from mind_mem.block_store_postgres import _block_to_row

        with pytest.raises(ValueError, match="_id"):
            _block_to_row({"Statement": "no id field"})

    def test_empty_statement_falls_back_to_json_text(self) -> None:
        from mind_mem.block_store_postgres import _block_to_row

        block = {"_id": "D-1", "_source_file": "f.md", "Status": "active"}
        _id, _fp, content, _meta = _block_to_row(block)
        assert "active" in content

    def test_statement_preferred_over_content(self) -> None:
        from mind_mem.block_store_postgres import _block_to_row

        block = {
            "_id": "D-2",
            "Statement": "Prefer this",
            "content": "not this",
        }
        _id, _fp, content, _meta = _block_to_row(block)
        assert content == "Prefer this"

    def test_private_keys_excluded_from_metadata(self) -> None:
        from mind_mem.block_store_postgres import _block_to_row

        block = {
            "_id": "D-3",
            "_source_file": "f.md",
            "_created_at": "2026-01-01",
            "Status": "active",
        }
        _id, _fp, _content, meta_json = _block_to_row(block)
        meta = json.loads(meta_json)
        assert "_id" not in meta
        assert "_source_file" not in meta
        assert "Status" in meta


# ===========================================================================
# 2. block_store_postgres_replica — circuit-breaker recovery after cooldown
# ===========================================================================


class TestCircuitBreakerCooldownExpiry:
    """After cooldown expires a previously-tripped replica becomes healthy again."""

    def test_replica_healthy_after_cooldown_expires(self) -> None:
        from mind_mem.block_store_postgres_replica import _ReplicaState

        mock_store = MagicMock()
        state = _ReplicaState(store=mock_store)

        # Trip the breaker.
        for _ in range(3):
            state.record_failure()
        assert not state.healthy

        # Wind the clock back: set cooling_until to the past.
        state.cooling_until = time.time() - 1.0
        assert state.healthy

    def test_record_success_resets_failure_count(self) -> None:
        from mind_mem.block_store_postgres_replica import _ReplicaState

        mock_store = MagicMock()
        state = _ReplicaState(store=mock_store)
        state.record_failure()
        state.record_failure()
        state.record_success()
        assert state.failure_count == 0
        assert state.cooling_until == 0.0


# ===========================================================================
# 3. recall_cache — missing branches
# ===========================================================================


class TestLRUUpdateExistingKey:
    """set() on an already-present key must update it and move to end."""

    def test_overwrite_moves_to_end(self) -> None:
        from mind_mem.recall_cache import LRUCache

        cache = LRUCache(max_entries=3)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        # Overwrite "a" — it moves to the end (most-recent) position.
        cache.set("a", "updated")
        # Now add "d" — the LRU entry ("b") should be evicted, not "a".
        cache.set("d", "4")
        assert cache.get("a") == "updated"
        assert cache.get("b") is None

    def test_concurrent_set_no_corruption(self) -> None:
        """Multiple threads writing distinct keys must not corrupt the cache."""
        from mind_mem.recall_cache import LRUCache

        cache = LRUCache(max_entries=200)
        errors: list[Exception] = []

        def _write(n: int) -> None:
            try:
                cache.set(f"key-{n}", f"val-{n}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_max_entries_one_keeps_only_last(self) -> None:
        """An LRU with max_entries=1 retains only the most-recently-set entry."""
        from mind_mem.recall_cache import LRUCache

        cache = LRUCache(max_entries=1)
        cache.set("a", "first")
        cache.set("b", "second")
        assert cache.get("a") is None
        assert cache.get("b") == "second"


class TestRecallCacheMetrics:
    """RecallCache hit/miss increments metrics correctly."""

    def test_miss_increments_miss_counter(self) -> None:
        from mind_mem.recall_cache import RecallCache

        cache = RecallCache()
        with patch("mind_mem.recall_cache.metrics") as mock_metrics:
            cache.get("nonexistent-key")
            mock_metrics.inc.assert_called_with("recall_cache_misses_total")

    def test_hit_increments_hit_counter(self) -> None:
        from mind_mem.recall_cache import RecallCache

        cache = RecallCache()
        cache.set("hit-key", "value")
        with patch("mind_mem.recall_cache.metrics") as mock_metrics:
            result = cache.get("hit-key")
        assert result == "value"
        mock_metrics.inc.assert_called_with("recall_cache_hits_total")


# ===========================================================================
# 4. mcp/tools/public — error envelope for invalid dispatch modes
# ===========================================================================


class TestPublicDispatcherErrorEnvelopes:
    """Every dispatcher must return a JSON error envelope for unknown modes."""

    def _parse(self, result: str) -> dict:
        return json.loads(result)

    def test_recall_unknown_mode_returns_error(self) -> None:
        from mind_mem.mcp.tools.public import recall

        result = self._parse(recall.__wrapped__("q", mode="no-such-mode"))  # type: ignore[attr-defined]
        assert "error" in result
        assert "valid_modes" in result

    def test_recall_similar_missing_block_id_returns_error(self) -> None:
        from mind_mem.mcp.tools.public import recall

        result = self._parse(recall.__wrapped__("q", mode="similar", block_id=""))  # type: ignore[attr-defined]
        assert "error" in result
        assert "block_id" in result["error"]

    def test_staged_change_unknown_phase_returns_error(self) -> None:
        from mind_mem.mcp.tools.public import staged_change

        result = self._parse(staged_change.__wrapped__(phase="unknown"))  # type: ignore[attr-defined]
        assert "error" in result
        assert "valid_phases" in result

    def test_staged_change_propose_missing_statement_returns_error(self) -> None:
        from mind_mem.mcp.tools.public import staged_change

        result = self._parse(
            staged_change.__wrapped__(phase="propose", block_type="decision", statement="")  # type: ignore[attr-defined]
        )
        assert "error" in result

    def test_memory_verify_unknown_mode_returns_error(self) -> None:
        from mind_mem.mcp.tools.public import memory_verify

        result = self._parse(memory_verify.__wrapped__(mode="bogus"))  # type: ignore[attr-defined]
        assert "error" in result
        assert "valid_modes" in result

    def test_memory_verify_merkle_missing_args_returns_error(self) -> None:
        from mind_mem.mcp.tools.public import memory_verify

        result = self._parse(memory_verify.__wrapped__(mode="merkle", block_id="", content_hash=""))  # type: ignore[attr-defined]
        assert "error" in result

    def test_kernels_get_missing_name_returns_error(self) -> None:
        from mind_mem.mcp.tools.public import kernels

        result = self._parse(kernels.__wrapped__(action="get", name=""))  # type: ignore[attr-defined]
        assert "error" in result

    def test_kernels_unknown_action_returns_error(self) -> None:
        from mind_mem.mcp.tools.public import kernels

        result = self._parse(kernels.__wrapped__(action="delete"))  # type: ignore[attr-defined]
        assert "error" in result
        assert "valid_actions" in result


# ===========================================================================
# 5. telemetry — _fire_metric for non-recall spans
# ===========================================================================


class TestFireMetricAllSpans:
    """_fire_metric must dispatch to the right Prometheus counter for each span."""

    def _run_with_mock_counters(self, span_name: str) -> MagicMock:
        from mind_mem import telemetry

        counter = MagicMock()
        sentinel = MagicMock()  # non-None value so _init_prom_metrics fast-paths out
        original_prom = telemetry._HAS_PROM
        orig_rd = telemetry._recall_duration  # must be non-None to skip _init_prom_metrics
        orig_pu = telemetry._propose_update_total
        orig_sc = telemetry._scan_total
        orig_ap = telemetry._apply_total
        orig_rb = telemetry._apply_rollback_total
        try:
            telemetry._HAS_PROM = True  # type: ignore[assignment]
            # Set _recall_duration to a non-None sentinel so _init_prom_metrics
            # fast-paths out without overwriting our injected mocks.
            telemetry._recall_duration = sentinel
            telemetry._propose_update_total = counter
            telemetry._scan_total = counter
            telemetry._apply_total = counter
            telemetry._apply_rollback_total = counter
            telemetry._fire_metric(span_name, 0.05)
        finally:
            telemetry._HAS_PROM = original_prom  # type: ignore[assignment]
            telemetry._recall_duration = orig_rd
            telemetry._propose_update_total = orig_pu
            telemetry._scan_total = orig_sc
            telemetry._apply_total = orig_ap
            telemetry._apply_rollback_total = orig_rb
        return counter

    def test_propose_update_increments_counter(self) -> None:
        counter = self._run_with_mock_counters("propose_update")
        counter.inc.assert_called()

    def test_scan_increments_counter(self) -> None:
        counter = self._run_with_mock_counters("scan")
        counter.inc.assert_called()

    def test_approve_apply_increments_counter(self) -> None:
        counter = self._run_with_mock_counters("approve_apply")
        counter.inc.assert_called()

    def test_rollback_proposal_increments_counter(self) -> None:
        counter = self._run_with_mock_counters("rollback_proposal")
        counter.inc.assert_called()

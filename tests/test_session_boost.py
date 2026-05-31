"""v3.3.0 Tier 2 #5 — session-boundary preservation via recall-side boost.

Given top-N seed results in session S, bump other same-session
candidates. Idempotent under no session metadata; fail-open;
re-sorts by boosted score.
"""

from __future__ import annotations

import pytest

from mind_mem.session_boost import (
    apply_session_boost,
    is_session_boost_enabled,
    resolve_session_boost_config,
)


def _block(bid: str, score: float, **extra) -> dict:
    return {"_id": bid, "score": score, **extra}


class TestSessionExtraction:
    def test_session_from_dia_id(self) -> None:
        from mind_mem.session_boost import _session_of

        assert _session_of({"dia_id": "D1:3"}) == "D1"
        assert _session_of({"dia_id": "DIA-D2-10"}) == "D2"

    def test_session_from_explicit_field(self) -> None:
        from mind_mem.session_boost import _session_of

        assert _session_of({"SessionId": "sess-17"}) == "sess-17"
        assert _session_of({"session_id": "Z42"}) == "Z42"

    def test_session_from_block_id_prefix(self) -> None:
        from mind_mem.session_boost import _session_of

        assert _session_of({"_id": "DIA-D5-7"}) == "D5"

    def test_no_session_info_returns_none(self) -> None:
        from mind_mem.session_boost import _session_of

        assert _session_of({"_id": "D-20260420-001"}) is None
        assert _session_of({}) is None


class TestApplySessionBoost:
    def test_no_session_info_unchanged(self) -> None:
        results = [_block("x", 5.0), _block("y", 3.0)]
        out = apply_session_boost(list(results))
        assert [(b["_id"], b["score"]) for b in out] == [("x", 5.0), ("y", 3.0)]

    def test_same_session_blocks_boosted(self) -> None:
        results = [
            _block("a", 10.0, dia_id="D1:3"),
            _block("b", 8.0, dia_id="D2:5"),
            _block("c", 7.0, dia_id="D1:7"),
            _block("d", 6.0, dia_id="D3:1"),
        ]
        # top seed count 1 → only D1 is the active session.
        out = apply_session_boost(list(results), top_seed_count=1, boost=0.5)
        # D1 members (a, c) boosted; D2 (b), D3 (d) unchanged.
        by_id = {b["_id"]: b for b in out}
        assert by_id["a"]["_session_boost"] == 0.5
        assert by_id["c"]["_session_boost"] == 0.5
        assert "_session_boost" not in by_id["b"]
        assert "_session_boost" not in by_id["d"]
        # a: 10 * 1.5 = 15; c: 7 * 1.5 = 10.5.
        assert by_id["a"]["score"] == pytest.approx(15.0)
        assert by_id["c"]["score"] == pytest.approx(10.5)

    def test_multiple_seed_sessions_both_active(self) -> None:
        results = [
            _block("a", 10.0, dia_id="D1:3"),
            _block("b", 9.0, dia_id="D2:5"),
            _block("c", 7.0, dia_id="D1:7"),  # same session as a
            _block("d", 6.0, dia_id="D2:9"),  # same session as b
            _block("e", 5.0, dia_id="D3:1"),  # different
        ]
        out = apply_session_boost(list(results), top_seed_count=2, boost=0.5)
        # D1 and D2 both active (from top 2 seeds).
        by_id = {b["_id"]: b for b in out}
        assert "_session_boost" in by_id["a"]
        assert "_session_boost" in by_id["b"]
        assert "_session_boost" in by_id["c"]
        assert "_session_boost" in by_id["d"]
        assert "_session_boost" not in by_id["e"]

    def test_resort_after_boost(self) -> None:
        """If the boost lifts a lower-ranked same-session block above
        a higher-ranked other-session block, the order updates."""
        results = [
            _block("seed", 10.0, dia_id="D1:3"),
            _block("other", 9.0, dia_id="D2:5"),
            _block("cousin", 7.0, dia_id="D1:7"),  # boosted to 10.5
        ]
        out = apply_session_boost(list(results), top_seed_count=1, boost=0.5)
        ids = [b["_id"] for b in out]
        # After boost: seed=15, cousin=10.5, other=9.
        assert ids == ["seed", "cousin", "other"]

    def test_zero_boost_noop(self) -> None:
        results = [_block("a", 10.0, dia_id="D1:3"), _block("b", 5.0, dia_id="D1:5")]
        out = apply_session_boost(list(results), boost=0.0)
        assert [b["_id"] for b in out] == ["a", "b"]
        assert all("_session_boost" not in b for b in out)

    def test_empty_results_unchanged(self) -> None:
        assert apply_session_boost([]) == []


class TestEnableResolution:
    def test_off_without_config(self) -> None:
        assert is_session_boost_enabled(None, []) is False
        assert is_session_boost_enabled({}, []) is False

    def test_explicit_enabled_regardless_of_results(self) -> None:
        cfg = {"retrieval": {"session_boost": {"enabled": True}}}
        assert is_session_boost_enabled(cfg, []) is True

    def test_auto_enable_requires_session_info(self) -> None:
        cfg = {"retrieval": {"session_boost": {}}}
        # No dia_id → skip auto-enable.
        assert is_session_boost_enabled(cfg, [{"_id": "D-20260420-001", "score": 1.0}]) is False
        # With dia_id → fire.
        assert is_session_boost_enabled(cfg, [{"_id": "DIA-D1-3", "score": 1.0}]) is True

    def test_auto_enable_false_overrides(self) -> None:
        cfg = {"retrieval": {"session_boost": {"auto_enable": False}}}
        assert is_session_boost_enabled(cfg, [{"dia_id": "D1:3", "score": 1.0}]) is False


class TestConfigResolution:
    def test_defaults(self) -> None:
        assert resolve_session_boost_config(None) == {"top_seed_count": 3, "boost": 0.3}

    def test_custom_values(self) -> None:
        cfg = {"retrieval": {"session_boost": {"top_seed_count": 5, "boost": 0.8}}}
        out = resolve_session_boost_config(cfg)
        assert out["top_seed_count"] == 5
        assert out["boost"] == pytest.approx(0.8)

    def test_invalid_values_fall_back(self) -> None:
        cfg = {"retrieval": {"session_boost": {"top_seed_count": -1, "boost": 999}}}
        assert resolve_session_boost_config(cfg) == {"top_seed_count": 3, "boost": 0.3}

"""v3.3.0 Tier 4 #10 — per-tier learned weights override.

Operators can now override the hard-coded tier-boost multipliers via
``retrieval.tier_boost_weights`` in ``mind-mem.json``. The offline
training loop that grid-searches optimal values from LoCoMo results
is in ``benchmarks/tier_weight_search.py`` (shipped separately);
these tests cover only the in-code resolution contract.
"""

from __future__ import annotations

import pytest

from mind_mem.tier_recall import _TIER_BOOST, resolve_tier_weights


class TestResolveTierWeights:
    def test_baseline_when_no_config(self) -> None:
        assert resolve_tier_weights(None) == _TIER_BOOST
        assert resolve_tier_weights({}) == _TIER_BOOST

    def test_override_by_name(self) -> None:
        cfg = {
            "retrieval": {
                "tier_boost_weights": {
                    "WORKING": 0.5,
                    "VERIFIED": 2.5,
                },
            }
        }
        w = resolve_tier_weights(cfg)
        assert w[1] == pytest.approx(0.5)  # WORKING
        assert w[2] == _TIER_BOOST[2]  # SHARED unchanged
        assert w[3] == _TIER_BOOST[3]  # LONG_TERM unchanged
        assert w[4] == pytest.approx(2.5)  # VERIFIED

    def test_override_by_integer_key(self) -> None:
        cfg = {"retrieval": {"tier_boost_weights": {1: 0.3, 4: 3.0}}}
        w = resolve_tier_weights(cfg)
        assert w[1] == pytest.approx(0.3)
        assert w[4] == pytest.approx(3.0)

    def test_override_by_string_digit(self) -> None:
        """JSON config keys are strings; '1' should resolve to tier 1."""
        cfg = {"retrieval": {"tier_boost_weights": {"1": 0.9}}}
        w = resolve_tier_weights(cfg)
        assert w[1] == pytest.approx(0.9)

    def test_unknown_tier_name_ignored(self) -> None:
        cfg = {"retrieval": {"tier_boost_weights": {"NONSENSE": 99.0}}}
        w = resolve_tier_weights(cfg)
        assert w == _TIER_BOOST  # unchanged, no garbage value injected

    def test_non_numeric_value_ignored(self) -> None:
        cfg = {"retrieval": {"tier_boost_weights": {"WORKING": "not-a-float"}}}
        w = resolve_tier_weights(cfg)
        assert w[1] == _TIER_BOOST[1]

    def test_non_positive_value_ignored(self) -> None:
        """Boosts multiply scores — a 0 or negative multiplier is nonsensical."""
        cfg = {"retrieval": {"tier_boost_weights": {"WORKING": 0, "SHARED": -0.5}}}
        w = resolve_tier_weights(cfg)
        assert w[1] == _TIER_BOOST[1]
        assert w[2] == _TIER_BOOST[2]

    def test_case_insensitive_tier_name(self) -> None:
        cfg = {"retrieval": {"tier_boost_weights": {"working": 0.6, "verified": 2.1}}}
        w = resolve_tier_weights(cfg)
        assert w[1] == pytest.approx(0.6)
        assert w[4] == pytest.approx(2.1)

    def test_malformed_retrieval_section_falls_back(self) -> None:
        cfg = {"retrieval": "not-a-dict"}
        assert resolve_tier_weights(cfg) == _TIER_BOOST

    def test_malformed_weights_section_falls_back(self) -> None:
        cfg = {"retrieval": {"tier_boost_weights": "not-a-dict"}}
        assert resolve_tier_weights(cfg) == _TIER_BOOST

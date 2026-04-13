# Copyright 2026 STARGA, Inc.
"""Tests for speculative prefetch predictor (v2.0.0b1)."""

from __future__ import annotations

import threading

import pytest

from mind_mem.speculative_prefetch import (
    PrefetchPredictor,
    PrefetchStats,
    get_default_predictor,
    reset_default_predictor,
    signature,
)


# ---------------------------------------------------------------------------
# signature()
# ---------------------------------------------------------------------------


class TestSignature:
    def test_same_intent_same_signature_despite_phrasing(self) -> None:
        a = signature("How do I reset my JWT token?")
        b = signature("Reset the JWT token for me")
        c = signature("jwt token reset")
        assert a == b == c

    def test_stopwords_stripped(self) -> None:
        # Truly empty / whitespace-only queries hash to the sentinel.
        assert signature("") == "signature:empty"
        assert signature("   ") == "signature:empty"

    def test_all_stopword_queries_get_distinct_buckets(self) -> None:
        # Noise queries with different surface text must NOT collide on
        # the sentinel bucket — pooling them breaks prediction accuracy.
        s1 = signature("the a an is it")
        s2 = signature("on or so that")
        assert s1 != s2
        assert s1.startswith("empty:") and s2.startswith("empty:")

    def test_case_insensitive(self) -> None:
        assert signature("JWT Token") == signature("jwt token")

    def test_non_alphanumeric_stripped(self) -> None:
        # Non-alphanumeric-only queries tokenise to an empty list, so
        # they go through the raw-text fallback (still not the sentinel).
        assert signature("!@#$%").startswith("empty:")
        assert signature("jwt?!?!") == signature("jwt")

    def test_different_intents_different_signatures(self) -> None:
        assert signature("jwt token") != signature("password reset")

    def test_signature_length_bounded(self) -> None:
        assert len(signature("the quick brown fox")) == 16

    def test_long_input_is_truncated_not_oomed(self) -> None:
        # 10MB query must not burn CPU indefinitely on the regex engine.
        huge = "jwt " * 2_500_000
        sig = signature(huge)
        # Same tokens as the short form because truncation drops the
        # rest; the signature is deterministic regardless of tail.
        assert sig == signature("jwt")

    def test_unicode_tokens_distinct(self) -> None:
        # Non-ASCII queries must not all collapse onto the empty bucket.
        a = signature("你好 世界")
        b = signature("密码 重置")
        assert a != b
        assert a != "signature:empty"
        assert b != "signature:empty"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_rejects_zero_max_signatures(self) -> None:
        with pytest.raises(ValueError, match="max_signatures"):
            PrefetchPredictor(max_signatures=0)

    def test_rejects_zero_per_bucket_cap(self) -> None:
        with pytest.raises(ValueError, match="per_bucket_cap"):
            PrefetchPredictor(per_bucket_cap=0)


# ---------------------------------------------------------------------------
# Observe + predict
# ---------------------------------------------------------------------------


class TestObservePredict:
    def test_unseen_query_returns_empty(self) -> None:
        p = PrefetchPredictor()
        assert p.predict("unseen query") == []

    def test_single_observation_produces_prediction(self) -> None:
        p = PrefetchPredictor()
        p.observe("JWT auth", ["D-001", "D-002"])
        predicted = p.predict("JWT auth")
        assert set(predicted) == {"D-001", "D-002"}

    def test_prediction_ranked_by_frequency(self) -> None:
        p = PrefetchPredictor()
        p.observe("JWT auth", ["D-001", "D-002"])
        p.observe("JWT auth", ["D-001"])
        p.observe("JWT auth", ["D-001"])
        ranked = p.predict("JWT auth", limit=2)
        assert ranked[0] == "D-001"

    def test_tiebreak_by_block_id(self) -> None:
        p = PrefetchPredictor()
        p.observe("JWT", ["D-002", "D-001"])
        ranked = p.predict("JWT", limit=2)
        # Both seen once; ordering should be deterministic (ascending id).
        assert ranked == ["D-001", "D-002"]

    def test_different_phrasing_shares_bucket(self) -> None:
        p = PrefetchPredictor()
        p.observe("JWT token reset", ["A"])
        # Same signature despite different wording
        assert p.predict("reset jwt token") == ["A"]

    def test_limit_caps_prediction(self) -> None:
        p = PrefetchPredictor()
        p.observe("topic", [f"D-{i}" for i in range(10)])
        assert len(p.predict("topic", limit=3)) == 3

    def test_limit_zero_returns_empty(self) -> None:
        p = PrefetchPredictor()
        p.observe("topic", ["A"])
        assert p.predict("topic", limit=0) == []

    def test_falsy_block_ids_skipped(self) -> None:
        p = PrefetchPredictor()
        p.observe("topic", ["", None, "A"])  # type: ignore[list-item]
        assert p.predict("topic") == ["A"]


# ---------------------------------------------------------------------------
# Bucket trimming
# ---------------------------------------------------------------------------


class TestTrimming:
    def test_per_bucket_cap_enforced(self) -> None:
        p = PrefetchPredictor(per_bucket_cap=3)
        p.observe("topic", ["A", "B", "C", "D", "E"])
        predicted = p.predict("topic", limit=10)
        # Cap=3 — after re-hydration, the top-3 most-observed remain.
        assert len(predicted) == 3

    def test_signature_lru_trim(self) -> None:
        p = PrefetchPredictor(max_signatures=2)
        p.observe("one topic", ["X"])
        p.observe("two topic", ["Y"])
        p.observe("three topic", ["Z"])
        # "one topic" should have been evicted as LRU.
        assert p.predict("one topic") == []
        assert p.predict("two topic") == ["Y"]
        assert p.predict("three topic") == ["Z"]

    def test_recently_used_signature_survives_trim(self) -> None:
        p = PrefetchPredictor(max_signatures=2)
        p.observe("one topic", ["X"])
        p.observe("two topic", ["Y"])
        # Touch "one topic" via a predict so it becomes MRU.
        p.predict("one topic")
        p.observe("three topic", ["Z"])
        # "two topic" should be the LRU victim now.
        assert p.predict("one topic") == ["X"]
        assert p.predict("two topic") == []
        assert p.predict("three topic") == ["Z"]

    def test_bucket_trim_preserves_survivor_frequencies(self) -> None:
        """Survivors must keep their observation counts after trim.

        The earlier reset-to-1 policy froze the bucket on whatever top-N
        happened to arrive first and destroyed frequency information.
        """
        p = PrefetchPredictor(per_bucket_cap=3)
        # A is observed heavily, then B, C, D arrive once each.
        for _ in range(10):
            p.observe("topic", ["A"])
        p.observe("topic", ["B"])
        p.observe("topic", ["C"])
        p.observe("topic", ["D"])  # triggers trim — lowest-count victim is D
        p.observe("topic", ["E"])  # triggers trim again — victim is C or B
        # A should still be the top prediction because its count wasn't reset.
        assert p.predict("topic", limit=1) == ["A"]


# ---------------------------------------------------------------------------
# Evaluate / hit rate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_hits_counted_when_actual_matches_prediction(self) -> None:
        p = PrefetchPredictor()
        p.observe("JWT", ["A", "B", "C"])
        hits = p.evaluate("JWT", ["A", "B", "D"])
        assert hits == 2
        stats = p.stats()
        assert stats.prefetch_hits == 2
        assert stats.prefetch_misses == 1  # C was predicted but not seen

    def test_empty_prediction_returns_zero_hits(self) -> None:
        p = PrefetchPredictor()
        hits = p.evaluate("unseen", ["A", "B"])
        assert hits == 0
        # No prediction → no check recorded (idle signatures don't poison hit rate).
        assert p.stats().prefetch_checks == 0

    def test_hit_rate_calculation(self) -> None:
        p = PrefetchPredictor()
        p.observe("JWT", ["A"])
        p.evaluate("JWT", ["A"])  # 1 hit, 0 miss
        p.observe("JWT", ["A"])
        p.evaluate("JWT", ["B"])  # 0 hit, 1 miss
        stats = p.stats()
        assert stats.hit_rate == 0.5

    def test_evaluate_with_explicit_predicted_set(self) -> None:
        """Passing the exact prefetched list must not re-run predict().

        The prior evaluate() always re-predicted at per_bucket_cap, which
        double-counted _predictions and measured efficacy against the
        wrong set when the caller had used a tighter limit.
        """
        p = PrefetchPredictor()
        p.observe("JWT", ["A", "B", "C"])
        before = p.stats().predictions
        hits = p.evaluate("JWT", ["A"], predicted=["A"])  # caller only warmed A
        after = p.stats()
        assert hits == 1
        assert after.prefetch_hits == 1
        # One warmed block, one hit → zero misses.
        assert after.prefetch_misses == 0
        # evaluate() must not have run a fresh predict().
        assert after.predictions == before


# ---------------------------------------------------------------------------
# Stats / clear
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_initial_values(self) -> None:
        p = PrefetchPredictor()
        s = p.stats()
        assert s.signatures == 0
        assert s.observations == 0
        assert s.predictions == 0
        assert s.prefetch_hits == 0
        assert s.prefetch_misses == 0
        assert s.hit_rate == 0.0

    def test_stats_as_dict(self) -> None:
        p = PrefetchPredictor()
        p.observe("topic", ["A", "B"])
        p.predict("topic")
        d = p.stats().as_dict()
        assert d["signatures"] == 1
        assert d["observations"] == 2
        assert d["predictions"] == 1
        assert "hit_rate" in d

    def test_clear_resets_everything(self) -> None:
        p = PrefetchPredictor()
        p.observe("topic", ["A", "B"])
        p.predict("topic")
        p.clear()
        s = p.stats()
        assert s.signatures == 0
        assert s.observations == 0
        assert s.predictions == 0


# ---------------------------------------------------------------------------
# Default predictor singleton
# ---------------------------------------------------------------------------


class TestDefaultPredictor:
    def setup_method(self) -> None:
        reset_default_predictor()

    def test_default_predictor_singleton(self) -> None:
        p1 = get_default_predictor()
        p2 = get_default_predictor()
        assert p1 is p2

    def test_reset_drops_instance(self) -> None:
        p1 = get_default_predictor()
        reset_default_predictor()
        p2 = get_default_predictor()
        assert p1 is not p2


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_observe_and_predict(self) -> None:
        p = PrefetchPredictor()
        errors: list[BaseException] = []
        barrier = threading.Barrier(16)

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for i in range(100):
                    p.observe(f"topic-{tid % 4}", [f"blk-{tid}-{i}"])
                    _ = p.predict(f"topic-{tid % 4}")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Worker errors: {errors[:3]}"
        # All 4 signature buckets should have accumulated observations.
        assert p.stats().signatures == 4

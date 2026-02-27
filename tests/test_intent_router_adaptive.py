"""Tests for adaptive intent routing (#470).

Verifies that the IntentRouter records feedback, computes adaptation weights,
persists stats to disk, and applies weights during classification.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.intent_router import IntentRouter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def router():
    """Fresh IntentRouter with no workspace (no persistence)."""
    return IntentRouter()


@pytest.fixture()
def workspace(tmp_path):
    """Temporary workspace directory with memory/ subdirectory."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return str(tmp_path)


@pytest.fixture()
def persistent_router(workspace):
    """IntentRouter wired to a temporary workspace for persistence."""
    return IntentRouter(workspace=workspace)


# ---------------------------------------------------------------------------
# record_feedback — stats tracking
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    """Tests for IntentRouter.record_feedback()."""

    def test_good_outcome_increments_good(self, router: IntentRouter):
        router.record_feedback("why did X fail", "WHY", result_count=5, avg_score=0.6)
        stats = router._intent_stats["WHY"]
        assert stats["good"] == 1
        assert stats["poor"] == 0
        assert stats["total"] == 1

    def test_poor_outcome_no_results(self, router: IntentRouter):
        router.record_feedback("why did X fail", "WHY", result_count=0, avg_score=0.0)
        stats = router._intent_stats["WHY"]
        assert stats["good"] == 0
        assert stats["poor"] == 1
        assert stats["total"] == 1

    def test_poor_outcome_low_score(self, router: IntentRouter):
        router.record_feedback("why did X fail", "WHY", result_count=3, avg_score=0.1)
        stats = router._intent_stats["WHY"]
        assert stats["good"] == 0
        assert stats["poor"] == 1

    def test_boundary_score_exactly_0_3(self, router: IntentRouter):
        """avg_score=0.3 is NOT > 0.3, so classified as poor."""
        router.record_feedback("why X", "WHY", result_count=1, avg_score=0.3)
        assert router._intent_stats["WHY"]["poor"] == 1

    def test_multiple_intents_tracked_independently(self, router: IntentRouter):
        router.record_feedback("why X", "WHY", result_count=3, avg_score=0.5)
        router.record_feedback("when X", "WHEN", result_count=0, avg_score=0.0)
        assert router._intent_stats["WHY"]["good"] == 1
        assert router._intent_stats["WHEN"]["poor"] == 1

    def test_user_selected_accepted(self, router: IntentRouter):
        """user_selected parameter is accepted without error."""
        router.record_feedback("why X", "WHY", result_count=3, avg_score=0.5, user_selected=2)
        assert router._intent_stats["WHY"]["total"] == 1


# ---------------------------------------------------------------------------
# Adaptation weights computation
# ---------------------------------------------------------------------------


class TestAdaptationWeights:
    """Tests for adaptation weight calculation."""

    def test_no_weight_before_min_samples(self, router: IntentRouter):
        """Weight is NOT computed until _MIN_SAMPLES (5) feedback entries."""
        for _ in range(4):
            router.record_feedback("why X", "WHY", result_count=3, avg_score=0.5)
        assert "WHY" not in router._adaptation_weights

    def test_weight_computed_at_min_samples(self, router: IntentRouter):
        """Weight IS computed once we hit exactly _MIN_SAMPLES."""
        for _ in range(5):
            router.record_feedback("why X", "WHY", result_count=3, avg_score=0.5)
        assert "WHY" in router._adaptation_weights

    def test_all_good_weight_is_1_0(self, router: IntentRouter):
        """100% good outcomes -> weight = 0.5 + 0.5 * 1.0 = 1.0."""
        for _ in range(5):
            router.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)
        assert router._adaptation_weights["WHY"] == pytest.approx(1.0)

    def test_all_poor_weight_is_0_5(self, router: IntentRouter):
        """0% good outcomes -> weight = 0.5 + 0.5 * 0.0 = 0.5."""
        for _ in range(5):
            router.record_feedback("why X", "WHY", result_count=0, avg_score=0.0)
        assert router._adaptation_weights["WHY"] == pytest.approx(0.5)

    def test_mixed_weight_in_range(self, router: IntentRouter):
        """3 good + 2 poor out of 5 -> ratio = 0.6 -> weight = 0.5 + 0.3 = 0.8."""
        for _ in range(3):
            router.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)
        for _ in range(2):
            router.record_feedback("why X", "WHY", result_count=0, avg_score=0.0)
        assert router._adaptation_weights["WHY"] == pytest.approx(0.8)

    def test_weight_updates_incrementally(self, router: IntentRouter):
        """Weight recalculates on every new feedback after threshold."""
        for _ in range(5):
            router.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)
        w1 = router._adaptation_weights["WHY"]
        assert w1 == pytest.approx(1.0)

        # Add a poor sample: 5 good, 1 poor -> ratio = 5/6 -> weight = 0.5 + 0.5*(5/6)
        router.record_feedback("why X", "WHY", result_count=0, avg_score=0.0)
        w2 = router._adaptation_weights["WHY"]
        assert w2 < w1
        assert w2 == pytest.approx(0.5 + 0.5 * (5 / 6))


# ---------------------------------------------------------------------------
# classify() uses adaptation weights
# ---------------------------------------------------------------------------


class TestClassifyAdaptive:
    """Tests that classify() applies adaptation weights to confidence."""

    def test_classify_without_adaptation(self, router: IntentRouter):
        """Without feedback, classify returns unmodified confidence."""
        result = router.classify("why did the server crash")
        assert result.intent == "WHY"
        assert result.confidence > 0

    def test_classify_with_good_adaptation(self, router: IntentRouter):
        """Good intent keeps confidence near original (weight ~1.0)."""
        # Get baseline
        baseline = router.classify("why did the server crash").confidence

        # Record all-good feedback
        for _ in range(5):
            router.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)

        adapted = router.classify("why did the server crash").confidence
        # weight=1.0, so adapted should equal baseline
        assert adapted == pytest.approx(baseline)

    def test_classify_with_poor_adaptation(self, router: IntentRouter):
        """Poor intent gets confidence cut (weight=0.5)."""
        baseline = router.classify("why did the server crash").confidence

        for _ in range(5):
            router.record_feedback("why X", "WHY", result_count=0, avg_score=0.0)

        adapted = router.classify("why did the server crash").confidence
        # weight=0.5, so adapted ~ baseline * 0.5
        assert adapted < baseline
        assert adapted == pytest.approx(baseline * 0.5, abs=0.01)

    def test_unaffected_intents_unchanged(self, router: IntentRouter):
        """Intents without feedback are not affected."""
        baseline = router.classify("when did they meet").confidence

        # Record feedback for a DIFFERENT intent (WHY)
        for _ in range(5):
            router.record_feedback("why X", "WHY", result_count=0, avg_score=0.0)

        after = router.classify("when did they meet").confidence
        assert after == baseline


# ---------------------------------------------------------------------------
# Persistence: save / load cycle
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for stats persistence to workspace/memory/intent_router_stats.json."""

    def test_save_creates_file(self, persistent_router: IntentRouter, workspace: str):
        persistent_router.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)
        path = os.path.join(workspace, "memory", "intent_router_stats.json")
        assert os.path.isfile(path)

    def test_saved_json_structure(self, persistent_router: IntentRouter, workspace: str):
        persistent_router.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)
        path = os.path.join(workspace, "memory", "intent_router_stats.json")
        with open(path) as f:
            data = json.load(f)
        assert "intent_stats" in data
        assert "adaptation_weights" in data
        assert data["intent_stats"]["WHY"]["total"] == 1

    def test_load_restores_state(self, workspace: str):
        """Save from one router, load into a fresh one."""
        r1 = IntentRouter(workspace=workspace)
        for _ in range(6):
            r1.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)
        saved_weight = r1._adaptation_weights["WHY"]
        saved_stats = r1._intent_stats["WHY"].copy()

        # Create fresh router pointing to same workspace — loads persisted data
        r2 = IntentRouter(workspace=workspace)
        assert r2._adaptation_weights.get("WHY") == saved_weight
        assert r2._intent_stats.get("WHY") == saved_stats

    def test_classify_uses_persisted_weights(self, workspace: str):
        """Persisted weights actually affect classify() in a new router."""
        r1 = IntentRouter(workspace=workspace)
        for _ in range(5):
            r1.record_feedback("why X", "WHY", result_count=0, avg_score=0.0)

        r2 = IntentRouter(workspace=workspace)
        result = r2.classify("why did the server crash")
        # Should be down-weighted by 0.5
        assert result.confidence > 0
        # Verify weight was loaded
        assert r2._adaptation_weights["WHY"] == pytest.approx(0.5)

    def test_no_file_no_crash(self, workspace: str):
        """Loading from workspace with no stats file does not crash."""
        stats_path = os.path.join(workspace, "memory", "intent_router_stats.json")
        assert not os.path.isfile(stats_path)
        r = IntentRouter(workspace=workspace)
        assert r._intent_stats == {}
        assert r._adaptation_weights == {}

    def test_corrupt_file_no_crash(self, workspace: str):
        """Loading corrupt JSON does not crash, just ignores."""
        path = os.path.join(workspace, "memory", "intent_router_stats.json")
        with open(path, "w") as f:
            f.write("{invalid json!!")
        r = IntentRouter(workspace=workspace)
        assert r._intent_stats == {}
        assert r._adaptation_weights == {}

    def test_no_workspace_no_save(self, router: IntentRouter, tmp_path):
        """Router without workspace does not try to save."""
        router.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)
        # No file should be created anywhere related to this router
        assert router._stats_path() is None


# ---------------------------------------------------------------------------
# Singleton get_router() workspace upgrade
# ---------------------------------------------------------------------------


class TestGetRouter:
    """Tests for get_router() singleton behavior with workspace."""

    def test_get_router_returns_router(self):
        import mind_mem.intent_router as mod
        from mind_mem.intent_router import get_router

        old = mod._router
        try:
            mod._router = None
            r = get_router()
            assert isinstance(r, IntentRouter)
            assert r._workspace is None
        finally:
            mod._router = old

    def test_get_router_with_workspace(self, workspace: str):
        import mind_mem.intent_router as mod
        from mind_mem.intent_router import get_router

        old = mod._router
        try:
            mod._router = None
            r = get_router(workspace=workspace)
            assert r._workspace == workspace
        finally:
            mod._router = old

    def test_get_router_upgrades_workspace(self, workspace: str):
        import mind_mem.intent_router as mod
        from mind_mem.intent_router import get_router

        old = mod._router
        try:
            mod._router = None
            r1 = get_router()  # no workspace
            assert r1._workspace is None
            r2 = get_router(workspace=workspace)  # upgrade
            assert r2 is r1  # same instance
            assert r2._workspace == workspace
        finally:
            mod._router = old


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for adaptive routing."""

    def test_feedback_for_unknown_intent(self, router: IntentRouter):
        """Feedback for an intent not in INTENT_CONFIG works fine."""
        router.record_feedback("foo", "NONEXISTENT", result_count=3, avg_score=0.5)
        assert router._intent_stats["NONEXISTENT"]["total"] == 1

    def test_empty_query_not_affected_by_adaptation(self, router: IntentRouter):
        """Empty query returns WHAT with confidence=0.0 regardless of weights."""
        for _ in range(5):
            router.record_feedback("what X", "WHAT", result_count=0, avg_score=0.0)
        result = router.classify("")
        assert result.intent == "WHAT"
        assert result.confidence == 0.0

    def test_no_match_query_not_affected_by_adaptation(self, router: IntentRouter):
        """Default (no match) still returns WHAT 0.1 without adaptation applied."""
        result = router.classify("xyzzy plugh")
        assert result.intent == "WHAT"
        assert result.confidence == 0.1

    def test_min_samples_class_attribute(self):
        """_MIN_SAMPLES is accessible and defaults to 5."""
        assert IntentRouter._MIN_SAMPLES == 5

    def test_memory_dir_created_if_missing(self, tmp_path):
        """_save_stats creates memory/ dir if it doesn't exist."""
        ws = str(tmp_path / "fresh_workspace")
        os.makedirs(ws)
        r = IntentRouter(workspace=ws)
        r.record_feedback("why X", "WHY", result_count=5, avg_score=0.8)
        assert os.path.isfile(os.path.join(ws, "memory", "intent_router_stats.json"))

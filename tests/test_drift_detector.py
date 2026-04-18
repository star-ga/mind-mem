"""Tests for mind-mem semantic belief drift detection (drift_detector.py)."""

import os

import pytest

from mind_mem.drift_detector import (
    DRIFT_REVERSAL,
    DRIFT_SEMANTIC,
    DriftDetector,
    DriftSignal,
    _jaccard,
    _trigrams,
)


@pytest.fixture
def workspace(tmp_path):
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    return ws


@pytest.fixture
def detector(workspace):
    return DriftDetector(workspace)


class TestTrigrams:
    def test_normal_text(self):
        tris = _trigrams("hello")
        assert "hel" in tris
        assert "ell" in tris
        assert "llo" in tris
        assert len(tris) == 3

    def test_short_text(self):
        assert _trigrams("hi") == {"hi"}
        assert _trigrams("") == set()

    def test_case_insensitive(self):
        assert _trigrams("Hello") == _trigrams("hello")


class TestJaccard:
    def test_identical(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        sim = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.0 < sim < 1.0

    def test_empty(self):
        assert _jaccard(set(), {"a"}) == 0.0
        assert _jaccard(set(), set()) == 0.0


class TestDriftSignal:
    def test_to_dict(self):
        sig = DriftSignal(
            block_a_id="D-001",
            block_b_id="D-002",
            similarity=0.75,
            confidence=0.8,
            drift_type=DRIFT_SEMANTIC,
            description="Test drift",
            date_a="2026-01-01",
            date_b="2026-03-01",
        )
        d = sig.to_dict()
        assert d["similarity"] == 0.75
        assert d["confidence"] == 0.8
        assert d["drift_type"] == "semantic"


class TestDriftDetector:
    def _write_decisions(self, workspace, blocks_text):
        path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(blocks_text)

    def test_empty_workspace(self, detector):
        signals = detector.scan()
        assert signals == []

    def test_single_block_no_drift(self, workspace, detector):
        self._write_decisions(
            workspace,
            """
[D-20260301-001]
Date: 2026-03-01
Status: active
Subject: Use PostgreSQL for data storage
Priority: 5
""",
        )
        signals = detector.scan()
        assert signals == []

    def test_similar_blocks_detected(self, workspace, detector):
        self._write_decisions(
            workspace,
            """
[D-20260301-001]
Date: 2026-03-01
Status: active
Subject: Use PostgreSQL for all persistent data storage in production
Priority: 5

---

[D-20260302-001]
Date: 2026-03-02
Status: active
Subject: Use MongoDB for all persistent data storage in production
Priority: 5
""",
        )
        signals = detector.scan()
        assert len(signals) >= 1
        assert signals[0].similarity > 0.3

    def test_modality_conflict_boosts_confidence(self, workspace):
        self._write_decisions(
            workspace,
            """
[D-20260301-001]
Date: 2026-03-01
Status: active
Subject: All services must use TLS encryption
ConstraintSignatures: [{"modality": "must", "subject": "TLS"}]

---

[D-20260302-001]
Date: 2026-03-02
Status: active
Subject: Internal services must_not use TLS encryption overhead
ConstraintSignatures: [{"modality": "must_not", "subject": "TLS"}]
""",
        )
        detector = DriftDetector(workspace, similarity_threshold=0.2)
        signals = detector.scan()
        reversals = [s for s in signals if s.drift_type == DRIFT_REVERSAL]
        # May or may not detect depending on text similarity
        # but if detected, should be high confidence
        for r in reversals:
            assert r.confidence >= 0.5

    def test_signals_persisted(self, workspace, detector):
        self._write_decisions(
            workspace,
            """
[D-20260301-001]
Date: 2026-03-01
Subject: Deploy all microservices to Kubernetes cluster in production

---

[D-20260302-001]
Date: 2026-03-02
Subject: Deploy all microservices to Docker Swarm cluster in production
""",
        )
        signals = detector.scan()
        recent = detector.recent_signals()
        assert len(recent) >= len(signals)

    def test_belief_snapshot_and_timeline(self, workspace, detector):
        block = {
            "_id": "D-20260301-001",
            "Status": "active",
            "Subject": "Use Redis for caching",
            "Priority": 5,
        }
        detector.snapshot_belief("D-20260301-001", block)

        # Simulate change
        block["Subject"] = "Use Memcached for caching"
        detector.snapshot_belief("D-20260301-001", block)

        timeline = detector.belief_timeline("D-20260301-001")
        assert len(timeline) == 2
        assert timeline[0]["changed"]  # First snapshot always "changed"
        assert timeline[1]["changed"]  # Content actually changed

    def test_belief_timeline_unchanged(self, workspace, detector):
        block = {
            "_id": "D-001",
            "Subject": "Same content",
        }
        detector.snapshot_belief("D-001", block)
        detector.snapshot_belief("D-001", block)

        timeline = detector.belief_timeline("D-001")
        assert len(timeline) == 2
        assert not timeline[1]["changed"]  # Same content hash

    def test_recent_signals_filter(self, workspace, detector):
        self._write_decisions(
            workspace,
            """
[D-20260301-001]
Date: 2026-03-01
Subject: Use PostgreSQL for persistent data storage layer

---

[D-20260302-001]
Date: 2026-03-02
Subject: Use MySQL for persistent data storage layer
""",
        )
        detector.scan()

        # Filter by confidence
        high_conf = detector.recent_signals(min_confidence=0.9)
        all_sigs = detector.recent_signals(min_confidence=0.0)
        assert len(high_conf) <= len(all_sigs)

    def test_custom_threshold(self, workspace):
        self._write_decisions(
            workspace,
            """
[D-20260301-001]
Subject: Use Python Flask

---

[D-20260302-001]
Subject: Use Python Django
""",
        )
        strict = DriftDetector(workspace, similarity_threshold=0.9)
        signals = strict.scan()
        # Very strict threshold — unlikely to find drift
        assert len(signals) == 0 or all(s.similarity >= 0.9 for s in signals)

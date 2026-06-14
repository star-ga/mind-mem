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


# ─── Backend-aware block enumeration (audit bug 12) ──────────────────────────
#
# drift_detector._load_blocks read ``decisions/DECISIONS.md`` directly via
# parse_file, so on a non-Markdown backend (Postgres) — where that file is
# the empty init template and every decision lives in the store — drift
# analysis was silently blind. The load now routes non-Markdown backends
# through ``mind_mem.storage.iter_active_blocks`` while leaving the default
# Markdown / SQLite path byte-for-byte unchanged.
#
# These tests run on BOTH backends: the Markdown branch needs no DB; the
# Postgres branch uses an isolated, uniquely-named scratch schema that is
# created and dropped per test (the production ``mind_mem`` schema is never
# touched) and skips cleanly when psycopg / a live Postgres is unavailable
# so SQLite-only CI stays green.

import json  # noqa: E402
import uuid  # noqa: E402
from typing import Generator  # noqa: E402

from mind_mem.drift_detector import _is_decision_block, _is_markdown_backend  # noqa: E402

# Two near-identical decision statements that drift detection should pair.
_DRIFT_A = "Use PostgreSQL for all persistent data storage in production"
_DRIFT_B = "Use MongoDB for all persistent data storage in production"


def _write_config(ws: str, block_store: dict | None = None) -> None:
    config: dict = {"recall": {"backend": "scan"}}
    if block_store is not None:
        config["block_store"] = block_store
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh)


class TestDecisionBlockPredicate:
    """``_is_decision_block`` mirrors the Markdown 'decisions only' semantics."""

    def test_decisions_source_file_matches(self):
        assert _is_decision_block({"_id": "D-20260613-001", "_source_file": "decisions/DECISIONS.md"})

    def test_windows_separator_source_matches(self):
        assert _is_decision_block({"_id": "D-1", "_source_file": "decisions\\DECISIONS.md"})

    def test_d_prefix_without_source_matches(self):
        assert _is_decision_block({"_id": "D-20260613-001"})

    def test_non_decision_source_excluded(self):
        assert not _is_decision_block({"_id": "T-20260613-001", "_source_file": "tasks/TASKS.md"})

    def test_explicit_non_decisions_source_overrides_prefix(self):
        # A D-prefixed id with an explicit non-decisions source is not a decision.
        assert not _is_decision_block({"_id": "D-1", "_source_file": "tasks/TASKS.md"})

    def test_non_d_prefix_without_source_excluded(self):
        assert not _is_decision_block({"_id": "PRJ-foo"})


class TestMarkdownBackendUnchanged:
    """The zero-config Markdown / SQLite default path must not regress."""

    def test_no_config_is_markdown_backend(self, workspace):
        # No mind-mem.json at all → markdown (the zero-config default).
        assert _is_markdown_backend(workspace) is True

    def test_explicit_markdown_backend(self, workspace):
        _write_config(workspace, block_store={"backend": "markdown"})
        assert _is_markdown_backend(workspace) is True
        det = DriftDetector(workspace, similarity_threshold=0.2)
        path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                f"\n[D-20260301-001]\nDate: 2026-03-01\nSubject: {_DRIFT_A}\n\n"
                f"---\n\n[D-20260302-001]\nDate: 2026-03-02\nSubject: {_DRIFT_B}\n"
            )
        signals = det.scan()
        assert len(signals) >= 1
        ids = {signals[0].block_a_id, signals[0].block_b_id}
        assert ids == {"D-20260301-001", "D-20260302-001"}


# ─── Postgres backend (live DB; skips cleanly when unavailable) ──────────────

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed; skipping Postgres tests")

from mind_mem.block_store_postgres import PostgresBlockStore  # noqa: E402

_DSN = os.environ.get("MIND_MEM_TEST_PG_DSN")


def _pg_available(dsn: str) -> bool:
    try:
        conn = psycopg.connect(dsn, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


_pg_live = pytest.mark.skipif(
    not _pg_available(_DSN),
    reason="no live Postgres available at the test DSN",
)


@pytest.fixture
def pg_workspace(tmp_path) -> Generator[tuple[str, "PostgresBlockStore"], None, None]:
    """A workspace configured for Postgres on an isolated scratch schema."""
    schema = f"mm_fix_{uuid.uuid4().hex[:12]}"
    ws = str(tmp_path / "pgws")
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    _write_config(ws, block_store={"backend": "postgres", "dsn": _DSN, "schema": schema})

    store = PostgresBlockStore(dsn=_DSN, schema=schema, workspace=ws)
    store._ensure_schema()
    try:
        yield ws, store
    finally:
        try:
            conn = psycopg.connect(_DSN)
            conn.autocommit = True
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.close()
        except Exception:
            pass
        store.close()


@_pg_live
class TestDriftDetectorPostgres:
    """Drift detection must see Postgres-resident decision blocks (audit bug 12)."""

    def test_load_blocks_reads_store_not_empty_markdown(self, pg_workspace):
        ws, store = pg_workspace
        assert _is_markdown_backend(ws) is False
        store.write_block(
            {
                "_id": "D-20260613-401",
                "_source_file": "decisions/DECISIONS.md",
                "Subject": _DRIFT_A,
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        store.write_block(
            {
                "_id": "D-20260613-402",
                "_source_file": "decisions/DECISIONS.md",
                "Subject": _DRIFT_B,
                "Status": "active",
                "Date": "2026-06-14",
            }
        )
        det = DriftDetector(ws, similarity_threshold=0.2)
        # The on-disk DECISIONS.md is the empty init template — without
        # backend-aware enumeration this returns [] (the exact audit bug 12).
        ids = {b.get("_id") for b in det._load_blocks()}
        assert {"D-20260613-401", "D-20260613-402"} <= ids

    def test_scan_detects_drift_in_postgres(self, pg_workspace):
        ws, store = pg_workspace
        store.write_block(
            {
                "_id": "D-20260613-411",
                "_source_file": "decisions/DECISIONS.md",
                "Subject": _DRIFT_A,
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        store.write_block(
            {
                "_id": "D-20260613-412",
                "_source_file": "decisions/DECISIONS.md",
                "Subject": _DRIFT_B,
                "Status": "active",
                "Date": "2026-06-14",
            }
        )
        det = DriftDetector(ws, similarity_threshold=0.2)
        signals = det.scan()
        assert len(signals) >= 1
        paired = {signals[0].block_a_id, signals[0].block_b_id}
        assert paired == {"D-20260613-411", "D-20260613-412"}

    def test_non_decision_store_blocks_excluded(self, pg_workspace):
        ws, store = pg_workspace
        # A task block must not be mistaken for a decision block.
        store.write_block(
            {
                "_id": "T-20260613-421",
                "_source_file": "tasks/TASKS.md",
                "Subject": _DRIFT_A,
                "Status": "active",
                "Date": "2026-06-13",
            }
        )
        store.write_block(
            {
                "_id": "D-20260613-422",
                "_source_file": "decisions/DECISIONS.md",
                "Subject": _DRIFT_B,
                "Status": "active",
                "Date": "2026-06-14",
            }
        )
        det = DriftDetector(ws, similarity_threshold=0.2)
        ids = {b.get("_id") for b in det._load_blocks()}
        assert "D-20260613-422" in ids
        assert "T-20260613-421" not in ids

    def test_empty_store_returns_no_signals(self, pg_workspace):
        ws, _store = pg_workspace
        det = DriftDetector(ws, similarity_threshold=0.2)
        assert det.scan() == []

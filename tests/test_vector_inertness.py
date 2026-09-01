"""The vector-leg honesty gauge.

Measured 2026-08-31 (``benchmarks/embed_augmentation_ab.py``, all-MiniLM-L6-v2):
a near-duplicate corpus ranks at CHANCE (12.5% top1 over 8 blocks, negative
mean margin) with an inter-block cosine spread of 0.002, against 0.108 for
prose. RRF nevertheless fused that noise at full weight and the answer kept the
"hybrid" label.

These tests pin the three properties that make the gauge trustworthy:
it fires on genuinely indistinguishable vectors, it does NOT fire on a healthy
corpus, and it ABSTAINS rather than guessing when it cannot tell — because
silently disabling a working retrieval leg is a worse failure than the one the
gauge exists to catch.
"""

from __future__ import annotations

import os
import random
import sqlite3
import struct

import pytest

from mind_mem import vector_inertness as vi


@pytest.fixture(autouse=True)
def _clean_cache():
    vi.reset_cache()
    yield
    vi.reset_cache()


def _write_index(ws, vectors: list[list[float]]) -> None:
    """Populate a workspace's embedding_cache with *vectors*."""
    d = os.path.join(str(ws), ".mind-mem-index")
    os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(os.path.join(d, "recall.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedding_cache ("
        "block_id TEXT NOT NULL, content_hash TEXT NOT NULL, model_name TEXT NOT NULL, "
        "dimension INTEGER NOT NULL, embedding BLOB NOT NULL, "
        "PRIMARY KEY (block_id, model_name))"
    )
    for i, vec in enumerate(vectors):
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache VALUES (?,?,?,?,?)",
            (f"B-{i:04d}", f"h{i}", "test-model", len(vec), struct.pack(f"{len(vec)}f", *vec)),
        )
    conn.commit()
    conn.close()


def _boilerplate(n: int = 12, dim: int = 64) -> list[list[float]]:
    """Near-identical vectors: the 512-mind `format!`-one-flat-string shape."""
    rng = random.Random(11)
    base = [rng.gauss(0, 1) for _ in range(dim)]
    return [[v + rng.gauss(0, 1e-4) for v in base] for _ in range(n)]


def _prose(n: int = 12, dim: int = 64) -> list[list[float]]:
    rng = random.Random(23)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


class TestSpread:
    def test_identical_vectors_have_zero_spread(self) -> None:
        v = [[1.0, 0.0, 0.0]] * 5
        assert vi.spread_of(v) == pytest.approx(0.0, abs=1e-9)

    def test_varied_vectors_have_real_spread(self) -> None:
        assert vi.spread_of(_prose()) > 0.05

    def test_a_single_vector_cannot_have_a_spread(self) -> None:
        assert vi.spread_of([[1.0, 2.0]]) == 0.0

    def test_high_mean_similarity_alone_is_not_inertness(self) -> None:
        """A topically tight corpus can still rank fine — VARIANCE is what matters.

        Two tight clusters are all mutually similar, but the pairwise spread is
        real, so ordering still carries information.
        """
        rng = random.Random(5)
        a = [1.0] * 32
        cluster1 = [[v + rng.gauss(0, 0.01) for v in a] for _ in range(6)]
        cluster2 = [[v + rng.gauss(0, 0.01) for v in ([1.0] * 16 + [-1.0] * 16)] for _ in range(6)]
        assert vi.spread_of(cluster1 + cluster2) > vi.DEFAULT_SPREAD_FLOOR


class TestVerdict:
    def test_boilerplate_is_called_inert(self, tmp_path) -> None:
        _write_index(tmp_path, _boilerplate())
        r = vi.measure(str(tmp_path))
        assert r.inert is True
        assert r.spread is not None and r.spread < vi.DEFAULT_SPREAD_FLOOR
        assert "BM25-only" in r.reason

    def test_a_healthy_corpus_is_not_disabled(self, tmp_path) -> None:
        _write_index(tmp_path, _prose())
        r = vi.measure(str(tmp_path))
        assert r.inert is False
        assert r.spread is not None and r.spread >= vi.DEFAULT_SPREAD_FLOOR

    def test_too_few_vectors_abstains_rather_than_guessing(self, tmp_path) -> None:
        _write_index(tmp_path, _boilerplate(n=vi.MIN_SAMPLE - 1))
        r = vi.measure(str(tmp_path))
        assert r.inert is False, "must not disable a leg on a sample too small to judge"
        assert r.spread is None
        assert "abstains" in r.reason

    def test_a_missing_index_abstains(self, tmp_path) -> None:
        r = vi.measure(str(tmp_path / "nothing-here"))
        assert r.inert is False and r.sampled == 0

    def test_an_unreadable_index_abstains_instead_of_raising(self, tmp_path) -> None:
        d = tmp_path / ".mind-mem-index"
        d.mkdir()
        (d / "recall.db").write_text("this is not a database", encoding="utf-8")
        r = vi.measure(str(tmp_path))
        assert r.inert is False and r.sampled == 0


class TestCache:
    def test_repeat_calls_reuse_the_measurement(self, tmp_path) -> None:
        _write_index(tmp_path, _prose())
        first = vi.inertness_for(str(tmp_path))
        assert vi.inertness_for(str(tmp_path)) is first

    def test_a_changed_index_is_re_measured(self, tmp_path) -> None:
        """Stamped on (size, mtime_ns) -- an appended index must not answer stale."""
        _write_index(tmp_path, _prose())
        healthy = vi.inertness_for(str(tmp_path))
        assert healthy.inert is False

        db = tmp_path / ".mind-mem-index" / "recall.db"
        os.utime(db, (0, 0))  # force a different stamp deterministically
        _write_index(tmp_path, _boilerplate(n=40))
        again = vi.inertness_for(str(tmp_path))
        assert again is not healthy, "cache did not notice the index changed"

    def test_the_cache_is_bounded(self, tmp_path) -> None:
        for i in range(vi._CACHE_MAX + 5):
            ws = tmp_path / f"ws{i}"
            _write_index(ws, _prose(n=vi.MIN_SAMPLE))
            vi.inertness_for(str(ws))
        assert len(vi._CACHE) <= vi._CACHE_MAX


class TestReportShape:
    def test_as_dict_carries_the_evidence_not_just_the_verdict(self, tmp_path) -> None:
        """A verdict with no number is an assertion; this one must be arguable."""
        _write_index(tmp_path, _boilerplate())
        d = vi.measure(str(tmp_path)).as_dict()
        assert set(d) == {"inert", "spread", "sampled", "floor", "reason"}
        assert d["sampled"] >= vi.MIN_SAMPLE
        assert d["floor"] == vi.DEFAULT_SPREAD_FLOOR

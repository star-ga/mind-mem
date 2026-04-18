# Copyright 2026 STARGA, Inc.
"""Tests for interaction signal capture + A/B eval (v2.1.0)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

import pytest

from mind_mem.interaction_signals import (
    ABResult,
    Signal,
    SignalStore,
    SignalType,
    classify,
    evaluate_ab,
    jaccard_similarity,
)

# ---------------------------------------------------------------------------
# Tokenisation + similarity
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_queries_score_one(self) -> None:
        assert jaccard_similarity("JWT token reset", "JWT token reset") == 1.0

    def test_disjoint_queries_score_zero(self) -> None:
        assert jaccard_similarity("JWT auth", "postgres vacuum") == 0.0

    def test_partial_overlap_between_zero_and_one(self) -> None:
        sim = jaccard_similarity("JWT token reset", "reset my JWT token")
        assert 0.0 < sim <= 1.0

    def test_empty_query_returns_zero(self) -> None:
        assert jaccard_similarity("", "anything") == 0.0
        assert jaccard_similarity("anything", "") == 0.0

    def test_stopwords_stripped(self) -> None:
        # "the", "a", etc. should be discarded, so these two collapse.
        s1 = "the JWT token"
        s2 = "a JWT token"
        assert jaccard_similarity(s1, s2) == 1.0


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


class TestClassify:
    def test_identical_phrasing_is_re_query(self) -> None:
        assert classify("JWT token", "JWT token") is SignalType.RE_QUERY

    def test_unrelated_queries_return_none(self) -> None:
        assert classify("JWT", "postgres vacuum") is None

    def test_refinement_detected(self) -> None:
        # Related but narrower
        assert classify("auth decision", "auth decision database") in (
            SignalType.RE_QUERY,
            SignalType.REFINEMENT,
        )

    def test_correction_markers_detected(self) -> None:
        assert classify("JWT", "no, I meant oauth2 token exchange") is SignalType.CORRECTION

    def test_empty_inputs_return_none(self) -> None:
        assert classify("", "anything") is None
        assert classify("anything", "") is None


# ---------------------------------------------------------------------------
# SignalStore
# ---------------------------------------------------------------------------


@pytest.fixture()
def store_path():
    with tempfile.TemporaryDirectory() as td:
        yield os.path.join(td, "signals.jsonl")


class TestSignalStore:
    def test_observe_persists_to_disk(self, store_path: str) -> None:
        store = SignalStore(store_path)
        rec = store.observe(
            session_id="s1",
            previous_query="JWT",
            new_query="JWT",
            signal_type=SignalType.RE_QUERY,
            similarity=1.0,
        )
        assert rec is not None
        assert os.path.isfile(store_path)
        lines = Path(store_path).read_text().splitlines()
        assert len(lines) == 1
        on_disk = json.loads(lines[0])
        assert on_disk["signal_id"] == rec.signal_id
        assert on_disk["signal_type"] == "re_query"

    def test_duplicate_signal_id_is_skipped(self, store_path: str) -> None:
        store = SignalStore(store_path)
        first = store.observe(
            session_id="s1",
            previous_query="JWT",
            new_query="JWT",
            signal_type=SignalType.RE_QUERY,
            similarity=1.0,
            timestamp="2026-04-13T00:00:00Z",
        )
        second = store.observe(
            session_id="s1",
            previous_query="JWT",
            new_query="JWT",
            signal_type=SignalType.RE_QUERY,
            similarity=1.0,
            timestamp="2026-04-13T00:00:00Z",
        )
        assert first is not None
        assert second is None  # idempotent replay
        lines = Path(store_path).read_text().splitlines()
        assert len(lines) == 1

    def test_reload_preserves_seen_ids(self, store_path: str) -> None:
        SignalStore(store_path).observe(
            session_id="s1",
            previous_query="JWT",
            new_query="JWT",
            signal_type=SignalType.RE_QUERY,
            similarity=1.0,
            timestamp="2026-04-13T00:00:00Z",
        )
        # Fresh store over the same file must refuse the same signal_id.
        store2 = SignalStore(store_path)
        retry = store2.observe(
            session_id="s1",
            previous_query="JWT",
            new_query="JWT",
            signal_type=SignalType.RE_QUERY,
            similarity=1.0,
            timestamp="2026-04-13T00:00:00Z",
        )
        assert retry is None

    def test_observe_pair_classifies_and_persists(self, store_path: str) -> None:
        store = SignalStore(store_path)
        result = store.observe_pair(
            session_id="s1",
            previous_query="JWT",
            new_query="JWT token",
        )
        assert result is not None
        assert result.signal_type in {SignalType.RE_QUERY, SignalType.REFINEMENT}

    def test_observe_pair_returns_none_for_unrelated(self, store_path: str) -> None:
        store = SignalStore(store_path)
        result = store.observe_pair(
            session_id="s1",
            previous_query="JWT",
            new_query="postgres vacuum",
        )
        assert result is None

    def test_stats_counts_by_type_and_sessions(self, store_path: str) -> None:
        store = SignalStore(store_path)
        store.observe(
            session_id="s1",
            previous_query="JWT",
            new_query="JWT",
            signal_type=SignalType.RE_QUERY,
            similarity=1.0,
        )
        store.observe(
            session_id="s2",
            previous_query="auth",
            new_query="auth flow",
            signal_type=SignalType.REFINEMENT,
            similarity=0.5,
        )
        store.observe(
            session_id="s2",
            previous_query="x",
            new_query="no i meant y",
            signal_type=SignalType.CORRECTION,
            similarity=0.1,
        )
        stats = store.stats()
        assert stats.total == 3
        assert stats.re_query == 1
        assert stats.refinement == 1
        assert stats.correction == 1
        assert stats.unique_sessions == 2

    def test_non_utf8_bytes_in_store_do_not_crash(self, store_path: str) -> None:
        """Audit regression (v2.1.0): binary corruption must not break load."""
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        # Real record, then a line with a stray non-UTF-8 byte sequence,
        # then another real record.
        good_line = (
            '{"signal_id": "x", "timestamp": "t", "session_id": "s", '
            '"signal_type": "re_query", "previous_query": "p", '
            '"new_query": "n", "previous_results": [], "similarity": 0.5}\n'
        )
        mixed = good_line.encode("utf-8") + b"\xff\xfe not utf8 \n" + good_line.encode("utf-8")
        Path(store_path).write_bytes(mixed)
        # Must not raise.
        store = SignalStore(store_path)
        signals = store.all_signals()
        # The two valid records should come back; the garbage is skipped.
        assert len(signals) == 2

    def test_wrongdoing_does_not_false_positive_correction(self) -> None:
        """Audit regression: word-boundary regex prevents 'wrongdoing' → CORRECTION."""
        # "wrong" as a literal substring used to trigger CORRECTION.
        # After the fix, classify must not treat these as corrections.
        assert classify("ethics policy", "ethical wrongdoing report") != SignalType.CORRECTION

    def test_correction_markers_require_word_boundaries(self) -> None:
        # Audit regression: "wrong" as a prefix of another word
        # (wrongdoing, wrongful) must not trigger CORRECTION.
        assert classify("policy", "wrongdoing policy update") != SignalType.CORRECTION
        assert classify("policy", "wrongful termination laws") != SignalType.CORRECTION

    def test_malformed_lines_ignored_on_load(self, store_path: str) -> None:
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        Path(store_path).write_text(
            '{ not json }\n{"signal_id": "x", "timestamp": "t", '
            '"session_id": "s", "signal_type": "re_query", '
            '"previous_query": "p", "new_query": "n", '
            '"previous_results": [], "similarity": 0.5}\n',
            encoding="utf-8",
        )
        store = SignalStore(store_path)
        signals = store.all_signals()
        assert len(signals) == 1
        assert signals[0].signal_type is SignalType.RE_QUERY

    def test_concurrent_observe_no_loss(self, store_path: str) -> None:
        store = SignalStore(store_path)
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for i in range(25):
                    store.observe(
                        session_id=f"s-{tid}",
                        previous_query=f"q {tid}",
                        new_query=f"q {tid} {i}",
                        signal_type=SignalType.REFINEMENT,
                        similarity=0.5,
                        timestamp=f"2026-04-13T{tid:02d}:{i:02d}:00Z",
                    )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # 8 threads * 25 records, all unique.
        assert store.stats().total == 200


# ---------------------------------------------------------------------------
# A/B evaluation
# ---------------------------------------------------------------------------


class TestABEval:
    def _make_signals(self) -> list[Signal]:
        return [
            Signal(
                signal_id="s1",
                timestamp="t",
                session_id="x",
                signal_type=SignalType.RE_QUERY,
                previous_query="q",
                new_query="q",
                previous_results=("A", "B"),
                similarity=1.0,
            ),
            Signal(
                signal_id="s2",
                timestamp="t",
                session_id="x",
                signal_type=SignalType.REFINEMENT,
                previous_query="q",
                new_query="q tighter",
                previous_results=("C",),
                similarity=0.5,
            ),
        ]

    def test_candidate_ranking_target_higher_wins(self) -> None:
        signals = self._make_signals()

        def baseline(q: str) -> list[str]:
            return ["Z", "Y", "A", "C"]  # A at 3, C at 4

        def candidate(q: str) -> list[str]:
            return ["A", "C", "Z"]  # A at 1, C at 2

        result = evaluate_ab(signals, baseline=baseline, candidate=candidate)
        assert result.signals_scored == 2
        assert result.candidate_mrr > result.baseline_mrr
        assert result.winner == "candidate"

    def test_correction_signals_excluded(self) -> None:
        signals = [
            Signal(
                signal_id="c",
                timestamp="t",
                session_id="x",
                signal_type=SignalType.CORRECTION,
                previous_query="q",
                new_query="no i meant r",
                previous_results=("A",),
                similarity=0.1,
            )
        ]

        def baseline(q: str) -> list[str]:
            return ["A"]

        def candidate(q: str) -> list[str]:
            return ["A"]

        result = evaluate_ab(signals, baseline=baseline, candidate=candidate)
        assert result.signals_scored == 0

    def test_tie_reported_when_identical(self) -> None:
        signals = self._make_signals()

        def same(q: str) -> list[str]:
            return ["A", "C"]

        result = evaluate_ab(signals, baseline=same, candidate=same)
        assert result.winner == "tie"

    def test_result_as_dict_has_expected_keys(self) -> None:
        r = ABResult(signals_scored=1, baseline_mrr=0.5, candidate_mrr=0.25)
        d = r.as_dict()
        assert set(d.keys()) == {
            "signals_scored",
            "baseline_mrr",
            "candidate_mrr",
            "delta",
            "winner",
        }
        assert d["winner"] == "baseline"

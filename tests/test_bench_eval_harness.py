"""Tests for the LongMemEval consolidation harness (self-asserting adapters).

Uses a tiny synthetic dataset shaped like LongMemEval-S so the harness is
exercised end-to-end with no external data. The real dataset is not
redistributed; the loader is tested for its fail-clear behavior.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.bench.eval_adapter import PipelineProbe, SessionDoc, config_sha256
from mind_mem.bench.eval_adapters import Bm25BaselineAdapter, MindMemAdapter, get_adapter
from mind_mem.bench.eval_scorer import aggregate, score_question
from mind_mem.bench.longmemeval_suite import (
    DatasetNotFoundError,
    build_session_docs,
    render_scorecard,
    resolve_data_path,
    run_suite,
    write_ndjson,
)


def _synthetic_dataset() -> list[dict]:
    return [
        {
            "question_id": "q1",
            "question": "What is the capital of France?",
            "question_type": "single-session-user",
            "answer_session_ids": ["s_gold_1"],
            "haystack_session_ids": ["s_noise_a", "s_gold_1", "s_noise_b"],
            "haystack_sessions": [
                [{"role": "user", "content": "I love hiking in the mountains every summer."}],
                [
                    {"role": "user", "content": "Tell me about France."},
                    {"role": "assistant", "content": "The capital of France is Paris."},
                ],
                [{"role": "user", "content": "My favorite programming language is Python."}],
            ],
        },
        {
            "question_id": "q2",
            "question": "Which programming language does the user prefer?",
            "question_type": "preference",
            "answer_session_ids": ["s_gold_2"],
            "haystack_session_ids": ["s_gold_2", "s_noise_c"],
            "haystack_sessions": [
                [{"role": "user", "content": "My favorite programming language is Python for data science."}],
                [{"role": "user", "content": "The weather in Paris was lovely."}],
            ],
        },
        {
            # abstention question — must be skipped by the pool filter
            "question_id": "q3_abs",
            "question": "unanswerable",
            "question_type": "abstention",
            "answer_session_ids": [],
            "haystack_sessions": [],
        },
    ]


# --------------------------------------------------------------------------
# session doc construction
# --------------------------------------------------------------------------


def test_build_session_docs_all_turns():
    q = _synthetic_dataset()[0]
    docs = build_session_docs(q, turns="all")
    assert [d.doc_id for d in docs] == ["s_noise_a", "s_gold_1", "s_noise_b"]
    assert "Paris" in docs[1].text
    assert "(assistant)" in docs[1].text


def test_build_session_docs_user_only():
    q = _synthetic_dataset()[0]
    docs = build_session_docs(q, turns="user")
    # user-only drops the assistant turn that holds "Paris"
    assert "Paris" not in docs[1].text
    assert "Tell me about France" in docs[1].text


# --------------------------------------------------------------------------
# BM25 baseline adapter
# --------------------------------------------------------------------------


def test_bm25_baseline_ranks_gold_first():
    q = _synthetic_dataset()[0]
    docs = build_session_docs(q, turns="all")
    adapter = Bm25BaselineAdapter()
    state = adapter.init(docs, None)
    hits = adapter.query(q["question"], state, k=3)
    assert hits, "baseline returned no hits"
    assert hits[0]["doc_id"] == "s_gold_1"
    assert state.probe.effective_backend == "bm25_inmemory"
    assert state.probe.mismatch is False
    adapter.teardown(state)


def test_get_adapter_unknown_raises():
    with pytest.raises(KeyError):
        get_adapter("nope")


# --------------------------------------------------------------------------
# mind-mem adapter — real store + real recall path
# --------------------------------------------------------------------------


def test_mindmem_adapter_hits_gold_via_real_recall():
    q = _synthetic_dataset()[0]
    docs = build_session_docs(q, turns="all")
    adapter = MindMemAdapter()
    state = adapter.init(docs, None)
    try:
        hits = adapter.query(q["question"], state, k=5)
        retrieved = [h["doc_id"] for h in hits]
        assert "s_gold_1" in retrieved, f"gold not retrieved: {retrieved}"
        # default config declares sqlite; probe must reflect what recall resolves to
        assert state.probe.declared_backend == "sqlite"
        assert state.probe.effective_backend in ("sqlite", "scan")
    finally:
        adapter.teardown(state)


def test_mindmem_adapter_flags_declared_vs_effective_mismatch():
    """Declaring a backend recall() has no dispatch for (``hybrid``) must be
    caught as a pipeline mismatch — this is the false-green tripwire."""
    q = _synthetic_dataset()[0]
    docs = build_session_docs(q, turns="all")
    adapter = MindMemAdapter()
    state = adapter.init(docs, {"recall": {"backend": "hybrid"}})
    try:
        assert state.probe.declared_backend == "hybrid"
        # mind-mem's recall() facade has no "hybrid" dispatch → falls to scan
        assert state.probe.effective_backend == "scan"
        assert state.probe.mismatch is True
    finally:
        adapter.teardown(state)


def test_config_sha256_stable_and_order_independent():
    a = config_sha256({"recall": {"backend": "sqlite", "knee_cutoff": False}})
    b = config_sha256({"recall": {"knee_cutoff": False, "backend": "sqlite"}})
    assert a == b
    assert a != config_sha256({"recall": {"backend": "scan"}})


# --------------------------------------------------------------------------
# scorer — dual protocol
# --------------------------------------------------------------------------


def test_score_question_dual_protocol():
    # gold = {a, b}; retrieved ranks a at 1, b at 4
    qs = score_question(
        question_id="q",
        question_type="t",
        retrieved_doc_ids=["a", "x", "y", "b", "z"],
        gold_doc_ids={"a", "b"},
        latency_ms=1.0,
    )
    assert qs.first_gold_rank == 1
    assert qs.reciprocal_rank == 1.0
    assert qs.hit is True
    # any@k: a present from k=1; all@k needs both a and b (b at rank 4)
    assert qs.recall_any_at_k[1] == 1
    assert qs.recall_all_at_k[1] == 0
    assert qs.recall_all_at_k[3] == 0
    assert qs.recall_all_at_k[5] == 1
    # precision@1 = 1/1, precision@5 = 2/5; recall@5 = 2/2
    assert qs.precision_at_k[1] == pytest.approx(1.0)
    assert qs.precision_at_k[5] == pytest.approx(0.4)
    assert qs.recall_at_k[5] == pytest.approx(1.0)


def test_score_question_miss():
    qs = score_question("q", "t", ["x", "y"], {"a"}, latency_ms=0.5)
    assert qs.first_gold_rank is None
    assert qs.hit is False
    assert qs.reciprocal_rank == 0.0
    assert qs.recall_any_at_k[10] == 0


def test_aggregate_means():
    a = score_question("q1", "t1", ["g"], {"g"}, 1.0)
    b = score_question("q2", "t2", ["x", "g"], {"g"}, 3.0)
    agg = aggregate([a, b])
    assert agg["n"] == 2
    assert agg["overall"]["hit_rate"] == 1.0
    assert set(agg["by_type"]) == {"t1", "t2"}


# --------------------------------------------------------------------------
# end-to-end suite + artifacts
# --------------------------------------------------------------------------


def test_run_suite_and_ndjson_carries_pipeline(tmp_path):
    ds = _synthetic_dataset()
    result = run_suite("bm25_baseline", ds, k=5, turns="all")
    # abstention q3 filtered out of the pool → 2 scored
    assert result.evaluated == 2
    assert len(result.probes) == 2

    ndjson = tmp_path / "out.ndjson"
    write_ndjson(result, str(ndjson))
    rows = [json.loads(line) for line in ndjson.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert "pipeline" in row
        assert row["pipeline"]["effective_backend"] == "bm25_inmemory"
        assert "config_sha256" in row["pipeline"]
        assert "recall_any_at_k" in row
        assert "recall_all_at_k" in row


def test_render_scorecard_honesty_rails(tmp_path):
    ds = _synthetic_dataset()
    result = run_suite("bm25_baseline", ds, k=5, turns="all")
    card = render_scorecard(result, dataset_path="longmemeval_s.json", k=5, embedder="none (BM25-only)")
    assert "85.3" in card and "UNREPRODUCED" in card
    assert "recall_any@k" in card and "recall_all@k" in card
    assert "No new competitor comparison" in card
    # no mismatch on the baseline
    assert "PIPELINE MISMATCH" not in card


def test_scorecard_surfaces_mismatch():
    from mind_mem.bench.longmemeval_suite import SuiteResult

    probe = PipelineProbe(
        adapter="mind_mem",
        declared_backend="hybrid",
        effective_backend="scan",
        vector_available=False,
        config_sha256="deadbeefdeadbeef",
    )
    score = score_question("q", "t", ["g"], {"g"}, 1.0)
    result = SuiteResult("mind_mem", "all", [score], [probe], 1, 0, 0.1)
    card = render_scorecard(result, dataset_path="x.json", k=5, embedder="none")
    assert "PIPELINE MISMATCH" in card


def test_resolve_data_path_fails_clear(tmp_path, monkeypatch):
    monkeypatch.delenv("LONGMEMEVAL_DATA", raising=False)
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(DatasetNotFoundError) as exc:
        resolve_data_path(str(missing))
    assert "does not redistribute" in str(exc.value)


def test_session_doc_dataclass():
    d = SessionDoc(doc_id="s1", text="hello")
    assert d.doc_id == "s1" and d.text == "hello"

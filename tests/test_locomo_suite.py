"""Tests for the LoCoMo recall harness (self-asserting adapters).

Runs entirely on a tiny committed fixture shaped like LoCoMo
(`tests/fixtures/locomo_mini.json`): ZERO external dataset, ZERO API calls,
ZERO network. The real LoCoMo JSON is not redistributed; the loader is tested
for its fail-clear behaviour. No LoCoMo number is fabricated — every assertion
is over the fixture the harness actually scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.bench.eval_adapter import PipelineProbe
from mind_mem.bench.eval_scorer import aggregate
from locomo_suite import (
    DatasetNotFoundError,
    build_session_docs,
    evidence_session_id,
    extract_sessions,
    flatten_questions,
    load_dataset,
    render_scorecard,
    resolve_data_path,
    run_suite,
)
from mind_mem.bench.longmemeval_suite import SuiteResult

FIXTURE = Path(__file__).parent / "fixtures" / "locomo_mini.json"


def _load() -> list[dict]:
    return load_dataset(str(FIXTURE))


# --------------------------------------------------------------------------
# load_dataset
# --------------------------------------------------------------------------


def test_load_dataset_returns_sample_list():
    ds = _load()
    assert isinstance(ds, list)
    assert len(ds) == 2
    assert ds[0]["sample_id"] == "conv-1"
    assert "conversation" in ds[0] and "qa" in ds[0]


def test_load_dataset_rejects_non_list(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        load_dataset(str(bad))


# --------------------------------------------------------------------------
# evidence + session extraction
# --------------------------------------------------------------------------


def test_evidence_session_id_parses_dialogue_ids():
    assert evidence_session_id("D3:2") == "3"
    assert evidence_session_id("d12:1") == "12"
    assert evidence_session_id("5") == "5"
    assert evidence_session_id("") is None
    assert evidence_session_id("nonsense") is None


def test_extract_sessions_orders_and_ignores_metadata():
    conv = _load()[0]["conversation"]
    sessions = extract_sessions(conv)
    assert [s["session_id"] for s in sessions] == ["1", "2", "3"]
    # metadata keys (speaker_a, session_N_date_time) never become sessions
    assert all(s["session_id"].isdigit() for s in sessions)
    assert len(sessions[0]["turns"]) == 2


# --------------------------------------------------------------------------
# flatten (nested LoCoMo -> flat question records)
# --------------------------------------------------------------------------


def test_flatten_questions_shapes_records_and_gold():
    qs = flatten_questions(_load())
    # 4 QA in conv-1 + 2 in conv-2 = 6 total (adversarial one kept here, filtered in run_suite)
    assert len(qs) == 6
    q0 = qs[0]
    assert q0["question_id"] == "conv-1::q0"
    assert q0["question_type"] == "category_1"
    assert q0["gold_session_ids"] == ["1"]
    # the adversarial question (empty evidence) yields no gold session
    adversarial = [q for q in qs if q["question_type"] == "category_5"][0]
    assert adversarial["gold_session_ids"] == []


# --------------------------------------------------------------------------
# docs builder
# --------------------------------------------------------------------------


def test_build_session_docs_all_turns():
    q = flatten_questions(_load())[2]  # conv-1 promotion question, gold session 3
    docs = build_session_docs(q, turns="all")
    assert [d.doc_id for d in docs] == ["1", "2", "3"]
    assert "senior engineer" in docs[2].text
    assert "(Melanie)" in docs[2].text and "(Caroline)" in docs[2].text


def test_build_session_docs_speaker_filter():
    q = flatten_questions(_load())[2]
    docs = build_session_docs(q, turns="Melanie")
    # session 3: Melanie speaks the promotion line; Caroline's reply is dropped
    assert "senior engineer" in docs[2].text
    assert "(Caroline)" not in docs[2].text


# --------------------------------------------------------------------------
# run_suite over the fixture (bm25_baseline)
# --------------------------------------------------------------------------


def test_run_suite_bm25_scores_and_filters_adversarial():
    result = run_suite("bm25_baseline", _load(), k=5, turns="all")
    # 6 QA total; the one adversarial question (no gold) is filtered → 5 scored
    assert result.evaluated == 5
    assert len(result.probes) == 5
    assert result.any_mismatch is False
    for p in result.probes:
        assert p.effective_backend == "bm25_inmemory"


def test_run_suite_bm25_retrieves_gold_sessions():
    result = run_suite("bm25_baseline", _load(), k=5, turns="all")
    # each fixture question shares distinctive tokens with exactly its gold
    # session, so the honest BM25 floor must place gold in the top-k.
    for s in result.scores:
        assert s.recall_any_at_k[5] == 1, f"gold session missed for {s.question_id}"
    agg = aggregate(result.scores)
    assert agg["overall"]["recall_any@5"] == 1.0
    # per-category breakdown present (LoCoMo categories 1 and 2 in the fixture)
    assert set(agg["by_type"]) == {"category_1", "category_2"}


def test_run_suite_per_type_is_deterministic_and_seeded():
    a = run_suite("bm25_baseline", _load(), k=5, per_type=1, seed=42)
    b = run_suite("bm25_baseline", _load(), k=5, per_type=1, seed=42)
    assert [s.question_id for s in a.scores] == [s.question_id for s in b.scores]
    # up to 1 per category, two categories with gold → 2 evaluated
    assert a.evaluated == 2
    assert {s.question_type for s in a.scores} == {"category_1", "category_2"}


def test_run_suite_ndjson_carries_pipeline(tmp_path):
    from locomo_suite import write_ndjson

    result = run_suite("bm25_baseline", _load(), k=5, turns="all")
    out = tmp_path / "locomo.ndjson"
    write_ndjson(result, str(out))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 5
    for row in rows:
        assert row["pipeline"]["effective_backend"] == "bm25_inmemory"
        assert "config_sha256" in row["pipeline"]
        assert "recall_any_at_k" in row and "recall_all_at_k" in row


# --------------------------------------------------------------------------
# scorer output aggregation
# --------------------------------------------------------------------------


def test_scorer_aggregate_over_fixture():
    result = run_suite("bm25_baseline", _load(), k=5, turns="all")
    agg = aggregate(result.scores)
    assert agg["n"] == 5
    o = agg["overall"]
    # every gold session retrieved → hit_rate and any@5 both perfect on the floor
    assert o["hit_rate"] == 1.0
    assert o["recall_any@5"] == 1.0
    # metrics are bounded probabilities, never NaN/None
    for key, val in o.items():
        if key == "n":
            continue
        assert isinstance(val, (int, float))
        assert 0.0 <= val or key == "mean_latency_ms"


# --------------------------------------------------------------------------
# render_scorecard — honesty rails + disclosure
# --------------------------------------------------------------------------


def test_render_scorecard_discloses_sample_size_and_vector_deps():
    result = run_suite("bm25_baseline", _load(), k=5, turns="all")
    card = render_scorecard(result, dataset_path="locomo.json", k=5, embedder="none (BM25-only)", sampling="full set")
    assert "LoCoMo recall scorecard" in card
    assert "Questions evaluated:** 5" in card
    assert "Vector deps available" in card
    # The probe answers "is the embedder importable", so the scorecard
    # must not claim the leg ran: with the default zero-dep config no
    # embedder is constructed, and the box the harness runs on may well
    # have `sentence_transformers` installed.
    assert "leg exercised" not in card
    # honesty rails
    assert "recall_any@k" in card and "recall_all@k" in card
    assert "No prior recall number is reproduced" in card
    assert "77.9" in card  # prior LLM-judge figure named only to disown it as a recall number
    assert "No competitor comparison" in card
    # the honest BM25 floor must not trip the mismatch banner
    assert "PIPELINE MISMATCH" not in card


def test_scorecard_surfaces_mismatch():
    from mind_mem.bench.eval_scorer import score_question

    probe = PipelineProbe(
        adapter="mind_mem",
        declared_backend="hybrid",
        effective_backend="scan",
        vector_available=False,
        config_sha256="deadbeefdeadbeef",
    )
    score = score_question("q", "category_1", ["1"], {"1"}, 1.0)
    result = SuiteResult("mind_mem", "all", [score], [probe], 1, 0, 0.1)
    card = render_scorecard(result, dataset_path="locomo.json", k=5, embedder="none")
    assert "PIPELINE MISMATCH" in card


def test_scorecard_empty_scores_is_honest():
    result = SuiteResult("bm25_baseline", "all", [], [], 0, 0, 0.0)
    card = render_scorecard(result, dataset_path="locomo.json", k=5, embedder="none")
    assert "No questions scored" in card
    assert "nothing scored" in card
    assert "leg exercised" not in card


def test_scorecard_never_claims_an_unexercised_vector_leg():
    """A BM25-only run must not be reported as the full stack.

    ``PipelineProbe.vector_available`` is ``find_spec(...) is not None``
    — the package being installed, which says nothing about whether the
    run built an embedder. The scorecard reports it as availability.
    """
    from mind_mem.bench.eval_scorer import score_question

    probe = PipelineProbe(
        adapter="mind_mem",
        declared_backend="sqlite",
        effective_backend="sqlite",
        vector_available=True,  # installed on this box...
        config_sha256="deadbeefdeadbeef",
    )
    score = score_question("q", "category_1", ["1"], {"1"}, 1.0)
    result = SuiteResult("mind_mem", "all", [score], [probe], 1, 0, 0.1)
    card = render_scorecard(result, dataset_path="locomo.json", k=5, embedder="none (BM25-only)")
    # ...but never constructed, so the scorecard may not say it ran.
    assert "leg exercised" not in card
    assert "Vector deps available (installed, not necessarily used):** `True`" in card


# --------------------------------------------------------------------------
# DatasetNotFoundError path (mirrors LongMemEval)
# --------------------------------------------------------------------------


def test_resolve_data_path_fails_clear(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCOMO_DATA", raising=False)
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(DatasetNotFoundError) as exc:
        resolve_data_path(str(missing))
    assert "does not redistribute" in str(exc.value)


def test_resolve_data_path_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCOMO_DATA", str(FIXTURE))
    assert resolve_data_path() == str(FIXTURE)

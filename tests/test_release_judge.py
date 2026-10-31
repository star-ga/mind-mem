"""Synthetic scoring protocol tests; no network or model needed."""

import pytest

from benchmarks.release_judge import judge, parse_score
from benchmarks.release_runtime import require_pin


def test_structured_rejection_scoring():
    assert parse_score('{"entity_correct": true, "rejection_quality": 10}', True)["score"] == 80
    assert parse_score('{"entity_correct": true, "wrong_entity_hallucination": true}', True)["score"] == 0
    assert parse_score('```json\n{"score": 51}\n```', False)["score"] == 51


def test_negative_votes_get_majority_review():
    row = {"benchmark": "generic", "generated": "blue", "question": "Color?", "reference": "red", "abstention": False}
    calls = []

    def call(messages, n):
        calls.append(messages)
        return "yes" if len(calls) == 2 else "no"

    ok, votes = judge(row, {"LME_JUDGE": "{q} {a} {r}"}, call)
    assert not ok and len(votes) == 3


def test_mutable_model_aliases_cannot_run():
    for model, revision in (("release-pending", "abc"), ("model", "main"), ("model-latest", "abc"), ("model", None)):
        with pytest.raises(ValueError):
            require_pin(model, revision)
    require_pin("organization/model", "0123456789abcdef")

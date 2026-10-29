"""Synthetic tests of evidence binding, correction selection and offline metrics."""

import copy
import hashlib
import json

import pytest

from benchmarks.release_scoring import CorrectedKey, aggregate, digest, require_no_regression, text_metrics, unique, verify_bundle


def answer():
    return {
        "id": "example",
        "question": "What is the color?",
        "reference": "blue",
        "generated": "green",
        "partition": "held_out",
        "official_ok": True,
    }


def correction():
    return {
        "id": "example",
        "question": "What is the color?",
        "official_gold": "blue",
        "verdict": "GOLD_WRONG",
        "accepted_answers": ["red"],
        "evidence": [{"quote": "The color is red."}],
    }


def adjudication(key, ok):
    return {
        "id": "example",
        "answer": "green",
        "answer_sha256": hashlib.sha256(b"green").hexdigest(),
        "key_sha256": digest(key),
        "corrected_ok": ok,
    }


def test_negative_correction_and_answer_binding():
    key = correction()
    scorer = CorrectedKey([key], [adjudication(key, False)])
    row = answer()
    assert scorer.score(row) is False
    row["generated"] = "not red"
    with pytest.raises(ValueError, match="Unreviewed"):
        scorer.score(row)
    row["generated"] = "red"
    assert scorer.score(row) is True


def test_key_revision_invalidates_adjudication():
    key = correction()
    reviewed = adjudication(key, True)
    key["accepted_answers"] = ["purple"]
    with pytest.raises(ValueError, match="Unreviewed"):
        CorrectedKey([key], [reviewed]).score(answer())


def test_reference_identity_and_duplicate_rejected():
    row = answer()
    row["question"] = "A different question"
    with pytest.raises(ValueError, match="mismatch"):
        CorrectedKey([correction()], []).score(row)
    with pytest.raises(ValueError, match="duplicate"):
        unique([row, row])


def test_metrics_and_aggregate_only_output():
    assert text_metrics("The red boat [[source]].", ["red boat"]) == (1.0, 1.0)
    assert text_metrics("red red boat", ["red boat"]) == (0.0, 0.8)
    result = aggregate([answer()])
    assert result["held_out"]["correct"] == 1
    assert result["dev"]["accuracy"] is None
    assert "question" not in json.dumps(result)


def test_held_out_change_policy():
    before = aggregate([answer()])
    after = copy.deepcopy(before)
    after["held_out"]["correct"] = 0
    with pytest.raises(ValueError, match="regression"):
        require_no_regression(before, after)
    require_no_regression(before, before)


def test_manifest_tampering_and_vote_integrity(tmp_path):
    prompt = "A generic evidence prompt"
    h = hashlib.sha256(prompt.encode()).hexdigest()
    row = {**answer(), "prompt_sha256": h, "votes": [{"ok": False}]}
    (tmp_path / "answers.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "prompts.jsonl").write_text(json.dumps({"id": row["id"], "prompt_sha256": h, "messages": [{"content": prompt}]}) + "\n")
    manifest = {"rows": 1, "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="majority"):
        verify_bundle(tmp_path)
    (tmp_path / "answers.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(tmp_path)


def test_drop_in_key_cannot_change_unreviewed_content(tmp_path):
    from benchmarks.release_scoring import verify_correction_evidence

    row = {**correction(), "id": "example-0-0", "corrected_answer": "red"}
    row["evidence"] = [{"conversation": 0, "sample_id": "sample", "turn_id": "D1:1", "quote": "The color is red."}]
    original = [
        {
            "sample_id": "sample",
            "qa": [{"question": row["question"], "answer": "blue"}],
            "conversation": {"session_1": [{"dia_id": "D1:1", "text": "The color is red."}]},
        }
    ]
    revised = copy.deepcopy(original)
    revised[0]["qa"][0]["answer"] = "red"
    evidence = {"id": row["id"], "source_row_sha256": digest(row), "quoted_turns": row["evidence"], "dia_ids": ["D1:1"]}
    (tmp_path / "locomo10.official.json").write_text(json.dumps(original))
    (tmp_path / "locomo10.corrected.json").write_text(json.dumps(revised))
    (tmp_path / "correction_evidence.jsonl").write_text(json.dumps(evidence) + "\n")
    assert verify_correction_evidence(tmp_path, [row])["literal_citations"] == 1
    revised[0]["conversation"]["session_1"][0]["text"] = "Changed without review"
    (tmp_path / "locomo10.corrected.json").write_text(json.dumps(revised))
    with pytest.raises(ValueError, match="unreviewed changes"):
        verify_correction_evidence(tmp_path, [row])

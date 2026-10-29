"""Synthetic coverage for download-only dataset reconstruction."""

import copy

import pytest

from benchmarks.release_inputs import corrected_dataset, source_blocks, text_hash


def test_source_blocks_reconstruct_format_and_stable_ids():
    sample = {
        "conversation": {
            "speaker_a": "Alex",
            "session_1_date_time": "2024-01-02",
            "session_1": [{"dia_id": "D1:1", "speaker": "speaker_a", "text": "A boat.", "blip_caption": "Blue hull."}],
        }
    }
    assert source_blocks(sample, "locomo") == {"DIA-D1-1": ("[Alex] A boat. [shared image: Blue hull.]", "2024-01-02")}
    sample = {"haystack_dates": ["2024/01/02"], "haystack_sessions": [[{"role": "user", "content": "one\n[two]"}]]}
    assert source_blocks(sample, "longmemeval_s") == {"D-20240102-001": ("user: one / (two)", "2024-01-02")}


def test_corrected_dataset_changes_only_the_bound_answer():
    original = [{"qa": [{"question": "Which color?", "answer": "blue"}], "conversation": {"session_1": []}}]
    pristine = copy.deepcopy(original)
    row = {"id": "example-0-0", "question_sha256": text_hash("Which color?"), "official_gold": "blue", "corrected_answer": "red"}
    expected = copy.deepcopy(original)
    expected[0]["qa"][0]["answer"] = "red"
    assert corrected_dataset(original, [row]) == expected
    assert original == pristine
    with pytest.raises(ValueError, match="downloaded source"):
        corrected_dataset(original, [{**row, "question_sha256": text_hash("Other?")}])


def test_text_free_prompt_manifest_still_binds_answers(tmp_path):
    import json

    from benchmarks.release_scoring import sha256, verify_bundle

    row = {"id": "example", "prompt_sha256": text_hash("synthetic"), "votes": [{"ok": True}], "official_ok": True}
    (tmp_path / "answers.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "prompt_hashes.jsonl").write_text(json.dumps({"id": "example", "prompt_sha256": text_hash("changed")}) + "\n")
    manifest = {"rows": 1, "artifacts": {p.name: sha256(p) for p in tmp_path.iterdir()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binding mismatch"):
        verify_bundle(tmp_path)

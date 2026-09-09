"""Integrity controls for the existing 4B evaluation runners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.eval_receipt import bindings_match, build_receipt, receipt_is_valid, require_complete


def _receipt(tmp_path: Path) -> tuple[dict, Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    # The holdout runner writes its report beside full-FT weights; it is not a
    # model input and must not invalidate the model binding after evaluation.
    (model / "eval_holdout_report.json").write_text("{}\n", encoding="utf-8")
    dataset = tmp_path / "corpus.jsonl"
    dataset.write_text('{"messages":[]}\n', encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    receipt = build_receipt(
        repo_root=root,
        model_root=model,
        dataset_root=dataset,
        source_paths=(root / "train/eval_receipt.py",),
        probe_sets={"main": [["question", ["answer"]]]},
        command="fixture",
    )
    assert list(receipt["model"]["files"]) == ["config.json"]
    return receipt, model, dataset


def test_receipt_is_complete_and_rehashes_current_bindings(tmp_path: Path) -> None:
    receipt, model, dataset = _receipt(tmp_path)
    assert receipt["complete"] is True
    assert receipt_is_valid(receipt)
    assert bindings_match(receipt)

    (model / "config.json").write_text("changed\n", encoding="utf-8")
    assert not bindings_match(receipt)

    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    dataset.write_text('{"messages":[1]}\n', encoding="utf-8")
    assert not bindings_match(receipt)


def test_receipt_self_digest_rejects_edit(tmp_path: Path) -> None:
    receipt, _, _ = _receipt(tmp_path)
    tampered = json.loads(json.dumps(receipt))
    tampered["probes"]["main"] = "0" * 64
    assert not receipt_is_valid(tampered)


def test_missing_dataset_is_incomplete(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    receipt = build_receipt(
        repo_root=root,
        model_root=model,
        dataset_root=tmp_path / "missing.jsonl",
        source_paths=(root / "train/eval_receipt.py",),
        probe_sets={"holdout": []},
        command="fixture",
    )
    assert receipt["complete"] is False
    with pytest.raises(ValueError, match="model/source/dataset"):
        require_complete(receipt)

"""Regression controls for the post-training receipt comparison shell path."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from train import eval_receipt as E

REPO = Path(__file__).resolve().parents[1]
CHAIN = REPO / "train" / "post_train_chain.sh"


def _write_report(root: Path, *, tokenizer: Path, report_path: Path, suite: str) -> None:
    model = root / "model"
    corpus = root / "corpus.jsonl"
    model.mkdir(parents=True, exist_ok=True)
    tokenizer.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"same model")
    (tokenizer / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    corpus.write_text("{}\n", encoding="utf-8")

    selection = E.ModelSelection(kind="full-ft", model_path=model, tokenizer_path=tokenizer)
    probes = {"probe": [("question", ["answer"])]}
    captured = E.capture_inputs(
        selection,
        repo_root=REPO,
        dataset_root=corpus,
        source_paths=E.eval_source_paths(REPO),
        probe_sets=probes,
    )
    receipt = E.build_receipt(
        repo_root=REPO,
        suite=suite,
        captured=captured,
        probe_counts={"probe": (1, 1)},
        probe_sets=probes,
        command=f"python3 train/eval_{suite}.py",
        run_id=f"{suite}-run",
        started_at="2026-09-12T00:00:00+00:00",
        ended_at="2026-09-12T00:01:00+00:00",
        status="completed",
    )
    report = E.finalize_report({"probe": {"hits": 1, "total": 1, "accuracy": 1.0, "items": []}}, receipt)
    report_path.write_text(json.dumps(report), encoding="utf-8")


def _run_chain(main_report: Path, holdout_report: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "MM_POST_TRAIN_EVAL_REPORT": str(main_report),
            "MM_POST_TRAIN_HOLDOUT_REPORT": str(holdout_report),
        }
    )
    return subprocess.run(["bash", str(CHAIN), "--validate-receipts"], cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8")


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="POSIX training chain requires bash")
def test_post_train_chain_refuses_mismatched_tokenizer_and_accepts_match(tmp_path: Path) -> None:
    model_root = tmp_path / "candidate"
    tokenizer_a = tmp_path / "tokenizer-a"
    tokenizer_b = tmp_path / "tokenizer-b"
    main_report = tmp_path / "main.json"
    holdout_report = tmp_path / "holdout.json"

    _write_report(model_root, tokenizer=tokenizer_a, report_path=main_report, suite="main")
    _write_report(model_root, tokenizer=tokenizer_b, report_path=holdout_report, suite="holdout")
    rejected = _run_chain(main_report, holdout_report)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "different tokenizers" in rejected.stdout

    _write_report(model_root, tokenizer=tokenizer_a, report_path=holdout_report, suite="holdout")
    accepted = _run_chain(main_report, holdout_report)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert "same checkpoint" in accepted.stdout

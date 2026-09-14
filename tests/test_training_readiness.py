"""Data-contract controls for the deterministic training corpus manifest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "train" / "build_corpus.py"
MANIFEST = REPO / "train" / "training_readiness_manifest.py"
LEAKED_HOLDOUT = "How does an operator change the workspace eviction policy at runtime without restarting?"


@pytest.fixture(scope="module")
def generated_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("training-readiness") / "corpus.jsonl"
    env = os.environ.copy()
    env.pop("MM_CORPUS", None)
    env["MM_CORPUS_OUT"] = str(output)
    result = subprocess.run(
        [sys.executable, str(BUILD)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file()
    return output


def _manifest(corpus: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MANIFEST), "--corpus", str(corpus), "--output", str(output)],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )


def _read_manifest(result: subprocess.CompletedProcess[str], output: Path) -> dict:
    assert output.is_file(), (
        f"manifest was not written (returncode={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r})"
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_manifest_flags_paraphrase_contamination_even_when_exact_holdout_is_unseen(generated_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    result = _manifest(generated_corpus, output)
    assert result.returncode == 1, result.stdout + result.stderr
    manifest = _read_manifest(result, output)

    surface = manifest["live_mcp_surface"]
    binding = manifest["evaluation_probe_binding"]
    assert surface["registered_tool_count"] == 103
    assert surface["tools_missing_from_corpus"] == []
    assert binding["holdout_exact_overlap_count"] == 0
    assert binding["holdout_exact_string_status"] == "PASS"
    assert binding["targeted_training_harvests"] == ["_harvest_v4_retry2g_holdout_paraphrase"]
    assert binding["semantic_contamination_status"] == "CONFIRMED"
    assert binding["training_eval_separation_status"] == "CONTAMINATED_DEVELOPMENT"
    assert binding["independence_ready"] is False
    assert binding["main_eval_is_training_overlap"] is True
    assert binding["main_eval_status"] == "TRAINING_OVERLAP"


def test_manifest_fails_closed_when_a_holdout_prompt_is_added(generated_corpus: Path, tmp_path: Path) -> None:
    poisoned = tmp_path / "poisoned-corpus.jsonl"
    poisoned.write_bytes(generated_corpus.read_bytes())
    with poisoned.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": LEAKED_HOLDOUT},
                        {"role": "assistant", "content": "set_active_policy"},
                    ]
                }
            )
            + "\n"
        )

    output = tmp_path / "poisoned-manifest.json"
    result = _manifest(poisoned, output)
    assert result.returncode == 1, result.stdout + result.stderr
    manifest = _read_manifest(result, output)
    binding = manifest["evaluation_probe_binding"]
    assert binding["holdout_exact_overlap_count"] == 1
    assert binding["holdout_exact_string_status"] == "FAIL"
    assert binding["training_eval_separation_status"] == "CONTAMINATED_DEVELOPMENT"

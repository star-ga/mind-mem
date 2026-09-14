"""Data-contract controls for the deterministic training corpus manifest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import count_mcp_tools
from train import training_readiness_manifest as readiness

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
    # Bind the manifest to the live AST registration inventory independently
    # of the manifest's own ``registered_tools`` helper.  The four entity
    # merge dispatchers were added after the 103-tool training-card claim;
    # require their corpus coverage explicitly rather than freezing another
    # stale total in this test.
    live_tool_count = count_mcp_tools.count_tools()
    assert surface["registered_tool_count"] == live_tool_count
    live_tool_names = {name for path in count_mcp_tools._tool_source_files() for name in count_mcp_tools._tool_names(path)}
    new_entity_merge_tools = {
        "propose_entity_merge",
        "list_entity_merge_proposals",
        "approve_entity_merge",
        "reverse_entity_merge",
    }
    assert new_entity_merge_tools <= live_tool_names
    mention_counts = surface["examples_mentioning_tool"]
    assert all(mention_counts.get(name, 0) > 0 for name in new_entity_merge_tools)
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


def _fact_sources(root: Path, *, readme: str, model_card: str) -> None:
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)
    (train / "README.md").write_text(readme, encoding="utf-8")
    (train / "HF_MODEL_CARD_v4.md").write_text(model_card, encoding="utf-8")


def test_model_facts_accept_wrapped_markdown_and_explicit_base_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fact_sources(
        tmp_path,
        readme="The current default base is `base-A`.\n",
        model_card=("---\nbase_model: base-A\n---\nThese weights were trained against an\n> **83-tool** surface.\nbase_model: base-A\n"),
    )
    monkeypatch.setattr(readiness, "REPO", tmp_path)
    monkeypatch.setenv("MM_BASE_MODEL", "/models/operator-selected")

    facts = readiness._read_model_facts()

    assert facts["base_model_default"] == "base-A"
    assert facts["base_model_requested"] == "/models/operator-selected"
    assert facts["historical_model_card_trained_tool_count"] == 83
    assert facts["historical_model_card_base_model"] == "base-A"


def test_new_training_default_does_not_rewrite_published_weights_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fact_sources(
        tmp_path,
        readme="The current default base is `next-base`.\n",
        model_card="base_model: \"published-base\"\nbase_model: 'published-base'\ntrained against an **83-tool** surface.\n",
    )
    monkeypatch.setattr(readiness, "REPO", tmp_path)
    monkeypatch.delenv("MM_BASE_MODEL", raising=False)
    facts = readiness._read_model_facts()
    assert facts["base_model_default"] == facts["base_model_requested"] == "next-base"
    assert facts["historical_model_card_base_model"] == "published-base"


@pytest.mark.parametrize(
    ("readme", "model_card", "message"),
    [
        (
            "The current default base is `base-A`.\nThe current default base is `base-B`.\n",
            "base_model: base-A\ntrained against an **83-tool** surface.\n",
            "conflicting default base-model",
        ),
        (
            "The current default base is `base-A`.\n",
            "base_model: base-A\ntrained against an **83-tool** surface.\ntrained against an **96-tool** surface.\n",
            "conflicting trained-tool count",
        ),
        (
            "The current default base is `base-A`.\n",
            "base_model: base-A\nbase_model: base-B\ntrained against an **83-tool** surface.\n",
            "conflicting model-card base-model",
        ),
    ],
)
def test_model_facts_reject_conflicting_source_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    readme: str,
    model_card: str,
    message: str,
) -> None:
    _fact_sources(tmp_path, readme=readme, model_card=model_card)
    monkeypatch.setattr(readiness, "REPO", tmp_path)

    with pytest.raises(SystemExit, match=message):
        readiness._read_model_facts()


@pytest.mark.parametrize(
    ("readme", "model_card"),
    [
        (
            "No current base is documented.\n",
            "base_model: base-A\ntrained against an **83-tool** surface.\n",
        ),
        (
            "The current default base is `base-A`.\n",
            "base_model: base-A\nNo trained surface is documented.\n",
        ),
        (
            "The current default base is `base-A`.\n",
            "base_model:\ntrained against an **83-tool** surface.\n",
        ),
    ],
)
def test_model_facts_reject_missing_or_malformed_required_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    readme: str,
    model_card: str,
) -> None:
    _fact_sources(tmp_path, readme=readme, model_card=model_card)
    monkeypatch.setattr(readiness, "REPO", tmp_path)

    with pytest.raises(SystemExit, match="could not bind"):
        readiness._read_model_facts()

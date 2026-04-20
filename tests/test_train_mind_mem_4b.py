"""Smoke tests for benchmarks/train_mind_mem_4b.py.

Only covers the pure-Python helpers (``_format_example``,
``iter_examples``, ``load_train_config``). The ``train()`` function
requires torch + transformers and is exercised in the Runpod run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH_DIR))

# Importing the script is safe — heavy deps are lazy-imported inside train().
import train_mind_mem_4b as trainer  # noqa: E402


class TestFormatExample:
    def test_dispatcher_example(self) -> None:
        ex = {
            "prompt": "What did Alice say?",
            "expected_call": {
                "tool": "recall",
                "args": {"mode": "similar", "query": "Alice"},
            },
        }
        out = trainer._format_example(ex)
        assert out is not None
        assert out["prompt"] == "What did Alice say?"
        assert json.loads(out["target"])["tool"] == "recall"

    def test_retrieval_example(self) -> None:
        ex = {
            "task": "query_decomposition",
            "prompt": "A and B",
            "expected_output": ["A and B", "A", "B"],
        }
        out = trainer._format_example(ex)
        assert out is not None
        assert out["prompt"].startswith("[task=query_decomposition]")
        assert json.loads(out["target"]) == ["A and B", "A", "B"]

    def test_raw_prompt_target_passthrough(self) -> None:
        """Operator-supplied custom data using the shared shape goes through."""
        ex = {"prompt": "Custom prompt", "target": "Custom target"}
        out = trainer._format_example(ex)
        assert out == {"prompt": "Custom prompt", "target": "Custom target"}

    def test_unknown_shape_returns_none(self) -> None:
        assert trainer._format_example({"just": "noise"}) is None


class TestIterExamples:
    def test_reads_jsonl(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "\n".join(
                [
                    json.dumps({"prompt": "p1", "expected_call": {"tool": "recall", "args": {}}}),
                    json.dumps({"task": "query_reformulation", "prompt": "x", "expected_output": ["x"]}),
                    "",  # blank line — skipped silently
                    json.dumps({"prompt": "raw", "target": "custom"}),
                ]
            ),
            encoding="utf-8",
        )
        examples = list(trainer.iter_examples(str(corpus)))
        assert len(examples) == 3
        assert examples[0]["prompt"] == "p1"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "not json\n" + json.dumps({"prompt": "ok", "expected_call": {"tool": "recall", "args": {}}}) + "\n" + "{broken\n",
            encoding="utf-8",
        )
        examples = list(trainer.iter_examples(str(corpus)))
        assert len(examples) == 1


class TestLoadTrainConfig:
    def test_defaults_when_no_path(self) -> None:
        cfg = trainer.load_train_config(None)
        assert cfg.base_model == "star-ga/mind-mem-4b"
        assert cfg.num_train_epochs == 3

    def test_yaml_override(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "num_train_epochs: 5\nlearning_rate: 1.0e-5\nextra_thing: hello\n",
            encoding="utf-8",
        )
        cfg = trainer.load_train_config(str(yaml_path))
        assert cfg.num_train_epochs == 5
        assert cfg.learning_rate == pytest.approx(1.0e-5)
        # Unknown keys land in extra_config rather than failing.
        assert cfg.extra_config == {"extra_thing": "hello"}

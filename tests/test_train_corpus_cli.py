"""Corpus CLI inspection and bad arguments must never regenerate training data.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "train" / "build_corpus.py"


def _run(output: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO,
        env={**os.environ, "MM_CORPUS_OUT": str(output)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("args,code", [(("--help",), 0), (("--outpt", "mistyped.jsonl"), 2)])
def test_inspection_or_invalid_arguments_do_not_touch_output(tmp_path: Path, existing: bool, args: tuple[str, ...], code: int) -> None:
    output = tmp_path / "uncreated" / "corpus.jsonl"
    original = b"existing corpus must survive\n"
    if existing:
        output.parent.mkdir()
        output.write_bytes(original)
    result = _run(output, *args)
    assert result.returncode == code, result.stdout + result.stderr
    if existing:
        assert output.read_bytes() == original
    else:
        assert not output.parent.exists()
    assert "wrote " not in result.stdout


def test_explicit_output_overrides_environment_and_matches_default_generation(tmp_path: Path) -> None:
    default = tmp_path / "environment.jsonl"
    selected = tmp_path / "selected" / "corpus.jsonl"
    sentinel = b"keep the configured corpus\n"
    default.write_bytes(sentinel)
    result = _run(default, "--output", str(selected))
    assert result.returncode == 0, result.stdout + result.stderr
    assert default.read_bytes() == sentinel
    selected_bytes = selected.read_bytes()
    rows = [json.loads(line) for line in selected_bytes.splitlines()]
    assert len(rows) > 100
    assert all(isinstance(row.get("messages"), list) and row["messages"] for row in rows)
    result = _run(default)
    assert result.returncode == 0, result.stdout + result.stderr
    assert default.read_bytes() == selected_bytes

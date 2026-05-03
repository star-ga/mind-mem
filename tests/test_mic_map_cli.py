"""Integration tests for the ``mm mic`` CLI subcommand.

Covers:

* ``mm mic convert <file> --to micb`` and ``--to mic2`` produce
  byte-identical output to the in-process ``emit_micb`` /
  ``emit_mic2`` calls.
* ``mm mic inspect`` text + JSON output shapes.
* Auto-detect: the same ``mm mic`` command handles both mic@2 and
  mic-b inputs based on the magic bytes / header line.
* Error paths: malformed input, missing file, unrecognised payload.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mind_mem.mic_map import (
    Arg,
    Graph,
    Node,
    Param,
    Type,
    emit_mic2,
    emit_micb,
)


def _residual_block() -> Graph:
    return Graph(
        types=[
            Type(dtype="f16", dims=("128", "128")),
            Type(dtype="f16", dims=("128",)),
        ],
        values=[
            Arg(name="X", type_idx=0),
            Param(name="W", type_idx=0),
            Param(name="b", type_idx=1),
            Node(opcode="m", inputs=(0, 1)),
            Node(opcode="+", inputs=(3, 2)),
            Node(opcode="r", inputs=(4,)),
            Node(opcode="+", inputs=(5, 0)),
        ],
        output=6,
    )


def _run_mm(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Invoke `mm` via the installed entry point, capturing stdout/err."""
    return subprocess.run(
        ["mm", *args],
        capture_output=True,
        check=False,
        text=False,  # bytes — mic-b output is binary
        **kwargs,
    )


@pytest.fixture
def mic_files(tmp_path: Path) -> dict[str, Path]:
    """Write the residual block in both formats; return paths."""
    g = _residual_block()
    mic2_path = tmp_path / "graph.mic2"
    micb_path = tmp_path / "graph.micb"
    mic2_path.write_text(emit_mic2(g), encoding="utf-8")
    micb_path.write_bytes(emit_micb(g))
    return {"graph": g, "mic2": mic2_path, "micb": micb_path}


# ---------------------------------------------------------------------------
# convert — mic2 ↔ micb round-trip
# ---------------------------------------------------------------------------


class TestMicConvert:
    def test_mic2_to_micb_byte_identical(self, mic_files: dict, tmp_path: Path) -> None:
        out_path = tmp_path / "out.micb"
        result = _run_mm(["mic", "convert", str(mic_files["mic2"]), "--to", "micb", "-o", str(out_path)])
        assert result.returncode == 0, result.stderr.decode()
        assert out_path.read_bytes() == mic_files["micb"].read_bytes()

    def test_micb_to_mic2_byte_identical(self, mic_files: dict, tmp_path: Path) -> None:
        out_path = tmp_path / "out.mic2"
        result = _run_mm(["mic", "convert", str(mic_files["micb"]), "--to", "mic2", "-o", str(out_path)])
        assert result.returncode == 0, result.stderr.decode()
        assert out_path.read_bytes() == mic_files["mic2"].read_bytes()

    def test_convert_to_stdout_text(self, mic_files: dict) -> None:
        result = _run_mm(["mic", "convert", str(mic_files["micb"]), "--to", "mic2"])
        assert result.returncode == 0
        assert result.stdout == mic_files["mic2"].read_bytes()

    def test_convert_to_stdout_binary(self, mic_files: dict) -> None:
        result = _run_mm(["mic", "convert", str(mic_files["mic2"]), "--to", "micb"])
        assert result.returncode == 0
        assert result.stdout == mic_files["micb"].read_bytes()


# ---------------------------------------------------------------------------
# inspect — text + JSON
# ---------------------------------------------------------------------------


class TestMicInspect:
    def test_inspect_text_mic2(self, mic_files: dict) -> None:
        result = _run_mm(["mic", "inspect", str(mic_files["mic2"])])
        assert result.returncode == 0
        out = result.stdout.decode()
        assert "format:        mic2" in out
        assert "types:         2" in out
        assert "values:        7" in out
        assert "output:        #6" in out
        # Each value type appears at least once.
        assert "arg     X" in out
        assert "param   W" in out
        assert "node    m" in out

    def test_inspect_text_micb(self, mic_files: dict) -> None:
        result = _run_mm(["mic", "inspect", str(mic_files["micb"])])
        assert result.returncode == 0
        out = result.stdout.decode()
        assert "format:        micb" in out
        assert "types:         2" in out
        assert "values:        7" in out

    def test_inspect_json(self, mic_files: dict) -> None:
        result = _run_mm(["mic", "inspect", str(mic_files["mic2"]), "--json"])
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert parsed["format"] == "mic2"
        assert parsed["type_count"] == 2
        assert parsed["value_count"] == 7
        assert parsed["output_idx"] == 6
        # Values are tagged by kind.
        kinds = [v["kind"] for v in parsed["values"]]
        assert kinds == ["arg", "param", "param", "node", "node", "node", "node"]
        assert parsed["values"][3]["opcode"] == "m"


# ---------------------------------------------------------------------------
# Errors — missing file, malformed input
# ---------------------------------------------------------------------------


class TestMicErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        result = _run_mm(["mic", "convert", str(tmp_path / "nope.mic2"), "--to", "micb"])
        assert result.returncode != 0
        assert b"error:" in result.stderr

    def test_unrecognised_payload(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.txt"
        bad.write_text("this is not a mic payload\n", encoding="utf-8")
        result = _run_mm(["mic", "inspect", str(bad)])
        assert result.returncode != 0
        assert b"not a recognised" in result.stderr

    def test_corrupted_micb(self, tmp_path: Path) -> None:
        # MICB magic but truncated body.
        bad = tmp_path / "bad.micb"
        bad.write_bytes(b"MICB\x02")  # magic + version, nothing else
        result = _run_mm(["mic", "inspect", str(bad)])
        assert result.returncode != 0
        assert b"parse failed" in result.stderr

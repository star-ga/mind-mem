"""Integration tests for the MIC/MAP MCP tools (``mic_convert_tool``,
``mic_inspect_tool``).

Both tools take a ``str`` payload (text for mic@2, base64 for mic-b)
and return a JSON-string envelope. Tests exercise:

* convert round-trips byte-for-byte for both directions.
* inspect returns the same structural summary regardless of input
  format.
* auto-detect handles both header-tagged text and base64-encoded
  binary on the same code path.
* size guard rejects payloads above ``MAX_INPUT_BYTES``.
* error paths return ``ok: False`` with a structured ``error`` field
  rather than raising.
"""

from __future__ import annotations

import base64
import json

import pytest

from mind_mem.mcp.tools.mic_map import (
    MAX_INPUT_BYTES,
    mic_convert_tool,
    mic_inspect_tool,
)
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


@pytest.fixture
def mic_payloads() -> dict:
    g = _residual_block()
    text = emit_mic2(g)
    binary = emit_micb(g)
    return {
        "graph": g,
        "mic2_text": text,
        "micb_bytes": binary,
        "micb_b64": base64.b64encode(binary).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# mic_convert_tool — both directions, byte-identical
# ---------------------------------------------------------------------------


class TestMicConvertTool:
    def test_mic2_to_micb_byte_identical(self, mic_payloads: dict) -> None:
        result = json.loads(
            mic_convert_tool(mic_payloads["mic2_text"], input_format="mic2", output_format="micb")
        )
        assert result["ok"] is True
        assert result["input_format"] == "mic2"
        assert result["output_format"] == "micb"
        out_bytes = base64.b64decode(result["output"])
        assert out_bytes == mic_payloads["micb_bytes"]
        assert result["byte_size"] == len(mic_payloads["micb_bytes"])
        assert result["value_count"] == 7
        assert result["type_count"] == 2

    def test_micb_to_mic2_byte_identical(self, mic_payloads: dict) -> None:
        result = json.loads(
            mic_convert_tool(mic_payloads["micb_b64"], input_format="micb", output_format="mic2")
        )
        assert result["ok"] is True
        assert result["output"] == mic_payloads["mic2_text"]

    def test_auto_detect_mic2(self, mic_payloads: dict) -> None:
        # auto-detect on text — leading "mic@2"
        result = json.loads(
            mic_convert_tool(mic_payloads["mic2_text"], input_format="auto", output_format="micb")
        )
        assert result["input_format"] == "mic2"
        assert base64.b64decode(result["output"]) == mic_payloads["micb_bytes"]

    def test_auto_detect_micb(self, mic_payloads: dict) -> None:
        # auto-detect on base64 mic-b — MICB magic after decode
        result = json.loads(
            mic_convert_tool(mic_payloads["micb_b64"], input_format="auto", output_format="mic2")
        )
        assert result["input_format"] == "micb"
        assert result["output"] == mic_payloads["mic2_text"]


# ---------------------------------------------------------------------------
# mic_inspect_tool — same summary regardless of format
# ---------------------------------------------------------------------------


class TestMicInspectTool:
    def test_inspect_mic2(self, mic_payloads: dict) -> None:
        result = json.loads(mic_inspect_tool(mic_payloads["mic2_text"], input_format="mic2"))
        assert result["ok"] is True
        assert result["input_format"] == "mic2"
        assert result["type_count"] == 2
        assert result["value_count"] == 7
        assert result["output_idx"] == 6
        kinds = [v["kind"] for v in result["values"]]
        assert kinds == ["arg", "param", "param", "node", "node", "node", "node"]

    def test_inspect_micb_matches_mic2(self, mic_payloads: dict) -> None:
        a = json.loads(mic_inspect_tool(mic_payloads["mic2_text"], input_format="mic2"))
        b = json.loads(mic_inspect_tool(mic_payloads["micb_b64"], input_format="micb"))
        # Strip format-specific fields, compare structure.
        for k in ("input_format",):
            a.pop(k, None)
            b.pop(k, None)
        assert a == b


# ---------------------------------------------------------------------------
# Errors + size guard
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_invalid_input_format(self, mic_payloads: dict) -> None:
        result = json.loads(
            mic_convert_tool(mic_payloads["mic2_text"], input_format="bogus", output_format="micb")
        )
        assert result["ok"] is False
        assert "input_format" in result["error"]

    def test_invalid_output_format(self, mic_payloads: dict) -> None:
        result = json.loads(
            mic_convert_tool(mic_payloads["mic2_text"], input_format="mic2", output_format="json")
        )
        assert result["ok"] is False
        assert "output_format" in result["error"]

    def test_corrupted_micb(self) -> None:
        # MICB magic + version, then nothing — parser fails inside.
        corrupted = base64.b64encode(b"MICB\x02").decode("ascii")
        result = json.loads(mic_convert_tool(corrupted, input_format="micb", output_format="mic2"))
        assert result["ok"] is False
        assert "parse failed" in result["error"]

    def test_invalid_base64(self) -> None:
        result = json.loads(mic_convert_tool("not!valid!base64!", input_format="micb", output_format="mic2"))
        assert result["ok"] is False
        assert "base64" in result["error"]

    def test_size_guard(self) -> None:
        # 9 MiB of plain text — over MAX_INPUT_BYTES (8 MiB).
        oversize = "x" * (MAX_INPUT_BYTES + 1024 * 1024)
        result = json.loads(mic_convert_tool(oversize, input_format="mic2", output_format="micb"))
        assert result["ok"] is False
        assert "exceeds" in result["error"]

    def test_inspect_corrupted(self) -> None:
        result = json.loads(mic_inspect_tool("not a mic payload", input_format="mic2"))
        assert result["ok"] is False
        assert "parse failed" in result["error"]

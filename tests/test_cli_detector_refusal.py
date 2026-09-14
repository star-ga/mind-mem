# Copyright 2026 STARGA, Inc.
"""CLI detector failures are stable refusals, never plugin tracebacks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mind_mem import mm_cli
from mind_mem.init_workspace import init

CANARY = "CLI-CANARY"
PLUGIN_DETAIL = "private-detector-detail"


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, failure: str = "raise") -> Path:
    package = tmp_path / f"cli_detector_{failure}"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    if failure == "raise":
        scan_body = f'        raise RuntimeError("{PLUGIN_DETAIL}: " + text)\n'
    elif failure == "malformed":
        scan_body = "        return [object()]\n"
    else:
        scan_body = (
            '        start = text.find("CLI-CANARY")\n'
            "        if start < 0:\n"
            "            return []\n"
            '        return [Finding(start, start + len("CLI-CANARY"), self.name, self.category)]\n'
        )
    (package / "detector.py").write_text(
        "from mind_mem.compliance.detectors import CATEGORY_SECRET, Detector, Finding\n"
        "class CliDetector(Detector):\n"
        "    name = 'cli_runtime_detector'\n"
        "    category = CATEGORY_SECRET\n"
        "    def scan(self, text):\n"
        f"{scan_body}"
        "        return []\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, f"cli_detector_{failure}.detector", raising=False)
    monkeypatch.delitem(sys.modules, f"cli_detector_{failure}", raising=False)
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("v4", {})["redaction"] = {
        "enabled": True,
        "mode": "redact",
        "detectors": [],
        "plugins": [f"cli_detector_{failure}.detector:CliDetector"],
    }
    config["v4"]["compliance_export"] = {"enabled": True}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(config_path))
    return workspace


def _args(command: str, output: Path) -> list[str]:
    text = f"safe {CANARY}"
    if command == "redact":
        return ["compliance", "redact", "--text", text, "--json"]
    if command == "screen":
        return ["compliance", "screen", "--text", text, "--json"]
    return ["export", "--policy", "redacted", "--out", str(output)]


@pytest.mark.parametrize("failure", ["raise", "malformed"])
@pytest.mark.parametrize("command", ["redact", "screen", "export"])
def test_runtime_detector_failure_is_a_stable_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    command: str,
) -> None:
    workspace = _workspace(tmp_path, monkeypatch, failure=failure)
    output = tmp_path / "existing-bundle.jsonl"
    output.write_bytes(b"preserve-this-output")
    if command == "export":
        (workspace / "decisions" / "DECISIONS.md").write_text(
            f"[D-CLI-FAILURE]\nStatus: active\nDate: 2026-09-14\nStatement: {CANARY}\n", encoding="utf-8"
        )

    result = mm_cli.main(_args(command, output))
    captured = capsys.readouterr()

    assert result == 4
    combined = captured.out + captured.err
    assert "compliance_detector_failed" in combined
    assert "Traceback" not in combined
    assert CANARY not in combined
    assert PLUGIN_DETAIL not in combined
    if command == "export":
        assert output.read_bytes() == b"preserve-this-output"
    else:
        payload = json.loads(captured.out)
        assert payload["error"] == "compliance_detector_failed"


def test_redact_in_place_preserves_source_on_detector_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, monkeypatch, failure="raise")
    source = tmp_path / "input.txt"
    original = f"safe {CANARY}\n".encode()
    source.write_bytes(original)

    assert mm_cli.main(["compliance", "redact", "--file", str(source), "--in-place", "--json"]) == 4
    captured = capsys.readouterr()
    assert "compliance_detector_failed" in captured.out
    assert source.read_bytes() == original
    assert CANARY not in captured.out + captured.err


@pytest.mark.parametrize("command", ["redact", "screen", "export"])
def test_same_plugin_success_remains_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    workspace = _workspace(tmp_path, monkeypatch, failure="success")
    output = tmp_path / "bundle.jsonl"
    if command == "export":
        (workspace / "decisions" / "DECISIONS.md").write_text(
            f"[D-CLI-1]\nStatus: active\nDate: 2026-09-14\nStatement: {CANARY}\n", encoding="utf-8"
        )
    result = mm_cli.main(_args(command, output))
    captured = capsys.readouterr()

    assert result == 0
    if command == "export":
        assert CANARY.encode() not in output.read_bytes()
        assert json.loads(captured.out)["block_count"] == 1
    elif command == "redact":
        assert json.loads(captured.out)["changed"] is True
    else:
        assert CANARY not in captured.out


@pytest.mark.parametrize("policy", ["full", "redacted", "metadata-only"])
@pytest.mark.parametrize("fmt", ["jsonl", "markdown"])
@pytest.mark.parametrize("since", [None, "2026-09-14"])
def test_export_cli_policy_format_since_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    policy: str,
    fmt: str,
    since: str | None,
) -> None:
    workspace = _workspace(tmp_path, monkeypatch, failure="success")
    (workspace / "decisions" / "DECISIONS.md").write_text(
        f"[D-CLI-MATRIX]\nStatus: active\nDate: 2026-09-14\nStatement: {CANARY}\n", encoding="utf-8"
    )
    output = tmp_path / f"matrix.{fmt}"
    args = ["export", "--policy", policy, "--format", fmt, "--out", str(output)]
    if since is not None:
        args.extend(["--since", since])
    assert mm_cli.main(args) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["policy"] == policy
    assert envelope["format"] == fmt
    assert envelope["block_count"] == 1
    assert output.stat().st_size > 0

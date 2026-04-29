"""Tests for the arch-mind MCP tool wrapper.

Covers:
* binary discovery (env var + PATH fallback + missing-binary error)
* every tool's argv shape (subprocess invocation is mocked)
* error paths (non-zero exit, timeout)
* registration (the seven tools land on a FastMCP instance)

The wrapper itself is a thin subprocess shim — every tool just
shells out to the ``arch-mind`` binary and bundles the result. We
mock ``subprocess.run`` so these tests run on any machine without
needing arch-mind installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Module under test (loaded via importlib so we don't depend on the rest of
# the mind-mem MCP server being importable in the test environment).
# ---------------------------------------------------------------------------


SRC = Path(__file__).resolve().parents[1] / "src" / "mind_mem" / "mcp" / "tools" / "arch_mind.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_test_arch_mind", SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Provide a fake parent package + observability stub so the relative
    # import in arch_mind.py resolves without pulling in the full
    # mind-mem MCP infrastructure.
    import sys
    import types

    pkg = types.ModuleType("_test_arch_mind_pkg")
    infra = types.ModuleType("_test_arch_mind_pkg.infra")
    obs = types.ModuleType("_test_arch_mind_pkg.infra.observability")

    def _noop(fn):
        return fn

    obs.mcp_tool_observe = _noop
    pkg.infra = infra
    infra.observability = obs
    sys.modules["_test_arch_mind_pkg"] = pkg
    sys.modules["_test_arch_mind_pkg.infra"] = infra
    sys.modules["_test_arch_mind_pkg.infra.observability"] = obs

    src_text = SRC.read_text().replace(
        "from ..infra.observability import mcp_tool_observe",
        "from _test_arch_mind_pkg.infra.observability import mcp_tool_observe",
    )

    code = compile(src_text, str(SRC), "exec")
    exec(code, mod.__dict__)
    return mod


arch_mind = _load_module()


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def test_resolve_binary_via_env(monkeypatch, tmp_path):
    fake = tmp_path / "arch-mind"
    fake.write_text("")
    monkeypatch.setenv("ARCH_MIND_BIN", str(fake))
    assert arch_mind._resolve_binary() == str(fake)


def test_resolve_binary_via_path(monkeypatch, tmp_path):
    monkeypatch.delenv("ARCH_MIND_BIN", raising=False)
    monkeypatch.setattr(arch_mind.shutil, "which", lambda *_: "/fake/arch-mind")
    assert arch_mind._resolve_binary() == "/fake/arch-mind"


def test_resolve_binary_missing(monkeypatch):
    monkeypatch.delenv("ARCH_MIND_BIN", raising=False)
    monkeypatch.setattr(arch_mind.shutil, "which", lambda *_: None)
    with pytest.raises(arch_mind.ArchMindError):
        arch_mind._resolve_binary()


# ---------------------------------------------------------------------------
# _run helper
# ---------------------------------------------------------------------------


def _make_subprocess_result(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["arch-mind"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_run_success_parses_json(monkeypatch):
    monkeypatch.setenv("ARCH_MIND_BIN", "/bin/true")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: _make_subprocess_result(0, '{"baseline_id": 7}\n')
    )
    result = arch_mind._run(["baseline"])
    assert result["ok"] is True
    assert result["code"] == 0
    assert result["json"] == {"baseline_id": 7}


def test_run_failure_returns_structured(monkeypatch):
    monkeypatch.setenv("ARCH_MIND_BIN", "/bin/true")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: _make_subprocess_result(2, "", "boom\n")
    )
    result = arch_mind._run(["scan"])
    assert result["ok"] is False
    assert result["code"] == 2
    assert "boom" in result["stderr"]
    assert result["json"] is None


def test_run_timeout(monkeypatch):
    monkeypatch.setenv("ARCH_MIND_BIN", "/bin/true")

    def boom(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["arch-mind"], timeout=1)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(arch_mind.ArchMindError) as exc:
        arch_mind._run(["scan"], timeout=1)
    assert "timed out" in str(exc.value)


# ---------------------------------------------------------------------------
# Per-tool argv shape — captures the command the wrapper would invoke.
# ---------------------------------------------------------------------------


def _capture_argv(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured["kwargs"] = kwargs
        return _make_subprocess_result(0, "{}\n")

    monkeypatch.setenv("ARCH_MIND_BIN", "/fake/arch-mind")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_arch_baseline_argv(monkeypatch):
    captured = _capture_argv(monkeypatch)
    arch_mind.arch_baseline(repo="/r", fixture="/f.json")
    assert captured["args"][0] == "/fake/arch-mind"
    assert captured["args"][1] == "baseline"
    assert "/f.json" in captured["args"]
    assert "/r/.arch-mind/baseline.json" in captured["args"]


def test_arch_delta_argv(monkeypatch):
    captured = _capture_argv(monkeypatch)
    arch_mind.arch_delta(repo="/r", before="/a.json", after="/b.json")
    assert captured["args"][1] == "delta"
    assert "/a.json" in captured["args"]
    assert "/b.json" in captured["args"]


def test_arch_history_with_days_and_verify(monkeypatch):
    captured = _capture_argv(monkeypatch)
    arch_mind.arch_history(repo="/r", days=30, verify=True)
    assert captured["args"][1] == "history"
    assert "--repo" in captured["args"]
    assert "--days" in captured["args"]
    assert "30" in captured["args"]
    assert "--verify" in captured["args"]


def test_arch_history_default_no_days(monkeypatch):
    captured = _capture_argv(monkeypatch)
    arch_mind.arch_history(repo="/r")
    assert captured["args"][1] == "history"
    assert "--days" not in captured["args"]
    assert "--verify" not in captured["args"]


def test_arch_check_rules_default_rules_path(monkeypatch):
    captured = _capture_argv(monkeypatch)
    arch_mind.arch_check_rules(repo="/r", fixture="/f.json")
    assert captured["args"][1] == "check-rules"
    # No explicit --rules: arch-mind binary defaults to <repo>/.arch-mind/rules.mind
    assert "--rules" not in captured["args"]
    assert "report" in captured["args"]


def test_arch_check_rules_explicit(monkeypatch):
    captured = _capture_argv(monkeypatch)
    arch_mind.arch_check_rules(
        repo="/r", fixture="/f.json", rules="/x/rules.mind", mode="enforce"
    )
    assert "--rules" in captured["args"]
    assert "/x/rules.mind" in captured["args"]
    assert "enforce" in captured["args"]


def test_arch_session_start_argv(monkeypatch):
    captured = _capture_argv(monkeypatch)
    arch_mind.arch_session_start(
        repo="/r", fixture="/f.json", agent_id="claude-test", commit_sha="deadbeef" * 5
    )
    assert captured["args"][1] == "session-start"
    assert "--agent" in captured["args"]
    assert "claude-test" in captured["args"]
    assert "--commit" in captured["args"]


def test_arch_session_end_argv(monkeypatch):
    captured = _capture_argv(monkeypatch)
    arch_mind.arch_session_end(repo="/r", fixture="/f.json")
    assert captured["args"][1] == "session-end"
    assert "--repo" in captured["args"]
    assert "--fixture" in captured["args"]


def test_arch_metric_explain_filters(monkeypatch):
    monkeypatch.setenv("ARCH_MIND_BIN", "/fake/arch-mind")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _make_subprocess_result(
            0,
            (
                "Scan: f.json\n"
                "  modularity_q16  q16=327680000  raw≈5000.00\n"
                "  acyclicity_q16  q16=655360000  raw≈10000.00\n"
            ),
        ),
    )
    result = arch_mind.arch_metric_explain(metric="modularity_q16", fixture="/f.json")
    assert result["ok"] is True
    assert result["metric"] == "modularity_q16"
    assert any("modularity_q16" in line for line in result["metric_lines"])
    # Make sure unrelated metrics are filtered out.
    assert not any("acyclicity_q16" in line for line in result["metric_lines"])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_wires_seven_tools():
    seen: list[str] = []

    class FakeMcp:
        def tool(self, fn):
            seen.append(fn.__name__)
            return fn

    arch_mind.register(FakeMcp())
    assert sorted(seen) == sorted(
        [
            "arch_baseline",
            "arch_delta",
            "arch_history",
            "arch_check_rules",
            "arch_session_start",
            "arch_session_end",
            "arch_metric_explain",
        ]
    )

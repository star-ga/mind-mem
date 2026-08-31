# Copyright 2026 STARGA, Inc.
"""Regression tests: the quality-gate preview must agree with enforcement.

``mcp.tools.quality.validate_block`` exists to pre-flight a statement
before ``propose_update`` stages it. It read no workspace configuration at
all, so in a workspace configured ``{"quality_gate": {"mode": "strict"}}``
it answered ``accept: true`` for statements ``propose_update`` rejects
with ``quality_gate_rejection`` — a preview of a decision it was not
previewing. Its docstring also advertised a flat ``quality_gate_mode``
key that nothing in the codebase reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.mcp.infra.config import _get_quality_gate_mode
from mind_mem.mcp.tools.quality import validate_block

# Fires the ``too_short`` rule: fewer than 32 non-whitespace characters.
_SHORT = "too short"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A workspace whose ``quality_gate.mode`` the test chooses."""

    def _configure(mode: str | None) -> Path:
        config: dict = {"version": "4.0.0"}
        if mode is not None:
            config["quality_gate"] = {"mode": mode}
        (tmp_path / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("MIND_MEM_ACL_DISABLED", "true")
        return tmp_path

    return _configure


def _verdict(**kwargs) -> dict:
    return json.loads(validate_block(_SHORT, **kwargs))


class TestPreviewFollowsWorkspaceMode:
    def test_strict_workspace_previews_as_reject(self, workspace) -> None:
        """The case that mattered: enforcement rejects, so the preview must."""
        workspace("strict")
        assert _get_quality_gate_mode() == "strict"
        verdict = _verdict()
        assert verdict["accept"] is False
        assert verdict["mode"] == "strict"
        assert verdict["strict"] is True
        assert any("too_short" in r for r in verdict["reasons"])

    def test_advisory_workspace_previews_as_accept(self, workspace) -> None:
        workspace("advisory")
        verdict = _verdict()
        assert verdict["accept"] is True
        assert verdict["mode"] == "advisory"
        assert verdict["strict"] is False
        assert any("too_short" in a for a in verdict["advisory"])

    def test_unconfigured_workspace_stays_advisory(self, workspace) -> None:
        workspace(None)
        verdict = _verdict()
        assert verdict["accept"] is True
        assert verdict["mode"] == "advisory"

    def test_off_workspace_never_rejects(self, workspace) -> None:
        """``off`` means propose_update skips the gate — so must the preview."""
        workspace("off")
        verdict = _verdict()
        assert verdict["accept"] is True
        assert verdict["mode"] == "off"

    def test_explicit_strict_arg_still_wins(self, workspace) -> None:
        workspace("advisory")
        verdict = _verdict(strict=True)
        assert verdict["accept"] is False
        assert verdict["strict"] is True

    def test_force_still_accepts_in_a_strict_workspace(self, workspace) -> None:
        workspace("strict")
        verdict = _verdict(force=True)
        assert verdict["accept"] is True
        assert verdict["forced"] is True

    def test_clean_text_accepts_in_a_strict_workspace(self, workspace) -> None:
        workspace("strict")
        text = "The connection manager now closes every read connection it handed out."
        verdict = json.loads(validate_block(text))
        assert verdict["accept"] is True
        assert verdict["reasons"] == []

    def test_non_string_input_still_reports_an_error(self, workspace) -> None:
        workspace("strict")
        payload = json.loads(validate_block(123))  # type: ignore[arg-type]
        assert payload["error"] == "text must be a string"


class TestDocumentedKeyIsTheKeyThatIsRead:
    def test_documented_key_appears_in_the_module_docstring(self) -> None:
        """The docstring must name the nested key the config layer reads."""
        import mind_mem.mcp.tools.quality as quality_module

        doc = quality_module.__doc__ or ""
        assert '"quality_gate": {"mode": "strict"}' in doc
        assert "quality_gate_mode" not in doc.replace("_get_quality_gate_mode", "")

    def test_flat_key_changes_nothing(self, workspace, tmp_path: Path) -> None:
        """An operator writing the once-documented flat key must not be told
        it is enforcing when it is not — the mode stays the default."""
        workspace(None)
        (tmp_path / "mind-mem.json").write_text(json.dumps({"quality_gate_mode": "strict"}), encoding="utf-8")
        assert _get_quality_gate_mode() == "advisory"
        assert _verdict()["mode"] == "advisory"

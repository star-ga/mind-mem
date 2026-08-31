"""``mm bind`` — the missing command that arms GovernanceGate step 1.

``GovernanceGate.admit`` verifies the config spec-hash only when
``.spec_binding.json`` exists.  ``SpecBindingManager.bind`` wrote it, but
nothing in the shipped CLI or MCP surface ever called ``bind`` — only
tests did — and ``init_workspace`` does not write one.  So on every
workspace the shipped tooling can produce, step 1 was inert: an operator
or agent editing ``governance_mode`` / ``mcp_acl.admin_tools`` /
``proposal_budget`` in ``mind-mem.json`` was never caught, while the
module docstring described the gate as verifying spec-hash consistency.

The mechanism these tests check is not "the command returns 0" — it is
that after running it, ``admit`` *raises* on a config edit it previously
admitted.  ``test_gate_admits_a_tampered_config_before_binding`` is the
control that shows the hole is real; it is the reason the rest matter.

Every test here fails on the pre-fix tree with ImportError /
AttributeError: ``_cmd_bind`` did not exist.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from mind_mem.governance_gate import GovernanceBypassError, GovernanceGate
from mind_mem.mm_cli import _cmd_bind


def _args(workspace: str, *, rebind: bool = False, config: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(workspace=workspace, config=config, rebind=rebind, json=True)


def _out(capsys: pytest.CaptureFixture[str]) -> dict:
    captured = capsys.readouterr()
    return json.loads(captured.out or captured.err)


@pytest.fixture()
def ws(tmp_path: Path) -> str:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "mind-mem.json").write_text(json.dumps({"governance_mode": "strict"}), encoding="utf-8")
    return str(workspace)


def _admit(workspace: str, block_id: str = "D-20260830-001") -> None:
    gate = GovernanceGate(workspace)
    try:
        gate.admit(action="WRITE", block_id=block_id, content="body", actor="test")
    finally:
        gate.close()


def _tamper(workspace: str) -> None:
    """The edit the gate exists to catch."""
    cfg = Path(workspace) / "mind-mem.json"
    cfg.write_text(json.dumps({"governance_mode": "off"}), encoding="utf-8")


class TestBindArmsTheGate:
    def test_gate_admits_a_tampered_config_before_binding(self, ws: str) -> None:
        """The hole, stated. No binding → step 1 does not run."""
        assert not os.path.isfile(os.path.join(ws, ".spec_binding.json"))
        _tamper(ws)
        _admit(ws)  # must not raise — this is the pre-fix behaviour everywhere

    def test_binding_then_tampering_blocks_the_write(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert _cmd_bind(_args(ws)) == 0
        payload = _out(capsys)
        assert payload["bound"] is True and payload["changed"] is True
        assert os.path.isfile(os.path.join(ws, ".spec_binding.json"))

        _tamper(ws)
        with pytest.raises(GovernanceBypassError):
            _admit(ws)

    def test_binding_leaves_an_untampered_workspace_writable(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        """No-regression control: arming must not break normal writes."""
        assert _cmd_bind(_args(ws)) == 0
        capsys.readouterr()
        _admit(ws)


class TestBindRefusesToLaunderDrift:
    def test_second_bind_on_an_unchanged_config_is_a_no_op(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert _cmd_bind(_args(ws)) == 0
        first = _out(capsys)

        assert _cmd_bind(_args(ws)) == 0
        second = _out(capsys)

        assert second["changed"] is False
        assert second["spec_hash"] == first["spec_hash"]

    def test_drift_exits_three_and_does_not_rebind(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        """Silently re-attesting a drifted config would launder the edit."""
        assert _cmd_bind(_args(ws)) == 0
        original = _out(capsys)
        _tamper(ws)

        assert _cmd_bind(_args(ws)) == 3
        payload = _out(capsys)
        assert payload["error"] == "drifted"

        # The stored binding still names the reviewed config, so the gate
        # keeps refusing until a human says otherwise.
        stored = json.loads((Path(ws) / ".spec_binding.json").read_text(encoding="utf-8"))
        assert stored["spec_hash"] == original["spec_hash"]
        with pytest.raises(GovernanceBypassError):
            _admit(ws)

    def test_explicit_rebind_reattests_and_unblocks(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert _cmd_bind(_args(ws)) == 0
        original = _out(capsys)
        _tamper(ws)

        assert _cmd_bind(_args(ws, rebind=True)) == 0
        payload = _out(capsys)
        assert payload["changed"] is True
        assert payload["spec_hash"] != original["spec_hash"]

        _admit(ws)  # reviewed and accepted → writes flow again


class TestBindInputHandling:
    def test_missing_config_is_an_error_not_a_binding(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _cmd_bind(_args(str(empty))) == 2
        assert _out(capsys)["error"] == "config_missing"
        assert not (empty / ".spec_binding.json").exists()

    def test_corrupt_binding_is_not_reinterpreted_as_absent(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        (Path(ws) / ".spec_binding.json").write_text("{not json", encoding="utf-8")

        assert _cmd_bind(_args(ws)) == 3
        assert _out(capsys)["error"] == "binding_corrupted"

        assert _cmd_bind(_args(ws, rebind=True)) == 0
        assert _out(capsys)["changed"] is True

    def test_tilde_workspace_is_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        home = tmp_path / "home"
        (home / "ws").mkdir(parents=True)
        (home / "ws" / "mind-mem.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))

        assert _cmd_bind(_args("~/ws")) == 0
        assert (home / "ws" / ".spec_binding.json").is_file()

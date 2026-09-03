# Copyright 2026 STARGA, Inc.
"""``mm config set`` — the configuration path that keeps the binding current.

``GovernanceGate`` step 1 verifies ``mind-mem.json`` against
``.spec_binding.json``, and as of 5.0.2 ``init`` arms every workspace it
creates.  Until this command existed, the only way to change a documented
setting was to hand-edit the file — which *is* the drift the gate exists
to catch.  Under ``enforce`` that made every setting change a total write
outage until someone remembered ``mm bind --rebind``; under
``detect_only`` it wrote a ``DRIFT`` row for a change nobody was hiding.
Remembering is not a mechanism.  This command is: the write and the
re-attestation are one step, so the supported path is also the correct
one and there is nothing left to remember.

The mechanism under test is never "the command returns 0".  It is that a
governed write **admits** after ``config set`` and **raises** after the
same edit made by hand — :class:`TestTheBindingStaysCurrent` holds both
halves, and the hand-edit half is the positive control that proves the
admit half is not vacuous.

Every test here fails on the pre-fix tree with ``ImportError``:
``config_set`` / ``_cmd_config_set`` did not exist.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pytest

from mind_mem.governance_gate import GovernanceBypassError, GovernanceGate
from mind_mem.init_workspace import init
from mind_mem.mm_cli import ConfigSetError, _cmd_config_set, config_set
from mind_mem.spec_binding import SpecBindingManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def ws(tmp_path: Path) -> str:
    """A real, armed workspace — ``init`` writes the binding itself."""
    workspace = str(tmp_path / "ws")
    init(workspace)
    binding = os.path.join(workspace, ".spec_binding.json")
    assert os.path.isfile(binding), "init did not arm the workspace; every test below would be vacuous"
    return workspace


def _config_path(workspace: str) -> str:
    return os.path.join(workspace, "mind-mem.json")


def _read_config(workspace: str) -> dict[str, Any]:
    with open(_config_path(workspace), encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, dict)
    return data


def _hand_edit(workspace: str, key: str, value: Any) -> None:
    """Change a setting the way an operator did before this command existed."""
    config = _read_config(workspace)
    config[key] = value
    with open(_config_path(workspace), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def _admit(workspace: str, block_id: str = "D-20260901-001") -> None:
    """One governed write, on a gate built fresh so nothing is cached."""
    gate = GovernanceGate(workspace)
    try:
        gate.admit(action="WRITE", block_id=block_id, content="body", actor="test")
    finally:
        gate.close()


def _binding_is_current(workspace: str) -> bool:
    valid, _reason = SpecBindingManager(_config_path(workspace)).verify()
    return valid


def _args(workspace: str, key: str, value: str, **overrides: Any) -> argparse.Namespace:
    fields: dict[str, Any] = {
        "workspace": workspace,
        "config": None,
        "key": key,
        "value": value,
        "raw_string": False,
        "json": True,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def _out(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    captured = capsys.readouterr()
    payload = json.loads(captured.out or captured.err)
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# The gate this command exists to keep satisfied
# ---------------------------------------------------------------------------


class TestTheBindingStaysCurrent:
    """The two halves that only mean something together."""

    def test_config_set_leaves_the_binding_current_and_the_workspace_writable(self, ws: str) -> None:
        config_set(_config_path(ws), "governance_mode", "enforce")

        assert _read_config(ws)["governance_mode"] == "enforce"
        assert _binding_is_current(ws), "config set left the binding behind its own write"
        _admit(ws)  # must not raise: the setting changed, nothing drifted

    def test_positive_control_the_same_edit_by_hand_is_refused_under_enforce(self, ws: str) -> None:
        """Proof the assertion above is not vacuous.

        Same workspace, same key, same value — reached by hand instead.
        If this admitted too, the test above would pass on a gate that
        checks nothing.
        """
        _hand_edit(ws, "governance_mode", "enforce")

        assert not _binding_is_current(ws), "the hand edit did not even drift; this control proves nothing"
        with pytest.raises(GovernanceBypassError, match="spec-hash drifted"):
            _admit(ws)

    def test_a_second_set_over_the_first_still_leaves_it_current(self, ws: str) -> None:
        """Repeated configuration is the normal case, not a one-shot."""
        config_set(_config_path(ws), "governance_mode", "enforce")
        config_set(_config_path(ws), "proposal_budget.backlog_limit", 500)
        config_set(_config_path(ws), "v4.multi_modal.enabled", True)

        assert _binding_is_current(ws)
        _admit(ws)

    def test_set_under_enforce_stays_writable(self, ws: str) -> None:
        """The mode that raises on drift is the one that must keep working."""
        config_set(_config_path(ws), "governance_mode", "enforce")
        config_set(_config_path(ws), "recall.backend", "sqlite")

        assert _read_config(ws)["governance_mode"] == "enforce"
        _admit(ws, "D-20260901-002")


# ---------------------------------------------------------------------------
# It must not become a laundering tool
# ---------------------------------------------------------------------------


class TestItRefusesToLaunderDrift:
    def test_a_config_that_already_drifted_is_refused(self, ws: str) -> None:
        _hand_edit(ws, "governance_mode", "off")

        with pytest.raises(ConfigSetError, match="already drifted"):
            config_set(_config_path(ws), "proposal_budget.backlog_limit", 500)

    def test_the_refusal_writes_nothing_at_all(self, ws: str) -> None:
        _hand_edit(ws, "governance_mode", "off")
        before_config = Path(_config_path(ws)).read_bytes()
        before_binding = Path(ws, ".spec_binding.json").read_bytes()

        with pytest.raises(ConfigSetError):
            config_set(_config_path(ws), "proposal_budget.backlog_limit", 500)

        assert Path(_config_path(ws)).read_bytes() == before_config
        assert Path(ws, ".spec_binding.json").read_bytes() == before_binding

    def test_the_unreviewed_edit_is_still_refused_by_the_gate_afterwards(self, ws: str) -> None:
        """The drift survives the attempt — it was not quietly attested."""
        config_set(_config_path(ws), "governance_mode", "enforce")
        _hand_edit(ws, "proposal_budget", {"backlog_limit": 1})

        with pytest.raises(ConfigSetError):
            config_set(_config_path(ws), "v4.multi_modal.enabled", True)
        with pytest.raises(GovernanceBypassError):
            _admit(ws)


# ---------------------------------------------------------------------------
# Armed-ness is never changed as a side effect
# ---------------------------------------------------------------------------


class TestArmednessIsPreserved:
    def test_an_unbound_workspace_is_not_armed_by_a_set(self, tmp_path: Path) -> None:
        workspace = tmp_path / "unbound"
        workspace.mkdir()
        config = workspace / "mind-mem.json"
        config.write_text(json.dumps({"governance_mode": "detect_only"}), encoding="utf-8")

        result = config_set(str(config), "governance_mode", "enforce")

        assert result["rebound"] is False
        assert result["spec_hash"] is None
        assert not (workspace / ".spec_binding.json").exists(), "config set armed a workspace `mm bind` had not"

    def test_an_armed_workspace_stays_armed(self, ws: str) -> None:
        config_set(_config_path(ws), "governance_mode", "enforce")
        assert os.path.isfile(os.path.join(ws, ".spec_binding.json"))


# ---------------------------------------------------------------------------
# Key and value handling
# ---------------------------------------------------------------------------


class TestKeysAndValues:
    def test_a_dotted_key_creates_the_nesting(self, ws: str) -> None:
        config_set(_config_path(ws), "v4.multi_modal.enabled", True)
        assert _read_config(ws)["v4"]["multi_modal"] == {"enabled": True}

    def test_a_dotted_key_merges_into_an_existing_object(self, ws: str) -> None:
        config_set(_config_path(ws), "v4.multi_modal.enabled", True)
        config_set(_config_path(ws), "v4.ingest_serve.enabled", True)

        v4 = _read_config(ws)["v4"]
        assert v4["multi_modal"] == {"enabled": True}
        assert v4["ingest_serve"] == {"enabled": True}

    def test_untouched_keys_survive(self, ws: str) -> None:
        before = _read_config(ws)
        config_set(_config_path(ws), "governance_mode", "enforce")
        after = _read_config(ws)

        assert set(after) >= set(before)
        for key, value in before.items():
            if key != "governance_mode":
                assert after[key] == value, f"config set collaterally changed {key!r}"

    def test_descending_into_a_scalar_is_refused(self, ws: str) -> None:
        config_set(_config_path(ws), "recall", "sqlite")
        with pytest.raises(ConfigSetError, match="cannot descend"):
            config_set(_config_path(ws), "recall.backend", "sqlite")
        assert _read_config(ws)["recall"] == "sqlite", "the refused set damaged the value it refused to descend into"

    @pytest.mark.parametrize("key", ["", "   ", "a..b", ".a", "a."])
    def test_a_malformed_key_is_refused(self, ws: str, key: str) -> None:
        with pytest.raises(ConfigSetError):
            config_set(_config_path(ws), key, 1)

    def test_a_missing_config_is_refused_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / "nowhere" / "mind-mem.json"
        with pytest.raises(ConfigSetError, match="no config to set"):
            config_set(str(missing), "governance_mode", "enforce")
        assert not missing.exists()

    def test_a_non_object_config_is_refused_not_overwritten(self, tmp_path: Path) -> None:
        config = tmp_path / "mind-mem.json"
        config.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ConfigSetError, match="not a JSON object"):
            config_set(str(config), "governance_mode", "enforce")
        assert config.read_text(encoding="utf-8") == "[1, 2, 3]"

    def test_unparseable_json_is_refused_not_overwritten(self, tmp_path: Path) -> None:
        config = tmp_path / "mind-mem.json"
        config.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigSetError, match="not valid JSON"):
            config_set(str(config), "governance_mode", "enforce")
        assert config.read_text(encoding="utf-8") == "{not json"

    def test_the_report_names_what_changed(self, ws: str) -> None:
        result = config_set(_config_path(ws), "governance_mode", "enforce")
        assert result["previous"] == "detect_only"
        assert result["value"] == "enforce"
        assert result["changed"] is True

    def test_a_key_that_was_absent_reports_no_previous(self, ws: str) -> None:
        result = config_set(_config_path(ws), "brand_new_key", 1)
        assert "previous" not in result


# ---------------------------------------------------------------------------
# The argparse surface
# ---------------------------------------------------------------------------


class TestTheCommandSurface:
    def test_the_command_writes_rebinds_and_reports(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert _cmd_config_set(_args(ws, "governance_mode", "enforce")) == 0
        payload = _out(capsys)

        assert payload["rebound"] is True
        assert payload["value"] == "enforce"
        assert _read_config(ws)["governance_mode"] == "enforce"
        _admit(ws)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("false", False), ("500", 500), ("null", None), ('{"a": 1}', {"a": 1}), ("enforce", "enforce")],
    )
    def test_a_value_is_json_when_it_parses_and_a_string_otherwise(
        self, ws: str, raw: str, expected: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _cmd_config_set(_args(ws, "probe", raw)) == 0
        capsys.readouterr()
        assert _read_config(ws)["probe"] == expected

    def test_raw_string_keeps_a_json_looking_value_as_text(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert _cmd_config_set(_args(ws, "probe", "true", raw_string=True)) == 0
        capsys.readouterr()
        assert _read_config(ws)["probe"] == "true"

    def test_a_refusal_exits_two_and_says_why(self, ws: str, capsys: pytest.CaptureFixture[str]) -> None:
        _hand_edit(ws, "governance_mode", "off")
        assert _cmd_config_set(_args(ws, "probe", "1")) == 2
        payload = _out(capsys)
        assert payload["changed"] is False
        assert "already drifted" in payload["error"]

    def test_an_explicit_config_path_is_honoured(self, ws: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"governance_mode": "detect_only"}), encoding="utf-8")

        assert _cmd_config_set(_args(ws, "governance_mode", "enforce", config=str(other))) == 0
        capsys.readouterr()

        assert json.loads(other.read_text(encoding="utf-8"))["governance_mode"] == "enforce"
        assert _read_config(ws)["governance_mode"] == "detect_only", "the workspace config was written instead of --config"

    def test_the_parser_registers_the_subcommand(self) -> None:
        """Registration is a gate of its own: an unregistered command is unreachable."""
        from mind_mem.mm_cli import build_parser

        args = build_parser().parse_args(["config", "set", "governance_mode", "enforce"])
        assert args.func is _cmd_config_set
        assert (args.key, args.value) == ("governance_mode", "enforce")

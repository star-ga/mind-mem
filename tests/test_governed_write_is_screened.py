# Copyright 2026 STARGA, Inc.
"""The GOVERNED write path runs the compliance pre-write controls.

The compliance package shipped wired to the CLI and the export pipeline
first. That made the honest claim "the chain is wired", not "every write is
screened" -- and a compliance control the governed path does not run is a
feature, not a control. These tests pin the door itself.

Two defects were found by writing them, both in the wiring rather than in the
compliance package:

* resolving the policy raised out of ``propose_update`` on a malformed
  config, which would have taken down the tool rather than refusing one
  proposal;
* the provenance dict was keyed by the Python parameter names
  (``actor_id``) while the policy keys off the block-metadata names
  (``ActorId``), so every field read as absent and a ``required`` policy
  refused an ATTRIBUTED write exactly as loudly as an unattributed one.

The second is the reason ``test_the_same_write_with_attribution_is_admitted``
exists. Without it, a refusal looks like a working gate; with it, a gate that
refuses everything is distinguishable from a gate that refuses the right
thing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.init_workspace import init


def _ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **provenance: object) -> str:
    ws = str(tmp_path / "ws")
    init(ws)
    cfg = os.path.join(ws, "mind-mem.json")
    data = json.loads(Path(cfg).read_text(encoding="utf-8"))
    v4 = data.setdefault("v4", {})
    v4["provenance"] = provenance or {"enabled": False}
    v4["redaction"] = {"enabled": False}
    Path(cfg).write_text(json.dumps(data, indent=1), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
    monkeypatch.setenv("MIND_MEM_CONFIG", cfg)
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return ws


def _propose(**kwargs: str) -> dict:
    from mind_mem.mcp.tools.governance import propose_update

    kwargs.setdefault("block_type", "decision")
    kwargs.setdefault("rationale", "agreed in the review meeting")
    return json.loads(propose_update(**kwargs))


class TestTheGovernedWritePathIsScreened:
    def test_a_plain_write_is_admitted_when_the_policy_is_off(self, tmp_path, monkeypatch):
        """Positive control. A gate that refuses everything is not a gate."""
        _ws(tmp_path, monkeypatch, enabled=False)
        assert "error" not in _propose(statement="We ship the release on Friday.")

    def test_an_unattributed_write_is_refused_under_required(self, tmp_path, monkeypatch):
        _ws(tmp_path, monkeypatch, enabled=True, policy="required", fields=["ActorId"])
        assert _propose(statement="We ship on Monday.")["error"] == "provenance_required"

    def test_the_same_write_with_attribution_is_admitted(self, tmp_path, monkeypatch):
        """The control that caught the field-name defect.

        Identical to the refusal case except for the attribution, so a gate
        that refuses regardless of attribution fails HERE rather than passing
        as a working control.
        """
        _ws(tmp_path, monkeypatch, enabled=True, policy="required", fields=["ActorId"])
        assert "error" not in _propose(statement="We ship on Monday.", actor_id="nikolai")

    def test_a_malformed_policy_refuses_structurally_rather_than_crashing(self, tmp_path, monkeypatch):
        """Fails CLOSED: a policy that cannot be read is not a policy of 'off'."""
        _ws(tmp_path, monkeypatch, enabled=True, policy="off")
        assert _propose(statement="We ship on Tuesday.", actor_id="nikolai")["error"] == "compliance_config_invalid"

    def test_an_unknown_required_field_is_refused_rather_than_ignored(self, tmp_path, monkeypatch):
        """A field name outside the five is unenforceable, so it is an error."""
        _ws(tmp_path, monkeypatch, enabled=True, policy="required", fields=["actor_id"])
        assert _propose(statement="We ship later.", actor_id="nikolai")["error"] == "compliance_config_invalid"


class TestTheDoorIsWhereTheOtherGateIs:
    def test_the_screen_call_is_inside_propose_update(self):
        """Anchored on the enclosing function, not on a line number.

        An edit that moves the call out of the governed path -- or into a
        helper nothing calls -- makes this fail rather than silently
        un-screening every write.
        """
        import ast
        import inspect

        from mind_mem.mcp.tools import governance

        tree = ast.parse(inspect.getsource(governance))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "propose_update")
        called = {
            n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "") for n in ast.walk(fn) if isinstance(n, ast.Call)
        }
        assert "screen" in called, "propose_update no longer screens the write"
        assert "validate_block" in called, "the quality gate moved; the two belong at one door"

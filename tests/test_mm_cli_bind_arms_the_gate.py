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

from mind_mem import governance_gate
from mind_mem.evidence_objects import EvidenceAction
from mind_mem.governance_gate import GovernanceBypassError, GovernanceGate
from mind_mem.hash_chain_v2 import HashChainV2
from mind_mem.init_workspace import init
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


def _set_mode(workspace: str, *, mode: str | None = None, **settings: object) -> None:
    """Edit the config the way an operator does: by hand, in place.

    There is no ``mm config set``; hand-editing ``mind-mem.json`` is the
    documented and only configuration path, which is exactly why the
    drift response has to distinguish "an operator changed a setting"
    from "an attacker rewrote the governance rules".
    """
    cfg = Path(workspace) / "mind-mem.json"
    config = json.loads(cfg.read_text(encoding="utf-8"))
    if mode is not None:
        config["governance_mode"] = mode
    config.update(settings)
    cfg.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _wrote_block(workspace: str, block_id: str) -> bool:
    """True when *block_id* reached the hash chain — the ledger, not a log."""
    rows = HashChainV2(str(Path(workspace) / "memory" / "hash_chain_v2.db")).get_block_chain(block_id)
    return bool(rows)


def _drift_rows(workspace: str) -> list[dict]:
    """Evidence rows recording a detected config drift, in order."""
    path = Path(workspace) / "memory" / "evidence_chain.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("metadata", {}).get("action_verb") == "DRIFT"]


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


class TestInitArmsTheGate:
    """A workspace made by the shipped tooling is armed from birth.

    Through 5.0.1 ``init_workspace`` wrote no binding, so step 1 was
    inert on every workspace this product could produce: the
    ``TestBindArmsTheGate`` cases above all begin by hand-writing a
    ``mind-mem.json``, which is what a real ``init`` never did for them.
    """

    def test_init_writes_a_binding(self, tmp_path: Path) -> None:
        ws = tmp_path / "fresh"
        created, _skipped = init(str(ws))
        assert (ws / ".spec_binding.json").is_file()
        assert "file: .spec_binding.json" in created, "and says so, rather than arming silently"

    def test_a_fresh_workspace_detects_a_config_edit(self, tmp_path: Path) -> None:
        """The property, not the file: an edit is now *caught*.

        Under ``detect_only`` catching means recording, so the assertion
        is that a DRIFT row appears — not that the write dies. The
        ``enforce`` case below is the twin that shows a mode can still
        block.
        """
        ws = tmp_path / "fresh"
        init(str(ws))
        _admit(str(ws), "D-20260902-001")
        assert _drift_rows(str(ws)) == [], "control: an untouched config records no drift"

        _set_mode(str(ws), auto_recall=False)
        _admit(str(ws), "D-20260902-002")

        rows = _drift_rows(str(ws))
        assert len(rows) == 1, "the config edit was detected and recorded"
        assert rows[0]["metadata"]["governance_mode"] == "detect_only"

    def test_init_over_an_existing_workspace_does_not_rebind(self, tmp_path: Path) -> None:
        """Re-running init must not rubber-stamp a config it never reviewed.

        ``mm bind`` refuses to re-attest drift without ``--rebind``; init
        must not become the back door around that refusal.
        """
        ws = tmp_path / "fresh"
        init(str(ws))
        original = json.loads((ws / ".spec_binding.json").read_text(encoding="utf-8"))["spec_hash"]

        _set_mode(str(ws), auto_recall=False)
        created, skipped = init(str(ws))

        assert "mind-mem.json" in skipped, "precondition: the second init did not author the config"
        assert "file: .spec_binding.json" not in created
        stored = json.loads((ws / ".spec_binding.json").read_text(encoding="utf-8"))["spec_hash"]
        assert stored == original, "the binding still names the config that was reviewed"
        assert _drift_rows(str(ws)) == [], "…and nothing was written yet — the drift shows up on the next admit"
        _admit(str(ws), "D-20260902-003")
        assert len(_drift_rows(str(ws))) == 1


class TestDriftHonoursGovernanceMode:
    """Step 1 used to raise unconditionally, ignoring the mode entirely.

    That is why arming at birth was a write outage: hand-editing
    ``mind-mem.json`` is the only configuration path this product offers,
    and under the old behaviour a bound workspace that used it lost every
    governed write — in the shipped default ``detect_only``.
    """

    def test_detect_only_records_the_drift_and_admits(self, tmp_path: Path) -> None:
        ws = tmp_path / "fresh"
        init(str(ws))
        _set_mode(str(ws), auto_recall=False)

        _admit(str(ws), "D-20260902-010")  # must not raise

        rows = _drift_rows(str(ws))
        assert len(rows) == 1
        assert rows[0]["action"] == EvidenceAction.DRIFT.value, "recorded under the DRIFT member, which already existed"
        assert "spec_hash" not in rows[0]["metadata"], "a record of failed verification must not claim it verified"
        assert "config hash mismatch" in rows[0]["metadata"]["drift_reason"]

    def test_enforce_still_blocks_the_write(self, tmp_path: Path) -> None:
        """Positive control: ``detect_only`` admitting is a mode, not a hole."""
        ws = tmp_path / "fresh"
        init(str(ws))
        _set_mode(str(ws), mode="enforce", auto_recall=False)

        with pytest.raises(GovernanceBypassError):
            _admit(str(ws), "D-20260902-011")

    def test_the_refusal_records_the_drift_it_refused_on(self, tmp_path: Path) -> None:
        """The loudest response must not be the one that leaves no evidence.

        A refusal used to raise and write nothing: the tamper it caught
        survived only in a log line, and a ledger read afterwards showed
        no sign that anything had been detected. The record is written
        before the mode is consulted, so it lands under every response.
        """
        ws = tmp_path / "fresh"
        init(str(ws))
        _set_mode(str(ws), mode="enforce", auto_recall=False)

        with pytest.raises(GovernanceBypassError):
            _admit(str(ws), "D-20260902-014")

        rows = _drift_rows(str(ws))
        assert len(rows) == 1, "the refusal left a record of what it refused on"
        assert rows[0]["metadata"]["governance_mode"] == "enforce", "…naming the response that was applied"
        assert not _wrote_block(str(ws), "D-20260902-014"), "positive control: the write really was blocked"

    def test_a_relaxed_response_is_visible_in_the_record(self, tmp_path: Path) -> None:
        """The mode is downgradable by the edit it judges; the record is not.

        An attacker who sets ``governance_mode`` to ``detect_only`` in the
        same edit picks the lenient response — and cannot stop the ledger
        from naming both the drift and the fact that enforcement was
        relaxed for it.
        """
        ws = tmp_path / "fresh"
        init(str(ws))
        _set_mode(str(ws), mode="detect_only", proposal_budget={"per_run": 9999})

        _admit(str(ws), "D-20260902-015")

        rows = _drift_rows(str(ws))
        assert len(rows) == 1
        assert rows[0]["metadata"]["governance_mode"] == "detect_only"

    def test_an_unparseable_config_enforces(self, tmp_path: Path) -> None:
        """Fail closed: a config that cannot state its mode does not get the lenient one."""
        ws = tmp_path / "fresh"
        init(str(ws))
        (ws / "mind-mem.json").write_text("{ not json", encoding="utf-8")

        with pytest.raises(GovernanceBypassError):
            _admit(str(ws), "D-20260902-012")

    def test_a_config_with_no_mode_takes_the_shipped_default(self, tmp_path: Path) -> None:
        """An absent key is not an unreadable config.

        ``governance_mode`` has one shipped default in this package —
        ``DEFAULT_CONFIG`` writes ``detect_only``, ``apply_engine`` and
        ``intel_scan`` both default to it — and the gate applies the same
        one. Reading its absence as ``enforce`` instead was measured to
        refuse 69 governed writes across 24 test files whose workspaces
        carry a config without the key; that is not fail-closed, it is
        this module disagreeing with the package.
        """
        ws = tmp_path / "fresh"
        init(str(ws))
        config = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
        config.pop("governance_mode")
        (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")

        _admit(str(ws), "D-20260902-013")  # must not raise

        assert governance_gate.DEFAULT_MODE in governance_gate.DETECT_ONLY_MODES
        rows = _drift_rows(str(ws))
        assert len(rows) == 1, "the edit is still detected and recorded — only the response is lenient"
        assert rows[0]["metadata"]["governance_mode"] == governance_gate.DEFAULT_MODE

    def test_a_non_string_mode_enforces(self, tmp_path: Path) -> None:
        """Present but malformed is unreadable, and unreadable fails closed."""
        ws = tmp_path / "fresh"
        init(str(ws))
        config = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
        config["governance_mode"] = {"mode": "detect_only"}
        (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(GovernanceBypassError):
            _admit(str(ws), "D-20260902-016")

    def test_an_unbound_workspace_is_unchanged(self, ws: str) -> None:
        """No binding still means no drift check and no DRIFT row.

        The mode is consulted only *after* drift is found, so a workspace
        made before 5.0.2 behaves exactly as it did.
        """
        _tamper(ws)
        _admit(ws)
        assert _drift_rows(ws) == []

    def test_one_drift_observation_records_one_row(self, tmp_path: Path) -> None:
        """A bound-and-edited workspace must not flood the ledger.

        Under ``detect_only`` every admission re-detects the same drift.
        Recording it per admission would bury the auditor in copies of one
        finding; recording it per *observation* keeps the ledger readable.
        """
        ws = tmp_path / "fresh"
        init(str(ws))
        _set_mode(str(ws), auto_recall=False)

        gate = GovernanceGate(str(ws))
        try:
            for i in range(5):
                gate.admit(action="WRITE", block_id=f"D-20260902-02{i}", content="body", actor="test")
            assert len(_drift_rows(str(ws))) == 1

            # A *different* edit is a new observation, and records again.
            _set_mode(str(ws), auto_capture=False)
            gate.admit(action="WRITE", block_id="D-20260902-030", content="body", actor="test")
            assert len(_drift_rows(str(ws))) == 2
        finally:
            gate.close()


class TestMutationTwinDriftMode:
    """Restore the unconditional raise; the detect_only case must go red."""

    def test_detect_only_admitting_depends_on_the_mode_check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty the lenient set — the pre-5.0.2 unconditional raise, exactly.

        This is the outage that made arming at birth unshippable: a
        workspace in the shipped default mode, configured the only way
        this product allows, loses every governed write.
        """
        monkeypatch.setattr(governance_gate, "DETECT_ONLY_MODES", frozenset())
        ws = tmp_path / "fresh"
        init(str(ws))
        _set_mode(str(ws), auto_recall=False)

        with pytest.raises(GovernanceBypassError):
            _admit(str(ws), "D-20260902-040")
        assert not _wrote_block(str(ws), "D-20260902-040"), "twin precondition: the write died"

        # Which is what test_detect_only_records_the_drift_and_admits asserts
        # against — it calls _admit with no pytest.raises around it at all.
        with pytest.raises(GovernanceBypassError):
            _admit(str(ws), "D-20260902-041")

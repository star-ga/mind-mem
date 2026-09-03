# Copyright 2026 STARGA, Inc.
"""``provenance: off | recommended | required`` — the policy, and the door it closes.

The five provenance fields have shipped for releases; what did not exist
was the rule that decides whether a write may arrive without them. So the
tests here are about the *policy*, and the one that matters most is the
round trip: ``required`` refuses a write, and the same write with
attribution is admitted. A refusal state nothing can leave is a one-way
door, and a policy that only ever refuses is indistinguishable from a
broken one.

The pre-write door is tested through the CLI verb that calls it, because
"imported" is not "wired": the assertions run
``mm compliance screen`` / ``mm compliance provenance`` end to end and
read the process exit code, which is the contract a hook or an agent
actually depends on.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem import mm_cli
from mind_mem.compliance.prewrite import PreWritePolicy, screen
from mind_mem.compliance.provenance_policy import (
    POLICY_OFF,
    POLICY_RECOMMENDED,
    POLICY_REQUIRED,
    PROVENANCE_FIELDS,
    ProvenanceConfigError,
    ProvenanceRequired,
    evaluate_provenance,
    require_provenance,
    resolve_policy,
    resolve_required_fields,
)

FULL = {name: f"value-{name}" for name in PROVENANCE_FIELDS}
PARTIAL = {"ActorId": "agent-7", "ActorRole": "planner"}

ATTRIBUTED_BLOCK = """[DEC-20260105-001]
Title: Attributed
Status: active
Date: 2026-01-05
ActorId: agent-7
ActorRole: planner
SessionId: sess-1
ToolId: mm
Purpose: record the decision
Body: fine
"""

UNATTRIBUTED_BLOCK = """[DEC-20260106-002]
Title: Unattributed
Status: active
Date: 2026-01-06
Body: who wrote this
"""


def _workspace(tmp_path: Path, **v4: object) -> str:
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": v4}), encoding="utf-8")
    return str(tmp_path)


def _corpus(tmp_path: Path, blocks: str) -> None:
    (tmp_path / "decisions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions" / "DECISIONS.md").write_text(blocks, encoding="utf-8")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class TestPolicyResolution:
    def test_absent_flag_means_off(self, tmp_path: Path) -> None:
        assert resolve_policy(_workspace(tmp_path)) == POLICY_OFF

    def test_enabled_defaults_to_recommended(self, tmp_path: Path) -> None:
        """The migration-safe default: a corpus written before the policy
        existed must not become un-writable the day it is switched on."""
        assert resolve_policy(_workspace(tmp_path, provenance={"enabled": True})) == POLICY_RECOMMENDED

    def test_required_is_read_from_the_workspace(self, tmp_path: Path) -> None:
        assert resolve_policy(_workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_REQUIRED})) == POLICY_REQUIRED

    def test_a_bare_true_cannot_switch_it_on(self, tmp_path: Path) -> None:
        assert resolve_policy(_workspace(tmp_path, provenance=True)) == POLICY_OFF

    def test_an_unknown_policy_value_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ProvenanceConfigError):
            resolve_policy(_workspace(tmp_path, provenance={"enabled": True, "policy": "mostly"}))

    def test_all_five_fields_are_required_by_default(self, tmp_path: Path) -> None:
        assert resolve_required_fields(_workspace(tmp_path, provenance={"enabled": True})) == PROVENANCE_FIELDS

    def test_a_declared_subset_is_honoured_in_canonical_order(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, provenance={"enabled": True, "fields": ["Purpose", "ActorId"]})
        assert resolve_required_fields(ws) == ("ActorId", "Purpose")

    def test_a_field_nobody_checks_is_refused(self, tmp_path: Path) -> None:
        """A policy naming an unenforceable field reads as protection and is not."""
        ws = _workspace(tmp_path, provenance={"enabled": True, "fields": ["ActorId", "Vibes"]})
        with pytest.raises(ProvenanceConfigError) as excinfo:
            resolve_required_fields(ws)
        assert "Vibes" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_off_admits_everything(self) -> None:
        decision = evaluate_provenance({}, policy=POLICY_OFF, required=())
        assert decision.admitted is True
        assert decision.warnings == ()

    def test_recommended_warns_and_admits(self) -> None:
        decision = evaluate_provenance(PARTIAL, policy=POLICY_RECOMMENDED)
        assert decision.admitted is True
        assert decision.missing == ("SessionId", "ToolId", "Purpose")
        assert decision.warnings and "SessionId" in decision.warnings[0]

    def test_required_refuses_what_recommended_admits(self) -> None:
        """The two policies must differ, or one of them is decoration."""
        assert evaluate_provenance(PARTIAL, policy=POLICY_RECOMMENDED).admitted is True
        assert evaluate_provenance(PARTIAL, policy=POLICY_REQUIRED).admitted is False

    def test_required_admits_a_fully_attributed_write(self) -> None:
        decision = require_provenance(FULL, policy=POLICY_REQUIRED)
        assert decision.missing == ()
        assert decision.admitted is True

    def test_the_refusal_names_what_is_missing(self) -> None:
        with pytest.raises(ProvenanceRequired) as excinfo:
            require_provenance(PARTIAL, policy=POLICY_REQUIRED)
        assert "SessionId" in str(excinfo.value)

    def test_whitespace_is_not_attribution(self) -> None:
        blanks = {name: "   " for name in PROVENANCE_FIELDS}
        assert evaluate_provenance(blanks, policy=POLICY_REQUIRED).missing == PROVENANCE_FIELDS

    def test_a_non_string_value_is_not_attribution(self) -> None:
        assert evaluate_provenance({"ActorId": 7}, policy=POLICY_REQUIRED).missing == PROVENANCE_FIELDS


# ---------------------------------------------------------------------------
# The pre-write door: order, and the OFF short-circuit
# ---------------------------------------------------------------------------


class TestTheDoor:
    def test_both_controls_off_leaves_the_door_inert(self, tmp_path: Path) -> None:
        policy = PreWritePolicy.resolve(_workspace(tmp_path))
        assert policy.inert is True
        assert policy.detectors == () and policy.required_fields == ()

    def test_provenance_is_judged_before_the_text_is_scanned(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Order matters: a refused write must not have been scanned or hashed."""
        ws = _workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_REQUIRED}, redaction={"enabled": True})

        def _explode(*_a: object, **_k: object) -> list[object]:
            raise AssertionError("the text was scanned for a write that was already refused")

        monkeypatch.setattr("mind_mem.compliance.redaction.scan_text", _explode)
        with pytest.raises(ProvenanceRequired):
            screen("secret AKIAIOSFODNN7EXAMPLE", policy=PreWritePolicy.resolve(ws), provenance=PARTIAL)

    def test_an_attributed_write_passes_the_whole_door(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_REQUIRED}, redaction={"enabled": True})
        screening = screen("mail ops@example.com", policy=PreWritePolicy.resolve(ws), provenance=FULL, target="notes/x.md")
        assert "[REDACTED:email]" in screening.text
        assert screening.provenance.admitted is True
        assert screening.audit_seq == 1

    def test_a_read_only_pass_writes_no_ledger(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, redaction={"enabled": True})
        screening = screen("mail ops@example.com", policy=PreWritePolicy.resolve(ws), record=False)
        assert screening.audit_seq is None
        assert not (Path(ws) / ".mind-mem-audit").exists()
        # POSITIVE CONTROL: the recording pass on the same input does write.
        screen("mail ops@example.com", policy=PreWritePolicy.resolve(ws), record=True, target="notes/y.md")
        assert (Path(ws) / ".mind-mem-audit" / "chain.jsonl").is_file()


# ---------------------------------------------------------------------------
# The CLI is the entry point
# ---------------------------------------------------------------------------


class TestTheCommandLine:
    def test_screen_refuses_an_unattributed_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ws = _workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_REQUIRED})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["compliance", "screen", "--text", "hello"]) == 4
        assert "ActorId" in capsys.readouterr().err

    def test_screen_admits_the_same_write_with_attribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The round trip. Without it, 'refuses' could just mean 'broken'."""
        ws = _workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_REQUIRED})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        code = mm_cli.main(
            [
                "compliance",
                "screen",
                "--text",
                "hello",
                "--actor-id",
                "a",
                "--actor-role",
                "planner",
                "--session-id",
                "s",
                "--tool-id",
                "mm",
                "--purpose",
                "p",
            ]
        )
        assert code == 0
        assert capsys.readouterr().out == "hello"

    def test_screen_refuses_when_no_control_is_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("MIND_MEM_WORKSPACE", _workspace(tmp_path))
        assert mm_cli.main(["compliance", "screen", "--text", "hello"]) == 3
        assert "provenance" in capsys.readouterr().err

    def test_provenance_report_exits_one_on_an_unattributed_corpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _corpus(tmp_path, ATTRIBUTED_BLOCK + "\n" + UNATTRIBUTED_BLOCK)
        ws = _workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_REQUIRED})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["compliance", "provenance", "--json"]) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["admitted_blocks"] == 2
        assert report["blocks_missing_provenance"] == 1
        assert report["blocks"][0]["id"] == "DEC-20260106-002"

    def test_the_same_corpus_passes_once_it_is_attributed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Positive control for the exit-1 above: the check can also pass."""
        _corpus(tmp_path, ATTRIBUTED_BLOCK)
        ws = _workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_REQUIRED})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["compliance", "provenance", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["blocks_missing_provenance"] == 0

    def test_recommended_reports_but_does_not_fail_the_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _corpus(tmp_path, UNATTRIBUTED_BLOCK)
        ws = _workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_RECOMMENDED})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["compliance", "provenance", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["blocks_missing_provenance"] == 1

    def test_the_report_refuses_when_the_policy_is_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _corpus(tmp_path, ATTRIBUTED_BLOCK)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", _workspace(tmp_path))
        assert mm_cli.main(["compliance", "provenance"]) == 3
        assert "provenance" in capsys.readouterr().err

    def test_a_withheld_block_is_not_judged_by_the_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The report runs over the admitted set, like every other read path."""
        _corpus(tmp_path, ATTRIBUTED_BLOCK + "\n" + UNATTRIBUTED_BLOCK.replace("Status: active", "Status: quarantined"))
        ws = _workspace(tmp_path, provenance={"enabled": True, "policy": POLICY_REQUIRED})
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        assert mm_cli.main(["compliance", "provenance", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["withheld_count"] == 1
        assert report["blocks_missing_provenance"] == 0

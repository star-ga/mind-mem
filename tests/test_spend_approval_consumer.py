"""Controls for the approval parser and the RunPod launch boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from train import runpod_deploy, spend_guard


def _digest(*, tag: str = "launch-1") -> str:
    return runpod_deploy._launch_config_digest(
        gpu_type=runpod_deploy.DEFAULT_GPU_TYPE,
        image=runpod_deploy.DEFAULT_IMAGE,
        version_tag=tag,
        skip_upload=False,
    )


def _marker(path: Path, *, tag: str = "launch-1", budget: str = "5", config: str | None = None) -> None:
    lines = [f"tag: {tag}", f"budget_usd: {budget}"]
    if config is not None:
        lines.append(f"config_sha256: {config}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _argv(approval: Path, budget: str = "5") -> list[str]:
    return [
        "runpod_deploy.py",
        "--approval-file",
        str(approval),
        "--budget-usd",
        budget,
        "--version-tag",
        "launch-1",
    ]


def test_parser_requires_exact_fields_and_rejects_prefix_or_comment_attacks(tmp_path: Path) -> None:
    cases = (
        "tag: launch-1-extra\nbudget_usd: 5\nconfig_sha256: " + "a" * 64 + "\n",
        "tag: launch-1 # launch-1\nbudget_usd: 5\nconfig_sha256: " + "a" * 64 + "\n",
        "tag: launch-1\nbudget_usd: 50\nconfig_sha256: " + "a" * 64 + "\n",
        "tag: launch-1\nbudget_usd: 5\nbudget_usd: 5\nconfig_sha256: " + "a" * 64 + "\n",
    )
    for index, content in enumerate(cases):
        marker = tmp_path / f"approval-{index}.yml"
        marker.write_text(content, encoding="utf-8")
        with pytest.raises(spend_guard.ApprovalError):
            spend_guard.validate_approval(marker, expected_tag="launch-1", expected_budget_usd=5.0)


@pytest.mark.parametrize("missing", ["tag", "budget_usd"])
def test_parser_refuses_missing_required_field_with_config(tmp_path: Path, missing: str) -> None:
    fields = {"tag": "launch-1", "budget_usd": "5", "config_sha256": "a" * 64}
    del fields[missing]
    marker = tmp_path / "approval.yml"
    marker.write_text("".join(f"{key}: {value}\n" for key, value in fields.items()), encoding="utf-8")
    with pytest.raises(spend_guard.ApprovalError, match="exactly tag and budget_usd"):
        spend_guard.parse_approval_file(marker)


def test_spend_guard_preflight_uses_same_parser_and_records_marker_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    approval = tmp_path / "approval.yml"
    _marker(approval, config="a" * 64)
    ledger = tmp_path / "ledger.jsonl"
    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.setattr(spend_guard, "LEDGER", ledger)
    monkeypatch.setattr(spend_guard, "WEIGHT_ROOT", weights)
    spend_guard.preflight(
        SimpleNamespace(
            tag="launch-1",
            budget_usd=5.0,
            approval_file=str(approval),
            prev_run_tag=None,
            config_sha256="a" * 64,
        )
    )
    entry = ledger.read_text(encoding="utf-8").strip()
    assert spend_guard.parse_approval_file(approval).marker_sha256 in entry
    assert '"config_sha256": "' + "a" * 64 + '"' in entry


def test_runpod_refuses_invalid_approval_before_provision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_BASE_MODEL", "Qwen/Qwen3.8-4B")
    approval = tmp_path / "approval.yml"
    _marker(approval, tag="launch-1", budget="50", config="a" * 64)
    created: list[tuple[str, str]] = []
    monkeypatch.setattr(runpod_deploy, "provision", lambda **kwargs: created.append(("create", "called")) or "pod")
    monkeypatch.setattr(sys, "argv", _argv(approval))

    with pytest.raises(SystemExit, match="SPEND-GUARD REFUSED"):
        runpod_deploy.main()
    assert created == []


def test_runpod_refuses_missing_approval_before_provision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_BASE_MODEL", "Qwen/Qwen3.8-4B")
    created: list[str] = []
    monkeypatch.setattr(runpod_deploy, "provision", lambda **kwargs: created.append("create") or "pod")
    monkeypatch.setattr(sys, "argv", _argv(tmp_path / "missing.yml"))

    with pytest.raises(SystemExit, match="approval file cannot be read"):
        runpod_deploy.main()
    assert created == []


@pytest.mark.parametrize("approved,requested", [("100000000000000000000", "100000000000000000001"), ("5", "5.0000000000000001")])
def test_runpod_preserves_decimal_budget_before_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved: str, requested: str
) -> None:
    monkeypatch.setenv("MM_BASE_MODEL", "offline-test-base")
    marker = tmp_path / "approval.yml"
    _marker(marker, budget=approved, config=_digest())
    monkeypatch.setattr(sys, "argv", _argv(marker, requested))
    monkeypatch.setattr(runpod_deploy, "provision", lambda **_kwargs: pytest.fail("must not provision"))
    with pytest.raises(SystemExit, match="does not match requested"):
        runpod_deploy.main()


def test_preflight_cli_preserves_exact_decimal_in_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "approval.yml"
    budget = "5.0000000000000001"
    _marker(marker, budget=budget, config="a" * 64)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(spend_guard, "LEDGER", ledger)
    monkeypatch.setattr(spend_guard, "WEIGHT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "spend_guard.py", "preflight", "--tag", "launch-1", "--budget-usd", budget,
        "--approval-file", str(marker), "--config-sha256", "a" * 64,
    ])
    spend_guard.main()
    assert json.loads(ledger.read_text())["budget_usd"] == budget


def test_preflight_cli_refuses_missing_expected_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "approval.yml"
    _marker(marker)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(spend_guard, "LEDGER", ledger)
    monkeypatch.setattr(sys, "argv", [
        "spend_guard.py", "preflight", "--tag", "launch-1", "--budget-usd", "5",
        "--approval-file", str(marker),
    ])
    with pytest.raises(SystemExit) as exc:
        spend_guard.main()
    assert exc.value.code == 2
    assert not ledger.exists()


def test_runpod_refuses_unbound_marker_before_provision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_BASE_MODEL", "Qwen/Qwen3.8-4B")
    approval = tmp_path / "approval.yml"
    _marker(approval)
    created: list[str] = []
    monkeypatch.setattr(runpod_deploy, "provision", lambda **kwargs: created.append("create") or "pod")
    monkeypatch.setattr(sys, "argv", _argv(approval))

    with pytest.raises(SystemExit, match="SPEND-GUARD REFUSED"):
        runpod_deploy.main()
    assert created == []


def test_runpod_refuses_wrong_config_before_provision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_BASE_MODEL", "Qwen/Qwen3.8-4B")
    approval = tmp_path / "approval.yml"
    _marker(approval, config="b" * 64)
    created: list[str] = []
    monkeypatch.setattr(runpod_deploy, "provision", lambda **kwargs: created.append("create") or "pod")
    monkeypatch.setattr(sys, "argv", _argv(approval))

    with pytest.raises(SystemExit, match="SPEND-GUARD REFUSED"):
        runpod_deploy.main()
    assert created == []


def test_runpod_requires_explicit_base_before_provision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MM_BASE_MODEL", raising=False)
    approval = tmp_path / "approval.yml"
    _marker(approval, config="a" * 64)
    created: list[str] = []
    monkeypatch.setattr(runpod_deploy, "provision", lambda **kwargs: created.append("create") or "pod")
    monkeypatch.setattr(sys, "argv", _argv(approval))

    with pytest.raises(SystemExit, match="MM_BASE_MODEL is required"):
        runpod_deploy.main()
    assert created == []


def test_runpod_requires_config_bound_marker_and_accepts_exact_marker_before_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MM_BASE_MODEL", "Qwen/Qwen3.8-4B")
    monkeypatch.delenv("MM_RUNPOD_CLOUD", raising=False)
    approval = tmp_path / "approval.yml"
    config = _digest()
    _marker(approval, config=config)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    token = tmp_path / "token"
    token.write_text("secret-not-printed", encoding="utf-8")
    ssh_key = tmp_path / "id_ed25519"
    ssh_key.write_text("private", encoding="utf-8")
    (tmp_path / "id_ed25519.pub").write_text("public", encoding="utf-8")
    created: list[str] = []
    monkeypatch.setattr(runpod_deploy, "CORPUS", corpus)
    monkeypatch.setattr(runpod_deploy, "HF_TOKEN_FILE", token)
    monkeypatch.setattr(runpod_deploy, "SSH_KEY", ssh_key)
    monkeypatch.setattr(runpod_deploy, "provision", lambda **kwargs: created.append("create") or "pod")
    monkeypatch.setattr(runpod_deploy, "wait_ssh", lambda pod: ("127.0.0.1", 22))
    monkeypatch.setattr(sys, "argv", _argv(approval) + ["--provision-only"])

    runpod_deploy.main()
    assert created == ["create"]


def test_print_config_is_side_effect_free(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("MM_BASE_MODEL", "Qwen/Qwen3.8-4B")
    created: list[str] = []
    monkeypatch.setattr(runpod_deploy, "provision", lambda **kwargs: created.append("create") or "pod")
    monkeypatch.setattr(sys, "argv", ["runpod_deploy.py", "--print-approval-config"])

    runpod_deploy.main()
    assert created == []
    assert "config_sha256" in capsys.readouterr().out

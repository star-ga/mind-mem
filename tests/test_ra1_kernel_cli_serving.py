"""Acceptance controls for the primary ``mm recall --kernel`` serving door.

The kernel API remains a ``KernelResult``.  This file exercises the actual CLI
door, which owns the existing recall attestation and served-ledger attachment.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import date
from pathlib import Path

import pytest
from test_v4_kernels_wiring import _build_workspace, _write_config

from mind_mem import mm_cli
from mind_mem.served_ledger import read_served_runs, row_hash
from mind_mem.v4.cognitive_kernel import mind_recall


def _args() -> Namespace:
    return Namespace(query="PostgreSQL", kernel="default", limit=10)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    _build_workspace(root)
    _write_config(root, kernels=True)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(root / "mind-mem.json"))
    return root


def _run(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    assert mm_cli._cmd_kernel_recall(_args()) == 0
    return json.loads(capsys.readouterr().out)


def test_kernel_cli_binds_engine_instant_and_records_exact_output(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The row is the final CLI output, with one captured instant end-to-end."""
    import importlib

    from mind_mem.v4 import cognitive_kernel

    _recall_core = importlib.import_module("mind_mem._recall_core")

    seen: list[date | None] = []
    real_recall = _recall_core.recall

    def observe(*args: object, **kwargs: object) -> list[dict[str, object]]:
        value = kwargs.get("scoring_instant")
        seen.append(value if isinstance(value, date) else None)
        return real_recall(*args, **kwargs)

    from mind_mem import scoring_instant

    monkeypatch.setattr(scoring_instant, "resolve_scoring_instant", lambda _value: date(2026, 9, 14))
    monkeypatch.setattr(_recall_core, "recall", observe)
    with cognitive_kernel._registry_lock:
        previous = cognitive_kernel._registry.get(cognitive_kernel.KernelKind.DEFAULT)
        cognitive_kernel._registry[cognitive_kernel.KernelKind.DEFAULT] = cognitive_kernel._default_kernel
    try:
        payload = _run(capsys)
    finally:
        with cognitive_kernel._registry_lock:
            if previous is None:
                cognitive_kernel._registry.pop(cognitive_kernel.KernelKind.DEFAULT, None)
            else:
                cognitive_kernel._registry[cognitive_kernel.KernelKind.DEFAULT] = previous

    assert seen == [date(2026, 9, 14)], "the kernel ranked without the CLI's captured instant"
    assert payload["count"] == 1
    hits = payload["hits"]
    assert isinstance(hits, list) and hits[0]["block_id"] == "D-20260101-001"
    attestation = payload["attestation"]
    assert isinstance(attestation, dict)
    assert attestation["served_proof"] == "recorded"
    assert attestation["scoring_instant"] == "2026-09-14"

    rows = read_served_runs(str(workspace))
    assert len(rows) == 1
    row = rows[0]
    assert row.ids == tuple(hit["block_id"] for hit in hits)
    assert row.served_digest == attestation["results_digest"]
    assert attestation["served_row_hash"] == row_hash(row)
    assert attestation["served_seq"] == row.seq


def test_kernel_cli_adds_receipt_without_changing_kernel_hits(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The additive receipt does not change the public kernel result."""
    fixed = date(2026, 9, 14)
    from mind_mem import scoring_instant

    monkeypatch.setattr(scoring_instant, "resolve_scoring_instant", lambda _value: fixed)
    expected = mind_recall(str(workspace), "PostgreSQL", kernel="default", scoring_instant=fixed)
    payload = _run(capsys)
    got = payload["hits"]
    assert isinstance(got, list)
    assert [(h["block_id"], h["score"]) for h in got] == [(h.block_id, h.score) for h in expected.hits]
    assert isinstance(payload["attestation"], dict)


def test_kernel_cli_reports_recorder_failure_while_serving_answer(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A downstream ledger failure is explicit and cannot erase the answer."""
    import mind_mem.served_ledger as ledger

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("fixture recorder failure")

    monkeypatch.setattr(ledger, "append_served_run", fail)
    payload = _run(capsys)
    assert payload["count"] == 1
    attestation = payload["attestation"]
    assert isinstance(attestation, dict)
    assert attestation["served_proof"] == "unproven"
    assert "fixture recorder failure" in attestation["ledger_error"]
    assert read_served_runs(str(workspace)) == ()


def test_kernel_cli_disabled_ledger_is_explicitly_unproven(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = json.loads((workspace / "mind-mem.json").read_text(encoding="utf-8"))
    config["served_ledger"] = {"enabled": False}
    (workspace / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    payload = _run(capsys)
    assert payload["count"] == 1
    attestation = payload["attestation"]
    assert isinstance(attestation, dict)
    assert attestation["served_proof"] == "unproven"
    assert "disabled" in attestation["ledger_error"]
    assert read_served_runs(str(workspace)) == ()


def test_kernel_cli_feature_flag_off_keeps_operator_refusal(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = json.loads((workspace / "mind-mem.json").read_text(encoding="utf-8"))
    config["v4"]["cognitive_kernel"] = {"enabled": False}
    (workspace / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    assert mm_cli._cmd_kernel_recall(_args()) == 64
    assert "disabled" in capsys.readouterr().err.lower()

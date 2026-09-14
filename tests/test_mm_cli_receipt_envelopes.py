# Copyright 2026 STARGA, Inc.
"""Opt-in CLI receipt envelopes preserve the existing ranked evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mind_mem.init_workspace import init
from mind_mem.recall import ServedResults
from mind_mem.served_ledger import read_served_runs, row_hash

REPO = Path(__file__).resolve().parents[1]


def _workspace(root: Path, *, count: int = 1, long: bool = False) -> Path:
    root.mkdir()
    init(str(root))
    lines: list[str] = []
    for number in range(count):
        statement = "receipt envelope compiler evidence"
        if long:
            statement += " " + ("bounded context projection " * 8)
        lines.append(f"[D-CLI-{number:03d}]\nDate: 2026-09-14\nStatus: active\nStatement: {statement}\nTags: receipt, compiler\n\n")
    (root / "decisions" / "DECISIONS.md").write_text("".join(lines), encoding="utf-8", newline="\n")
    return root


def _run_cli(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "MIND_MEM_WORKSPACE": str(workspace),
        "PYTHONPATH": str(REPO / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "mind_mem.mm_cli", *args],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def _payload(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), result.stderr
    return json.loads(result.stdout)


def test_recall_envelope_carries_the_existing_one_row_receipt(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "recall")
    result = _run_cli(workspace, "recall", "receipt envelope", "--limit", "3", "--receipt-envelope")
    payload = _payload(result)

    assert payload["schema"] == "mind-mem/cli-receipt-envelope@1"
    assert payload["surface"] == "recall"
    assert payload["projection"] == {
        "kind": "ranked_recall",
        "final_included_ids": ["D-CLI-000"],
        "dropped_ids": [],
        "rendered_text_attested": False,
    }
    evidence = payload["ranked_recall_evidence"]
    assert evidence["status"] == "recorded"
    attestation = evidence["attestation"]
    assert attestation["served_proof"] == "recorded"

    rows = read_served_runs(str(workspace))
    assert len(rows) == 1
    row = rows[0]
    assert tuple(payload["projection"]["final_included_ids"]) == row.ids
    assert attestation["results_digest"] == row.served_digest
    assert attestation["served_row_hash"] == row_hash(row)
    assert attestation["served_seq"] == row.seq


def test_recall_default_stdout_remains_a_plain_json_list(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "default")
    result = _run_cli(workspace, "recall", "receipt envelope", "--limit", "3")
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert isinstance(value, list)
    assert value[0]["_id"] == "D-CLI-000"
    assert "ranked_recall_evidence" not in value


def test_context_envelope_separates_ranked_receipt_from_pack_projection(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "context", count=3, long=True)
    result = _run_cli(workspace, "context", "receipt envelope", "--limit", "3", "--max-tokens", "100", "--receipt-envelope")
    payload = _payload(result)

    packed = payload["result"]
    included_ids = [item["_id"] for item in packed["included"]]
    dropped_ids = [item["_id"] for item in packed["dropped"]]
    assert included_ids, "fixture must exercise a nonempty packed result"
    assert dropped_ids, "fixture must exercise the dropped side of the pack"
    projection = payload["projection"]
    assert projection["kind"] == "context_pack"
    assert projection["final_included_ids"] == included_ids
    assert projection["dropped_ids"] == dropped_ids
    assert projection["rendered_text_attested"] is False

    evidence = payload["ranked_recall_evidence"]
    assert evidence["status"] == "recorded"
    rows = read_served_runs(str(workspace))
    assert len(rows) == 1
    assert tuple(included_ids + dropped_ids) == rows[0].ids
    assert tuple(included_ids) != rows[0].ids, "pack projection must not be confused with ranked receipt scope"


def test_inject_envelope_marks_rendered_text_as_unsealed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "inject")
    result = _run_cli(workspace, "inject", "receipt envelope", "--agent", "generic", "--receipt-envelope")
    payload = _payload(result)

    assert payload["surface"] == "inject"
    assert payload["result"]["rendered_text"]
    assert payload["projection"]["final_included_ids"] == ["D-CLI-000"]
    assert payload["projection"]["rendered_text_attested"] is False
    assert "rendered_text_digest" not in payload["projection"]
    assert payload["ranked_recall_evidence"]["status"] == "recorded"


@pytest.mark.parametrize("surface", ["recall", "context", "inject"])
def test_missing_receipt_is_explicitly_unproven(surface: str) -> None:
    from mind_mem import mm_cli

    results = ServedResults([{"_id": "D-CLI-MISSING", "score": 1.0}])
    results.attestation = None
    if surface == "recall":
        response: object = list(results)
    elif surface == "context":
        response = {"included": list(results), "dropped": []}
    else:
        response = {"rendered_text": "[framed data]"}
    envelope = mm_cli._receipt_envelope(
        surface,
        "missing receipt",
        results,
        response,
        final_included_ids=["D-CLI-MISSING"],
        projection_kind=surface,
    )
    evidence = envelope["ranked_recall_evidence"]
    assert evidence["status"] == "unproven"
    assert evidence["attestation"] is None
    assert evidence["reason"] == "serving receipt unavailable"


def test_degraded_receipt_is_explicitly_unproven() -> None:
    from mind_mem import mm_cli

    results = ServedResults([{"_id": "D-CLI-DEGRADED", "score": 1.0}])
    results.attestation = {"served_proof": "recorded", "results_digest": "digest"}
    results.degraded = {"leg": "vector", "reason": "deadline_exceeded"}
    envelope = mm_cli._receipt_envelope(
        "recall",
        "degraded receipt",
        results,
        list(results),
        final_included_ids=["D-CLI-DEGRADED"],
        projection_kind="ranked_recall",
    )
    evidence = envelope["ranked_recall_evidence"]
    assert evidence["status"] == "unproven"
    assert evidence["attestation"]["served_proof"] == "recorded"
    assert evidence["reason"] == "deadline_exceeded"


def test_receipt_envelope_is_advertised_and_unknown_flag_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "help")
    help_result = _run_cli(workspace, "context", "--help")
    assert help_result.returncode == 0
    assert "--receipt-envelope" in help_result.stdout

    bad_result = _run_cli(workspace, "context", "query", "--receipt-envelope-extra")
    assert bad_result.returncode == 2
    assert "--receipt-envelope" in bad_result.stderr


def test_kernel_and_ordinary_receipt_envelope_are_explicitly_exclusive(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "exclusive")
    result = _run_cli(workspace, "recall", "receipt envelope", "--kernel", "default", "--receipt-envelope")
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
    assert result.stdout == ""
    assert read_served_runs(workspace) == ()

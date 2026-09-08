# Copyright 2026 STARGA, Inc.
"""The Bandit gate, executed against every way a scan can go wrong.

History, because each repair uncovered the next hole:

  1. `bandit -r src ... || true` -- success for findings, for crashes, and for
     not running at all.
  2. capture-and-re-raise -- the exit code was written to the environment and
     never read, and `set -uo pipefail` does not clear an inherited errexit,
     so under `bash -e` the failing command ended the step before the capture.
  3. the parser read `runs: []` as "0 high", so a stale or truncated report
     from a crashed scanner passed.

All three are gone, and this file is why the third cannot come back quietly:
the decision lives in a script that can be run against stubs rather than in
YAML that only ever executes in CI.

Severity is read from the field Bandit actually emits. Measured against the
installed tool rather than assumed -- `properties.issue_severity` is "HIGH",
`level` is "error", and `rule.defaultConfiguration.level` is None, which is
what the first version of the parser tried to use.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_GATE = _REPO / "scripts" / "bandit_gate.py"

#: The shape the installed Bandit really emits, measured:
#:     tool.driver.name / version, an explicit results list, and
#:     invocations[0].executionSuccessful.
#: The first version of these fixtures had none of it -- they were exactly the
#: structurally-empty reports that turned out to be a false green, so they
#: could never have caught it.
_DRIVER = {"name": "Bandit", "version": "1.9.4", "semanticVersion": "1.9.4"}
_INVOCATIONS = [{"executionSuccessful": True, "endTimeUtc": "2026-09-08T14:36:44Z"}]


def _sarif(results: list[dict]) -> dict:
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{"tool": {"driver": dict(_DRIVER)}, "invocations": [dict(i) for i in _INVOCATIONS], "results": results}],
    }


_LOW = {"ruleId": "B404", "level": "note", "properties": {"issue_severity": "LOW"}, "message": {"text": "import subprocess"}}
_HIGH = {
    "ruleId": "B602",
    "level": "error",
    "properties": {"issue_severity": "HIGH"},
    "message": {"text": "subprocess with shell=True"},
}

CLEAN = _sarif([])
LOW_ONLY = _sarif([_LOW])
HIGH = _sarif([_LOW, _HIGH])
NO_RUNS = {"runs": []}


def _run(tmp_path: pathlib.Path, sarif, exit_code: int) -> subprocess.CompletedProcess:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "bandit.sarif"
    if sarif is not None:
        path.write_text(sarif if isinstance(sarif, str) else json.dumps(sarif), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(_GATE), "--sarif", str(path), "--exit-code", str(exit_code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_clean_scan_passes(tmp_path) -> None:
    r = _run(tmp_path, CLEAN, 0)
    assert r.returncode == 0, r.stderr
    assert "0 high-severity" in r.stdout


def test_low_only_findings_pass_and_are_reported(tmp_path) -> None:
    """Low severity reaches code scanning without blocking the build."""
    r = _run(tmp_path, LOW_ONLY, 1)
    assert r.returncode == 0, r.stderr
    assert "1 finding" in r.stdout and "0 high-severity" in r.stdout


def test_a_high_finding_fails(tmp_path) -> None:
    r = _run(tmp_path, HIGH, 1)
    assert r.returncode == 1
    assert "B602" in r.stdout
    assert "1 high-severity" in r.stdout


def test_a_crashed_scanner_with_a_STALE_nonempty_report_fails(tmp_path) -> None:
    """The hole root found: a plausible report from a scan that never ran.

    The report here is a perfectly valid CLEAN sarif. Only the exit code says
    the scanner died, so a gate that read the report alone would pass.
    """
    r = _run(tmp_path, CLEAN, 127)
    assert r.returncode == 2, r.stdout
    assert "did not complete" in r.stderr
    assert "clean scan" in r.stderr


def test_an_empty_runs_list_is_not_zero_findings(tmp_path) -> None:
    """`runs: []` is what a truncated or aborted scan leaves, not a clean one."""
    r = _run(tmp_path, NO_RUNS, 0)
    assert r.returncode == 2, r.stdout
    assert "no runs" in r.stderr


def test_a_malformed_report_fails(tmp_path) -> None:
    r = _run(tmp_path, "{not json", 0)
    assert r.returncode == 2
    assert "not readable JSON" in r.stderr


def test_a_missing_report_fails(tmp_path) -> None:
    r = _run(tmp_path, None, 0)
    assert r.returncode == 2
    assert "did not produce a report" in r.stderr


def test_an_empty_report_file_fails(tmp_path) -> None:
    r = _run(tmp_path, "", 0)
    assert r.returncode == 2
    assert "empty" in r.stderr


def test_the_three_outcomes_are_distinguishable(tmp_path) -> None:
    """Pass, findings and broken must never share an exit code.

    Collapsing any two is how the original `|| true` behaved, one level up.
    """
    codes = {}
    for name, sarif, exit_code in (("clean", CLEAN, 0), ("high", HIGH, 1), ("broken", CLEAN, 127)):
        codes[name] = _run(tmp_path / name, sarif, exit_code).returncode
    assert len(set(codes.values())) == 3, f"outcomes are not distinguishable: {codes}"


def test_the_workflow_reads_the_code_it_captured() -> None:
    """The second repair's defect: bandit_exit written, never read."""
    wf = (_REPO / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
    assert "bandit_exit=$rc" in wf, "the exit code is no longer persisted"
    assert "env.bandit_exit" in wf, "the persisted exit code is never read again"
    recipe = [ln for ln in wf.splitlines() if "bandit -r src" in ln and not ln.lstrip().startswith("#")]
    assert recipe, "the bandit job no longer scans src"
    for ln in recipe:
        assert "|| true" not in ln, f"the exit code is masked again: {ln.strip()}"
    assert "if bandit -r src" in wf, "capture is not errexit-proof: use an explicit if/else, not a bare command"


@pytest.mark.parametrize("field", ["issue_severity", "level"])
def test_severity_is_read_from_a_field_bandit_really_emits(tmp_path, field) -> None:
    """Verified against the installed tool, not assumed.

    Measured: a HIGH finding carries properties.issue_severity == "HIGH" AND
    level == "error"; rule.defaultConfiguration.level is None, which is what
    the first parser tried to read.
    """
    result = {"ruleId": "B602", "message": {"text": "x"}}
    if field == "issue_severity":
        result["properties"] = {"issue_severity": "HIGH"}
    else:
        result["level"] = "error"
    r = _run(tmp_path, _sarif([result]), 1)
    assert r.returncode == 1, f"a HIGH finding expressed via {field} was not caught: {r.stdout} {r.stderr}"


# ---------------------------------------------------------------------------
# Shaped-but-empty reports. Root demonstrated both against the first version.
# ---------------------------------------------------------------------------

def _real(results: list[dict]) -> dict:
    """The same real shape the fixtures above use."""
    return _sarif(results)


def test_a_structurally_empty_run_is_not_a_clean_scan(tmp_path) -> None:
    """`{"runs": [{}]}` with exit 0 passed. It carries no evidence of anything.

    No tool identity, no results list, no invocation. Reading it as "zero
    findings" fabricates execution evidence out of an absence.
    """
    r = _run(tmp_path, {"runs": [{}]}, 0)
    assert r.returncode == 2, r.stdout
    assert "no tool identity" in r.stderr or "names no tool" in r.stderr


def test_findings_exit_with_an_empty_report_is_incoherent(tmp_path) -> None:
    """`{"runs":[{"results":[]}]}` with exit 1 passed. Bandit exits 1 only on a finding."""
    r = _run(tmp_path, {"runs": [{"results": []}]}, 1)
    assert r.returncode == 2, r.stdout


def test_a_report_from_another_tool_is_refused(tmp_path) -> None:
    doc = _real([])
    doc["runs"][0]["tool"] = {"driver": {"name": "SomeOtherScanner", "version": "1.0"}}
    r = _run(tmp_path, doc, 0)
    assert r.returncode == 2
    assert "not Bandit" in r.stderr


@pytest.mark.parametrize("mutation", ["no_results_key", "null_results", "no_invocations", "unsuccessful"])
def test_each_missing_piece_of_completed_scan_evidence_is_refused(tmp_path, mutation) -> None:
    doc = _real([])
    run = doc["runs"][0]
    if mutation == "no_results_key":
        del run["results"]
    elif mutation == "null_results":
        run["results"] = None
    elif mutation == "no_invocations":
        del run["invocations"]
    else:
        run["invocations"] = [{"executionSuccessful": False}]
    r = _run(tmp_path / mutation, doc, 0)
    assert r.returncode == 2, f"{mutation} was accepted: {r.stdout}"


def test_a_real_shaped_clean_report_still_passes(tmp_path) -> None:
    """The whole point of the shape checks is to keep honest reports green."""
    assert _run(tmp_path, _real([]), 0).returncode == 0


def test_a_real_shaped_low_only_report_still_passes(tmp_path) -> None:
    low = [{"ruleId": "B404", "level": "note", "properties": {"issue_severity": "LOW"}, "message": {"text": "x"}}]
    assert _run(tmp_path, _real(low), 1).returncode == 0

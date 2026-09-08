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

CLEAN = {"runs": [{"tool": {"driver": {"name": "Bandit"}}, "results": []}]}
LOW_ONLY = {
    "runs": [
        {
            "tool": {"driver": {"name": "Bandit"}},
            "results": [
                {"ruleId": "B404", "level": "note", "properties": {"issue_severity": "LOW"}, "message": {"text": "import subprocess"}}
            ],
        }
    ]
}
HIGH = {
    "runs": [
        {
            "tool": {"driver": {"name": "Bandit"}},
            "results": [
                {"ruleId": "B404", "level": "note", "properties": {"issue_severity": "LOW"}, "message": {"text": "import subprocess"}},
                {
                    "ruleId": "B602",
                    "level": "error",
                    "properties": {"issue_severity": "HIGH"},
                    "message": {"text": "subprocess with shell=True"},
                },
            ],
        }
    ]
}
NO_RUNS = {"runs": []}


def _run(tmp_path: pathlib.Path, sarif, exit_code: int) -> subprocess.CompletedProcess:
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
        d = tmp_path / name
        d.mkdir()
        codes[name] = _run(d, sarif, exit_code).returncode
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
    r = _run(tmp_path, {"runs": [{"tool": {"driver": {}}, "results": [result]}]}, 1)
    assert r.returncode == 1, f"a HIGH finding expressed via {field} was not caught: {r.stdout} {r.stderr}"

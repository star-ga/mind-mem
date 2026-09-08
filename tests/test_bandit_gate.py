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

Severity is read from the field Bandit actually emits. Measured by RUNNING
`bandit[sarif]` 1.9.4 over fixture files rather than assumed, and the run
corrected this file's own earlier assumption:

    HIGH    level "error"   issue_severity "HIGH"
    LOW     level "note"    issue_severity "LOW"
    MEDIUM  NO level key    issue_severity "MEDIUM"   <- not what this file said
    rule.defaultConfiguration = None, which the first parser tried to use

The MEDIUM row is the reason `issue_severity` is the authority: `level` is not
always emitted, so anything reading it first mis-classifies a real finding.
The version is the provenance of that measurement, not a pin.

A fourth hole, found by root after the third was fixed: ADMISSION was checking
shapes it could skip. `executionSuccessful: "false"` is a truthy string that
`bool()` accepted; a `null` second invocation was filtered out by an
`isinstance` guard so the good first one answered for the list; and a result
with no severity at all was counted as one non-HIGH finding. All three passed.
Absence is not evidence, and a record you cannot read is not one you may skip.
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
#: MEDIUM carries NO `level` key. Copied from real emitted output, not written
#: from the pattern the other two follow -- that pattern is exactly what the
#: measurement disproved.
_MEDIUM = {
    "ruleId": "B108",
    "properties": {"issue_severity": "MEDIUM", "issue_confidence": "MEDIUM"},
    "message": {"text": "hardcoded temp directory"},
}
_HIGH = {
    "ruleId": "B602",
    "level": "error",
    "properties": {"issue_severity": "HIGH"},
    "message": {"text": "subprocess with shell=True"},
}

CLEAN = _sarif([])
LOW_ONLY = _sarif([_LOW])
MEDIUM_ONLY = _sarif([_MEDIUM])
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
        encoding="utf-8",
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


def test_severity_is_read_from_the_field_bandit_really_emits(tmp_path) -> None:
    """`issue_severity` is the authority. Verified by running the tool.

    Measured on real output: every result carries
    `properties.issue_severity`, HIGH additionally carries `level: "error"`,
    and `rule.defaultConfiguration` is None -- which is what the first parser
    tried to read.
    """
    result = {"ruleId": "B602", "properties": {"issue_severity": "HIGH"}, "message": {"text": "x"}}
    r = _run(tmp_path, _sarif([result]), 1)
    assert r.returncode == 1, f"a HIGH finding was not caught: {r.stdout} {r.stderr}"


def test_a_level_without_a_severity_is_refused_rather_than_inferred(tmp_path) -> None:
    """DELIBERATELY STRICTER than the previous rule, and this records the change.

    The gate used to infer HIGH from `level: "error"` when `issue_severity`
    was missing. That fallback is gone: real Bandit emits `issue_severity` on
    EVERY result, measured across clean, LOW, MEDIUM and HIGH runs, so a
    result without it is not output this gate can classify. It is refused as
    unusable (exit 2) rather than guessed at in either direction.

    Exit 2 fails the step exactly as exit 1 does, so this is a tightening, not
    a hole: the report that used to be read as one HIGH finding is now read as
    a report that cannot be trusted to describe the run at all.
    """
    result = {"ruleId": "B602", "level": "error", "message": {"text": "x"}}
    r = _run(tmp_path, _sarif([result]), 1)
    assert r.returncode == 2, f"a severity-less result was classified rather than refused: {r.stdout} {r.stderr}"
    assert "issue_severity" in r.stderr


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


# ---------------------------------------------------------------------------
# Admission: a record you cannot read is not a record you may skip
#
# Every fixture below was demonstrated PASSING (exit 0) before this round.
# ---------------------------------------------------------------------------


def _run_with(tmp_path: pathlib.Path, run: dict, exit_code: int) -> subprocess.CompletedProcess:
    return _run(tmp_path, {"version": "2.1.0", "runs": [run]}, exit_code)


def _ok_run(**over) -> dict:
    run = {"tool": {"driver": dict(_DRIVER)}, "invocations": [dict(i) for i in _INVOCATIONS], "results": []}
    run.update(over)
    return run


def test_a_truthy_string_is_not_a_successful_execution(tmp_path) -> None:
    """`executionSuccessful: "false"` passed, because `bool("false")` is True.

    The tool emits a JSON boolean. Anything else is a report this gate cannot
    read, and reading a string as success is how a failed scan reports clean.
    """
    r = _run_with(tmp_path, _ok_run(invocations=[{"executionSuccessful": "false"}]), 0)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "not the boolean true" in r.stderr


@pytest.mark.parametrize("value", ["true", 1, "1", [], {}, None, 0])
def test_only_the_boolean_true_admits_an_invocation(tmp_path, value) -> None:
    """Identity, not truthiness -- and not falsiness either.

    `1` and `"true"` are truthy and are still not what the tool writes; a
    report using them is unusable rather than successful.
    """
    r = _run_with(tmp_path, _ok_run(invocations=[{"executionSuccessful": value}]), 0)
    assert r.returncode == 2, f"{value!r} was admitted: {r.stdout}{r.stderr}"


def test_a_malformed_invocation_cannot_be_skipped(tmp_path) -> None:
    """A good first entry answered for a null second one.

    The old guard filtered non-dict entries out of the list before checking,
    so the report that could NOT be read was the one that never got checked.
    """
    r = _run_with(tmp_path, _ok_run(invocations=[{"executionSuccessful": True}, None]), 0)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "not an object" in r.stderr

    # Positive control: the same report WITHOUT the malformed entry passes, so
    # the refusal is caused by that entry and not by the fixture in general.
    ok = _run_with(tmp_path, _ok_run(invocations=[{"executionSuccessful": True}]), 0)
    assert ok.returncode == 0, ok.stdout + ok.stderr


def test_a_result_with_no_severity_is_unusable_not_benign(tmp_path) -> None:
    """`results: [{}]` with exit 1 was reported as one non-HIGH finding.

    With no properties and no level there was nothing to compare, so the
    result fell through every branch and was counted as harmless. Absence of a
    severity is absence of evidence, not evidence of a low severity.
    """
    r = _run_with(tmp_path, _ok_run(results=[{}]), 1)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "issue_severity" in r.stderr


@pytest.mark.parametrize(
    "props",
    [
        None,
        {},
        {"issue_severity": None},
        {"issue_severity": ""},
        {"issue_severity": "CRITICAL"},
        {"issue_severity": 3},
        {"issue_severity": ["HIGH"]},
    ],
    ids=["absent", "empty", "null", "blank", "unknown", "int", "list"],
)
def test_an_unreadable_severity_never_becomes_low_by_omission(tmp_path, props) -> None:
    res = {"ruleId": "B000", "message": {"text": "x"}}
    if props is not None:
        res["properties"] = props
    r = _run_with(tmp_path, _ok_run(results=[res]), 1)
    assert r.returncode == 2, f"{props!r} was classified rather than refused: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("claimed", ["LOW", "MEDIUM"])
def test_an_error_level_finding_cannot_claim_a_milder_severity(tmp_path, claimed) -> None:
    """The downgrade direction is the dangerous one.

    A result carrying `level: "error"` while claiming a milder severity
    contradicts the tool's own mapping. Neither field is trusted over the
    other: the disagreement itself makes the report unusable.

    MEDIUM is the case that needs its own guard, and the mutation check is how
    that was established. LOW is caught by the severity->level map (LOW must be
    "note"), but MEDIUM emits no level at all, so the map has no entry and only
    the explicit "error implies HIGH" rule catches it. Testing LOW alone left
    that rule looking decorative.
    """
    liar = {"ruleId": "B602", "level": "error", "properties": {"issue_severity": claimed}, "message": {"text": "x"}}
    r = _run_with(tmp_path, _ok_run(results=[liar]), 1)
    assert r.returncode == 2, r.stdout + r.stderr


@pytest.mark.parametrize(
    "severity,level",
    [("LOW", "warning"), ("HIGH", "note")],
    ids=["low-with-a-nonsense-level", "high-claiming-a-mild-level"],
)
def test_a_level_that_disagrees_with_its_severity_is_unusable(tmp_path, severity, level) -> None:
    """The severity->level map, and the two cases that make it load-bearing.

    Both were found by mutation, not by reading: with the map check removed the
    rest of the file stayed green, because the separate "error implies HIGH"
    rule covers only levels that are literally "error".

      * LOW with level "warning" -- no rule fires, and the result is admitted
        as a benign finding. A gate that passes on a level it has never seen
        is guessing.
      * HIGH with level "note" -- the finding is still counted HIGH, so the
        step fails, but for the wrong reason and with the wrong message. A
        report whose two severity fields disagree does not describe the run.
    """
    res = {"ruleId": "B1", "properties": {"issue_severity": severity}, "level": level, "message": {"text": "x"}}
    r = _run_with(tmp_path, _ok_run(results=[res]), 1)
    assert r.returncode == 2, f"{severity}/{level} was classified rather than refused: {r.stdout}{r.stderr}"
    assert "the tool emits" in r.stderr, r.stderr


def test_a_medium_finding_has_no_level_and_still_passes(tmp_path) -> None:
    """Measured, and it contradicted this file's earlier assumption.

    A gate that required `level` on every result would refuse every real
    MEDIUM finding; one that read `level` first would classify it as unknown.
    It is a genuine, readable, non-HIGH finding and must pass.
    """
    assert "level" not in _MEDIUM, "the fixture stopped matching the measured shape"
    r = _run(tmp_path, MEDIUM_ONLY, 1)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 finding(s), 0 high-severity" in r.stdout


#: Every wrongly-typed value here is NON-EMPTY on purpose. A first draft used
#: `[]` and `{}`, and the mutation check showed those controls passing with the
#: type guard REMOVED: an empty container is falsy, so `x or {}` replaced it and
#: the report was refused for a different reason -- a missing tool name, an
#: unreadable severity. The control proved nothing about the guard it named.
#: A non-empty wrong type survives `or {}` and reaches the `.get`.
@pytest.mark.parametrize(
    "run",
    [
        {"tool": ["Bandit"], "invocations": [{"executionSuccessful": True}], "results": []},
        {"tool": {"driver": ["Bandit"]}, "invocations": [{"executionSuccessful": True}], "results": []},
        {
            "tool": {"driver": dict(_DRIVER)},
            "invocations": [{"executionSuccessful": True}],
            "results": [{"ruleId": "B1", "properties": [{"issue_severity": "HIGH"}]}],
        },
        {
            "tool": {"driver": dict(_DRIVER)},
            "invocations": [{"executionSuccessful": True}],
            "results": [{"ruleId": "B1", "properties": {"issue_severity": "HIGH"}, "message": "a string"}],
        },
    ],
    ids=["tool", "driver", "properties", "message"],
)
def test_a_wrongly_typed_nested_field_exits_two_rather_than_tracebacks(tmp_path, run) -> None:
    """Type-checked BEFORE `.get`, so the step fails cleanly.

    An AttributeError from inside the gate ends the CI step with a traceback
    and no verdict, which reads to an operator as tooling noise rather than as
    a refusal to certify the scan.
    """
    r = _run_with(tmp_path, run, 1)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "Traceback" not in r.stderr, r.stderr
    assert "not an object" in r.stderr, r.stderr


def test_a_non_string_level_is_refused_by_the_type_check_it_names(tmp_path) -> None:
    """A weaker control here would pass for the wrong reason.

    `level: 7` is also caught downstream by the severity/level agreement check
    (7 != "error"), so asserting only "exit 2" would stay green with the type
    guard removed. The MESSAGE is what distinguishes which guard fired.
    """
    res = {"ruleId": "B1", "properties": {"issue_severity": "HIGH"}, "level": 7, "message": {"text": "x"}}
    r = _run_with(tmp_path, _ok_run(results=[res]), 1)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "not a string" in r.stderr, r.stderr


def test_the_gate_does_not_pin_the_tool_version(tmp_path) -> None:
    """The measured CONTRACT is the guard, not the version that was measured.

    A version pin turns every Bandit upgrade into a red gate for a reason that
    has nothing to do with the code being scanned.
    """
    other = dict(_DRIVER, version="2.0.0", semanticVersion="2.0.0")
    r = _run_with(tmp_path, _ok_run(tool={"driver": other}), 0)
    assert r.returncode == 0, r.stdout + r.stderr

    src = (_REPO / "scripts" / "bandit_gate.py").read_text(encoding="utf-8")
    import ast

    tree = ast.parse(src)
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "1.9.4" not in literals, "the gate compares against a pinned version string"

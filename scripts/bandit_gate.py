#!/usr/bin/env python3
# Copyright 2026 STARGA, Inc.
"""Decide whether a Bandit run passes, from its exit code AND its SARIF.

The gate used to be `bandit -r src ... || true`, which reported success for a
finding, for a crash, and for not running at all. The first repair captured the
exit code and never read it, and `set -uo pipefail` does not clear an inherited
`errexit`, so under `bash -e` the failing command ended the step before the
capture ran. Both were real and both are fixed here by moving the decision out
of YAML into something that can be executed against stubs.

TWO INPUTS, because either alone is a hole:

* the exit code separates "found something" (1) from "could not run" (anything
  else). A tool failure must never be reported as a clean scan;
* the SARIF separates HIGH from the rest, and must be PRESENT, non-empty,
  parseable, and carry at least one run. An empty `runs` list is not evidence
  of zero findings -- it is what a stale or truncated file looks like, and
  reading it as "0 high" is the same false green in a new place.

SEVERITY IS READ FROM THE FIELD BANDIT ACTUALLY EMITS, verified by running the
tool rather than assumed. Measured output, `bandit[sarif]` 1.9.4 (the version
is the PROVENANCE of the measurement, not a pin -- the contract below is the
guard, and a different version that still satisfies it passes):

    HIGH    result.level = "error"   properties.issue_severity = "HIGH"
    LOW     result.level = "note"    properties.issue_severity = "LOW"
    MEDIUM  result.level ABSENT      properties.issue_severity = "MEDIUM"
    rule.defaultConfiguration = None  <- unusable, and the first version of
                                         this gate relied on it

The MEDIUM row is why `issue_severity` is the authority and `level` is only
corroboration: `level` is not always emitted at all, so any rule that reads it
first silently classifies a MEDIUM finding as "no level".

AND SEVERITY MUST BE PRESENT. A `results: [{}]` report with exit 1 was incorrectly
accepted as one non-HIGH finding: with no `properties` and no `level`, a result
carrying no severity at all fell through every comparison and was counted as
benign. Absence is not a severity. A result whose severity cannot be read
makes the REPORT unusable (exit 2); it is never quietly treated as low.

Exit codes of this script: 0 pass, 1 high-severity findings, 2 the scan or its
report is unusable. The distinction is the point -- a broken scanner and a
clean codebase must never produce the same answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

#: Bandit's own exit codes. Anything else means it did not complete.
BANDIT_CLEAN = 0
BANDIT_FOUND_ISSUES = 1


#: Fields the installed Bandit really emits, measured rather than assumed:
#:     runs[].tool.driver.name        = "Bandit"
#:     runs[].tool.driver.version     = "1.9.4"
#:     runs[].results                 = [] (present and a list, even when clean)
#:     runs[].invocations[0].executionSuccessful = true
#: A structurally empty object like {"runs": [{}]} carries none of these, and
#: reading it as "zero findings" is fabricating execution evidence from an
#: absence. The malformed-report controls cover this case.
_EXPECTED_TOOL = "bandit"

#: Severities the tool really emits. A value outside this set is not a milder
#: finding -- it is a report this gate cannot classify, and guessing would be
#: the same false green in a new place.
_SEVERITIES = ("HIGH", "MEDIUM", "LOW")

#: The level each severity carries WHEN IT CARRIES ONE. Measured: MEDIUM emits
#: no `level` key, so this map is deliberately partial and absence is legal.
#: Used only to catch a CONTRADICTION -- an "error"-level result claiming LOW
#: severity, or a HIGH result claiming a milder level -- which is what a
#: doctored or mismatched report looks like.
_LEVEL_FOR_SEVERITY = {"HIGH": "error", "LOW": "note"}


def load_sarif(path: str) -> tuple[list[dict[str, Any]] | None, str]:
    """Return (runs, error). `runs` is None when the report is unusable.

    Every run must carry the evidence a completed Bandit scan actually
    produces. A report that merely has the right SHAPE is not a report: the
    first version accepted `{"runs": [{}]}` as a clean scan.
    """
    if not os.path.isfile(path):
        return None, f"no SARIF at {path}: the scanner did not produce a report"
    if os.path.getsize(path) == 0:
        return None, f"SARIF at {path} is empty"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, f"SARIF at {path} is not readable JSON: {exc}"
    if not isinstance(doc, dict):
        return None, "SARIF root is not an object"
    runs = doc.get("runs")
    if not isinstance(runs, list) or not runs:
        # NOT "zero findings". A report with no runs is what a stale,
        # truncated or aborted scan leaves behind.
        return None, "SARIF carries no runs; a scan that produced no run is not a scan that found nothing"

    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            return None, f"runs[{i}] is not an object"

        tool = run.get("tool")
        if tool is not None and not isinstance(tool, dict):
            return None, f"runs[{i}].tool is {type(tool).__name__}, not an object"
        driver = (tool or {}).get("driver")
        if driver is not None and not isinstance(driver, dict):
            # Guarded BEFORE `.get`, so an unusable report exits 2 rather than
            # ending the step with an AttributeError traceback.
            return None, f"runs[{i}].tool.driver is {type(driver).__name__}, not an object"
        driver = driver or {}
        name = str(driver.get("name") or "")
        if not name:
            return None, f"runs[{i}] names no tool; a report with no tool identity is not evidence of a scan"
        if name.strip().lower() != _EXPECTED_TOOL:
            return None, f"runs[{i}] was produced by {name!r}, not Bandit"

        # `results` must be PRESENT and a list. Absent or null is a report that
        # never recorded an outcome, which is not the same as recording none.
        if "results" not in run or not isinstance(run.get("results"), list):
            return None, f"runs[{i}] has no results list; the scan recorded no outcome, which is not an outcome of none"
        for j, res in enumerate(run["results"]):
            if not isinstance(res, dict):
                return None, f"runs[{i}].results[{j}] is not an object"
            bad = _severity_error(res, f"runs[{i}].results[{j}]")
            if bad:
                return None, bad
            message = res.get("message")
            if message is not None and not isinstance(message, dict):
                return None, f"runs[{i}].results[{j}].message is {type(message).__name__}, not an object"

        # Completed-scan evidence, from the field Bandit actually writes.
        #
        # Two demonstrated false greens are closed here, and both were about
        # ADMITTING rather than reading:
        #   * `executionSuccessful: "false"` -- a truthy STRING. `bool(x)` said
        #     the scan succeeded. The test is `is True`, so only the boolean
        #     the tool actually emits counts;
        #   * `invocations: [{...ok...}, null]` -- the malformed second entry
        #     was FILTERED OUT by an `if isinstance(inv, dict)` guard, and the
        #     surviving good one answered for the whole list. Skipping the
        #     record you cannot read is how a report with a broken invocation
        #     passes; every entry must be readable.
        invocations = run.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            return None, f"runs[{i}] records no invocation; there is no evidence the scan completed"
        for k, inv in enumerate(invocations):
            if not isinstance(inv, dict):
                return None, (
                    f"runs[{i}].invocations[{k}] is {type(inv).__name__}, not an object; "
                    f"a malformed invocation is not one that can be skipped"
                )
            if inv.get("executionSuccessful") is not True:
                return None, (
                    f"runs[{i}].invocations[{k}].executionSuccessful is "
                    f"{inv.get('executionSuccessful')!r}, not the boolean true the tool emits"
                )

    return runs, ""


def _severity_error(res: dict[str, Any], where: str) -> str:
    """Return an error string unless *res* carries a severity this gate can read.

    Called during ADMISSION, not during counting, so a result whose severity is
    absent, unknown or malformed makes the whole report unusable instead of
    being silently classified as benign. That was a measured false green:
    `results: [{}]` with exit 1 was reported as one non-HIGH finding and passed.
    """
    props = res.get("properties")
    if props is not None and not isinstance(props, dict):
        return f"{where}.properties is {type(props).__name__}, not an object"
    props = props or {}
    raw = props.get("issue_severity")
    if not isinstance(raw, str) or raw.strip().upper() not in _SEVERITIES:
        return (
            f"{where} carries no readable issue_severity ({raw!r}); one of {list(_SEVERITIES)} is required. "
            f"A result whose severity cannot be read is not a low-severity result."
        )
    severity = raw.strip().upper()

    level = res.get("level")
    if level is not None and not isinstance(level, str):
        return f"{where}.level is {type(level).__name__}, not a string"
    # MEDIUM legitimately carries no level, so absence is fine. A level that
    # CONTRADICTS the severity is not -- in either direction, because the
    # dangerous one is an error-level finding labelled LOW.
    expected = _LEVEL_FOR_SEVERITY.get(severity)
    if level is not None and expected is not None and level != expected:
        return f"{where} claims severity {severity} with level {level!r}; the tool emits {expected!r} for {severity}"
    if level == "error" and severity != "HIGH":
        return f"{where} has level 'error' but severity {severity}; an error-level finding is not a milder one"
    return ""


def high_findings(runs: list[dict[str, Any]]) -> list[str]:
    """Every HIGH-severity result.

    Severity has already been proven readable at admission, so this reads
    `issue_severity` alone: no `level` fallback, and therefore no path on which
    a missing severity silently becomes something other than HIGH.
    """
    out: list[str] = []
    for run in runs:
        for res in run.get("results") or []:
            severity = str((res.get("properties") or {}).get("issue_severity") or "").strip().upper()
            if severity == "HIGH":
                rule = res.get("ruleId") or "?"
                text = str((res.get("message") or {}).get("text") or "").strip().replace("\n", " ")
                out.append(f"{rule} [{severity}]: {text[:140]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bandit result gate")
    ap.add_argument("--sarif", required=True)
    ap.add_argument("--exit-code", required=True, type=int, help="the exit code bandit itself returned")
    a = ap.parse_args(argv)

    # A tool failure is decided FIRST and on its own. Reading the report of a
    # scan that crashed would let a stale file from an earlier step answer for
    # a run that never happened.
    if a.exit_code not in (BANDIT_CLEAN, BANDIT_FOUND_ISSUES):
        print(f"bandit did not complete: exit {a.exit_code}. This is a scanner failure, not a clean scan.", file=sys.stderr)
        return 2

    runs, error = load_sarif(a.sarif)
    if runs is None:
        print(f"bandit report unusable: {error}", file=sys.stderr)
        return 2

    findings = high_findings(runs)
    total = sum(len(r.get("results") or []) for r in runs)

    # Bandit exits 1 ONLY when it found something. Exit 1 with an empty report
    # is incoherent: either the report is not the one that run produced, or it
    # was truncated. Either way it is not a pass.
    if a.exit_code == BANDIT_FOUND_ISSUES and total == 0:
        print(
            "bandit exited 1 (findings) but its report contains none. The report does not describe that run.",
            file=sys.stderr,
        )
        return 2
    print(f"bandit: {total} finding(s), {len(findings)} high-severity")
    for f in findings:
        print(f"  {f}")

    if findings:
        print("FAILING: high-severity findings must be fixed or explicitly triaged.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

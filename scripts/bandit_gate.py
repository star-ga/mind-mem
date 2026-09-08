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

SEVERITY IS READ FROM THE FIELD BANDIT ACTUALLY EMITS, verified against the
installed tool rather than assumed. Measured on bandit with the sarif extra:

    result.properties.issue_severity  = "HIGH" / "MEDIUM" / "LOW"
    result.level                      = "error" for HIGH, "note" for LOW
    rule.defaultConfiguration.level   = None      <- unusable, and the first
                                                     version of this gate
                                                     relied on it

So `issue_severity` is authoritative and `level == "error"` corroborates it.

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
#: absence. Root demonstrated exactly that against the first version.
_EXPECTED_TOOL = "bandit"


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

        driver = ((run.get("tool") or {}).get("driver") or {}) if isinstance(run.get("tool"), dict) else {}
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

        # Completed-scan evidence, from the field Bandit actually writes.
        invocations = run.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            return None, f"runs[{i}] records no invocation; there is no evidence the scan completed"
        successful = [inv.get("executionSuccessful") for inv in invocations if isinstance(inv, dict)]
        if not successful or not all(bool(x) for x in successful):
            return None, f"runs[{i}] reports an invocation that did not complete successfully: {successful}"

    return runs, ""


def high_findings(runs: list[dict[str, Any]]) -> list[str]:
    """Every HIGH-severity result, by the field Bandit actually emits."""
    out: list[str] = []
    for run in runs:
        for res in run.get("results") or []:
            props = res.get("properties") or {}
            severity = str(props.get("issue_severity") or "").upper()
            if severity == "HIGH" or (not severity and str(res.get("level")) == "error"):
                rule = res.get("ruleId") or "?"
                text = ((res.get("message") or {}).get("text") or "").strip().replace("\n", " ")
                out.append(f"{rule} [{severity or res.get('level')}]: {text[:140]}")
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

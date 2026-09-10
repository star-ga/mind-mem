#!/usr/bin/env python3
"""Gate: every EVIDENCE.md claim's test must EXECUTE. A skip is a red build.

WHY THIS EXISTS, measured rather than argued.

`EVIDENCE.md` row 5 claims "MIND kernels equivalent to the Python baseline" and
offers `pytest tests/test_mind_ffi.py -q` as the command a third party runs to
check it. On this repository that command reports 9 passed. On a fresh clone,
on every CI row, and after `pip install`, it reports 8 passed and 1 SKIPPED --
because the path the loader probes is gitignored, the committed library sits at
a path the loader never probes, and the wheel ships no library at all.

So the claim was verified by a command that quietly tested nothing. That is
worse than an unverified claim: an unverified claim is honest, and this one was
unfalsifiable while looking checked.

The defect class is not "one bad row". It is that NOTHING NOTICED. The suite
was green, the doc cited a real file, the test existed and was collected. The
only signal was a `pytest.skip` nobody read. This gate turns that silence loud:

    an evidence test may not skip; a skip in the evidence set is a red build.

It is the sibling of `check_tool_surface.py --check-doc-names` (a documented
tool name must exist) and `check_reachable_modules.py` (a module must be
wired). Those assert that things EXIST and are WIRED. This one asserts that the
checks behind our public claims actually RUN.

USAGE
    python3 scripts/check_evidence_executes.py            # gate
    python3 scripts/check_evidence_executes.py --self-test # prove it can fail

EXIT
    0  every claimed test executed
    1  a claimed test skipped, was not collected, or the manifest disagrees
    2  harness error (could not run pytest at all)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "evidence_manifest.toml"


def load_manifest() -> dict:
    with open(MANIFEST, "rb") as fh:
        return tomllib.load(fh)


def run_pytest(node_ids: list[str], extra: list[str] | None = None) -> tuple[int, dict]:
    """Run pytest over *node_ids* and return (returncode, per-outcome counts).

    Uses -rs so runtime skips are REPORTED. A skip that only shows as an 's'
    in the progress line is exactly what went unnoticed for months.
    """
    report = ROOT / ".evidence-report.json"
    cmd = [
        sys.executable, "-m", "pytest", *node_ids,
        "-p", "no:randomly", "-q", "-rs",
        f"--junitxml={report.with_suffix('.xml')}",
    ] + (extra or [])
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    return proc.returncode, {"stdout": proc.stdout, "stderr": proc.stderr}


def parse_junit(path: Path) -> dict[str, str]:
    """node id -> outcome, read from the junit XML pytest just wrote."""
    import xml.etree.ElementTree as ET

    out: dict[str, str] = {}
    if not path.exists():
        return out
    for case in ET.parse(path).getroot().iter("testcase"):
        cls, name = case.get("classname", ""), case.get("name", "")
        key = f"{cls}::{name}" if cls else name
        outcome = "passed"
        for child in case:
            if child.tag in {"skipped", "failure", "error"}:
                outcome = child.tag
                break
        out[key] = outcome
    return out


def check(claims: dict, canary: bool = False) -> int:
    if canary:
        # Inject the canary AS A CLAIM, so the self-test exercises the exact
        # code path a real claim takes. An earlier version appended it only to
        # the pytest node list while the checking loop still walked the
        # manifest, so the canary was run and then never examined -- the gate
        # reported itself healthy while being blind. The self-test caught that,
        # which is the whole reason it exists.
        claims = dict(claims)
        claims["__canary__"] = {
            "claim": "canary: a deliberately skipping evidence test",
            "tests": ["tests/test_evidence_gate_canary.py::test_canary_evidence_test_that_skips"],
        }

    node_ids: list[str] = []
    for cid, entry in claims.items():
        node_ids.extend(entry["tests"])

    if not node_ids:
        print("FAIL: the manifest names no tests. A gate over nothing is not a gate.")
        return 1

    # A manifest entry pointing at a file that no longer exists makes pytest
    # exit before collecting ANYTHING ("no tests ran"), so one stale path hides
    # the status of every other claim. Report those as failed claims and run the
    # rest, rather than letting a stale path blind the gate.
    missing = [n for n in node_ids if not (ROOT / n.split("::")[0]).exists()]
    node_ids = [n for n in node_ids if n not in missing]
    if missing and not node_ids:
        print("EVIDENCE GATE FAILED — every claimed test path is missing:")
        for m in missing:
            print(f"  MISSING PATH  {m}")
        return 1

    rc, res = run_pytest(node_ids)
    outcomes = parse_junit((ROOT / ".evidence-report").with_suffix(".xml"))

    if not outcomes:
        print("HARNESS ERROR: pytest produced no junit report.")
        print(res["stdout"][-2000:])
        return 2

    # A node id in the manifest that pytest never collected is as bad as a skip:
    # the claim points at a test that does not exist any more.
    collected = set()
    for key in outcomes:
        collected.add(key.replace(".", "/", key.count(".") - 1) if "::" not in key else key)

    problems: list[str] = [f"  MISSING PATH   {m}  (claim points at a file that does not exist)" for m in missing]
    executed = 0
    for cid, entry in claims.items():
        for t in entry["tests"]:
            # A manifest entry is either a whole FILE ("tests/x.py") or a single
            # node ("tests/x.py::test_y"). junit keys look like
            # "tests.x::test_y", so match on the module for a file entry and on
            # the leaf for a node entry. Getting this wrong reports a healthy
            # test as NOT COLLECTED, which would train people to ignore the gate.
            if "::" in t:
                leaf = t.split("::")[-1]
                matches = [k for k in outcomes if k.endswith("::" + leaf)]
            else:
                # junit classname is "tests.mod" for a bare function but
                # "tests.mod.TestClass" for a method, so an equality test
                # silently misses every class-based test file. Match on the
                # module boundary instead.
                module = t.removesuffix(".py").replace("/", ".")
                matches = [
                    k for k in outcomes
                    if (cls := k.split("::")[0]) == module or cls.startswith(module + ".")
                ]
            if not matches:
                problems.append(f"  {cid}: NOT COLLECTED  {t}  (claim points at a test that no longer runs)")
                continue
            for m in matches:
                if outcomes[m] == "skipped":
                    problems.append(f"  {cid}: SKIPPED       {m}  (claim '{entry['claim']}' is unverified)")
                elif outcomes[m] in {"failure", "error"}:
                    problems.append(f"  {cid}: {outcomes[m].upper():<13}{m}")
                else:
                    executed += 1

    if canary:
        canary_skipped = any(
            k.endswith("::test_canary_evidence_test_that_skips") and v == "skipped"
            for k, v in outcomes.items()
        )
        if not canary_skipped:
            print("SELF-TEST HARNESS ERROR: the canary did not skip, so this proves nothing.")
            return 2
        if not any("__canary__" in p for p in problems):
            print("SELF-TEST FAILED: the gate did not flag a deliberately skipped evidence test.")
            print("A gate that cannot fail is not a gate.")
            return 1
        print(f"SELF-TEST PASSED: the gate flagged the skipped canary ({len(problems)} problem(s)).")
        return 0

    if problems:
        print(f"EVIDENCE GATE FAILED — {len(problems)} claim(s) are not actually verified:\n")
        print("\n".join(problems))
        print(
            "\nA claim whose test skips is unfalsifiable, not verified. Either make the "
            "test execute, or withdraw the claim from EVIDENCE.md."
        )
        return 1

    print(f"EVIDENCE GATE PASSED — {executed} test(s) executed across {len(claims)} claim(s), 0 skipped.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="add a deliberately-skipping canary and REQUIRE the gate to fail")
    args = ap.parse_args()
    try:
        manifest = load_manifest()
    except FileNotFoundError:
        print(f"HARNESS ERROR: no manifest at {MANIFEST}")
        return 2
    return check(manifest.get("claim", {}), canary=args.self_test)


if __name__ == "__main__":
    sys.exit(main())

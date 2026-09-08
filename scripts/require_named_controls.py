#!/usr/bin/env python3
# Copyright 2026 STARGA, Inc.
"""Assert that specific, named controls ran and passed.

A count is not a witness. The first version of this gate accepted any
`N passed` line, so a file that shrank to a single remaining test satisfied
it just as well as the full set -- and a skip, which is what an unconfigured
isolated database produces, reads as a pass in a green CI summary.

So the required cases are NAMED. Each must appear in the report, and must be
neither skipped, failed nor errored. Parsed from JUnit XML with the standard
library, from the SAME run that executed them, so nothing is executed twice
and the report cannot describe a different invocation.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

#: The controls this job exists to run. Named individually because the point
#: is that each specific guarantee was exercised, not that the file was.
REQUIRED = {
    "tests/test_pg_pool_autocommit_isolation.py": [
        "test_isolation_attestation_is_required_and_matches",
        "test_control_1_checkout_after_schema_setup_is_not_autocommit",
        "test_control_2_exception_path_does_not_leak_autocommit",
        "test_control_3_transaction_block_rolls_back_atomically",
        "test_control_4_failed_restoration_discards_the_connection",
        "test_control_5_scoped_restoration_holds_without_the_pool_callback",
    ],
}


def classname_for(path: str) -> str:
    """pytest's JUnit classname for a test file: the dotted module path.

    Measured against a real run: `tests/test_pg_pool_autocommit_isolation.py`
    reports `classname="tests.test_pg_pool_autocommit_isolation"`.
    """
    return path.removesuffix(".py").replace("/", ".")


def required_identities() -> set[tuple[str, str]]:
    """(classname, name) for every required control. Identity, not just a name."""
    out: set[tuple[str, str]] = set()
    for path, names in REQUIRED.items():
        cls = classname_for(path)
        for name in names:
            identity = (cls, name)
            if identity in out:
                raise SystemExit(f"REQUIRED lists {identity} twice; a duplicate requirement cannot be witnessed")
            out.add(identity)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: require_named_controls.py <junit.xml>", file=sys.stderr)
        return 2
    try:
        tree = ET.parse(argv[1])
    except (OSError, ET.ParseError) as exc:
        print(f"cannot read the JUnit report {argv[1]}: {exc}", file=sys.stderr)
        return 2

    # IDENTITY IS (classname, name), not a bare method name, and a FAILURE IS
    # NEVER OVERWRITTEN. Both were real false greens, demonstrated against the
    # first version: all six required names emitted under an unrelated
    # classname passed, and each genuinely FAILED control followed by an
    # unrelated same-named pass also passed, because the dict was keyed on the
    # bare name and the last write won.
    #
    # A parameterised case reports as `name[param]`, so the exact recorded name
    # is compared -- a required identity must match a case that really ran
    # under that identity, not one that merely shares a method name.
    seen: dict[tuple[str, str], str] = {}
    duplicates: set[tuple[str, str]] = set()
    for case in tree.iter("testcase"):
        identity = (case.get("classname") or "", case.get("name") or "")
        status = "passed"
        for child in case:
            if child.tag in ("skipped", "failure", "error"):
                status = child.tag
                break
        if identity in seen:
            duplicates.add(identity)
            # Worst outcome wins. A second, passing record for the same
            # identity must never erase the first one's failure.
            if seen[identity] == "passed":
                seen[identity] = status
        else:
            seen[identity] = status

    if not seen:
        print("the JUnit report contains no test cases; nothing was witnessed", file=sys.stderr)
        return 2

    problems: list[str] = []
    for cls, name in sorted(required_identities()):
        status = seen.get((cls, name))
        if status is None:
            near = [c for (c, n) in seen if n == name]
            hint = f" (a case of that name ran under {sorted(set(near))})" if near else ""
            problems.append(f"{cls}::{name} is ABSENT from the report{hint}")
        elif status != "passed":
            problems.append(f"{cls}::{name} was {status}; a skip in CI reads as a pass")

    for identity in sorted(duplicates & required_identities()):
        problems.append(f"{identity[0]}::{identity[1]} appears more than once; a required identity must be witnessed exactly once")

    print(f"required controls: {len(required_identities())}; cases in report: {len(seen)}")
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

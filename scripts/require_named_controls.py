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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: require_named_controls.py <junit.xml>", file=sys.stderr)
        return 2
    try:
        tree = ET.parse(argv[1])
    except (OSError, ET.ParseError) as exc:
        print(f"cannot read the JUnit report {argv[1]}: {exc}", file=sys.stderr)
        return 2

    seen: dict[str, str] = {}
    for case in tree.iter("testcase"):
        name = case.get("name") or ""
        status = "passed"
        for child in case:
            if child.tag in ("skipped", "failure", "error"):
                status = child.tag
                break
        seen[name] = status

    if not seen:
        print("the JUnit report contains no test cases; nothing was witnessed", file=sys.stderr)
        return 2

    problems: list[str] = []
    for path, names in REQUIRED.items():
        for name in names:
            status = seen.get(name)
            if status is None:
                problems.append(f"{path}::{name} is ABSENT from the report -- renamed, removed, or never selected")
            elif status != "passed":
                problems.append(f"{path}::{name} was {status}; a skip in CI reads as a pass")

    print(f"required controls: {sum(len(v) for v in REQUIRED.values())}; cases in report: {len(seen)}")
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

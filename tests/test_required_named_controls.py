# Copyright 2026 STARGA, Inc.
"""A count is not a witness: the CI receipt names the controls it requires.

The first version of the PG execution gate accepted any `N passed` line. A
file that shrank to one remaining test satisfied it exactly as well as the
full set, and a skip -- which is what an unconfigured isolated database
produces -- reads as a pass in a green summary. Neither witnesses that a
specific guarantee was exercised.

So the required cases are named and checked from the JUnit report of the SAME
run that executed them: nothing runs twice, and the report cannot describe a
different invocation than the one that produced it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "require_named_controls.py"


def _report(tmp_path: pathlib.Path, cases: list[tuple[str, str | None]]) -> str:
    suite = ET.Element("testsuite", name="pytest", tests=str(len(cases)))
    for name, status in cases:
        case = ET.SubElement(suite, "testcase", classname="tests.test_pg_pool_autocommit_isolation", name=name)
        if status:
            ET.SubElement(case, status, message="x")
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "r.xml"
    ET.ElementTree(suite).write(path)
    return str(path)


def _required() -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("rnc", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return [n for names in mod.REQUIRED.values() for n in names]


def _run(path: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_SCRIPT), path], capture_output=True, text=True, timeout=120)


def test_every_required_control_present_and_passed_is_accepted(tmp_path) -> None:
    r = _run(_report(tmp_path, [(n, None) for n in _required()]))
    assert r.returncode == 0, r.stderr


def test_a_missing_control_fails(tmp_path) -> None:
    """The case a file-count floor cannot catch: a method quietly removed."""
    names = _required()
    r = _run(_report(tmp_path, [(n, None) for n in names[1:]]))
    assert r.returncode == 1
    assert "ABSENT" in r.stderr


def test_a_skipped_control_fails(tmp_path) -> None:
    """A skip in CI reads as a pass; here it must not."""
    names = _required()
    r = _run(_report(tmp_path, [(names[0], "skipped")] + [(n, None) for n in names[1:]]))
    assert r.returncode == 1
    assert "skip" in r.stderr.lower()


def test_a_failed_or_errored_control_fails(tmp_path) -> None:
    for status in ("failure", "error"):
        names = _required()
        r = _run(_report(tmp_path / status, [(names[0], status)] + [(n, None) for n in names[1:]]))
        assert r.returncode == 1, f"{status} was accepted"


def test_an_empty_report_fails(tmp_path) -> None:
    """Nothing witnessed is not everything passing."""
    r = _run(_report(tmp_path, []))
    assert r.returncode == 2
    assert "no test cases" in r.stderr


def test_an_unreadable_report_fails(tmp_path) -> None:
    p = tmp_path / "bad.xml"
    p.write_text("<not xml", encoding="utf-8")
    r = _run(str(p))
    assert r.returncode == 2


def test_the_required_names_exist_in_the_test_file() -> None:
    """A required name that no longer exists would fail CI forever, silently blaming the wrong thing."""
    src = (_REPO / "tests" / "test_pg_pool_autocommit_isolation.py").read_text(encoding="utf-8")
    missing = [n for n in _required() if f"def {n}(" not in src]
    assert not missing, f"the receipt requires controls that do not exist: {missing}"


def test_the_workflow_does_not_run_the_pg_file_twice() -> None:
    """Root's point: inspect the first run's report rather than re-executing."""
    wf = (_REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    runs = [ln for ln in wf.splitlines() if "pytest" in ln and "test_pg_pool_autocommit_isolation" in ln]
    assert not runs, f"the PG file is executed a second time: {runs}"
    assert "--junitxml=pg-results.xml" in wf, "the single run produces no machine-readable report"
    assert "require_named_controls.py pg-results.xml" in wf, "the named-control receipt is not wired"

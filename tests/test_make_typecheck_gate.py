# Copyright 2026 STARGA, Inc.
"""`make typecheck` must fail when type checking fails, and when it cannot run.

The target was:

    python3 -m mypy src/ ... 2>/dev/null || echo "mypy not installed — skipping"

which turned every outcome into exit 0 -- a type error, a crash, and mypy
never running at all. An independent audit demonstrated it with a stub
`python3` returning 7: the target still exited 0 and printed the skip message.

Worse, measured while fixing it: `python3 -m mypy` is not importable on a
machine where mypy installs as a standalone executable, which is the ordinary
case. On such a machine the target has printed "skipping" and exited 0 every
time it has ever been invoked. The gate never type-checked anything.

Two controls, because one direction alone proves nothing: the target really
invokes a type checker, and a real type error really propagates.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


def _target_body() -> str:
    """Everything from the typecheck target to the NEXT TARGET.

    Not "to the next non-indented line": the target carries column-zero
    comments explaining why it looks the way it does, and stopping at the first
    of those returned an empty body -- which made every assertion below pass
    vacuously on nothing. A control's own parser needs the same suspicion as
    the thing it checks.
    """
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("typecheck:")), None)
    assert start is not None, "no typecheck target in the Makefile"
    body: list[str] = []
    for ln in lines[start + 1 :]:
        if re.match(r"^[A-Za-z0-9_.-]+:", ln):
            break
        body.append(ln)
    assert any(ln.strip() for ln in body), "the typecheck target has an empty body"
    return "\n".join(body)


def test_the_target_cannot_swallow_a_failure() -> None:
    """Structural: no `|| echo`, no `2>/dev/null` on the checker itself.

    Both were present and both are how the false green was built: the redirect
    hid the diagnosis and the `||` converted the failure into a success.
    """
    # RECIPE LINES ONLY -- the ones make actually executes, which in a Makefile
    # are the tab-indented ones. Scanning the whole body matches the comments
    # that DESCRIBE the old defect and convicts the target for explaining
    # itself. That is the second time today a text scan has done exactly that,
    # so it is written down rather than just fixed.
    recipe = [ln for ln in _target_body().splitlines() if ln.startswith("\t")]
    assert recipe, "the typecheck target has no recipe lines"
    body = "\n".join(recipe)

    # The invocation, whatever name the checker is held under -- the target
    # picks between an executable and a module, so matching a literal command
    # would break the moment that choice changes.
    checker_lines = [ln for ln in body.splitlines() if "src/" in ln and "--ignore-missing-imports" in ln]
    assert checker_lines, f"the target no longer type-checks src/:\n{body}"

    for line in checker_lines:
        assert "2>/dev/null" not in line, f"the checker's stderr is still discarded: {line.strip()}"
        assert "|| echo" not in line, f"the checker's failure is still converted into a message: {line.strip()}"
        assert not line.rstrip().endswith("|| true"), f"the checker's exit code is still masked: {line.strip()}"

    # And nowhere in the target does a failure become a printed reassurance.
    # The availability probe is allowed to branch; it exits non-zero when it
    # finds nothing, which the behavioural control does not cover because a
    # runner without mypy skips it.
    assert 'not installed — skipping' not in body, "the skip-on-failure message is back"
    assert "exit 2" in body, "a missing type checker no longer fails the target"


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not available on this runner")
def test_a_real_type_error_makes_the_gate_fail(tmp_path) -> None:
    """Behavioural, both directions, on a COPY so the checkout is never dirtied.

    The positive half is what stops this passing on a target that fails for
    some unrelated reason; the negative half is the gate itself.
    """
    if shutil.which("mypy") is None:
        try:
            subprocess.run([sys.executable, "-c", "import mypy"], check=True, capture_output=True)
        except Exception:
            pytest.skip("no type checker available, so neither direction can be measured")

    work = tmp_path / "tree"
    shutil.copytree(ROOT, work, symlinks=True, ignore=shutil.ignore_patterns(".git", "*.egg-info", "__pycache__", ".venv"))

    def run() -> subprocess.CompletedProcess:
        return subprocess.run(["make", "-s", "typecheck"], cwd=work, capture_output=True, text=True, timeout=1800)

    clean = run()
    assert clean.returncode == 0, f"the gate fails on a clean tree, so the negative half would prove nothing:\n{clean.stdout}\n{clean.stderr}"

    probe = work / "src" / "mind_mem" / "pipeline_hash.py"
    probe.write_text(probe.read_text(encoding="utf-8") + '\n\ndef _typecheck_probe() -> int:\n    return "not an int"\n', encoding="utf-8")

    broken = run()
    assert broken.returncode != 0, "a real type error did not fail the gate"
    assert "error:" in (broken.stdout + broken.stderr), "the gate failed without reporting the diagnosis"


def test_the_help_line_does_not_hardcode_a_test_count() -> None:
    """A number in help text goes stale silently and then lies to the reader."""
    text = MAKEFILE.read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("test:"))
    assert not re.search(r"\d{3,}", line), f"the help line hardcodes a count that will go stale: {line}"

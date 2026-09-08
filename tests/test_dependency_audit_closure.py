# Copyright 2026 STARGA, Inc.
"""The dependency audit must cover every extra that actually ships.

`.[all]` does not mean "all". Measured on this tree it omits 12 dependencies
of shipped extras -- cython, fastapi, httpx, python-jose, uvicorn, the four
opentelemetry/prometheus packages, and pgvector/psycopg/psycopg-pool -- so an
audit that installed `.[all]` alone left every one of those and their
transitive chains out of scope while reporting success.

The fix is not to redefine `.[all]`, which would change what users get from a
published extra. It is to name the audited closure EXPLICITLY in the workflow
and to fail here when a shipped extra is added without being added to it.

Dev-only extras are deliberately outside the closure: `test`, `benchmark` and
`red-team` are not installed by users of the library, and auditing a test
runner's dependencies as though they shipped would overstate the coverage in
the other direction.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import tomllib

_REPO = pathlib.Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "security.yml"

#: Extras a USER never installs. Everything else is shipped and must be audited.
DEV_ONLY = frozenset({"test", "benchmark", "red-team"})


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"].get("optional-dependencies", {})


def _audited_extras() -> set[str]:
    """Extras the pip-audit step actually installs, read from the workflow."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    install = [line for line in text.splitlines() if "pip install" in line and ".[" in line]
    assert install, "the pip-audit step no longer installs any extra"
    return set(re.findall(r"\.\[([a-z0-9_-]+)\]", " ".join(install)))


def test_every_shipped_extra_is_in_the_audited_closure() -> None:
    shipped = {name for name in _extras() if name not in DEV_ONLY and name != "all"}
    audited = _audited_extras()
    missing = sorted(shipped - audited)
    assert not missing, (
        f"these shipped extras are not installed by the dependency audit, so their dependencies "
        f"and transitive chains are unaudited while the job reports success: {missing}"
    )


def test_the_closure_is_named_not_inherited_from_all() -> None:
    """`.[all]` alone must not be the closure, because it is not one.

    Pinned so a later simplification back to a single `.[all]` install cannot
    quietly restore the gap.
    """
    audited = _audited_extras()
    assert audited != {"all"}, "the audit installs only .[all], which omits shipped extras"


def test_all_really_does_omit_shipped_dependencies() -> None:
    """The measurement behind all of this, asserted rather than remembered.

    If `.[all]` ever becomes a true closure this fails, and the right response
    is to simplify the workflow deliberately -- not to discover by accident
    that the reason for the explicit list is gone.
    """

    def names(reqs: list[str]) -> set[str]:
        return {re.split(r"[\[><=; ]", r.strip(), maxsplit=1)[0].strip() for r in reqs}

    extras = _extras()
    covered = names(extras["all"])
    gaps = {k: sorted(names(v) - covered) for k, v in extras.items() if k not in DEV_ONLY and k != "all"}
    gaps = {k: v for k, v in gaps.items() if v}
    assert gaps, "`.[all]` now covers every shipped extra; the explicit closure can be simplified deliberately"


def test_bandit_does_not_swallow_its_exit_code() -> None:
    """`bandit ... || true` reported success for every outcome, findings included."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    recipe = [line for line in text.splitlines() if "bandit -r src" in line and not line.lstrip().startswith("#")]
    assert recipe, "the bandit job no longer scans src"
    for line in recipe:
        assert "|| true" not in line, f"bandit's exit code is masked again: {line.strip()}"

    # Asserted by what the job RUNS, not by a step's display name -- a rename
    # would otherwise break this control while the gate still worked, and the
    # first version of it did exactly that.
    assert "scripts/bandit_gate.py" in text, "nothing evaluates the bandit result; findings cannot fail the job"
    assert "bandit_exit" in text, "the exit code is not persisted for the gate to read"


@pytest.mark.parametrize("claim", ["complete sast", "full sast", "comprehensive security"])
def test_no_workflow_ASSERTS_complete_security_coverage(claim) -> None:
    """A partial informational job must not be described as complete coverage.

    Scanned over the EXECUTABLE lines and step names only, not comments. The
    first version of this checked raw text and convicted the comment that
    DISCLAIMS the claim -- "not a claim of complete SAST coverage" -- which is
    the third time in one day a text scan of mine matched prose ABOUT a defect
    instead of the defect itself. The other two were `default=str` inside a
    docstring explaining why that module avoids it, and `|| echo` inside a
    Makefile comment explaining the false green it replaced.

    The pattern is now written down rather than fixed a third time in silence:
    a check whose subject is source behaviour must parse structure, and a check
    whose subject is a CLAIM must exclude the places where claims are
    discussed rather than made.
    """
    for wf in sorted((_REPO / ".github" / "workflows").glob("*.yml")):
        for lineno, line in enumerate(wf.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            body = line.split("#", 1)[0]
            assert claim not in body.lower(), f"{wf.name}:{lineno} asserts {claim!r}: {body.strip()}"

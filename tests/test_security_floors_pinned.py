"""Every advisory-bearing dependency keeps its declared security floor.

WHY THIS TEST EXISTS. A 2026-09-07 audit found eight packages in mind-mem's own
`[all]` closure carrying advisories -- authlib's CVE-2026-27962 among them, a JWK
header-injection AUTH BYPASS where the library verifies a token against a key
taken from the attacker's own header. Explicit floors were added to
pyproject.toml for all eight.

Nothing pinned them. A floor is a line in a file: a dependency bump, a merge, or
a well-meaning "loosen this to resolve" can lower one, and the next install
quietly resolves to a vulnerable version. That is the same shape as everything
else in this codebase's recent history -- a protection asserted in a document
with no code that fails when it stops being true.

And the audit that was meant to catch it could not: pyproject declares
`dependencies = []` (everything lives in an extra), so `pip-audit .` resolved an
EMPTY set and passed unconditionally, with `|| true` swallowing the rest. It had
never audited anything. This test is the cheap, offline complement to the
repaired CI job: it does not need a network or a resolver, so it cannot go
vacuous the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

#: package -> minimum version the 2026-09-07 advisory audit established.
#: Raising a floor here is fine; LOWERING one must fail this test.
_FLOORS = {
    "authlib": (1, 6, 12),          # CVE-2026-27962 JWK header-injection auth bypass
    "pyjwt": (2, 13, 0),            # 11 advisories
    "cryptography": (49, 0, 0),     # 8 advisories
    "starlette": (1, 3, 1),         # 7 advisories
    "python-multipart": (0, 0, 31), # 5 advisories
    "mcp": (1, 28, 1),              # 3 advisories
    "transformers": (5, 10, 0),     # 1 advisory
}


def _declared_floors(text: str) -> dict[str, tuple[int, ...]]:
    """package -> declared floor, parsed from the requirement strings."""
    out: dict[str, tuple[int, ...]] = {}
    for m in re.finditer(r'"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*>=\s*([0-9][0-9.]*)', text):
        name = m.group(1).strip().lower()
        ver = tuple(int(x) for x in m.group(2).split(".")[:3])
        # keep the LOWEST declaration: that is the one an install can resolve to
        if name not in out or ver < out[name]:
            out[name] = ver
    return out


def test_the_parser_finds_requirements_at_all():
    """POSITIVE CONTROL. A regex that matches nothing would pass every case."""
    found = _declared_floors(PYPROJECT.read_text(encoding="utf-8"))
    assert len(found) >= 5, f"parsed only {len(found)} floors; the regex is wrong"


@pytest.mark.parametrize(("pkg", "floor"), sorted(_FLOORS.items()))
def test_the_security_floor_is_declared_and_not_lowered(pkg, floor):
    declared = _declared_floors(PYPROJECT.read_text(encoding="utf-8"))
    assert pkg in declared, (
        f"{pkg} has advisories and NO declared floor in pyproject.toml. "
        f"A resolver free to pick any version will pick a vulnerable one."
    )
    assert declared[pkg] >= floor, (
        f"{pkg} floor was LOWERED to {'.'.join(map(str, declared[pkg]))}; the "
        f"advisory audit requires >= {'.'.join(map(str, floor))}."
    )


def test_a_lowered_floor_would_be_caught():
    """MUTATION CONTROL, in-memory: prove the comparison actually rejects.

    Done on a synthetic string rather than by editing pyproject, so the control
    cannot leave the repo dirty if it fails midway.
    """
    lowered = _declared_floors('"authlib>=1.0.0",')
    assert lowered["authlib"] < _FLOORS["authlib"], "the comparison is not ordering versions"


def test_aiohttp_is_still_absent_from_the_closure():
    """The audit's other half: aiohttp was NOT in the closure.

    If it ever appears as a direct requirement, it needs a floor decision rather
    than silently inheriting whatever a resolver picks.
    """
    text = PYPROJECT.read_text(encoding="utf-8").lower()
    hits = [ln for ln in text.splitlines()
            if "aiohttp" in ln and not ln.lstrip().startswith("#")]
    assert not hits, f"aiohttp entered the requirements without a floor decision: {hits}"

"""Every maintenance script listed must actually reach an install.

The defect this guards (measured 2026-08-29): pyproject's package-data was
``["py.typed", "*.mind", "*.pyx"]``, which matches no ``.sh`` file, so
``src/mind_mem/validate.sh`` never reached the wheel. ``init_workspace`` lists it
in ``MAINTENANCE_SCRIPTS`` and copied with ``if os.path.exists(src)`` and no
``else``, so on a pip install the copy was skipped in silence and every created
workspace had no validator while init still reported success.

Two independent checks, because either alone misses it: the file existing in the
source tree says nothing about whether it is packaged, and a glob that matches
says nothing about whether the file is there.
"""

from __future__ import annotations

import fnmatch
import os
import re

from mind_mem.init_workspace import MAINTENANCE_SCRIPTS, SCRIPT_DIR, TEMPLATE_DIR, TEMPLATE_MAP

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPROJECT = os.path.join(_REPO_ROOT, "pyproject.toml")


def _package_data_globs() -> list[str]:
    """The mind_mem package-data patterns from pyproject.

    Parsed with a regex rather than tomllib because this package supports 3.10,
    where tomllib does not exist and tomli may not be installed -- and skipping
    the check on the oldest supported interpreter is how it would rot.
    """
    with open(_PYPROJECT, encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"^mind_mem\s*=\s*\[(.*?)\]", text, re.M | re.S)
    assert match, "could not find the mind_mem package-data entry in pyproject.toml"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_every_maintenance_script_exists_in_the_source_tree() -> None:
    missing = [s for s in MAINTENANCE_SCRIPTS if not os.path.isfile(os.path.join(SCRIPT_DIR, s))]
    assert not missing, f"listed in MAINTENANCE_SCRIPTS but absent from the package: {missing}"


def test_every_maintenance_script_is_matched_by_package_data() -> None:
    """A .py file ships because setuptools packages modules; anything else needs a glob."""
    globs = _package_data_globs()
    unshipped = [s for s in MAINTENANCE_SCRIPTS if not s.endswith(".py") and not any(fnmatch.fnmatch(s, g) for g in globs)]
    assert not unshipped, (
        f"these maintenance scripts are NOT matched by pyproject package-data {globs} and would be missing from the wheel: {unshipped}"
    )


def test_a_missing_maintenance_script_is_reported_not_skipped(tmp_path, monkeypatch) -> None:
    """init must SAY a script is missing rather than carry on quietly."""
    from mind_mem import init_workspace as iw

    monkeypatch.setattr(iw, "MAINTENANCE_SCRIPTS", [*iw.MAINTENANCE_SCRIPTS, "definitely_not_shipped.sh"])
    result = iw.init(str(tmp_path / "ws"))
    created = result[0] if isinstance(result, tuple) else result
    blob = " ".join(created) if isinstance(created, (list, tuple)) else str(created)
    assert "definitely_not_shipped.sh" in blob and "MISSING" in blob, (
        "a listed-but-absent maintenance script must be reported; got: " + blob[:400]
    )


def test_template_dir_resolves_inside_the_package() -> None:
    """The defect that shipped a workspace with no corpus at all.

    TEMPLATE_DIR used to be dirname(dirname(SCRIPT_DIR))/templates, which lands
    on the repo root in a checkout and on <venv>/lib/pythonX.Y/templates in an
    install. Asserting containment is what makes the source-checkout-only path
    impossible to reintroduce -- the old value passes an "it exists" check when
    run from a checkout, which is exactly why it survived.
    """
    assert os.path.isdir(TEMPLATE_DIR), f"TEMPLATE_DIR does not exist: {TEMPLATE_DIR}"
    assert os.path.realpath(TEMPLATE_DIR).startswith(os.path.realpath(SCRIPT_DIR) + os.sep), (
        f"TEMPLATE_DIR must live INSIDE the package so it survives an install; got {TEMPLATE_DIR}"
    )


def test_every_template_the_map_needs_is_present_and_shipped() -> None:
    missing = [n for n in set(TEMPLATE_MAP.values()) if not os.path.isfile(os.path.join(TEMPLATE_DIR, n))]
    assert not missing, f"TEMPLATE_MAP names templates that are absent: {missing}"
    globs = _package_data_globs()
    unshipped = [n for n in sorted(set(TEMPLATE_MAP.values())) if not any(fnmatch.fnmatch(os.path.join("templates", n), g) for g in globs)]
    assert not unshipped, f"templates not matched by package-data {globs}; they would be missing from the wheel: {unshipped}"

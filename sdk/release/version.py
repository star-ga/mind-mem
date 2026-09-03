# Copyright 2026 STARGA, Inc.
"""Derive the SDK release identifiers from the one version that already exists.

Roadmap RM-0089 / RM-2322 ("publish the Go client as a Go module") and RM-2321
("JavaScript / TypeScript SDK") are both publish-step items. Neither can be
done correctly by hand-typing a tag, because the Go half had a latent defect
that only shows up at publish time:

    module github.com/star-ga/mind-mem/sdk/go       <- no major-version suffix

A subdirectory module is published by pushing a tag prefixed with its
directory, so the tag for this repository's 5.x line is ``sdk/go/v5.0.2``. Go
rejects a v2-or-higher version for a module path with no matching ``/vN``
suffix, so that tag would have resolved to nothing: ``go get`` would report
that the module path and version do not match, and the failure would land on
whoever tried to install it, not on us. The fix is the suffix (see
``sdk/go/go.mod``), and this module is what keeps the suffix honest — it
derives ``/vN`` from the package version instead of trusting a literal that
nobody re-reads after a major bump.

Reading the version
-------------------
``pyproject.toml`` is parsed with a small table-aware scan rather than
``tomllib``: ``tomllib`` is 3.11+, ``requires-python`` is ``>=3.10``, and the
CI matrix runs 3.10. ``tests/test_sdk_release_versioning.py`` cross-checks the
value against ``mind_mem.__version__`` so this parser cannot drift into
reading the wrong field.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
GO_MOD = REPO_ROOT / "sdk" / "go" / "go.mod"
JS_PACKAGE_JSON = REPO_ROOT / "sdk" / "js" / "package.json"

#: Module path without the major-version suffix. The suffix is derived, never
#: written by hand — see :func:`go_module_path`.
GO_MODULE_BASE = "github.com/star-ga/mind-mem/sdk/go"

#: Tag prefix for the Go subdirectory module. Go derives this from the
#: directory the ``go.mod`` lives in; it is not a free choice.
GO_TAG_PREFIX = "sdk/go"

_TABLE_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_VERSION_RE = re.compile(r"""^\s*version\s*=\s*["']([^"']+)["']\s*$""")
_MODULE_RE = re.compile(r"^\s*module\s+(\S+)\s*$")
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")


def read_package_version(pyproject: Path | None = None) -> str:
    """Return ``[project].version`` from *pyproject*.

    Only the ``[project]`` table is considered, so a ``version`` key in any
    other table (a tool section, a dependency group) cannot be picked up by
    accident.

    The path resolves at CALL time. A ``Path = PYPROJECT`` default would bake
    the module-level value into the signature, so redirecting it — which is
    how the positive controls prove these checks can fail — would silently
    keep reading the real tree and report success.
    """
    resolved = PYPROJECT if pyproject is None else pyproject
    table = ""
    for line in resolved.read_text(encoding="utf-8").splitlines():
        table_match = _TABLE_RE.match(line)
        if table_match:
            table = table_match.group(1).strip()
            continue
        if table != "project":
            continue
        version_match = _VERSION_RE.match(line)
        if version_match:
            return version_match.group(1)
    raise ValueError(f"no [project] version found in {resolved}")


def major_of(version: str) -> int:
    """Major component of a semver-shaped *version*."""
    match = _SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"version {version!r} is not MAJOR.MINOR.PATCH")
    return int(match.group(1))


def go_module_path(version: str) -> str:
    """Module path the Go client must declare to be installable at *version*.

    v0 and v1 take no suffix; v2+ take ``/vN``. This is a Go requirement, not
    a convention: the proxy refuses a mismatched pair.
    """
    major = major_of(version)
    return GO_MODULE_BASE if major < 2 else f"{GO_MODULE_BASE}/v{major}"


def go_tag(version: str) -> str:
    """The git tag that publishes the Go client at *version*."""
    return f"{GO_TAG_PREFIX}/v{version}"


def npm_version(version: str) -> str:
    """Version the npm package is published under.

    The client speaks one server's REST contract, so it carries that server's
    version. ``sdk/release/pack_js.py`` stamps this into the staged manifest at
    pack time; the manifest on disk is deliberately NOT the authority, so a
    release cannot ship a number somebody forgot to edit.
    """
    return version


def read_go_module_path(go_mod: Path | None = None) -> str:
    """Return the ``module`` path declared in *go_mod*.

    Resolves at call time — see :func:`read_package_version`.
    """
    resolved = GO_MOD if go_mod is None else go_mod
    for line in resolved.read_text(encoding="utf-8").splitlines():
        match = _MODULE_RE.match(line)
        if match:
            return match.group(1)
    raise ValueError(f"no module directive found in {resolved}")


def problems(version: str | None = None) -> list[str]:
    """Return every mismatch between the derived identifiers and the tree.

    Empty list means a release could be tagged today without producing an
    unresolvable module.
    """
    resolved = version if version is not None else read_package_version()
    found: list[str] = []

    expected_module = go_module_path(resolved)
    actual_module = read_go_module_path()
    if actual_module != expected_module:
        found.append(
            f"sdk/go/go.mod declares module {actual_module!r} but version {resolved} "
            f"requires {expected_module!r}; the tag {go_tag(resolved)!r} would not resolve"
        )
    return found


def summary(version: str | None = None) -> dict[str, str]:
    """Every derived identifier, for humans and for the release checklist."""
    resolved = version if version is not None else read_package_version()
    return {
        "package_version": resolved,
        "go_module_path": go_module_path(resolved),
        "go_tag": go_tag(resolved),
        "npm_version": npm_version(resolved),
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive / check SDK release identifiers")
    parser.add_argument("--check", action="store_true", help="exit 1 when the tree disagrees with the derivation")
    args = parser.parse_args(argv)

    for key, value in summary().items():
        print(f"{key}: {value}")

    if not args.check:
        return 0

    found = problems()
    for problem in found:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    raise SystemExit(_main())

"""Version consistency checker for mind-mem.

Verifies that version strings in pyproject.toml, __init__.py, and
CHANGELOG.md are all in sync.

With ``--expect VERSION`` the supplied string joins the set as a fourth
source, so the release workflow can require the git tag to agree with the
in-tree versions instead of trusting them on their own. That leg exists
because the tag is the only one of the four a reader cannot see from the
working tree, and it is the one the index and the GitHub Release are named
after: a tag that disagrees with pyproject.toml produces a release whose
filename and whose contents claim different versions.

Usage:
    python3 -m mind_mem.check_version
    python3 -m mind_mem.check_version --expect 5.2.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None  # type: ignore[assignment]


def get_pyproject_version() -> str | None:
    """Read version from pyproject.toml."""
    try:
        if tomllib is not None:
            with open("pyproject.toml", "rb") as f:
                data = tomllib.load(f)
            version = data.get("project", {}).get("version")
            return str(version) if version is not None else None
        # Regex fallback for Python <3.11 (no tomllib)
        content = Path("pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^\[project\].*?^version\s*=\s*"([^"]+)"', content, re.MULTILINE | re.DOTALL)
        return m.group(1) if m else None
    except FileNotFoundError:
        return None


def get_init_version() -> str | None:
    """Read __version__ from src/mind_mem/__init__.py."""
    try:
        content = Path("src/mind_mem/__init__.py").read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        return m.group(1) if m else None
    except FileNotFoundError:
        return None


def get_changelog_version() -> str | None:
    """Read latest version header from CHANGELOG.md."""
    try:
        content = Path("CHANGELOG.md").read_text(encoding="utf-8")
        m = re.search(r"##\s+\[?v?(\d+\.\d+\.\d+(?:[a-zA-Z0-9.+_-]*)?)", content)
        return m.group(1) if m else None
    except FileNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    """Check version consistency across project files (and an expected version).

    ``argv`` defaults to no arguments rather than to ``sys.argv[1:]``: callers
    that invoke ``main()`` programmatically (the existing gate tests do) must
    not inherit the enclosing process's command line. The console entry point
    below passes the real arguments explicitly.
    """
    args = list(argv if argv is not None else [])
    expected: str | None = None
    while args:
        arg = args.pop(0)
        if arg == "--expect":
            if not args:
                print("ERROR: --expect requires a version argument")
                return 1
            expected = args.pop(0).strip()
            # An empty --expect must not silently degrade to "no tag leg":
            # the release workflow passes a shell variable here, and an
            # unset variable would otherwise remove the gate it added.
            if not expected:
                print("ERROR: --expect was given an empty version string")
                return 1
        else:
            print(f"ERROR: unknown argument {arg!r} (expected: --expect VERSION)")
            return 1

    versions: dict[str, str | None] = {
        "pyproject.toml": get_pyproject_version(),
        "src/mind_mem/__init__.py": get_init_version(),
        "CHANGELOG.md": get_changelog_version(),
    }
    if expected is not None:
        versions["--expect (release tag)"] = expected

    print("Version check:")
    for source, ver in versions.items():
        status = ver or "NOT FOUND"
        print(f"  {source}: {status}")

    # Every source must resolve. Comparing only the readings that
    # happened to work means two unreadable sources leave one version in
    # the set and the mismatch branch can never fire — the checker would
    # report consensus from a single reading.
    missing = [source for source, ver in versions.items() if ver is None]
    if missing:
        print(f"\nERROR: No version string found in: {', '.join(missing)}")
        print("(run from the repository root — paths are resolved against the working directory)")
        return 1

    found = {v for v in versions.values() if v is not None}
    if len(found) > 1:
        print(f"\nERROR: Version mismatch: {found}")
        return 1

    print(f"\nOK: All versions consistent ({found.pop()})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

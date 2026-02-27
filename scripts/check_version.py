"""Version consistency checker for mind-mem.

Verifies that version strings in pyproject.toml, __init__.py, and
CHANGELOG.md are all in sync.

Usage:
    python3 scripts/check_version.py
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
        content = Path("pyproject.toml").read_text()
        m = re.search(r'^\[project\].*?^version\s*=\s*"([^"]+)"', content, re.MULTILINE | re.DOTALL)
        return m.group(1) if m else None
    except FileNotFoundError:
        return None


def get_init_version() -> str | None:
    """Read __version__ from scripts/__init__.py."""
    try:
        content = Path("scripts/__init__.py").read_text()
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        return m.group(1) if m else None
    except FileNotFoundError:
        return None


def get_changelog_version() -> str | None:
    """Read latest version header from CHANGELOG.md."""
    try:
        content = Path("CHANGELOG.md").read_text()
        m = re.search(r"##\s+\[?v?(\d+\.\d+\.\d+)", content)
        return m.group(1) if m else None
    except FileNotFoundError:
        return None


def main() -> int:
    """Check version consistency across project files."""
    versions: dict[str, str | None] = {
        "pyproject.toml": get_pyproject_version(),
        "scripts/__init__.py": get_init_version(),
        "CHANGELOG.md": get_changelog_version(),
    }

    print("Version check:")
    for source, ver in versions.items():
        status = ver or "NOT FOUND"
        print(f"  {source}: {status}")

    found = {v for v in versions.values() if v is not None}
    if len(found) == 0:
        print("\nERROR: No version strings found")
        return 1
    if len(found) > 1:
        print(f"\nERROR: Version mismatch: {found}")
        return 1

    print(f"\nOK: All versions consistent ({found.pop()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

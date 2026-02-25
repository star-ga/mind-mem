"""Tests for version consistency checker."""
import re

from scripts.check_version import (
    get_changelog_version,
    get_init_version,
    get_pyproject_version,
)


class TestVersionReaders:
    def test_pyproject_version_exists(self):
        ver = get_pyproject_version()
        assert ver is not None, "pyproject.toml should have a version"
        assert re.match(r"\d+\.\d+\.\d+", ver), f"Invalid version format: {ver}"

    def test_init_version_exists(self):
        ver = get_init_version()
        assert ver is not None, "__init__.py should have __version__"

    def test_changelog_version_exists(self):
        ver = get_changelog_version()
        assert ver is not None, "CHANGELOG.md should have a version header"

    def test_versions_match(self):
        pyproject = get_pyproject_version()
        init = get_init_version()
        changelog = get_changelog_version()
        found = {v for v in [pyproject, init, changelog] if v is not None}
        assert len(found) == 1, f"Version mismatch: pyproject={pyproject}, init={init}, changelog={changelog}"

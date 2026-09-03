# Copyright 2026 STARGA, Inc.
"""The Go module path must be derivable from the package version, not typed.

Roadmap RM-0089 / RM-2322 ("publish the Go client as a Go module") were filed
as publish-only items — source and tests already in tree, "only the publish
step is open". The publish step could not have worked. ``sdk/go/go.mod``
declared::

    module github.com/star-ga/mind-mem/sdk/go

A subdirectory module is published by pushing a tag prefixed with its
directory, so this repository's 5.x line publishes the client as
``sdk/go/v5.0.2``. Go refuses a v2-or-higher version for a module path with no
matching ``/vN`` suffix, so that tag would have resolved to nothing and
``go get`` would have failed for every consumer with a mismatch error. The
whole failure lands after the tag is pushed — and a pushed tag on a public
repository cannot be recalled from the module proxy.

So the suffix is now derived from the package version rather than written by
hand, and this module is the ratchet: when the package goes to 6.0.0 and
``go.mod`` still says ``/v5``, CI goes red here instead of the Go proxy going
red for a user.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from mind_mem import __version__ as PACKAGE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_MODULE = REPO_ROOT / "sdk" / "release" / "version.py"
GO_MOD = REPO_ROOT / "sdk" / "go" / "go.mod"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def version_module() -> Any:
    return _load(VERSION_MODULE, "mind_mem_sdk_version")


class TestVersionSource:
    def test_pyproject_version_matches_the_package(self, version_module: Any) -> None:
        # The derivation reads pyproject with a small table-aware scan rather
        # than tomllib (3.11+, and the CI matrix runs 3.10). This pins that
        # scan to the same answer the package itself reports, so it cannot
        # drift into reading some other table's `version` key.
        assert version_module.read_package_version() == PACKAGE_VERSION

    def test_reader_ignores_a_version_outside_the_project_table(self, version_module: Any, tmp_path: Path) -> None:
        # Positive control for "only [project] counts": a decoy first.
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.decoy]\nversion = "9.9.9"\n\n[project]\nname = "x"\nversion = "1.2.3"\n',
            encoding="utf-8",
        )
        assert version_module.read_package_version(pyproject) == "1.2.3"

    def test_reader_raises_when_there_is_no_project_version(self, version_module: Any, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.decoy]\nversion = "9.9.9"\n', encoding="utf-8")
        with pytest.raises(ValueError):
            version_module.read_package_version(pyproject)


class TestGoModulePathDerivation:
    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("0.1.0", "github.com/star-ga/mind-mem/sdk/go"),
            ("1.4.2", "github.com/star-ga/mind-mem/sdk/go"),
            ("2.0.0", "github.com/star-ga/mind-mem/sdk/go/v2"),
            ("5.0.1", "github.com/star-ga/mind-mem/sdk/go/v5"),
            ("5.0.2", "github.com/star-ga/mind-mem/sdk/go/v5"),
            ("11.3.0", "github.com/star-ga/mind-mem/sdk/go/v11"),
        ],
    )
    def test_suffix_appears_at_v2_and_above(self, version_module: Any, version: str, expected: str) -> None:
        assert version_module.go_module_path(version) == expected

    def test_tag_is_prefixed_with_the_module_directory(self, version_module: Any) -> None:
        assert version_module.go_tag("5.0.2") == "sdk/go/v5.0.2"

    def test_a_non_semver_version_is_refused(self, version_module: Any) -> None:
        with pytest.raises(ValueError):
            version_module.go_module_path("not-a-version")


class TestTreeMatchesTheDerivation:
    def test_go_mod_declares_the_derived_module_path(self, version_module: Any) -> None:
        assert version_module.read_go_module_path() == version_module.go_module_path(PACKAGE_VERSION)

    def test_no_problems_reported(self, version_module: Any) -> None:
        assert version_module.problems() == []

    def test_check_cli_exits_zero(self, version_module: Any) -> None:
        assert version_module._main(["--check"]) == 0

    def test_positive_control_a_stale_module_path_is_caught(self, version_module: Any, tmp_path: Path, monkeypatch: Any) -> None:
        # Reproduce exactly the state the tree was in: the 5.x package with an
        # unsuffixed module path. The gate must refuse it, and must name the
        # tag that would not have resolved.
        stale = tmp_path / "go.mod"
        stale.write_text("module github.com/star-ga/mind-mem/sdk/go\n\ngo 1.21\n", encoding="utf-8")
        monkeypatch.setattr(version_module, "GO_MOD", stale)

        found = version_module.problems()
        assert found, "the pre-fix go.mod passed the gate — the gate proves nothing"
        assert "sdk/go/v" in found[0]
        assert version_module._main(["--check"]) == 1

    def test_positive_control_a_next_major_is_caught(self, version_module: Any, tmp_path: Path, monkeypatch: Any) -> None:
        # The forward-looking half: today's go.mod against tomorrow's major.
        assert version_module.problems("6.0.0"), "a 6.x package with a /v5 module path passed the gate"

    def test_summary_lists_every_derived_identifier(self, version_module: Any) -> None:
        summary = version_module.summary()
        assert summary == {
            "package_version": PACKAGE_VERSION,
            "go_module_path": f"github.com/star-ga/mind-mem/sdk/go/v{PACKAGE_VERSION.split('.')[0]}",
            "go_tag": f"sdk/go/v{PACKAGE_VERSION}",
            "npm_version": PACKAGE_VERSION,
        }


class TestGoSourcesUseTheDeclaredModulePath:
    def test_the_test_import_matches_go_mod(self, version_module: Any) -> None:
        # A subdirectory module whose own tests import the pre-suffix path
        # compiles locally and breaks for everyone else, so this is checked
        # rather than assumed.
        declared = version_module.read_go_module_path()
        source = (REPO_ROOT / "sdk" / "go" / "client_test.go").read_text(encoding="utf-8")
        assert f'"{declared}"' in source, f"client_test.go does not import {declared}"

    def test_readme_install_line_matches_go_mod(self, version_module: Any) -> None:
        declared = version_module.read_go_module_path()
        readme = (REPO_ROOT / "sdk" / "go" / "README.md").read_text(encoding="utf-8")
        assert f"go get {declared}" in readme, f"sdk/go/README.md does not document `go get {declared}`"

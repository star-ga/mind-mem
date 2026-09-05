"""A package written to another drive must not take down the run.

Windows CI checks the repo out on D: while pytest's tmp_path lives on C:, and
``os.path.relpath`` RAISES across drives rather than returning something
awkward: "path is on mount 'C:', start on mount 'D:'". Seven tests in
test_repro_package.py died that way on all five Windows rows while every Linux
and macOS row passed, because the call sites are only building a human-readable
label for a command hint or a report header.

Reproduced here on any platform by making relpath raise, so the guard is not
dependent on a Windows runner to stay honest.
"""

import os
from unittest import mock

import pytest

from benchmarks.repro_manifest import display_path

CROSS_DRIVE = ValueError("path is on mount 'C:', start on mount 'D:'")


class TestDisplayPathSurvivesCrossDrive:
    def test_it_returns_an_absolute_path_instead_of_raising(self) -> None:
        with mock.patch("os.path.relpath", side_effect=CROSS_DRIVE):
            got = display_path(os.path.join(os.sep, "tmp", "pkg"), os.path.join(os.sep, "repo"))
        assert got.endswith("/tmp/pkg"), got
        assert "\\" not in got

    def test_it_does_not_swallow_the_normal_case(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert display_path(os.path.join(root, "benchmarks", "repro"), root) == "benchmarks/repro"

    def test_separators_are_normalised_either_way(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert "\\" not in display_path(os.path.join(root, "a", "b"), root)

    @pytest.mark.parametrize("bad", [ValueError("cross-drive"), CROSS_DRIVE])
    def test_any_valueerror_is_handled_not_just_the_windows_wording(self, bad: ValueError) -> None:
        with mock.patch("os.path.relpath", side_effect=bad):
            assert display_path("/x/y", "/z")


class TestTheVerifierItselfSurvives:
    def test_verify_reports_a_package_on_another_drive(self, tmp_path) -> None:
        """The end-to-end shape of the CI failure: verify a package whose path
        cannot be made relative to the repo root."""
        from benchmarks import repro_verify

        with mock.patch("os.path.relpath", side_effect=CROSS_DRIVE):
            # Must fail on CONTENT (no such package), never on path arithmetic.
            rc = repro_verify.main(["package", str(tmp_path / "nonexistent")])
        assert rc != 0

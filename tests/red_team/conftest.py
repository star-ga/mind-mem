"""pytest configuration for the red_team test package.

Registers the ``--petri-limit`` CLI option so the CI workflow can control
per-seed sample count without modifying source.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--petri-limit",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of samples per Petri seed (default: 5).",
    )


@pytest.fixture
def petri_limit(request: pytest.FixtureRequest) -> int:
    """Return the --petri-limit value for use in behavioral audit tests."""
    return int(request.config.getoption("--petri-limit"))
